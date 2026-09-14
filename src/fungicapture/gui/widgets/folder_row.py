"""
The folder chooser that sits at the top of every tab.

Why every tab has its own
-------------------------
The first version assumed one run from start to finish inside one project
folder. That was wrong: if you already have colony crops - from an earlier
session, or from another tool entirely - you should be able to point the
segmentation step at them and start there. Forcing the whole pipeline to run
again to produce files you already have is wasted hours.

So each step names its own input and output folder. They default to the
project's own sub-folders, so nothing changes for someone running straight
through, and the "Default" tag shows at a glance when a folder has been
pointed elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class FolderField(QWidget):
    """One labelled folder box with Browse and Default buttons."""

    changed = Signal()

    def __init__(
        self,
        label: str,
        *,
        tooltip: str = "",
        pick_file: bool = False,
        file_filter: str = "All files (*)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._default: Path | None = None
        self._pick_file = pick_file
        self._file_filter = file_filter

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)

        self.label = QLabel(label)
        self.label.setMinimumWidth(96)
        if tooltip:
            self.label.setToolTip(tooltip)
        layout.addWidget(self.label, 0, 0)

        self.field = QLineEdit()
        self.field.setPlaceholderText("(project default)")
        self.field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if tooltip:
            self.field.setToolTip(tooltip)
        self.field.editingFinished.connect(self.changed.emit)
        layout.addWidget(self.field, 0, 1)

        self.tag = QLabel("Default")
        self.tag.setStyleSheet("color:#777; font-size:11px;")
        layout.addWidget(self.tag, 0, 2)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout.addWidget(browse, 0, 3)

        reset = QPushButton("Default")
        reset.setToolTip("Go back to this project's own folder")
        reset.clicked.connect(self.reset_to_default)
        layout.addWidget(reset, 0, 4)

        layout.setColumnStretch(1, 1)
        self.field.textChanged.connect(self._refresh_tag)

    # ---------------- values ----------------

    def set_default(self, path: str | Path | None) -> None:
        """The project's own folder, used when the box is left empty."""
        self._default = Path(path) if path else None
        self._refresh_tag()

    def value(self) -> Path | None:
        """The chosen folder, or None to mean "use the default"."""
        text = self.field.text().strip()
        return Path(text).expanduser() if text else None

    def effective(self) -> Path | None:
        """What will actually be used."""
        return self.value() or self._default

    def set_value(self, path: str | Path | None) -> None:
        self.field.setText(str(path) if path else "")
        self._refresh_tag()

    def reset_to_default(self) -> None:
        self.field.clear()
        self._refresh_tag()
        self.changed.emit()

    # ---------------- display ----------------

    def _refresh_tag(self) -> None:
        chosen = self.value()
        if chosen is None:
            self.tag.setText("Default")
            self.tag.setStyleSheet("color:#777; font-size:11px;")
            self.field.setPlaceholderText(
                str(self._default) if self._default else "(project default)"
            )
            return

        if chosen.exists():
            self.tag.setText("Custom")
            self.tag.setStyleSheet("color:#1a6; font-size:11px;")
        else:
            # A folder that does not exist yet is fine for an output, and a
            # mistake for an input. Saying "will be created" rather than
            # showing an error keeps both cases honest.
            self.tag.setText("will be created")
            self.tag.setStyleSheet("color:#a70; font-size:11px;")

    def _browse(self) -> None:
        start = str(self.effective() or Path.home())
        if self._pick_file:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.label.text().rstrip(':')}", start,
                self._file_filter,
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {self.label.text().rstrip(':')}", start
            )
        if path:
            self.set_value(path)
            self.changed.emit()


class FolderPanel(QFrame):
    """
    The input/output block at the top of a tab.

    Also shows how many files the input folder currently holds, so a wrong
    folder is obvious before a long job is started rather than after it.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "FolderPanel { background:#f7f7f9; border:1px solid #dcdce2;"
            " border-radius:6px; }"
        )
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setVerticalSpacing(5)
        self._fields: dict[str, FolderField] = {}
        self._row = 0

        self.status = QLabel("")
        self.status.setStyleSheet("color:#555; font-size:11px;")
        self.status.setWordWrap(True)

    def add(
        self,
        key: str,
        label: str,
        *,
        tooltip: str = "",
        pick_file: bool = False,
        file_filter: str = "All files (*)",
    ) -> FolderField:
        field = FolderField(
            label, tooltip=tooltip, pick_file=pick_file, file_filter=file_filter
        )
        field.changed.connect(self._on_changed)
        self._layout.addWidget(field, self._row, 0)
        self._fields[key] = field
        self._row += 1
        return field

    def finish(self) -> None:
        """Call once after adding every field, to place the status line."""
        self._layout.addWidget(self.status, self._row, 0)
        self._row += 1

    def __getitem__(self, key: str) -> FolderField:
        return self._fields[key]

    def get(self, key: str) -> FolderField | None:
        return self._fields.get(key)

    def value(self, key: str) -> Path | None:
        field = self._fields.get(key)
        return field.value() if field else None

    def _on_changed(self) -> None:
        self.refresh_status()
        self.changed.emit()

    def refresh_status(self, counter=None) -> None:
        """Show what is in the input folder. ``counter`` returns a message."""
        if counter is not None:
            self._counter = counter
        counter = getattr(self, "_counter", None)
        if counter is None:
            return
        try:
            self.status.setText(counter())
        except Exception as error:  # pragma: no cover - never break the tab
            self.status.setText(f"Could not read the folder: {error}")


def count_images(folder: Path | None, suffixes=None) -> int:
    """How many images a folder holds, searching sub-folders too."""
    if folder is None or not Path(folder).exists():
        return 0
    suffixes = suffixes or {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sum(
        1
        for p in Path(folder).rglob("*")
        if p.is_file()
        and p.suffix.lower() in suffixes
        and "_overlay" not in p.stem
    )

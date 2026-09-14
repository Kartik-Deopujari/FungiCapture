"""
Tab 4 - Validate: look at every colony, mark pass or fail, export a PDF.

Replaces overlay_validator.py, which wrote 96 PNG files that a user then had to
open one by one. Here you click through them, and your verdict is recorded.

The verdict matters: a colony marked 'fail' is excluded from phenotype scoring
with the reason you typed attached, rather than being silently dropped or
silently kept.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.project import Project
from ...core.validate import VerdictStore
from ..widgets.folder_row import FolderPanel

STATUS_COLOURS = {"pass": "#228833", "fail": "#cc3311", "unreviewed": "#888888"}


class ValidateTab(QWidget):
    """Browse QC panels, mark verdicts, export reports."""

    request_run = Signal(str, dict)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.verdicts: VerdictStore | None = None
        self._pairs: list[tuple[Path, Path]] = []
        self._features = None
        self._current = -1

        outer = QVBoxLayout(self)

        # Like every other tab, this one names its own folders, so you can
        # review colonies and masks produced elsewhere.
        self.folders = FolderPanel()
        self.folders.add("crops", "Colony images:")
        self.folders.add("masks", "Masks:")
        self.folders.add("vectors", "Feature table:",
                         tooltip="Used to fill in the numbers beside each panel.")
        self.folders.add("qc", "Save reports to:")
        self.folders.finish()
        self.folders.refresh_status(self._count)
        self.folders.changed.connect(self._folders_changed)
        outer.addWidget(self.folders)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # ---- left: colony list ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show:"))
        self.filter = QComboBox()
        self.filter.addItems(["all", "unreviewed", "pass", "fail"])
        self.filter.currentTextChanged.connect(lambda _: self.refresh_list())
        filter_row.addWidget(self.filter, 1)
        left_layout.addLayout(filter_row)

        self.colony_list = QListWidget()
        self.colony_list.currentRowChanged.connect(self._select)
        left_layout.addWidget(self.colony_list, 1)

        self.counts = QLabel("")
        left_layout.addWidget(self.counts)

        verdict_box = QGroupBox("Verdict")
        verdict_layout = QVBoxLayout(verdict_box)
        buttons = QHBoxLayout()
        pass_button = QPushButton("✓ Pass")
        pass_button.clicked.connect(lambda: self._set_verdict("pass"))
        fail_button = QPushButton("✗ Fail")
        fail_button.clicked.connect(lambda: self._set_verdict("fail"))
        buttons.addWidget(pass_button)
        buttons.addWidget(fail_button)
        verdict_layout.addLayout(buttons)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Why? (kept with the record)")
        verdict_layout.addWidget(self.note)
        left_layout.addWidget(verdict_box)

        navigation = QHBoxLayout()
        previous = QPushButton("◀ Previous")
        previous.clicked.connect(lambda: self._step(-1))
        following = QPushButton("Next ▶")
        following.clicked.connect(lambda: self._step(1))
        navigation.addWidget(previous)
        navigation.addWidget(following)
        left_layout.addLayout(navigation)

        guide_button = QPushButton("❔ What do these features mean?")
        guide_button.setToolTip(
            "A short description of every measured feature, with a search box."
        )
        guide_button.clicked.connect(self._open_feature_guide)
        left_layout.addWidget(guide_button)

        report_box = QGroupBox("Export PDF report")
        report_layout = QVBoxLayout(report_box)
        for label, mode, tip in (
            ("Full report", "full",
             "One page per colony plus a summary — for supplementary material"),
            ("Failures only", "failures",
             "Just the problems — short enough to email"),
            ("Contact sheet", "contact_sheet",
             "24 thumbnails per page — for scanning a plate at a glance"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _, m=mode: self._export_report(m))
            report_layout.addWidget(button)
        left_layout.addWidget(report_box)

        splitter.addWidget(left)

        # ---- right: the panel ----
        right = QScrollArea()
        right.setWidgetResizable(True)
        self.panel_label = QLabel("Run the feature step, then come back here.")
        self.panel_label.setAlignment(Qt.AlignCenter)
        self.panel_label.setStyleSheet("background:#ffffff;")
        right.setWidget(self.panel_label)
        splitter.addWidget(right)
        splitter.setSizes([300, 1000])

    # ---------------- project ----------------

    def _count(self) -> str:
        return f"{len(self._pairs)} colony/mask pair(s) loaded"

    def _folders_changed(self) -> None:
        if self.project is None:
            return
        for key in ("crops", "masks", "vectors", "qc"):
            self.project.set_folder(key, self.folders[key].value())
        self.project.save()
        self.reload()

    def set_project(self, project: Project) -> None:
        self.project = project
        for key in ("crops", "masks", "vectors", "qc"):
            self.folders[key].set_default(project.root / project.DEFAULT_FOLDERS[key])
            self.folders[key].set_value(project.folder_overrides.get(key))
        self.verdicts = VerdictStore(project.qc_dir / VerdictStore.FILENAME)
        self.reload()

    def reload(self) -> None:
        """Re-scan for colonies. Called after the feature step finishes."""
        if self.project is None:
            return
        from ...core.pipeline import pair_crops_and_masks

        self._pairs = pair_crops_and_masks(
            self.project,
            self.folders["crops"].effective(),
            self.folders["masks"].effective(),
        )

        vectors = self.folders["vectors"].effective() or self.project.vectors_dir
        master = Path(vectors)
        if master.is_dir():
            master = master / "all_colony_features.csv"
        if master.exists():
            import pandas as pd

            self._features = pd.read_csv(master)
        self.folders.refresh_status()
        self.refresh_list()

    def refresh_list(self) -> None:
        from ...core.features.extract import normalise_stem

        if self.verdicts is None:
            return
        wanted = self.filter.currentText()
        self.colony_list.clear()

        for crop_path, _ in self._pairs:
            key = normalise_stem(crop_path.stem)
            status = self.verdicts.get(key).status
            if wanted != "all" and status != wanted:
                continue
            item = QListWidgetItem(f"{key}")
            item.setForeground(Qt.GlobalColor.black)
            item.setData(Qt.UserRole, str(crop_path))
            item.setToolTip(f"status: {status}")
            self.colony_list.addItem(item)

        counts = self.verdicts.status_counts()
        reviewed = counts.get("pass", 0) + counts.get("fail", 0)
        self.counts.setText(
            f"{len(self._pairs)} colonies   ✓ {counts.get('pass', 0)}   "
            f"✗ {counts.get('fail', 0)}   unreviewed {len(self._pairs) - reviewed}"
        )
        if self.colony_list.count():
            self.colony_list.setCurrentRow(0)

    # ---------------- panel ----------------

    def _select(self, row: int) -> None:
        if row < 0 or self.project is None:
            return
        item = self.colony_list.item(row)
        if item is None:
            return
        crop_path = Path(item.data(Qt.UserRole))
        mask_path = next(
            (m for c, m in self._pairs if c == crop_path), None
        )
        if mask_path is None:
            return
        self._current = row
        self._render(crop_path, mask_path)

    def _render(self, crop_path: Path, mask_path: Path) -> None:
        """Build the QC panel and show it.

        Rendered to a temporary PNG rather than embedded live: matplotlib's Qt
        backend and PySide6 can disagree about which event loop owns the
        figure, and a static image cannot crash the window.
        """
        import tempfile

        from ...core.features.extract import normalise_stem
        from ...core.validate import build_panel_data, make_panel, save_panel

        assert self.project is not None
        key = normalise_stem(crop_path.stem)

        values: dict = {}
        if self._features is not None and "base_key" in self._features.columns:
            match = self._features[self._features["base_key"] == key]
            if not match.empty:
                values = match.iloc[0].to_dict()

        try:
            data = build_panel_data(
                crop_path, mask_path, values, self.project.params
            )
            figure = make_panel(data, self.project.params, figsize=(15, 8.5))
            temporary = Path(tempfile.gettempdir()) / f"fungicapture_qc_{key}.png"
            save_panel(figure, temporary, dpi=100)
        except Exception as error:
            self.panel_label.setText(f"Could not build the panel:\n{error}")
            return

        from PySide6.QtGui import QPixmap

        self.panel_label.setPixmap(QPixmap(str(temporary)))
        verdict = self.verdicts.get(key) if self.verdicts else None
        self.note.setText(verdict.note if verdict else "")

    def _step(self, direction: int) -> None:
        row = self.colony_list.currentRow() + direction
        if 0 <= row < self.colony_list.count():
            self.colony_list.setCurrentRow(row)

    def _set_verdict(self, status: str) -> None:
        from ...core.features.extract import normalise_stem

        item = self.colony_list.currentItem()
        if item is None or self.verdicts is None:
            return
        key = normalise_stem(Path(item.data(Qt.UserRole)).stem)
        self.verdicts.set(key, status, self.note.text())
        self.verdicts.save()
        self.status_message.emit(f"{key} marked {status}")

        colour = STATUS_COLOURS.get(status, "#888888")
        item.setToolTip(f"status: {status}")
        from PySide6.QtGui import QColor

        item.setForeground(QColor(colour))

        counts = self.verdicts.status_counts()
        reviewed = counts.get("pass", 0) + counts.get("fail", 0)
        self.counts.setText(
            f"{len(self._pairs)} colonies   ✓ {counts.get('pass', 0)}   "
            f"✗ {counts.get('fail', 0)}   unreviewed {len(self._pairs) - reviewed}"
        )
        self._step(1)

    # ---------------- feature guide ----------------

    def _open_feature_guide(self) -> None:
        from ..widgets.feature_guide import FeatureGuideDialog

        columns = None
        if self._features is not None:
            columns = list(self._features.columns)
        FeatureGuideDialog(columns, self).exec()

    # ---------------- reports ----------------

    def _export_report(self, mode: str) -> None:
        if self.project is None or not self._pairs:
            QMessageBox.information(
                self, "Nothing to report",
                "No colonies found. Run cropping, segmentation and features first.",
            )
            return

        qc_dir = self.folders["qc"].effective() or self.project.qc_dir
        Path(qc_dir).mkdir(parents=True, exist_ok=True)
        default = Path(qc_dir) / f"qc_report_{mode}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF report", str(default), "PDF files (*.pdf)"
        )
        if path:
            self.request_run.emit(
                "report", {"path": path, "mode": mode, "pairs": self._pairs}
            )

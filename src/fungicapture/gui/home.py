"""
The home screen: one question before anything else.

**What does one photograph contain?**

Everything downstream branches on the answer - whether a grid is needed, where
names come from, whether a run is one image or a folder. Asking once, at the
start, is far better than letting a user discover halfway through that the tool
assumed the wrong thing.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.project import ImagingMode, PlateLayout, Project

RECENTS_FILENAME = "recent_projects.json"
MAX_RECENTS = 8


def recents_path() -> Path:
    from platformdirs import user_config_dir

    path = Path(user_config_dir("FungiCapture", "FungiCapture"))
    path.mkdir(parents=True, exist_ok=True)
    return path / RECENTS_FILENAME


def load_recents() -> list[dict]:
    path = recents_path()
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    # Drop projects that have been moved or deleted, so the list never offers
    # something that cannot open.
    return [e for e in entries if Path(e.get("root", "")).exists()]


def remember_project(project: Project) -> None:
    entries = [e for e in load_recents() if e.get("root") != str(project.root)]
    entries.insert(
        0,
        {
            "root": str(project.root),
            "name": project.name,
            "mode": project.mode.value,
            "mode_label": project.mode.label,
        },
    )
    try:
        recents_path().write_text(
            json.dumps(entries[:MAX_RECENTS], indent=2), encoding="utf-8"
        )
    except OSError:
        pass


# --------------------------------------------------------------------------
# Mode card
# --------------------------------------------------------------------------


class ModeCard(QFrame):
    """
    One clickable plate-shape card on the home screen, with a small diagram.

    The user picks a plate *shape* - round or rectangular/square - and, with a
    separate checkbox, whether a whole folder is processed. Those two choices
    map to the four imaging modes underneath (see ``ImagingMode.for_shape``).
    """

    clicked = Signal(bool)  # the card's ``round_plate`` value

    def __init__(self, round_plate: bool, parent=None):
        super().__init__(parent)
        self.round_plate = round_plate
        self.selected = False
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(250, 200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Round plates" if round_plate else "Rectangular / square plates")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        self._diagram = QWidget()
        self._diagram.setMinimumHeight(76)
        self._diagram.paintEvent = lambda event: self._paint_diagram(event)
        layout.addWidget(self._diagram, 1)

        description = QLabel(
            "Petri dishes / round plates. Each photograph holds one colony; the "
            "dish rim is found automatically — no grid to place."
            if round_plate else
            "Large rectangular or square plates holding many colonies in a grid. "
            "Place a grid or boxes once and cut every well."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)
        self._apply_style()

    def _paint_diagram(self, event) -> None:
        """A tiny picture of the plate shape - faster to read than words."""
        painter = QPainter(self._diagram)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self._diagram.rect()
        painter.setPen(QPen(QColor("#888"), 1.4))
        colony = QColor("#5a4632")
        cx, cy = rect.center().x(), rect.center().y()

        if self.round_plate:
            r = 30
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
            painter.setBrush(colony)
            painter.drawEllipse(
                int(cx - r * 0.32), int(cy - r * 0.32),
                int(r * 0.64), int(r * 0.64),
            )
        else:
            cols, rows = 4, 3
            width, height = 108, 62
            left, top = cx - width / 2, cy - height / 2
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(left), int(top), int(width), int(height))
            painter.setBrush(colony)
            for r in range(rows):
                for c in range(cols):
                    x = left + width * (c + 0.5) / cols
                    y = top + height * (r + 0.5) / rows
                    painter.drawEllipse(int(x - 3.5), int(y - 3.5), 7, 7)
        painter.end()

    def _apply_style(self) -> None:
        if self.selected:
            self.setStyleSheet(
                "ModeCard { border: 2px solid #2a7de1; border-radius: 8px;"
                " background: #eef5fd; }"
            )
        else:
            self.setStyleSheet(
                "ModeCard { border: 1px solid #ccc; border-radius: 8px;"
                " background: #fafafa; }"
            )

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._apply_style()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.round_plate)
        super().mousePressEvent(event)


# --------------------------------------------------------------------------
# Colour calibration prompt
# --------------------------------------------------------------------------


class ColourCalibrationDialog(QDialog):
    """
    Asked once when a project is created.

    Colour numbers are only comparable between photographs if the lighting was.
    Background normalisation and EXIF checking always run; a reference card is
    the one part that needs a change to how plates are photographed, so it is
    the one part worth asking about.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colour calibration")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        question = QLabel("Do your photographs contain a colour reference card?")
        font = QFont()
        font.setBold(True)
        question.setFont(font)
        layout.addWidget(question)

        self.use_card = QCheckBox("Process images with colour reference card")
        layout.addWidget(self.use_card)

        form = QFormLayout()
        self.card_path = QLineEdit()
        self.card_path.setPlaceholderText("Upload your card definition (.json or .csv)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.card_path)
        row.addWidget(browse)
        container = QWidget()
        container.setLayout(row)
        form.addRow("Card definition:", container)

        self.region = QComboBox()
        self.region.addItems(
            ["auto", "top-left", "top-right", "bottom-left", "bottom-right"]
        )
        form.addRow("Card position:", self.region)
        layout.addLayout(form)

        note = QLabel(
            "Recommended. Without a card, colour values can only be compared "
            "within one imaging session.\n\n"
            "FungiCapture ships no card values of its own — upload a definition "
            "of the card you own, listing each patch and its reference colour. "
            "Any card works.\n\n"
            "Agar background normalisation and EXIF drift checking run either way."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#eef5fd; border:1px solid #b8d4f0; border-radius:6px;"
            " padding:8px; color:#234;"
        )
        layout.addWidget(note)

        buttons = QDialogButtonBox()
        buttons.addButton("Skip for now", QDialogButtonBox.RejectRole)
        buttons.addButton("Continue", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.use_card.toggled.connect(self._toggle)
        self._toggle(False)

    def _toggle(self, checked: bool) -> None:
        self.card_path.setEnabled(checked)
        self.region.setEnabled(checked)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select your colour card definition", "",
            "Card definition (*.json *.csv);;All files (*)",
        )
        if path:
            self.card_path.setText(path)

    def settings(self) -> dict:
        return {
            "use_colour_card": self.use_card.isChecked(),
            "card_definition_path": self.card_path.text().strip() or None,
            "card_search_region": self.region.currentText(),
        }


# --------------------------------------------------------------------------
# New project dialog
# --------------------------------------------------------------------------


class NewProjectDialog(QDialog):
    """Name, location, images and plate size for a new project."""

    def __init__(self, mode: ImagingMode, parent=None):
        super().__init__(parent)
        self.mode = mode
        batch_suffix = " · batch" if mode.is_batch else ""
        self.setWindowTitle(f"New project — {mode.shape_label}{batch_suffix}")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name = QLineEdit("my_project")
        form.addRow("Project name:", self.name)

        self.location = QLineEdit(str(Path.home() / "FungiCapture"))
        location_browse = QPushButton("Browse…")
        location_browse.clicked.connect(
            lambda: self._pick_folder(self.location, "Where to keep the project")
        )
        form.addRow("Project folder:", self._row(self.location, location_browse))

        self.images = QLineEdit()
        self.images.setPlaceholderText("Folder holding your photographs")
        images_browse = QPushButton("Browse…")
        images_browse.clicked.connect(
            lambda: self._pick_folder(self.images, "Folder with your photographs")
        )
        form.addRow("Images:", self._row(self.images, images_browse))

        self.rows = QSpinBox()
        self.rows.setRange(1, 64)
        self.rows.setValue(8)
        self.cols = QSpinBox()
        self.cols.setRange(1, 64)
        self.cols.setValue(12)
        plate_row = QHBoxLayout()
        plate_row.addWidget(QLabel("rows"))
        plate_row.addWidget(self.rows)
        plate_row.addWidget(QLabel("columns"))
        plate_row.addWidget(self.cols)
        plate_row.addStretch()
        plate_widget = QWidget()
        plate_widget.setLayout(plate_row)
        self._plate_widget = plate_widget
        form.addRow("Plate layout:", plate_widget)
        # A dish photograph has no grid, so hide the control rather than show
        # one that does nothing.
        plate_widget.setVisible(mode.needs_grid)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field)
        row.addWidget(button)
        container = QWidget()
        container.setLayout(row)
        return container

    def _pick_folder(self, field: QLineEdit, caption: str) -> None:
        path = QFileDialog.getExistingDirectory(self, caption, field.text() or "")
        if path:
            field.setText(path)

    def _validate_and_accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "Name needed", "Please give the project a name.")
            return
        images = self.images.text().strip()
        if images and not Path(images).expanduser().exists():
            QMessageBox.warning(
                self, "Folder not found",
                f"There is no folder at:\n{images}\n\nPick one that exists.",
            )
            return
        self.accept()

    def create_project(self) -> Project:
        root = Path(self.location.text()).expanduser() / self.name.text().strip()
        project = Project.create(
            root,
            name=self.name.text().strip(),
            mode=self.mode,
            image_dir=self.images.text().strip() or None,
            layout=PlateLayout(rows=self.rows.value(), cols=self.cols.value()),
            exist_ok=True,
        )
        # The growth medium is set in the plate map editor, next to the strain
        # names it will be combined with. Starting from a sensible default
        # keeps the new-project screen to the questions that really must be
        # answered up front.
        project.normalise_media_label()
        project.save()
        return project


# --------------------------------------------------------------------------
# Home screen
# --------------------------------------------------------------------------


class HomeScreen(QWidget):
    """The mode picker and recent-projects list."""

    project_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._round_plate = False  # default: rectangular/square plates
        self._cards: list[ModeCard] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(14)

        title = QLabel("FungiCapture")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        prompt = QLabel("Start a new analysis — what shape are your plates?")
        prompt.setStyleSheet("color:#555; font-size:13px;")
        layout.addWidget(prompt)

        row = QHBoxLayout()
        for round_plate in (False, True):
            card = ModeCard(round_plate)
            card.clicked.connect(self._select_shape)
            self._cards.append(card)
            row.addWidget(card)
        layout.addLayout(row, 1)

        # The one remaining choice: one plate, or a whole folder of them.
        self.batch_check = QCheckBox(
            "Batch process — apply the same layout to every photograph in the folder"
        )
        self.batch_check.setToolTip(
            "On: the layout you set is used to crop every photograph in the "
            "folder, each into its own sub-folder.\n"
            "Off: only the plate you have open is cropped."
        )
        layout.addWidget(self.batch_check)

        layout.addWidget(QLabel("Recent projects"))
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(110)
        self.recent_list.itemDoubleClicked.connect(self._open_recent)
        layout.addWidget(self.recent_list)

        buttons = QHBoxLayout()
        open_button = QPushButton("Open existing project…")
        open_button.clicked.connect(self._open_existing)
        buttons.addWidget(open_button)
        buttons.addStretch()
        self.continue_button = QPushButton("Continue ▶")
        self.continue_button.setDefault(True)
        self.continue_button.clicked.connect(self._create_new)
        buttons.addWidget(self.continue_button)
        layout.addLayout(buttons)

        self._select_shape(self._round_plate)
        self.refresh_recents()

    def refresh_recents(self) -> None:
        self.recent_list.clear()
        for entry in load_recents():
            item = QListWidgetItem(
                f"{entry['name']}    —    {entry.get('mode_label', '')}    "
                f"{entry['root']}"
            )
            item.setData(Qt.UserRole, entry["root"])
            self.recent_list.addItem(item)

    def _select_shape(self, round_plate: bool) -> None:
        self._round_plate = round_plate
        for card in self._cards:
            card.set_selected(card.round_plate is round_plate)

    @property
    def _mode(self) -> ImagingMode:
        """The imaging mode from the chosen plate shape and batch checkbox."""
        return ImagingMode.for_shape(self._round_plate, self.batch_check.isChecked())

    def _create_new(self) -> None:
        dialog = NewProjectDialog(self._mode, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            project = dialog.create_project()
        except Exception as error:
            QMessageBox.critical(
                self, "Could not create the project", str(error)
            )
            return

        colour = ColourCalibrationDialog(self)
        if colour.exec() == QDialog.Accepted:
            settings = colour.settings()
            if settings["use_colour_card"] and not settings["card_definition_path"]:
                QMessageBox.information(
                    self, "No card definition",
                    "You ticked the colour card box but did not upload a "
                    "definition file, so card correction stays off. You can "
                    "add it later in Settings.",
                )
            else:
                from dataclasses import replace

                project.params = project.params.with_changes(
                    calibration=replace(
                        project.params.calibration, **settings
                    )
                )
                project.save()

        remember_project(project)
        self.project_ready.emit(project)

    def _open_existing(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a FungiCapture project")
        if path:
            self._open_path(path)

    def _open_recent(self, item: QListWidgetItem) -> None:
        self._open_path(item.data(Qt.UserRole))

    def _open_path(self, path: str) -> None:
        try:
            project = Project.load(path)
        except Exception as error:
            QMessageBox.critical(self, "Could not open the project", str(error))
            return
        remember_project(project)
        self.project_ready.emit(project)

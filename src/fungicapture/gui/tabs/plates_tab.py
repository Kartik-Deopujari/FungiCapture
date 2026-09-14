"""
Tab 1 - Plates: place the layout and export one image per colony.

Replaces GridShredder2.py. What is new here:

* **Auto-centering** - the grid is fitted to the colonies actually found,
  instead of being dragged into place by hand
* **Batch** - a layout set on one plate applies to all of them
* **A plate map editor** - strain names are typed in the application, with
  validation, rather than prepared in a spreadsheet elsewhere
* **Dish mode** - single-colony photographs need no grid at all
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.features.regions import load_rgb
from ...core.plate import BoxGeometry, GridGeometry
from ...core.platemap import PlateMap
from ...core.project import Project
from ..widgets.folder_row import FolderPanel, count_images
from ..widgets.plate_canvas import PlateCanvas


class PlateMapEditor(QDialog):
    """
    Edit which strain sits in which well, inside the application.

    Validation runs while the plate is in front of you, so a typo is caught
    here rather than surfacing three steps later as a colony called 'unknown'.
    """

    def __init__(self, plate_map: PlateMap, project: Project, parent=None):
        super().__init__(parent)
        self.plate_map = plate_map
        self.project = project
        self.setWindowTitle("Plate map")
        self.resize(560, 640)

        layout = QVBoxLayout(self)

        # The growth medium lives here rather than in the new-project dialog.
        # It belongs with the plate's identity: it goes into every exported
        # filename alongside the strain and the well, and it is the thing you
        # are most likely to change when setting up a new plate.
        medium_row = QHBoxLayout()
        medium_row.addWidget(QLabel("Growth medium:"))
        self.medium = QLineEdit(project.media_label)
        self.medium.setMaximumWidth(120)
        self.medium.setToolTip(
            "Two to four letters, for example SDA, PDA or MEX.\n"
            "This becomes part of every exported filename: STRAIN_MEDIUM_WELL.png"
        )
        medium_row.addWidget(self.medium)
        medium_hint = QLabel("used in every exported filename")
        medium_hint.setStyleSheet("color:#777; font-size:11px;")
        medium_row.addWidget(medium_hint)
        medium_row.addStretch()
        layout.addLayout(medium_row)

        buttons_row = QHBoxLayout()
        for label, slot in (
            ("Import CSV…", self._import),
            ("Export CSV…", self._export),
            ("Fill down column…", self._fill_down),
            ("Clear all", self._clear),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons_row.addWidget(button)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Well", "Strain", "Note"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        entries = list(self.plate_map)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            key_item = QTableWidgetItem(entry.key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, QTableWidgetItem(entry.strain))
            self.table.setItem(row, 2, QTableWidgetItem(entry.note))
        self._refresh_status()

    def _harvest(self) -> None:
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            strain = self.table.item(row, 1)
            note = self.table.item(row, 2)
            self.plate_map.set(
                key,
                strain.text() if strain else "",
                note.text() if note else "",
            )

    def _refresh_status(self) -> None:
        self._harvest()
        problems = self.plate_map.validate(self.project.layout)
        text = self.plate_map.summary()
        if problems:
            text += "\n\n" + "\n".join(f"• {p}" for p in problems)
            self.status.setStyleSheet("color:#8a5a00;")
        else:
            self.status.setStyleSheet("color:#227722;")
        self.status.setText(text)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import plate map", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            imported = PlateMap.load_csv(
                path, "well" if self.project.mode.needs_grid else "file"
            )
        except Exception as error:
            QMessageBox.warning(self, "Could not read that file", str(error))
            return
        for entry in imported:
            self.plate_map.set(entry.key, entry.strain, entry.note)
        self._populate()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plate map",
            str(self.project.root / "plate_map.csv"),
            "CSV files (*.csv)",
        )
        if path:
            self._harvest()
            self.plate_map.save_csv(path)
            QMessageBox.information(self, "Saved", f"Plate map written to:\n{path}")

    def _fill_down(self) -> None:
        """Copy the selected cell's value down the column - plates are usually
        filled in a pattern, and typing 96 names by hand is a good way to make
        a mistake."""
        selected = self.table.selectedRanges()
        if not selected:
            QMessageBox.information(
                self, "Select a cell first",
                "Click a strain cell, then use Fill down to copy it to every "
                "row below.",
            )
            return
        top = selected[0].topRow()
        column = max(1, selected[0].leftColumn())
        source = self.table.item(top, column)
        if source is None:
            return
        value = source.text()
        for row in range(top + 1, self.table.rowCount()):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self._refresh_status()

    def _clear(self) -> None:
        if QMessageBox.question(
            self, "Clear the plate map?",
            "This removes every strain name. Continue?",
        ) != QMessageBox.Yes:
            return
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(""))
            self.table.setItem(row, 2, QTableWidgetItem(""))
        self._refresh_status()

    def _accept(self) -> None:
        self._harvest()
        # Store and tidy the medium label. Normalising here means a value like
        # "sda " or "medium 1" is corrected while the user is still looking at
        # it, instead of silently producing filenames nothing can parse.
        self.project.media_label = self.medium.text().strip()
        problems = self.project.normalise_media_label()
        if problems:
            QMessageBox.information(
                self, "Growth medium adjusted", "\n\n".join(problems)
            )
        self.project.save()
        self.accept()


class PlatesTab(QWidget):
    """Load plates, place the layout, edit the plate map, export crops."""

    request_run = Signal(str, dict)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.plate_map: PlateMap | None = None
        self.grid: GridGeometry | None = None
        self.boxes: BoxGeometry | None = None
        self._images: list[Path] = []
        self._undo: list[tuple] = []
        # A small cache of decoded photographs, so clicking back to one you have
        # already opened is instant instead of re-decoding a 50-megapixel file.
        self._rgb_cache: OrderedDict[str, object] = OrderedDict()
        self._rgb_cache_max = 5

        outer = QVBoxLayout(self)

        # Each tab names its own folders, so this step can be run on its own.
        self.folders = FolderPanel()
        self.folders.add(
            "images", "Photographs:",
            tooltip="Folder of plate or dish photographs to cut up.",
        )
        self.folders.add(
            "crops", "Save colonies to:",
            tooltip="Where the one-colony images are written.",
        )
        self.folders.finish()
        self.folders.refresh_status(self._count_images)
        self.folders.changed.connect(self._folders_changed)
        outer.addWidget(self.folders)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # ---- left: plate list and controls ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_layout.addWidget(QLabel("Photographs"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._show_image)
        left_layout.addWidget(self.image_list, 1)

        layout_box = QGroupBox("Layout")
        layout_controls = QVBoxLayout(layout_box)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Grid lines", "Boxes"])
        self.mode_selector.currentTextChanged.connect(self._switch_layout_mode)
        layout_controls.addWidget(self.mode_selector)

        # Dish modes ("single plate" / "batch plate") start from one large box
        # and the user adds one box per colony. These two buttons are shown only
        # in those modes - the big gridded plates use the grid above instead.
        self.add_box_button = QPushButton("Add box")
        self.add_box_button.setToolTip(
            "Add another box for another colony on this plate."
        )
        self.add_box_button.clicked.connect(self._add_box)
        layout_controls.addWidget(self.add_box_button)

        self.remove_box_button = QPushButton("Remove selected box")
        self.remove_box_button.setToolTip(
            "Delete the selected box. You can also press Delete."
        )
        self.remove_box_button.clicked.connect(self._remove_box)
        layout_controls.addWidget(self.remove_box_button)

        self.apply_all_button = QPushButton("Apply layout to all plates")
        self.apply_all_button.setToolTip(
            "Use this exact layout for every photograph when you export."
        )
        self.apply_all_button.clicked.connect(self._apply_to_all)
        # Only meaningful in a batch mode - a single-plate mode crops just the
        # plate on screen, so "apply to all" would promise something it does
        # not do. It is shown or hidden per mode in ``set_project``.
        layout_controls.addWidget(self.apply_all_button)

        for label, slot, tip in (
            ("Reset layout", self._reset_layout, "Back to an even grid"),
            ("Undo (Ctrl+Z)", self._undo_last, "Undo the last layout change"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            layout_controls.addWidget(button)

        self.flip_columns = QCheckBox("A01 on the right (flip columns)")
        self.flip_columns.toggled.connect(self._toggle_flip)
        layout_controls.addWidget(self.flip_columns)

        self.show_labels = QCheckBox("Show well labels")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self._toggle_labels)
        layout_controls.addWidget(self.show_labels)
        left_layout.addWidget(layout_box)

        map_box = QGroupBox("Plate map")
        map_controls = QVBoxLayout(map_box)
        self.map_summary = QLabel("No plate map yet")
        self.map_summary.setWordWrap(True)
        map_controls.addWidget(self.map_summary)
        edit_map = QPushButton("Edit plate map…")
        edit_map.clicked.connect(self._edit_plate_map)
        map_controls.addWidget(edit_map)
        left_layout.addWidget(map_box)

        export_box = QGroupBox("Export")
        export_controls = QVBoxLayout(export_box)
        self.skip_edges = QCheckBox("Skip wells that touch the image edge")
        self.skip_edges.setToolTip(
            "Off by default. The original tool dropped these silently, so a "
            "whole outer row could vanish without warning."
        )
        export_controls.addWidget(self.skip_edges)
        self.export_button = QPushButton("Export colony crops")
        self.export_button.clicked.connect(self._export)
        export_controls.addWidget(self.export_button)
        left_layout.addWidget(export_box)

        splitter.addWidget(left)

        # ---- right: canvas ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        for label, slot in (
            ("Fit", lambda: self.canvas.fit_to_window()),
            ("+", lambda: self.canvas.zoom(1.25)),
            ("−", lambda: self.canvas.zoom(1 / 1.25)),
        ):
            button = QPushButton(label)
            button.setMaximumWidth(52)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.plate_status = QLabel("")
        toolbar.addWidget(self.plate_status)
        right_layout.addLayout(toolbar)

        self.canvas = PlateCanvas()
        self.canvas.layout_changed.connect(lambda: self.status_message.emit("Layout edited"))
        right_layout.addWidget(self.canvas, 1)
        splitter.addWidget(right)
        splitter.setSizes([320, 900])

    # ---------------- folders ----------------

    def _count_images(self) -> str:
        folder = self.folders["images"].effective()
        if folder is None:
            return "No photographs folder chosen yet."
        if not Path(folder).exists():
            return f"That folder does not exist: {folder}"
        return f"{count_images(folder)} photograph(s) found in {folder}"

    def _folders_changed(self) -> None:
        self.folders.refresh_status()
        if self.project is None:
            return
        images = self.folders["images"].value()
        if images is not None:
            self.project.image_dir = Path(images)
        self.project.set_folder("crops", self.folders["crops"].value())
        self.project.save()
        self.refresh_images()

    # ---------------- project ----------------

    def set_project(self, project: Project) -> None:
        self.project = project
        self.plate_map = PlateMap.blank_for_layout(project.layout)
        if project.plate_map_path and Path(project.plate_map_path).exists():
            try:
                self.plate_map = PlateMap.load_csv(
                    project.plate_map_path,
                    "well" if project.mode.needs_grid else "file",
                )
            except Exception:
                pass

        needs_grid = project.mode.needs_grid
        # The grid/boxes selector and column flip belong to the big gridded
        # plates. Dish modes use a hand-placed set of boxes instead, so hide
        # the grid controls and show the Add/Remove box buttons there.
        self.mode_selector.setEnabled(needs_grid)
        self.mode_selector.setVisible(needs_grid)
        self.flip_columns.setEnabled(needs_grid)
        self.flip_columns.setVisible(needs_grid)
        self.add_box_button.setVisible(not needs_grid)
        self.remove_box_button.setVisible(not needs_grid)

        # "Apply to all" and cropping every plate only make sense in a batch
        # mode. A single-plate/single-dish mode crops exactly the photograph on
        # screen, so hide the promise of applying to all and say so on the
        # export button.
        is_batch = project.mode.is_batch
        self.apply_all_button.setVisible(is_batch)
        self.export_button.setText(
            "Export colony crops (all plates)" if is_batch
            else "Export colony crops (this plate)"
        )

        if needs_grid:
            self.grid = GridGeometry.regular(
                project.layout.rows, project.layout.cols, project.layout.margin
            )
            self.boxes = None
        else:
            # Dish modes: start from one large box covering the plate; the user
            # adds a box per colony.
            self.grid = None
            self.boxes = BoxGeometry.single()
        self.canvas.flip_columns = project.layout.flip_columns
        self.flip_columns.setChecked(project.layout.flip_columns)

        self.folders["images"].set_default(project.image_dir)
        self.folders["images"].set_value(project.image_dir)
        self.folders["crops"].set_default(project.root / "crops")
        self.folders["crops"].set_value(project.folder_overrides.get("crops"))
        self.folders.refresh_status()

        self.refresh_images()
        self._update_map_summary()

    def refresh_images(self) -> None:
        self.image_list.clear()
        folder = self.folders["images"].effective() if hasattr(self, "folders") else None
        if self.project is None or folder is None:
            return
        try:
            self._images = self.project.find_images(folder)
        except Exception as error:
            self.status_message.emit(str(error))
            return
        for path in self._images:
            self.image_list.addItem(path.name)
        if self._images:
            self.image_list.setCurrentRow(0)

    def _load_rgb_cached(self, path: Path):
        """Decode a photograph, reusing a recent decode when possible."""
        key = str(path)
        cached = self._rgb_cache.get(key)
        if cached is not None:
            self._rgb_cache.move_to_end(key)
            return cached
        rgb = load_rgb(path)
        self._rgb_cache[key] = rgb
        while len(self._rgb_cache) > self._rgb_cache_max:
            self._rgb_cache.popitem(last=False)
        return rgb

    def _show_image(self, row: int) -> None:
        if not (0 <= row < len(self._images)):
            return
        try:
            rgb = self._load_rgb_cached(self._images[row])
        except Exception as error:
            QMessageBox.warning(self, "Could not open image", str(error))
            return
        self.canvas.set_image(rgb)
        if self.boxes is not None:
            self.canvas.set_boxes(self.boxes)
        elif self.grid is not None:
            self.canvas.set_grid(self.grid)
        self.plate_status.setText(
            f"{self._images[row].name}   {rgb.shape[1]} × {rgb.shape[0]} px"
        )

    # ---------------- layout ----------------

    def _push_undo(self) -> None:
        import copy

        self._undo.append((copy.deepcopy(self.grid), copy.deepcopy(self.boxes)))
        del self._undo[:-20]

    def _undo_last(self) -> None:
        if not self._undo:
            self.status_message.emit("Nothing to undo")
            return
        self.grid, self.boxes = self._undo.pop()
        if self.boxes is not None:
            self.canvas.set_boxes(self.boxes)
        elif self.grid is not None:
            self.canvas.set_grid(self.grid)
        self.status_message.emit("Undone")

    def _switch_layout_mode(self, text: str) -> None:
        if self.project is None or self.grid is None:
            return
        self._push_undo()
        if text == "Boxes":
            self.boxes = BoxGeometry.from_grid(
                self.grid, self.project.layout.rows, self.project.layout.cols
            )
            self.canvas.set_boxes(self.boxes)
        else:
            self.boxes = None
            self.canvas.set_grid(self.grid)

    def _apply_to_all(self) -> None:
        if self.grid is None and self.boxes is None:
            return
        QMessageBox.information(
            self, "Layout will be reused",
            "This exact layout will be used for every photograph when you "
            "export. Nothing is moved automatically — what you place is what is "
            "used, so line the plates up the same way when you photograph them.",
        )
        self.status_message.emit("Layout will be applied to all plates")

    def _reset_layout(self) -> None:
        if self.project is None:
            return
        self._push_undo()
        if self.project.mode.needs_grid:
            layout = self.project.layout
            self.grid = GridGeometry.regular(layout.rows, layout.cols, layout.margin)
            self.boxes = None
            self.mode_selector.setCurrentText("Grid lines")
            self.canvas.set_grid(self.grid)
        else:
            # Dish modes: back to one large box.
            self.grid = None
            self.boxes = BoxGeometry.single()
            self.canvas.set_boxes(self.boxes)

    def _add_box(self) -> None:
        """Dish modes: add a box for another colony."""
        if self.project is None:
            return
        if self.boxes is None:
            self.boxes = BoxGeometry.single()
        self._push_undo()
        # The canvas edits the same box set the tab holds. It only has that set
        # once an image is on screen, so hand it over if it does not yet.
        if self.canvas.boxes is not self.boxes:
            self.canvas.set_boxes(self.boxes)
        self.canvas.add_box()
        self.status_message.emit(f"{len(self.boxes.boxes)} box(es) on this plate")

    def _remove_box(self) -> None:
        """Dish modes: remove the selected box."""
        if self.boxes is None:
            return
        self._push_undo()
        if self.canvas.boxes is not self.boxes:
            self.canvas.set_boxes(self.boxes)
        self.canvas.delete_selected_boxes()

    def _toggle_flip(self, checked: bool) -> None:
        self.canvas.flip_columns = checked
        if self.project:
            self.project.layout.flip_columns = checked
            self.project.save()
        self.canvas.viewport().update()

    def _toggle_labels(self, checked: bool) -> None:
        self.canvas.show_labels = checked
        self.canvas.viewport().update()

    # ---------------- plate map ----------------

    def _edit_plate_map(self) -> None:
        if self.project is None or self.plate_map is None:
            return
        if not self.project.mode.needs_grid:
            self.plate_map = PlateMap.from_filenames(p.name for p in self._images)
        dialog = PlateMapEditor(self.plate_map, self.project, self)
        if dialog.exec() == QDialog.Accepted:
            path = self.plate_map.save_csv(self.project.root / "plate_map.csv")
            self.project.plate_map_path = path
            self.project.save()
            self._update_map_summary()

    def _update_map_summary(self) -> None:
        if self.plate_map is None:
            return
        problems = self.plate_map.validate(
            self.project.layout if self.project else None
        )
        text = self.plate_map.summary()
        if problems:
            text += f"\n{len(problems)} thing(s) to check"
        self.map_summary.setText(text)

    # ---------------- export ----------------

    def _selected_image(self) -> Path | None:
        if not self._images:
            return None
        row = self.image_list.currentRow()
        return self._images[row] if 0 <= row < len(self._images) else self._images[0]

    def layout_crop_options(self) -> dict:
        """The layout the user has placed, for the cropping step.

        Shared by the Export button and by "Run everything" so a dish plate's
        hand-placed boxes are honoured either way, rather than the whole-pipeline
        run silently falling back to automatic single-colony detection.
        """
        selected = self._selected_image()
        return {
            "grid": self.grid if self.boxes is None else None,
            "boxes": self.boxes,
            "plate_map": self.plate_map,
            "auto_centre": False,
            "skip_edge_wells": self.skip_edges.isChecked(),
            "image_path": str(selected) if selected is not None else None,
        }

    def pipeline_crop_options(self) -> dict:
        """Layout options for a whole-pipeline ("Run everything") run.

        Only the dish modes pass their hand-placed boxes here; the gridded
        plates keep their automatic grid fit, exactly as before, so a plain
        "Run everything" on a big plate is not changed.
        """
        selected = self._selected_image()
        options: dict = {"plate_map": self.plate_map}
        if selected is not None:
            options["image_path"] = str(selected)
        if self.project is not None and not self.project.mode.needs_grid:
            options["boxes"] = self.boxes
            options["auto_centre"] = False
            options["skip_edge_wells"] = self.skip_edges.isChecked()
        return options

    def _export(self) -> None:
        if self.project is None:
            return
        if not self._images:
            QMessageBox.warning(
                self, "No photographs",
                "No images were found. Check the project's image folder.",
            )
            return

        # In the single modes the export crops exactly the plate on screen, so
        # the pipeline is told which one that is. In batch modes it is ignored
        # and every photograph is cropped.
        if self.project.mode.is_batch:
            confirm = QMessageBox.question(
                self, "Crop every plate?",
                f"This will crop all {len(self._images)} photographs in the "
                "folder using this layout.\n\nContinue?",
            )
            if confirm != QMessageBox.Yes:
                return

        options = self.layout_crop_options()
        options["input_dir"] = self.folders["images"].effective()
        options["output_dir"] = self.folders["crops"].effective()
        self.request_run.emit("crop", options)

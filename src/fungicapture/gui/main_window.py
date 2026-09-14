"""
The main window: home screen, five tabs, one progress bar.

The window owns the project and the worker thread. Tabs never start jobs
themselves - they emit ``request_run`` and the window decides. That keeps all
the threading in one place, so two jobs can never run at once and fight over
the same files.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline import (
    crop_plates,
    extract_features,
    score_phenotypes,
    segment_crops,
)
from ..core.project import Project
from .home import HomeScreen
from .tabs.plates_tab import PlatesTab
from .tabs.simple_tabs import FeaturesTab, PhenotypesTab, SegmentTab
from .tabs.validate_tab import ValidateTab
from .workers import run_in_background

APP_VERSION = "0.1.0.dev0"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FungiCapture")
        self.resize(1400, 900)

        self.project: Project | None = None
        self._handle = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home = HomeScreen()
        self.home.project_ready.connect(self.open_project)
        self.stack.addWidget(self.home)

        self.workspace = self._build_workspace()
        self.stack.addWidget(self.workspace)

        self._build_menu()
        self._build_status_bar()

    # ---------------- construction ----------------

    def _build_workspace(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 10, 0)
        self.project_label = QLabel("")
        self.project_label.setStyleSheet("font-weight:600;")
        header.addWidget(self.project_label)
        header.addStretch()
        run_all = QPushButton("Run everything ▶")
        run_all.setToolTip("Crop, segment, measure and score, in one go")
        run_all.clicked.connect(self._run_all)
        header.addWidget(run_all)
        home_button = QPushButton("Home")
        home_button.clicked.connect(self._go_home)
        header.addWidget(home_button)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.plates_tab = PlatesTab()
        self.segment_tab = SegmentTab()
        self.features_tab = FeaturesTab()
        self.validate_tab = ValidateTab()
        self.phenotypes_tab = PhenotypesTab()

        for index, (widget, title) in enumerate(
            (
                (self.plates_tab, "1  Plates"),
                (self.segment_tab, "2  Segment"),
                (self.features_tab, "3  Features"),
                (self.validate_tab, "4  Validate"),
                (self.phenotypes_tab, "5  Phenotypes"),
            )
        ):
            self.tabs.addTab(widget, title)
            widget.request_run.connect(self._handle_request)
            widget.status_message.connect(self._set_status)

        layout.addWidget(self.tabs, 1)
        return container

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        for label, shortcut, slot in (
            ("New / open project…", QKeySequence.New, self._go_home),
            ("Open project folder…", QKeySequence.Open, self._open_folder),
            ("Save project", QKeySequence.Save, self._save_project),
        ):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About FungiCapture", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

        licensing = QAction("Licensing and citation", self)
        licensing.triggered.connect(self._licensing)
        help_menu.addAction(licensing)

    def _build_status_bar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)

        # Notes stay available on demand rather than interrupting.
        self.notes_button = QPushButton("")
        self.notes_button.setVisible(False)
        self.notes_button.setFlat(True)
        self.notes_button.setStyleSheet("color:#8a5a00;")
        self.notes_button.clicked.connect(self._show_notes)
        self._last_warnings: list[str] = []

        self.statusBar().addPermanentWidget(self.notes_button)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("Ready")

    def _show_notes(self) -> None:
        if not self._last_warnings:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Notes from the last run")
        box.setText(f"{len(self._last_warnings)} note(s) from the last run.")
        box.setDetailedText("\n\n".join(self._last_warnings))
        box.exec()

    # ---------------- project ----------------

    def open_project(self, project: Project) -> None:
        self.project = project
        project.ensure_dirs()
        for tab in (
            self.plates_tab, self.segment_tab, self.features_tab,
            self.validate_tab, self.phenotypes_tab,
        ):
            tab.set_project(project)

        self.project_label.setText(
            f"{project.name}   ·   {project.mode.label}   ·   {project.root}"
        )
        self.setWindowTitle(f"FungiCapture — {project.name}")
        self.stack.setCurrentWidget(self.workspace)
        self._set_status(f"Opened {project.name}")

    def _go_home(self) -> None:
        if self._busy():
            return
        self.home.refresh_recents()
        self.stack.setCurrentWidget(self.home)
        self.setWindowTitle("FungiCapture")

    def _open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a FungiCapture project")
        if not path:
            return
        try:
            self.open_project(Project.load(path))
        except Exception as error:
            QMessageBox.critical(self, "Could not open the project", str(error))

    def _save_project(self) -> None:
        if self.project:
            self.project.save()
            self._set_status("Project saved")

    # ---------------- running jobs ----------------

    def _busy(self) -> bool:
        if self._handle is not None and self._handle.is_running():
            QMessageBox.information(
                self, "Still working",
                "A job is running. Wait for it to finish, or press Cancel.",
            )
            return True
        return False

    def _handle_request(self, step: str, options: dict) -> None:
        if self.project is None or self._busy():
            return

        if step == "report":
            self._run_report(options)
            return

        functions = {
            "crop": crop_plates,
            "segment": segment_crops,
            "features": extract_features,
            "phenotypes": score_phenotypes,
        }
        function = functions.get(step)
        if function is None:
            return

        self._start(
            function,
            step=step,
            project=self.project,
            **options,
        )

    def _run_all(self) -> None:
        if self.project is None or self._busy():
            return
        if QMessageBox.question(
            self, "Run everything?",
            "This will crop the plates, segment every colony, measure the "
            "features and compute GWAS phenotypes.\n\nIt may take a while. You "
            "can cancel at any point.",
        ) != QMessageBox.Yes:
            return

        from ..core.pipeline import run_pipeline

        # Honour the layout the user placed on the Plates tab (a dish plate's
        # hand-placed boxes, or a gridded plate's grid) rather than re-detecting
        # from scratch.
        self._start(
            run_pipeline,
            step="everything",
            project=self.project,
            **self.plates_tab.pipeline_crop_options(),
        )

    def _run_report(self, options: dict) -> None:
        from ..core.report import build_context, write_report
        from ..core.validate import VerdictStore, build_panel_data

        assert self.project is not None
        project = self.project
        pairs = options["pairs"]
        mode = options["mode"]
        path = options["path"]

        # Load the measured feature values so the report's number panel and its
        # colour/zonation plots are filled in. Without this the PDF showed empty
        # panels: the report was built with no values at all, so every feature
        # was blank even though the CSV held them.
        feature_lookup = self._load_feature_values()

        def job(progress):
            from ..core.features.extract import normalise_stem

            store = VerdictStore(project.qc_dir / VerdictStore.FILENAME)
            panels = []
            for index, (crop_path, mask_path) in enumerate(pairs, start=1):
                if not progress(index, len(pairs), f"Preparing {crop_path.name}"):
                    break
                values = feature_lookup.get(normalise_stem(crop_path.stem), {})
                try:
                    panels.append(
                        build_panel_data(
                            crop_path, mask_path, values, project.params
                        )
                    )
                except Exception:
                    continue
            context = build_context(project, mode, len(panels), store)
            written = write_report(
                path, panels, project.params, context, mode=mode, verdicts=store
            )

            from ..core.pipeline import StepResult

            result = StepResult(
                step="report", input_noun="colony", output_noun="page"
            )
            result.n_inputs = len(pairs)
            result.n_outputs = len(panels)
            result.outputs = [written]
            return result

        self._start(job, step="report")

    def _load_feature_values(self) -> dict:
        """Map each colony's key to its measured feature values.

        Read from the same master feature table the Validate tab uses, so the
        PDF report shows exactly the numbers seen on screen. Returns an empty
        mapping (and the report still renders images, just without numbers) if
        the feature step has not run yet.
        """
        assert self.project is not None
        vectors = self.project.vectors_dir
        master = vectors / "all_colony_features.csv"
        if not master.exists():
            return {}
        try:
            import pandas as pd

            frame = pd.read_csv(master)
        except Exception:
            return {}
        if "base_key" not in frame.columns:
            return {}
        return {
            str(row["base_key"]): row.to_dict()
            for _, row in frame.iterrows()
        }

    def _start(self, function, step: str, **kwargs) -> None:
        self._current_step = step
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.cancel_button.setVisible(True)
        self._set_status(f"Running: {step}")

        self._handle = run_in_background(
            function,
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_failed=self._on_failed,
            **kwargs,
        )

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        if message:
            self.statusBar().showMessage(f"{message}  ({done}/{total})")

    def _on_finished(self, result) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)

        results = result if isinstance(result, list) else [result]
        lines = []
        for item in results:
            lines.append(item.summary())
            tab = {
                "crop": self.plates_tab,
                "segment": self.segment_tab,
                "features": self.features_tab,
                "phenotypes": self.phenotypes_tab,
            }.get(getattr(item, "step", ""))
            if tab is not None and hasattr(tab, "show_result"):
                tab.show_result(item)

        # New crops or features mean the Validate tab's list is stale.
        if any(getattr(i, "step", "") in ("features", "crop") for i in results):
            self.validate_tab.reload()

        warnings = [w for item in results for w in item.warnings]
        summary = "  |  ".join(lines)
        if warnings:
            summary += f"   —   {len(warnings)} note(s), see the tab log"
        self._set_status(summary)

        # Warnings go to the status bar and the tab log, not a modal dialog.
        # Most runs produce at least one note (the SAM 3 fallback message, for
        # instance), and a dialog that must be dismissed after every single run
        # trains people to click through without reading - which defeats the
        # purpose of warning them at all.
        #
        # Errors are different: those do interrupt, in _on_failed.
        self._last_warnings = warnings
        self.notes_button.setVisible(bool(warnings))
        self.notes_button.setText(f"⚠ {len(warnings)} note(s)")

    def _on_failed(self, message: str, trace: str) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self._set_status("Failed")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Something went wrong")
        box.setText(message)
        box.setDetailedText(trace)
        box.exec()

    def _cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._set_status("Cancelling…")

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    # ---------------- help ----------------

    def _about(self) -> None:
        QMessageBox.about(
            self, "About FungiCapture",
            f"<h3>FungiCapture {APP_VERSION}</h3>"
            "<p>High-throughput image-based phenotyping of fungal colonies — "
            "from an agar plate photograph to GWAS-ready phenotype scores.</p>"
            "<p>Shape, brightness, GLCM and LBP texture, and colour including "
            "melanization, measured on the colony interior and an adaptive "
            "boundary ring.</p>"
            "<p>Released under AGPL-3.0.</p>",
        )

    def _licensing(self) -> None:
        QMessageBox.information(
            self, "Licensing and citation",
            "FungiCapture is released under AGPL-3.0. This is required, not "
            "chosen: it depends on Ultralytics, which is AGPL-3.0.\n\n"
            "The SAM 3 weights are covered by Meta's own SAM License, a "
            "restricted research licence. FungiCapture does not redistribute "
            "them — you download them from Meta's gated page and accept that "
            "licence yourself.\n\n"
            "If you publish results made with SAM 3, the licence requires you "
            "to acknowledge it in your methods.\n\n"
            "See docs/LICENSING.md and docs/REFERENCES.md in the repository.",
        )

    def closeEvent(self, event) -> None:
        if self._handle is not None and self._handle.is_running():
            if QMessageBox.question(
                self, "A job is still running",
                "Quit anyway? The current job will be stopped.",
            ) != QMessageBox.Yes:
                event.ignore()
                return
            self._handle.cancel()
            self._handle.wait(3000)
        if self.project is not None:
            self.project.save()
        event.accept()

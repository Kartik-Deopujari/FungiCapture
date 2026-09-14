"""
The segmentation setup window.

This exists because of a real failure: the application installs its interface
and its science by default but NOT PyTorch, so SAM 3 could never load, the
classical CPU segmenter was used instead, and nothing said why. From the
outside that looks like "segmentation is broken and stuck on CPU".

The window answers three questions in order:

* what hardware is on this machine?
* what is installed, and what is missing?
* what will actually run, and what do I press to improve it?
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ...core.setup_env import check_environment, install_segmentation, restart_note
from ..workers import run_in_background


class SegmentationSetupDialog(QDialog):
    """Diagnose the machine, and install what is missing."""

    environment_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation setup")
        self.resize(720, 620)
        self._handle = None

        layout = QVBoxLayout(self)

        heading = QLabel("Segmentation setup")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

        blurb = QLabel(
            "FungiCapture can find colony outlines in two ways. The classical "
            "segmenter needs nothing and always works, but is weaker on faint "
            "filamentous edges. SAM 3 is much better on those, and needs "
            "PyTorch, Ultralytics and the model weights."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#555;")
        layout.addWidget(blurb)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QFont("monospace"))
        layout.addWidget(self.report, 1)

        self.force_cpu = QCheckBox(
            "Install the processor-only version (smaller, no GPU needed)"
        )
        self.force_cpu.setToolTip(
            "Tick this if you have no graphics card, or if the GPU version "
            "gives trouble. It is about 1 GB smaller."
        )
        layout.addWidget(self.force_cpu)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons_row = QHBoxLayout()
        self.recheck_button = QPushButton("Check again")
        self.recheck_button.clicked.connect(self.refresh)
        buttons_row.addWidget(self.recheck_button)

        self.install_button = QPushButton("Install PyTorch + Ultralytics")
        self.install_button.setToolTip(
            "Installs the deep-learning libraries SAM 3 needs - PyTorch, "
            "torchvision and Ultralytics (8.3.237+) - choosing the build that "
            "matches your graphics driver. About 2 GB, several minutes. If the "
            "GPU build cannot be installed it falls back to the processor build "
            "automatically."
        )
        self.install_button.clicked.connect(self._install)
        buttons_row.addWidget(self.install_button)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)

        self.refresh()

    # ---------------- checking ----------------

    def refresh(self) -> None:
        self.report.setPlainText("Checking…")
        report = check_environment()
        self.report.setPlainText(report.full_text())
        self.environment_changed.emit()

        if report.can_run_sam3:
            self.install_button.setText("Reinstall PyTorch + Ultralytics")
        else:
            self.install_button.setText("Install PyTorch + Ultralytics")

    # ---------------- installing ----------------

    def _install(self) -> None:
        if self._handle is not None and self._handle.is_running():
            return

        if (
            QMessageBox.question(
                self,
                "Install now?",
                "This downloads roughly 2 GB and can take several minutes on a "
                "normal connection.\n\nCarry on?",
            )
            != QMessageBox.Yes
        ):
            return

        self.install_button.setEnabled(False)
        self.recheck_button.setEnabled(False)
        self.progress.setVisible(True)
        self.report.setPlainText("Starting…\n")

        force_cpu = self.force_cpu.isChecked()

        def job(progress):
            # The worker calls progress(done, total, message); the installer
            # only has messages, so wrap it.
            def say(message: str) -> None:
                progress(0, 0, message)

            return install_segmentation(say, force_cpu=force_cpu)

        self._handle = run_in_background(
            job,
            on_progress=lambda done, total, message: self._append(message),
            on_finished=self._done,
            on_failed=self._failed,
        )

    def _append(self, message: str) -> None:
        if message:
            self.report.appendPlainText(message)

    def _done(self, outcome) -> None:
        succeeded, message = outcome
        self.progress.setVisible(False)
        self.install_button.setEnabled(True)
        self.recheck_button.setEnabled(True)

        box = QMessageBox(self)
        box.setWindowTitle("Finished" if succeeded else "Did not finish")
        box.setIcon(QMessageBox.Information if succeeded else QMessageBox.Warning)
        box.setText(message)
        if succeeded:
            box.setInformativeText(restart_note())
        box.exec()

        self.refresh()

    def _failed(self, message: str, trace: str) -> None:
        self.progress.setVisible(False)
        self.install_button.setEnabled(True)
        self.recheck_button.setEnabled(True)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Installation failed")
        box.setText(message)
        box.setDetailedText(trace)
        box.exec()

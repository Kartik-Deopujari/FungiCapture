"""
Application entry point.

Cross-platform notes
--------------------
The original script forced ``QT_QPA_PLATFORM=xcb``, which is Linux-only and
would break the application on macOS and Windows. Qt picks its own platform
plugin correctly on every system, so the right move is to say nothing.

The one thing worth clearing is ``QT_QPA_PLATFORM_PLUGIN_PATH``: OpenCV ships
its own Qt plugins, and if OpenCV is imported first that variable can point Qt
at the wrong ones, giving a confusing "could not load the Qt platform plugin"
failure at start-up.
"""

from __future__ import annotations

import os
import sys


def _prepare_environment() -> None:
    # Stop OpenCV's bundled Qt plugins from hijacking ours.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    # Crisp text on high-resolution screens.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def main(argv: list[str] | None = None) -> int:
    _prepare_environment()

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    QApplication.setApplicationName("FungiCapture")
    QApplication.setOrganizationName("FungiCapture")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()

    # Open a project passed on the command line, so double-clicking a project
    # folder can launch straight into it.
    arguments = (argv if argv is not None else sys.argv)[1:]
    if arguments:
        from ..core.project import Project

        try:
            window.open_project(Project.load(arguments[0]))
        except Exception:
            pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

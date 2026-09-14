"""
Entry point for the packaged application.

PyInstaller runs its target script as ``__main__``, with no package around it.
That breaks the relative imports inside ``fungicapture/gui/app.py``
(``from .main_window import MainWindow``), because there is no parent package
to resolve them against.

So the bundle points here instead: a tiny script that imports the real
application by its absolute name, which works the same whether it is frozen or
run from source.
"""

import multiprocessing
import sys


# Subcommands that mean "run the command line, not the window". Anything else
# on the command line is treated as a project folder to open in the window.
CLI_COMMANDS = {"new", "run", "report", "info", "gui", "--help", "-h", "--version"}


def main() -> int:
    # scikit-learn and NumPy can start worker processes. In a frozen bundle
    # each of those would otherwise re-launch the whole application instead of
    # starting a worker, which shows up as several copies of the window opening
    # at once. This call makes the child processes behave.
    multiprocessing.freeze_support()

    # One executable, two front ends. Double-clicking opens the window;
    # `FungiCapture run myproject` from a terminal runs the pipeline headless,
    # which is what a cluster job or a scripted re-analysis needs.
    if len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS:
        if sys.argv[1] == "--version":
            from fungicapture.gui.main_window import APP_VERSION

            print(f"FungiCapture {APP_VERSION}")
            return 0

        from fungicapture.cli.main import main as run_cli

        return run_cli(sys.argv[1:])

    from fungicapture.gui.app import main as run_application

    return run_application(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

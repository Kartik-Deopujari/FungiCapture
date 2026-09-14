#!/usr/bin/env bash
# ===================================================================
#  FungiCapture launcher for macOS
#
#  Double-click this file to start FungiCapture.
#
#  The first run sets everything up; later runs just open the program.
#  Everything is installed inside this folder, in ".venv". Nothing else
#  on the computer is changed, and deleting this folder removes it
#  completely.
#
#  Moving or renaming the folder is fine: the launcher notices and
#  repairs itself.
# ===================================================================

cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"

LOG="setup_log.txt"
VENV=".venv"
PYEXE="$VENV/bin/python"
STAMP="$VENV/.installed-at"

# macOS opens a Terminal window automatically for a .command file, so there
# is nothing to re-launch here.

say() { printf '%s\n' "$*"; }
pause_at_end() {
    say ""
    read -r -p "Press Enter to close this window. " _ 2>/dev/null || sleep 30
}

# Point Python at the source folder as well. This is a safety net: even if
# the installed copy is damaged, the program still starts from the source
# sitting right here.
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

start_program() {
    "$PYEXE" -m fungicapture.gui.app 2>>"$LOG"
}

# Is the existing setup usable? Checking that the file exists is not enough:
# a setup made in a different folder still leaves the file there but cannot
# import anything, which is exactly how this used to fail after someone moved
# the folder. So we test by actually importing.
setup_is_good() {
    [ -x "$PYEXE" ] || return 1
    [ -f "$STAMP" ] || return 1
    [ "$(cat "$STAMP" 2>/dev/null)" = "$HERE" ] || return 1
    "$PYEXE" -c "import fungicapture, PySide6" >/dev/null 2>&1
}

say ""
say " =========================================="
say "   FungiCapture"
say " =========================================="
say ""

if setup_is_good; then
    say " Starting FungiCapture..."
    say ""
    start_program && exit 0
    say ""
    say " --------------------------------------------------------------"
    say "  FungiCapture closed unexpectedly."
    say "  Details are in  $LOG"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi

# ---------- needs setting up, or repairing ----------
if [ -d "$VENV" ]; then
    say " The setup in this folder is out of date."
    say " That normally means the folder was moved or renamed since last time."
    say " Repairing it now - this takes a few minutes."
    say ""
    rm -rf "$VENV"
else
    say " First run - setting things up. This takes a few minutes."
    say " Please leave this window open."
    say ""
fi

# ---------- find Python ----------
say " [1/4] Looking for Python..."

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    say ""
    say " --------------------------------------------------------------"
    say "  Python 3.10 or newer was not found on this computer."
    say ""
    say "  Install it, then run this file again:"
    say ""
    say "    Easiest: download the installer from"
    say "      https://www.python.org/downloads/"
    say ""
    say "    Or, if you use Homebrew:"
    say "      brew install python@3.11"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi

say "       found $($PY --version 2>&1)"

# ---------- build the private environment ----------
say " [2/4] Creating a private setup inside this folder..."
if ! "$PY" -m venv "$VENV" >>"$LOG" 2>&1; then
    say ""
    say " --------------------------------------------------------------"
    say "  Could not create the private setup folder."
    say ""
    say "  This usually means the folder is read-only. Move the"
    say "  FungiCapture folder to your Documents folder and try again."
    say ""
    say "  Details are in  $LOG"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi

say " [3/4] Downloading the parts FungiCapture needs..."
say "       (this is the slow step - a few minutes)"
"$PYEXE" -m pip install --upgrade pip >>"$LOG" 2>&1

# A normal install, NOT "pip install -e". An editable install records the
# absolute path of this folder inside the setup; move the folder and every
# import breaks. A normal install copies the code in, so the setup keeps
# working wherever the folder ends up.
if ! "$PYEXE" -m pip install ".[gui]" >>"$LOG" 2>&1; then
    say ""
    say " --------------------------------------------------------------"
    say "  The download did not finish."
    say ""
    say "  Most often this is simply the internet connection dropping."
    say "  Check you are online and run this file again - it will carry"
    say "  on from where it stopped."
    say ""
    say "  If it keeps failing, send  $LOG  to whoever gave you"
    say "  FungiCapture."
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi

say " [4/4] Checking everything works..."
if ! "$PYEXE" -c "import fungicapture, PySide6" >>"$LOG" 2>&1; then
    say ""
    say " --------------------------------------------------------------"
    say "  Setup finished but the program will not start."
    say ""
    say "  If macOS blocked part of the download, try running this"
    say "  file again. If it still fails, send the log file on."
    say ""
    say "  Details are in  $LOG"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi
# Importing PySide6 is not enough - it succeeds even when a window cannot
# open. Actually create (and drop) a Qt application to be sure.
if ! "$PYEXE" -c "import sys; from PySide6.QtWidgets import QApplication; QApplication([]); sys.exit(0)" >>"$LOG" 2>&1; then
    say ""
    say " --------------------------------------------------------------"
    say "  Setup finished but a test window would not open."
    say ""
    say "  Try restarting the Mac and running this file again. If it"
    say "  persists, send the log file on."
    say ""
    say "  Details are in  $LOG"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi

# Remember where this setup was built, so a later move is detected.
printf '%s' "$HERE" > "$STAMP"

say ""
say " Setup finished."
say ""
say " Note: the deep-learning segmentation model (SAM 3) is NOT installed"
say " yet - it is about 2 GB, so it is kept optional. FungiCapture works"
say " without it using its built-in segmenter."
say ""
say " To add it: open the '2 Segment' tab and press 'Segmentation setup...'."
say " That checks your graphics card and installs the matching version."
say ""
say " Starting FungiCapture..."
say " Next time, this will open in a few seconds."
say ""
start_program

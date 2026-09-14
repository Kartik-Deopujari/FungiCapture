#!/usr/bin/env bash
# ===================================================================
#  FungiCapture — first-time setup for macOS
#
#  Double-click this file once, the first time, on a new Mac.
#  (If macOS refuses: right-click it, choose "Open", then "Open" again.)
#
#  It builds the private environment (.venv) inside this folder and
#  downloads everything FungiCapture needs, then checks a window can
#  really open. On macOS all the graphics libraries are bundled, so no
#  password and no system changes are needed.
#
#  After this finishes, start the program the normal way any time:
#      double-click  START FungiCapture (Mac).command
# ===================================================================

set -u
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"

LOG="setup_log.txt"
VENV=".venv"
PYEXE="$VENV/bin/python"
STAMP="$VENV/.installed-at"

say()  { printf '%s\n' "$*"; }
rule() { say " --------------------------------------------------------------"; }
pause_at_end() {
    say ""
    read -r -p "Press Enter to close this window. " _ 2>/dev/null || sleep 30
}

say ""
say " =========================================="
say "   FungiCapture — first-time setup (macOS)"
say " =========================================="
say ""
say " This runs once and needs an internet connection."
say ""

# ---------------------------------------------------------------
#  1. Find a suitable Python
# ---------------------------------------------------------------
say " [1/3] Looking for Python 3.10+..."
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
    rule
    say "  Python 3.10 or newer was not found."
    say ""
    say "  Install it, then run this file again:"
    say ""
    say "    Easiest:   download from  https://www.python.org/downloads/"
    say "    Homebrew:  brew install python"
    rule
    pause_at_end
    exit 1
fi
say "       found $($PY --version 2>&1)"

# ---------------------------------------------------------------
#  2. Build the private environment and install FungiCapture
# ---------------------------------------------------------------
say " [2/3] Building the private environment (a few minutes)..."
# Do NOT wipe a healthy environment: the optional SAM 3 libraries (PyTorch +
# Ultralytics, ~2 GB) are installed into this same .venv from inside the app,
# and deleting it on every run would throw that away. Rebuild only if missing
# or unusable.
if [ -x "$PYEXE" ] && "$PYEXE" -c "import fungicapture, PySide6" >/dev/null 2>&1; then
    say "       an existing environment is present and healthy - keeping it"
    say "       (this preserves the SAM 3 model if you have installed it)"
else
    [ -d "$VENV" ] && say "       the existing environment is unusable - rebuilding it"
    rm -rf "$VENV"
    if ! "$PY" -m venv "$VENV" >>"$LOG" 2>&1; then
        say ""
        rule
        say "  Could not create the private environment. Details are in  $LOG"
        rule
        pause_at_end
        exit 1
    fi
fi

"$PYEXE" -m pip install --upgrade pip >>"$LOG" 2>&1
# Normal (non-editable) install so the folder can be moved later.
if ! "$PYEXE" -m pip install ".[gui]" >>"$LOG" 2>&1; then
    say ""
    rule
    say "  The download did not finish - usually a dropped internet"
    say "  connection. Check you are online and run this file again."
    say "  Details are in  $LOG"
    rule
    pause_at_end
    exit 1
fi

# ---------------------------------------------------------------
#  3. Prove a window can actually open
# ---------------------------------------------------------------
say " [3/3] Checking a window can open..."
if ! "$PYEXE" - >>"$LOG" 2>&1 <<'PY'
import sys
from PySide6.QtWidgets import QApplication
QApplication([])
sys.exit(0)
PY
then
    say ""
    rule
    say "  Everything installed, but a test window would not open."
    say "  Details are in  $LOG - send it to whoever gave you FungiCapture."
    rule
    pause_at_end
    exit 1
fi

printf '%s' "$HERE" > "$STAMP"

say ""
rule
say "  Setup finished successfully."
rule
say ""
say " From now on, just start FungiCapture the normal way:"
say ""
say "     double-click  START FungiCapture (Mac).command"
say ""
say " (The deep-learning SAM 3 model is optional and about 2 GB. Add"
say "  it later from inside the app: the '2 Segment' tab, button"
say "  'Segmentation setup...'. FungiCapture works without it.)"
say ""

LAUNCHER="START FungiCapture (Mac).command"
if [ -f "$LAUNCHER" ]; then
    read -r -p " Start FungiCapture now? [Y/n] " ans 2>/dev/null || ans="n"
    case "${ans:-Y}" in
        n|N|no|No) say " You can start it any time by double-clicking the launcher." ;;
        *) exec bash "$LAUNCHER" ;;
    esac
fi

#!/usr/bin/env bash
# ===================================================================
#  FungiCapture — first-time setup for Linux
#
#  Run this ONCE, the first time, on a new computer:
#
#      bash "SETUP FungiCapture (Linux).sh"
#
#  It does the two things the everyday launcher cannot do on its own:
#
#    1. Installs the handful of SYSTEM libraries the graphical window
#       needs (Qt / xcb). These live outside this folder, so they need
#       your password once (sudo). Without them the program installs
#       fine but cannot open a window - the "could not load the Qt
#       platform plugin: xcb" error.
#
#    2. Builds the private environment (.venv) inside this folder and
#       downloads everything else FungiCapture needs.
#
#  After this finishes, start the program the normal way any time:
#
#      double-click  START FungiCapture (Linux).sh
#
#  Nothing here is destructive: re-running it is safe, and it only
#  installs well-known graphics libraries plus this project.
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
say "   FungiCapture — first-time setup"
say " =========================================="
say ""
say " This runs once and needs an internet connection."
say " It will ask for your password to install the system"
say " graphics libraries, then set everything else up here."
say ""

# ---------------------------------------------------------------
#  1. System graphics libraries (needs sudo, once)
# ---------------------------------------------------------------
# Which package manager is on this machine, and what does it call the
# libraries Qt needs? We keep the list minimal and boring: the xcb
# platform plugin plus its cursor/keyboard/EGL helpers.
PKG=""
INSTALL_CMD=""
GUI_PKGS=""
if command -v apt-get >/dev/null 2>&1 || command -v apt >/dev/null 2>&1; then
    PKG="apt"
    INSTALL_CMD="sudo apt-get install -y"
    GUI_PKGS="libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1 libgl1"
    PY_PKGS="python3 python3-venv python3-pip"
elif command -v dnf >/dev/null 2>&1; then
    PKG="dnf"
    INSTALL_CMD="sudo dnf install -y"
    GUI_PKGS="xcb-util-cursor libxkbcommon-x11 libglvnd-egl libglvnd-glx"
    PY_PKGS="python3 python3-pip"
elif command -v pacman >/dev/null 2>&1; then
    PKG="pacman"
    INSTALL_CMD="sudo pacman -S --needed --noconfirm"
    GUI_PKGS="xcb-util-cursor libxkbcommon-x11 libglvnd"
    PY_PKGS="python python-pip"
elif command -v zypper >/dev/null 2>&1; then
    PKG="zypper"
    INSTALL_CMD="sudo zypper install -y"
    GUI_PKGS="libxcb-cursor0 libxkbcommon-x11-0 libEGL1 libGL1"
    PY_PKGS="python3 python3-pip"
fi

say " [1/4] Installing system graphics libraries..."
if [ -z "$PKG" ]; then
    say ""
    rule
    say "  Could not recognise this Linux distribution's package"
    say "  manager, so the system libraries can't be installed"
    say "  automatically."
    say ""
    say "  Please install the Qt xcb libraries yourself (names vary by"
    say "  distribution: libxcb-cursor, libxkbcommon-x11, libEGL, libGL)"
    say "  and then run this file again."
    rule
    pause_at_end
    exit 1
fi

say "       using $PKG (you may be asked for your password)"
# python3-venv is bundled in $PY_PKGS so a bare-bones system can still build
# the environment in step 3.
if ! $INSTALL_CMD $PY_PKGS $GUI_PKGS >>"$LOG" 2>&1; then
    say ""
    rule
    say "  The system libraries did not install."
    say ""
    say "  If you were not asked for a password, or sudo is not set up,"
    say "  ask whoever administers this computer to run:"
    say ""
    say "    $INSTALL_CMD $GUI_PKGS"
    say ""
    say "  Details are in  $LOG"
    rule
    pause_at_end
    exit 1
fi
say "       done"

# ---------------------------------------------------------------
#  2. Find a suitable Python
# ---------------------------------------------------------------
say " [2/4] Looking for Python 3.10+..."
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
    say "  Python 3.10 or newer was not found, even after trying to"
    say "  install it. Install it manually and run this file again:"
    say ""
    say "    $INSTALL_CMD $PY_PKGS"
    rule
    pause_at_end
    exit 1
fi
say "       found $($PY --version 2>&1)"

# ---------------------------------------------------------------
#  3. Build the private environment and install FungiCapture
# ---------------------------------------------------------------
say " [3/4] Building the private environment (a few minutes)..."
# IMPORTANT: do NOT blindly wipe an existing environment. The optional SAM 3
# libraries (PyTorch + Ultralytics, ~2 GB) are installed into this same .venv
# from inside the app. Deleting it on every setup run would silently throw that
# away and force a slow re-download - which is exactly what made setup feel
# broken. So the venv is only rebuilt when it is actually missing or unusable.
if [ -x "$PYEXE" ] && "$PYEXE" -c "import fungicapture, PySide6" >/dev/null 2>&1; then
    say "       an existing environment is present and healthy - keeping it"
    say "       (this preserves the SAM 3 model if you have installed it)"
else
    [ -d "$VENV" ] && say "       the existing environment is unusable - rebuilding it"
    rm -rf "$VENV"
    if ! "$PY" -m venv "$VENV" >>"$LOG" 2>&1; then
        say ""
        rule
        say "  Could not create the private environment. On Debian/Ubuntu"
        say "  this usually means python3-venv is missing:"
        say ""
        say "    $INSTALL_CMD python3-venv"
        say ""
        say "  Details are in  $LOG"
        rule
        pause_at_end
        exit 1
    fi
fi

"$PYEXE" -m pip install --upgrade pip >>"$LOG" 2>&1
# A normal (non-editable) install so the folder can be moved later without
# breaking imports - the same choice the everyday launcher makes.
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
#  4. Prove a window can actually open
# ---------------------------------------------------------------
say " [4/4] Checking a window can open..."
if ! "$PYEXE" - >>"$LOG" 2>&1 <<'PY'
import sys
from PySide6.QtWidgets import QApplication
QApplication([])          # aborts here if the Qt platform plugin cannot load
sys.exit(0)
PY
then
    say ""
    rule
    say "  Everything installed, but a test window would not open."
    say ""
    say "  The graphics libraries may not have taken effect. Try"
    say "  logging out and back in, then run this file again. If it"
    say "  persists, install these by hand and retry:"
    say ""
    say "    $INSTALL_CMD $GUI_PKGS"
    say ""
    say "  Details are in  $LOG"
    rule
    pause_at_end
    exit 1
fi

# Remember where this was built so the launcher can detect a later move.
printf '%s' "$HERE" > "$STAMP"

say ""
rule
say "  Setup finished successfully."
rule
say ""
say " From now on, just start FungiCapture the normal way:"
say ""
say "     double-click  START FungiCapture (Linux).sh"
say ""
say " (The deep-learning SAM 3 model is optional and about 2 GB. Add"
say "  it later from inside the app: the '2 Segment' tab, button"
say "  'Segmentation setup...'. FungiCapture works without it.)"
say ""

# Offer to launch straight away if a launcher is present.
LAUNCHER="START FungiCapture (Linux).sh"
if [ -f "$LAUNCHER" ]; then
    read -r -p " Start FungiCapture now? [Y/n] " ans 2>/dev/null || ans="n"
    case "${ans:-Y}" in
        n|N|no|No) say " You can start it any time by double-clicking the launcher." ;;
        *) exec bash "$LAUNCHER" ;;
    esac
fi

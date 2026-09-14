#!/usr/bin/env bash
# ===================================================================
#  FungiCapture launcher for Linux
#
#  Double-click this file, or run:  bash "START FungiCapture (Linux).sh"
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

# When launched by double-click there may be no terminal attached, so
# reopen in one if we can - otherwise the user sees nothing at all.
if [ -z "${FUNGICAPTURE_IN_TERMINAL:-}" ] && [ ! -t 1 ]; then
    export FUNGICAPTURE_IN_TERMINAL=1
    for term in x-terminal-emulator gnome-terminal konsole xfce4-terminal xterm; do
        if command -v "$term" >/dev/null 2>&1; then
            exec "$term" -e bash "$0"
        fi
    done
    # No terminal found: carry on silently rather than doing nothing.
fi

say() { printf '%s\n' "$*"; }
pause_at_end() {
    say ""
    read -r -p "Press Enter to close this window. " _ 2>/dev/null || sleep 30
}

# The command that installs the graphical (Qt/xcb) system libraries PySide6
# needs but cannot bundle. These live OUTSIDE the .venv - they are system
# packages - which is why a perfectly good install can still fail to open a
# window after a system update or a graphics-driver reinstall.
gui_libs_command() {
    if command -v apt-get >/dev/null 2>&1 || command -v apt >/dev/null 2>&1; then
        printf 'sudo apt install libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1'
    elif command -v dnf >/dev/null 2>&1; then
        printf 'sudo dnf install xcb-util-cursor libxkbcommon-x11 libglvnd-egl'
    elif command -v pacman >/dev/null 2>&1; then
        printf 'sudo pacman -S --needed xcb-util-cursor libxkbcommon-x11 libglvnd'
    elif command -v zypper >/dev/null 2>&1; then
        printf 'sudo zypper install libxcb-cursor0 libxkbcommon-x11-0 libEGL1'
    else
        printf 'install the Qt xcb libraries for your distribution (libxcb-cursor, libxkbcommon-x11, libEGL)'
    fi
}

# Did the program fail because Qt could not start the graphical system, rather
# than because of a fault inside FungiCapture? That specific failure needs a
# SYSTEM library the .venv cannot provide, so it gets its own, actionable
# message instead of a generic "it crashed".
looks_like_missing_gui_libs() {
    tail -n 60 "$LOG" 2>/dev/null \
        | grep -qiE 'Qt platform plugin|xcb-cursor|no Qt platform plugin could be'
}

report_start_failure() {
    say ""
    say " --------------------------------------------------------------"
    if looks_like_missing_gui_libs; then
        say "  FungiCapture could not open its window."
        say ""
        say "  Some graphical system libraries that Qt needs are missing."
        say "  This commonly happens after a system update or after"
        say "  reinstalling graphics / NVIDIA drivers."
        say ""
        say "  Install them once with:"
        say ""
        say "    $(gui_libs_command)"
        say ""
        say "  then start FungiCapture again."
    else
        say "  FungiCapture closed unexpectedly."
        say "  Details are in  $LOG"
    fi
    say " --------------------------------------------------------------"
}

# Point Python at the source folder as well. This is a safety net: even if
# the installed copy is damaged, the program still starts from the source
# sitting right here.
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

start_program() {
    "$PYEXE" -m fungicapture.gui.app 2>>"$LOG"
}

# Actually open (and immediately close) a Qt window. Importing PySide6 is not a
# real test: the import succeeds even when the graphical system libraries are
# missing, and the failure only shows up later when a window is created. This
# catches that at setup time, while we can still tell the user what to install.
gui_preflight() {
    "$PYEXE" - >>"$LOG" 2>&1 <<'PY'
import sys
from PySide6.QtWidgets import QApplication
QApplication([])          # aborts here if the Qt platform plugin cannot load
sys.exit(0)
PY
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
    report_start_failure
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
    say "  Install it with one of these, then run this file again:"
    say ""
    say "    Ubuntu / Debian / Mint:"
    say "      sudo apt install python3 python3-venv python3-pip"
    say ""
    say "    Fedora:"
    say "      sudo dnf install python3 python3-pip"
    say ""
    say "    Arch:"
    say "      sudo pacman -S python python-pip"
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
    say "  On Ubuntu and Debian this usually means one package is"
    say "  missing. Install it and try again:"
    say ""
    say "    sudo apt install python3-venv"
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
    say "  Setup finished but FungiCapture itself could not be loaded."
    say "  Details are in  $LOG"
    say " --------------------------------------------------------------"
    pause_at_end
    exit 1
fi
# The import worked; now make sure a window can actually open. This is the
# check that catches missing system graphics libraries.
if ! gui_preflight; then
    say ""
    say " --------------------------------------------------------------"
    say "  Setup finished but the window could not open."
    say ""
    say "  Some graphical system libraries that Qt needs are missing."
    say "  Install them once with:"
    say ""
    say "    $(gui_libs_command)"
    say ""
    say "  then start FungiCapture again. Details are in  $LOG"
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
start_program || { report_start_failure; pause_at_end; exit 1; }

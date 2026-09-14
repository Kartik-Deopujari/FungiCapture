@echo off
REM ===================================================================
REM  FungiCapture launcher for Windows
REM
REM  Double-click this file. The first run sets everything up; later
REM  runs just open the program.
REM
REM  Everything is installed inside this folder, in a sub-folder called
REM  ".venv". Nothing else on the computer is changed, and deleting this
REM  folder removes it completely.
REM
REM  Moving or renaming the folder is fine: the launcher notices and
REM  repairs itself.
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
set "HERE=%CD%"

set "LOG=setup_log.txt"
set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.installed-at"

REM Point Python at the source folder as well. This is a safety net: even if
REM the installed copy is damaged, the program still starts from the source
REM sitting right here.
set "PYTHONPATH=%HERE%\src;%PYTHONPATH%"

echo.
echo  ==========================================
echo    FungiCapture
echo  ==========================================
echo.

REM ---------- is the existing setup usable? ----------
REM Checking that the file exists is not enough: a setup made in a different
REM folder still leaves the file there but cannot import anything, which is
REM exactly how this used to fail after someone moved the folder. So we check
REM where it was built, and then test by actually importing.
set "SETUP_OK="
if exist "%PYEXE%" (
    if exist "%STAMP%" (
        set /p BUILT_AT=<"%STAMP%"
        if /i "!BUILT_AT!"=="%HERE%" (
            "%PYEXE%" -c "import fungicapture, PySide6" >nul 2>&1
            if not errorlevel 1 set "SETUP_OK=1"
        )
    )
)

if defined SETUP_OK (
    echo  Starting FungiCapture...
    echo.
    "%PYEXE%" -m fungicapture.gui.app
    if errorlevel 1 goto :runfailed
    goto :end
)

REM ---------- needs setting up, or repairing ----------
if exist "%VENV%" (
    echo  The setup in this folder is out of date.
    echo  That normally means the folder was moved or renamed since last time.
    echo  Repairing it now - this takes a few minutes.
    echo.
    rmdir /s /q "%VENV%"
) else (
    echo  First run - setting things up. This takes a few minutes.
    echo  Please leave this window open.
    echo.
)

REM ---------- find Python ----------
echo  [1/4] Looking for Python...

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY goto :nopython

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :oldpython

for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%v"
echo        found Python !PYVER!

REM ---------- build the private environment ----------
echo  [2/4] Creating a private setup inside this folder...
%PY% -m venv "%VENV%" >>"%LOG%" 2>&1
if errorlevel 1 goto :venvfailed

echo  [3/4] Downloading the parts FungiCapture needs...
echo        (this is the slow step - a few minutes)
"%PYEXE%" -m pip install --upgrade pip >>"%LOG%" 2>&1

REM A normal install, NOT "pip install -e". An editable install records the
REM absolute path of this folder inside the setup; move the folder and every
REM import breaks. A normal install copies the code in, so the setup keeps
REM working wherever the folder ends up.
"%PYEXE%" -m pip install ".[gui]" >>"%LOG%" 2>&1
if errorlevel 1 goto :installfailed

echo  [4/4] Checking everything works...
"%PYEXE%" -c "import fungicapture, PySide6" >>"%LOG%" 2>&1
if errorlevel 1 goto :installfailed
REM Importing PySide6 is not enough - it succeeds even when a window cannot
REM open. Actually create (and drop) a Qt application to be sure.
"%PYEXE%" -c "import sys; from PySide6.QtWidgets import QApplication; QApplication([]); sys.exit(0)" >>"%LOG%" 2>&1
if errorlevel 1 goto :windowfailed

REM Remember where this setup was built, so a later move is detected.
> "%STAMP%" echo|set /p="%HERE%"

echo.
echo  Setup finished.
echo.
echo  Note: the deep-learning segmentation model (SAM 3) is NOT installed
echo  yet - it is about 2 GB, so it is kept optional. FungiCapture works
echo  without it using its built-in segmenter.
echo.
echo  To add it: open the "2 Segment" tab and press "Segmentation setup...".
echo  That checks your graphics card and installs the matching version.
echo.
echo  Starting FungiCapture...
echo  Next time, this will open in a few seconds.
echo.
"%PYEXE%" -m fungicapture.gui.app
if errorlevel 1 goto :runfailed
goto :end


REM ===================================================================
REM  Problems, with plain-language explanations
REM ===================================================================

:nopython
echo.
echo  --------------------------------------------------------------
echo   Python is not installed on this computer.
echo.
echo   FungiCapture needs it. Python is free and safe.
echo.
echo     1. Go to  https://www.python.org/downloads/
echo     2. Download Python 3.11 or newer.
echo     3. IMPORTANT: on the FIRST screen of the installer, tick
echo        "Add Python to PATH"  before clicking Install.
echo     4. Install it, then double-click this file again.
echo  --------------------------------------------------------------
echo.
pause
goto :end

:oldpython
echo.
echo  --------------------------------------------------------------
echo   The Python on this computer is too old.
echo   FungiCapture needs Python 3.10 or newer.
echo.
echo   Install a newer one from  https://www.python.org/downloads/
echo   and remember to tick "Add Python to PATH".
echo  --------------------------------------------------------------
echo.
pause
goto :end

:venvfailed
echo.
echo  --------------------------------------------------------------
echo   Could not create the private setup folder.
echo.
echo   This usually means one of two things:
echo     - This folder is read-only. Move FungiCapture to your
echo       Documents folder and try again.
echo     - Antivirus software blocked it. Allow the folder and retry.
echo.
echo   Details were written to  %LOG%
echo  --------------------------------------------------------------
echo.
pause
goto :end

:installfailed
echo.
echo  --------------------------------------------------------------
echo   The download did not finish.
echo.
echo   Most often this is simply the internet connection dropping.
echo   Check you are online and double-click this file again - it
echo   will carry on from where it stopped.
echo.
echo   If it keeps failing, send the file  %LOG%  to whoever gave
echo   you FungiCapture.
echo  --------------------------------------------------------------
echo.
pause
goto :end

:windowfailed
echo.
echo  --------------------------------------------------------------
echo   Setup finished, but a test window would not open.
echo.
echo   Try restarting the computer and running this file again.
echo   If it persists, send the file  %LOG%  to whoever gave you
echo   FungiCapture.
echo  --------------------------------------------------------------
echo.
pause
goto :end

:runfailed
echo.
echo  --------------------------------------------------------------
echo   FungiCapture closed unexpectedly.
echo   Details are in  %LOG%
echo.
echo   If this keeps happening, delete the ".venv" folder inside
echo   this folder and run this file again to rebuild the setup.
echo  --------------------------------------------------------------
echo.
"%PYEXE%" -m fungicapture.gui.app >>"%LOG%" 2>&1
pause
goto :end

:end
endlocal

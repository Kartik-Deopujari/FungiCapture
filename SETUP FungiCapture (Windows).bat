@echo off
REM ===================================================================
REM  FungiCapture - first-time setup for Windows
REM
REM  Double-click this file once, the first time, on a new PC.
REM  (If Windows shows "Windows protected your PC": click "More info",
REM   then "Run anyway". The file is just unsigned, nothing is wrong.)
REM
REM  It builds the private environment (.venv) inside this folder and
REM  downloads everything FungiCapture needs, then checks a window can
REM  really open. On Windows all the graphics libraries are bundled, so
REM  no administrator rights and no system changes are needed.
REM
REM  After this finishes, start the program the normal way any time:
REM      double-click  START FungiCapture (Windows).bat
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
set "HERE=%CD%"

set "LOG=setup_log.txt"
set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.installed-at"

echo.
echo  ==========================================
echo    FungiCapture - first-time setup (Windows)
echo  ==========================================
echo.
echo  This runs once and needs an internet connection.
echo.

REM ---------- find Python ----------
echo  [1/3] Looking for Python 3.10+...
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
REM Do NOT wipe a healthy environment: the optional SAM 3 libraries (PyTorch +
REM Ultralytics, ~2 GB) are installed into this same .venv from inside the app,
REM and deleting it every run would throw that away. Rebuild only if unusable.
echo  [2/3] Building the private environment (a few minutes)...
set "VENV_OK="
if exist "%PYEXE%" (
    "%PYEXE%" -c "import fungicapture, PySide6" >nul 2>&1
    if not errorlevel 1 set "VENV_OK=1"
)
if defined VENV_OK (
    echo        an existing environment is present and healthy - keeping it
    echo        ^(this preserves the SAM 3 model if you have installed it^)
) else (
    if exist "%VENV%" rmdir /s /q "%VENV%"
    %PY% -m venv "%VENV%" >>"%LOG%" 2>&1
    if errorlevel 1 goto :venvfailed
)

"%PYEXE%" -m pip install --upgrade pip >>"%LOG%" 2>&1
REM Normal (non-editable) install so the folder can be moved later.
"%PYEXE%" -m pip install ".[gui]" >>"%LOG%" 2>&1
if errorlevel 1 goto :installfailed

REM ---------- prove a window can actually open ----------
echo  [3/3] Checking a window can open...
"%PYEXE%" -c "import sys; from PySide6.QtWidgets import QApplication; QApplication([]); sys.exit(0)" >>"%LOG%" 2>&1
if errorlevel 1 goto :windowfailed

REM Remember where this setup was built, so a later move is detected.
> "%STAMP%" echo|set /p="%HERE%"

echo.
echo  --------------------------------------------------------------
echo   Setup finished successfully.
echo  --------------------------------------------------------------
echo.
echo  From now on, just start FungiCapture the normal way:
echo.
echo      double-click  START FungiCapture (Windows).bat
echo.
echo  (The deep-learning SAM 3 model is optional and about 2 GB. Add
echo   it later from inside the app: the "2 Segment" tab, button
echo   "Segmentation setup...". FungiCapture works without it.)
echo.

choice /c YN /n /m " Start FungiCapture now? [Y/N] "
if errorlevel 2 goto :dontstart
if not exist "START FungiCapture (Windows).bat" goto :end
call "START FungiCapture (Windows).bat"
goto :end

:dontstart
echo  You can start it any time by double-clicking the launcher.
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
echo   Check you are online and double-click this file again.
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
echo   Everything installed, but a test window would not open.
echo.
echo   Try restarting the computer and running this file again.
echo   If it persists, send the file  %LOG%  to whoever gave you
echo   FungiCapture.
echo  --------------------------------------------------------------
echo.
pause
goto :end

:end
endlocal

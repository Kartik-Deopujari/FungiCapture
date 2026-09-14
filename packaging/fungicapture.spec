# PyInstaller specification for FungiCapture.
#
# Built in ONE-FOLDER mode, not one-file. That is a licence requirement, not a
# preference: PySide6 is LGPL-3.0, which obliges us to let a user replace the Qt
# libraries. A one-file bundle unpacks to a temporary directory and makes that
# impossible.
#
# Build with:   pyinstaller packaging/fungicapture.spec --noconfirm
#
# The SAM 3 weights are deliberately NOT bundled. They are covered by Meta's
# restricted SAM License and are gated behind an access request, so the
# application downloads them on first run from Meta's own page. See
# docs/LICENSING.md.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_root = Path(SPECPATH).parent

hidden_imports = []
# scikit-image and scikit-learn load parts of themselves lazily, so PyInstaller
# cannot see those imports by static analysis and must be told.
for package in ("skimage", "sklearn", "scipy"):
    hidden_imports += collect_submodules(package)

datas = []
datas += collect_data_files("skimage", includes=["**/*.pyi"])

# Trim the bundle. These are large and never used by the application; leaving
# them in roughly doubles the download for no benefit.
excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "sphinx",
    "matplotlib.tests",
    "numpy.tests",
    # Video encoding and decoding, pulled in by imageio, which scikit-image
    # uses for file reading. FungiCapture only ever reads still images, so the
    # bundled ffmpeg binaries are 77 MB of dead weight.
    "imageio_ffmpeg",
    "imageio.plugins.ffmpeg",
    "av",
]

a = Analysis(
    # Not gui/app.py directly: PyInstaller runs its target as __main__, with no
    # package around it, and the relative imports inside app.py then fail. The
    # entry-point script imports by absolute name instead.
    [str(project_root / "packaging" / "entry_point.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Only one copy of OpenCV. Installing both opencv-python and
# opencv-python-headless - easy to do by accident, since different packages
# depend on each of them - makes PyInstaller bundle two complete builds of the
# same library, adding about 100 MB. FungiCapture needs the headless one; the
# GUI is Qt, not OpenCV's own windowing.
_seen_opencv = set()
_filtered_binaries = []
for _entry in a.binaries:
    _name = _entry[0].lower()
    if "opencv_python.libs" in _name or "/cv2/qt/" in _name.replace("\\", "/"):
        continue
    if "opencv" in _name:
        _key = _name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if _key in _seen_opencv:
            continue
        _seen_opencv.add(_key)
    _filtered_binaries.append(_entry)
a.binaries = _filtered_binaries

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FungiCapture",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts Qt libraries on macOS
    console=False,      # no terminal window on Windows
    disable_windowed_traceback=False,
    target_arch=None,   # set to "universal2" for a universal macOS build
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "packaging" / "icon.icns")
    if sys.platform == "darwin"
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FungiCapture",
)

# macOS: wrap the folder in a .app bundle so it behaves like a normal
# application. Without signing, Gatekeeper will warn on first launch - users
# right-click and choose Open, or you buy an Apple Developer ID.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="FungiCapture.app",
        icon=str(project_root / "packaging" / "icon.icns"),
        bundle_identifier="org.fungicapture.app",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )

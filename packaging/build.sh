#!/usr/bin/env bash
# Build FungiCapture for the current platform.
#
# Run from the repository root:  bash packaging/build.sh
set -euo pipefail

echo "==> Checking Python"
python3 --version

echo "==> Installing build dependencies"
python3 -m pip install --upgrade pip
# CPU-only torch keeps the bundle near 400 MB instead of several GB. Users with
# a CUDA GPU install the GPU build themselves afterwards.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu || true
python3 -m pip install -e ".[gui]"
python3 -m pip install pyinstaller pip-licenses

echo "==> Recording third-party licences"
pip-licenses --format=markdown --with-urls --output-file THIRD_PARTY_LICENSES.md
echo "    wrote THIRD_PARTY_LICENSES.md"

echo "==> Running tests"
python3 -m pytest -q

echo "==> Building"
pyinstaller packaging/fungicapture.spec --noconfirm

case "$(uname -s)" in
  Darwin)
    echo "==> Creating a disk image"
    hdiutil create -volname FungiCapture -srcfolder dist/FungiCapture.app \
            -ov -format UDZO dist/FungiCapture-macos.dmg
    echo "    dist/FungiCapture-macos.dmg"
    echo
    echo "    NOTE: unsigned. On first launch macOS will warn. Either"
    echo "    right-click the app and choose Open, or sign it with an"
    echo "    Apple Developer ID (99 USD/year)."
    ;;
  Linux)
    echo "==> Creating a tarball"
    (cd dist && tar czf FungiCapture-linux-x86_64.tar.gz FungiCapture)
    echo "    dist/FungiCapture-linux-x86_64.tar.gz"
    ;;
  *)
    echo "==> Windows: zip dist/FungiCapture, or build an installer with Inno Setup"
    ;;
esac

echo
echo "Done. The SAM 3 weights are NOT bundled - the application downloads them"
echo "on first run from Meta's gated page, as their licence requires."

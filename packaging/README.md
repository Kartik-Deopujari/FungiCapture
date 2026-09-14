# Packaging FungiCapture

## Build for the current platform

```bash
bash packaging/build.sh
```

Outputs land in `dist/`.

## What gets built

| Platform | Output | Notes |
|---|---|---|
| Windows | `dist/FungiCapture/` | Zip it, or wrap with Inno Setup for an installer |
| macOS | `dist/FungiCapture-macos.dmg` | Unsigned — see below |
| Linux | `dist/FungiCapture-linux-x86_64.tar.gz` | Runs on most distributions |

## Two things that are deliberate

**One-folder, not one-file.** PySide6 is LGPL-3.0, which obliges us to let a
user replace the Qt libraries. A one-file bundle unpacks to a temporary
directory and makes that impossible. This is a licence requirement, not a
preference.

**The SAM 3 weights are not bundled.** They are covered by Meta's restricted
SAM License and are gated behind an access request, so the application sends
the user to Meta's own page and caches the file locally. Never mirror the
weights in a release. See `docs/LICENSING.md`.

## macOS signing

Without an Apple Developer ID (99 USD/year), Gatekeeper shows a warning on
first launch. Users can right-click the app and choose Open to get past it, but
for a conference release signing is worth the money.

With a certificate:

```bash
codesign --deep --force --options runtime \
         --sign "Developer ID Application: YOUR NAME (TEAMID)" \
         dist/FungiCapture.app
xcrun notarytool submit dist/FungiCapture-macos.dmg \
      --apple-id YOU@EXAMPLE.COM --team-id TEAMID --wait
xcrun stapler staple dist/FungiCapture-macos.dmg
```

## Size

Roughly 400–600 MB unpacked, most of it PyTorch. If that is too much for your
users, build without the `model` extra: the application still runs with the
classical segmenter and drops to about 250 MB.

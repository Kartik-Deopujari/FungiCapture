# NOTICE — FungiCapture third-party attributions

FungiCapture

This program is licensed under the **GNU Affero General Public License, version 3
or later (AGPL-3.0-or-later)**. The complete, legally binding license text is in
[`LICENSE`](LICENSE). This NOTICE file lists the third-party components
FungiCapture depends on and the attribution each one requires. It is
informational; where it and a component's own license text differ, the license
text governs.

---

## 1. Why FungiCapture is AGPL-3.0

The optional SAM 3 segmentation backend imports from **Ultralytics**, which is
licensed **AGPL-3.0-only**. Under the conservative compliance interpretation,
that requires FungiCapture's own source to be released under AGPL-3.0. Ultralytics
and its deep-learning stack live in the optional `[model]` extra; the default
installation (classical Otsu segmentation backend) does **not** install any
AGPL-licensed code.

---

## 2. Core runtime dependencies

| Package | License (SPDX) |
|---|---|
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| matplotlib | Matplotlib License (PSF-derived, BSD-compatible) |
| Pillow | MIT-CMU (HPND) — bundles zlib, libjpeg |
| platformdirs | MIT |
| tomli-w | MIT |
| tomli (Python < 3.11 only) | MIT |

### Components with a wrapper / bundled-native split

| Package | Wrapper | Bundled native library | Native license | Required notice |
|---|---|---|---|---|
| `rawpy` | MIT | LibRaw | LGPL-2.1-only OR CDDL-1.0 | Retain the LibRaw notice; LibRaw is dynamically linked |
| `opencv-python-headless` | scripts MIT; OpenCV Apache-2.0 | **FFmpeg** | **LGPL-2.1** | Retain the FFmpeg notice; FFmpeg is dynamically linked |
| `imagecodecs` | BSD-3-Clause | libjpeg-turbo, libpng, OpenJPEG, zstd, bitshuffle, liblzf, libspng, and others | mixed permissive | Ship the bundled `imagecodecs/licenses/` folder. Includes the mandatory notice: *"This software is based in part on the work of the Independent JPEG Group."* |

---

## 3. Optional extra: `[gui]`

| Package | License | Choice recorded |
|---|---|---|
| PySide6 (Qt for Python) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (commercial option also exists) | **FungiCapture distributes PySide6 under the LGPL-3.0-only option.** |

---

## 4. Optional extra: `[model]` — SAM 3

| Component | License | Status |
|---|---|---|
| `ultralytics` (≥ 8.3.237) | AGPL-3.0-only (or paid Enterprise license) | Reason FungiCapture is AGPL-3.0 (§1) |
| `torch` | BSD-3-Clause | — |
| `torchvision` | BSD-3-Clause | — |
| **SAM 3 model weights** | **Meta SAM License** (see [`LICENSES/SAM_LICENSE.txt`](LICENSES/SAM_LICENSE.txt)) | **Not redistributed with FungiCapture** |

### The SAM 3 weights

Colony segmentation can use the **Segment Anything Model 3 (SAM 3)**, released by
Meta AI under the **SAM License** — a restricted research-and-community license,
**not** an OSI-approved open-source license. FungiCapture does **not** redistribute
the SAM 3 weights: the application directs each user to Meta's gated model page so
they request access and accept the license themselves. The full SAM License text
is included at [`LICENSES/SAM_LICENSE.txt`](LICENSES/SAM_LICENSE.txt) for
reference.


Model source and gated download: <https://github.com/facebookresearch/sam3> ·
<https://huggingface.co/facebook/sam3>

---

## 5. Build / development tools (not distributed in the running application)

`pytest`, `pytest-cov` (MIT); `ruff` (MIT); `PyInstaller`
(GPL-2.0-or-later WITH Bootloader-exception — the exception permits packaging a
non-GPL application).

---



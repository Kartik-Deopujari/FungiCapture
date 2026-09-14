# FungiCapture — Dependencies and Their Licences

**Date:** 14 September 2026
**Version covered:** `fungicapture` 0.1.0.dev0 (folder `fungicapture-v0.2.0`)
**Source of truth:** `pyproject.toml` and `src/fungicapture.egg-info/requires.txt`
as they exist in the repository today.

> **This is not legal advice.** It is an engineer's reading of published licence
> texts, prepared so that you can take a short, specific list of questions to
> your supervisor and to TU Wien's technology-transfer office. Every conclusion
> marked ⚠ needs a qualified opinion before public release.

---

## 1. How to read this document

A Python package can carry **two** licences that matter to you:

1. **The wrapper** — the Python code in the package itself.
2. **The native library it bundles** — compiled C or C++ code shipped inside
   the wheel.

These are often different, and the bundled one is usually the stricter of the
two. The red-team review was right to insist they be listed separately. Three
of your dependencies have exactly this split: `rawpy`, `opencv-python-headless`
and `imagecodecs`.

A second thing to keep straight: **"required" versus "optional"**. Your
`pyproject.toml` splits dependencies into a core set and three extras
(`gui`, `model`, `dev`). That split is not cosmetic — it is the single most
important fact in this whole document, and section 5 explains why.

---

## 2. Core runtime dependencies

These install with a plain `pip install fungicapture`. All of them are
permissive. None of them constrains what licence FungiCapture can carry.

| Package | Min version | SPDX licence | Bundled native code | Obligation on you |
|---|---|---|---|---|
| `numpy` | ≥ 1.24 | `BSD-3-Clause` | — | Keep the copyright notice |
| `scipy` | ≥ 1.10 | `BSD-3-Clause` | — | Keep the copyright notice |
| `pandas` | ≥ 2.0 | `BSD-3-Clause` | — | Keep the copyright notice |
| `scikit-image` | ≥ 0.22 | `BSD-3-Clause` | — | Keep the copyright notice |
| `scikit-learn` | ≥ 1.3 | `BSD-3-Clause` | — | Keep the copyright notice |
| `matplotlib` | ≥ 3.7 | Matplotlib License (PSF-derived, BSD-compatible) | — | Keep the notice |
| `Pillow` | ≥ 10.0 | `MIT-CMU` (HPND) | zlib, libjpeg | Keep the notice |
| `platformdirs` | ≥ 3.0 | `MIT` | — | Keep the notice |
| `tomli-w` | ≥ 1.0 | `MIT` | — | Keep the notice |
| `tomli` | ≥ 2.0, Python < 3.11 only | `MIT` | — | Keep the notice |

### The three with a wrapper/native split

| Package | Wrapper licence | Native library | Native licence | Notes |
|---|---|---|---|---|
| `rawpy` ≥ 0.19 | **`MIT`** | LibRaw | **`LGPL-2.1-only OR CDDL-1.0`** | Your August documents listed `rawpy` itself as LGPL. **That is wrong** — rawpy is MIT. The red-team review is correct here. The rawpy project explicitly excludes the GPL2/GPL3 demosaic packs *because* MIT is incompatible with GPL |
| `opencv-python-headless` ≥ 4.8 | Scripts `MIT`; OpenCV itself `Apache-2.0` | **FFmpeg** | **`LGPL-2.1`** | ⚠ Every wheel bundles FFmpeg under LGPLv2.1. This is a real LGPL obligation you currently do not document. The *headless* build correctly avoids the Qt 5 `LGPL-3.0` that the desktop build ships |
| `imagecodecs` ≥ 2023.1 | `BSD-3-Clause` | many (libjpeg-turbo, libpng, OpenJPEG, zstd, bitshuffle, liblzf, libspng, …) | mixed permissive | ⚠ This package is **not in any of your existing licence documents**. It bundles a large set of codec libraries, each with its own licence, in `imagecodecs/licenses/`. It also carries the mandatory notice "This software is based in part on the work of the Independent JPEG Group" |

**Action:** `imagecodecs` is a compliance gap. It is not dangerous — everything
in it is permissive — but a licence audit that misses an entire package with a
dozen bundled libraries looks careless. Add it.

---

## 3. Optional extras

### 3.1 `[gui]`

| Package | SPDX licence | Obligation |
|---|---|---|
| `PySide6` ≥ 6.6 | **`LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`** (plus a commercial option) | You must pick one and say which |

Your existing documents list PySide6 as "LGPL-3.0". That is incomplete — the
PyPI licence field is the three-way expression above. Record your choice
explicitly:

> FungiCapture distributes PySide6 under the **LGPL-3.0-only** option.

LGPL-3.0 requires that the user be able to **replace** the Qt libraries with
their own version. In practice this means:

* A one-folder PyInstaller build (Qt as separate `.so`/`.dll` files) —
  your `packaging/fungicapture.spec` should be checked to confirm it is
  one-folder, not one-file.
* Shipping the LGPL-3.0 text.
* Telling the user, in writing, that they may replace the Qt libraries.

⚠ A one-folder build makes replacement *possible*. It does not by itself prove
every LGPL clause is satisfied. Confirm with your institution.

### 3.2 `[model]` — the extra that decides everything

| Package | SPDX licence | Status |
|---|---|---|
| `ultralytics` ≥ 8.3.237 | **`AGPL-3.0-only`** (or a paid Enterprise licence) | ⚠ **This is why FungiCapture is AGPL** |
| `torch` ≥ 2.1 | `BSD-3-Clause` | Fine |
| `torchvision` ≥ 0.16 | `BSD-3-Clause` | Fine |

Ultralytics' published position is that any application depending on their
package must release its complete source under AGPL-3.0, or buy an Enterprise
licence. Your `pyproject.toml` already declares
`license = { text = "AGPL-3.0-or-later" }`, and the repository root already
carries the full AGPL-3.0 text in `LICENSE` (34 kB — this is the real text, not
a summary, which is correct).

**Note for your own understanding:** Ultralytics' position is a licensor's
interpretation, not a court ruling. How far AGPL reaches into a work that merely
*imports* a library is a genuinely contested question in software law, and it
depends on architecture and on how the work is distributed. You have chosen the
conservative route, which is the right call for an academic tool. Describe it as
"the conservative compliance interpretation", not as settled law.

### 3.3 `[dev]` — not distributed, but list them anyway

| Package | SPDX licence |
|---|---|
| `pytest` ≥ 7.4 | `MIT` |
| `pytest-cov` ≥ 4.1 | `MIT` |
| `hypothesis` ≥ 6.90 | `MPL-2.0` |
| `pip-licenses` ≥ 4.3 | `MIT` |

These are development-only and are not shipped to users, so they create no
distribution obligation. List them for completeness.

### 3.4 Build and packaging

| Tool | SPDX licence | Note |
|---|---|---|
| `setuptools`, `wheel` | `MIT` | Build only |
| `PyInstaller` | `GPL-2.0-or-later WITH Bootloader-exception` | The bootloader exception explicitly permits packaging non-GPL and commercially licensed programs. ⚠ The exception covers the **bootloader**; the licences of bundled runtime hooks from `pyinstaller-hooks-contrib` must be checked separately |

---

## 4. Model weights — not software, and not open source

| Artefact | Licence | Status |
|---|---|---|
| SAM 3 weights (`facebook/sam3`) | **Meta "SAM License"**, 19 Nov 2025 | ⚠ Restricted research/community licence. **Not OSI-approved. Never call it "open source."** |
| `sam3.1_multiplex.pt` (your fine-tune) | Derivative work of SAM 3 | ⚠ **Unresolved** |

Your code already handles this correctly. `core/segment.py` contains:

```python
SAM3_MODEL_PAGE = "https://huggingface.co/facebook/sam3"
SAM3_LICENCE_NOTE = (
    "The SAM 3 weights are released by Meta under the SAM License, which is a "
    "restricted research licence, not an open-source one. You must request "
    "access on Meta's model page and accept the licence yourself. FungiCapture "
    "does not redistribute the weights.\n\n"
    "If you publish results made with SAM 3, the licence requires you to "
    "acknowledge it in your methods."
)
```

That is the right design: the user goes to Meta's gated page and accepts the
licence themselves; you never mirror the file.

**One correction to your existing documents.** `FungiCapture_LICENSING.md` says
bundling the weights is "effectively required" not to do. The red-team review is
right that this overstates the legal position: Meta's licence *permits*
redistribution and derivative works provided they stay under its terms and the
licence text travels with them. Not bundling is a **prudent risk-control
decision**, not a universal legal prohibition. Reword it as *"recommended,
pending institutional legal review"*.

**The unresolved question — take this one to your supervisor:**

> `sam3.1_multiplex.pt` is a derivative work of SAM 3. Under the
> derivative-works clause of the SAM License, may we publish it, and under what
> terms? Separately: if we ship it alongside AGPL-3.0-licensed Ultralytics code
> as one combined work, are the two sets of conditions compatible? The SAM
> licence adds field-of-use restrictions (military, weapons, espionage,
> nuclear) that AGPL-3.0 does not permit to be added to AGPL-covered code.

That second half is the sharper question and the one I would put in writing.

---

## 5. The architectural fact that could change your licence later

This is worth understanding properly, because it is unusual and you built it
deliberately.

`ultralytics` is **not a core dependency**. It sits in the optional `[model]`
extra. Your own comment in `pyproject.toml` says:

```python
# core/ dependencies only. The GUI and the model backend are extras, so the
# science can be installed and run headless on a compute cluster.
```

And `core/segment.py` ships a **classical Otsu fallback backend** that needs no
model at all. So there exists a complete, working, useful installation of
FungiCapture — feature extraction, colour, calibration, phenotype scoring, CLI —
that never touches any AGPL code.

That does **not** mean you can relicense today. But it means:

* The question "is the whole of FungiCapture a derivative work of Ultralytics?"
  is a much more interesting question here than in a typical project, because
  the core genuinely runs without it.
* The escape route your `LICENSING.md` describes is shorter than it says: only
  the SAM 3 backend inside `core/segment.py` imports Ultralytics. Replacing that
  one code path with HuggingFace `transformers` (Apache-2.0) would remove the
  AGPL trigger entirely. The **weights** would still be restricted.

**My recommendation, unchanged from your August plan:** ship v1.0 as AGPL-3.0.
It is honest, it is normal for academic software, and it costs you nothing. Note
the `transformers` route as future work. Do not delay release over it.

---

## 6. Compliance gaps to close before release

| # | Gap | Effort |
|---|---|---|
| 1 | `imagecodecs` missing from every licence document | 10 min |
| 2 | FFmpeg (LGPL-2.1) inside `opencv-python-headless` not documented | 10 min |
| 3 | `rawpy` wrongly recorded as LGPL; it is MIT. LibRaw is the LGPL/CDDL part | 5 min |
| 4 | PySide6 recorded as LGPL-3.0 only; the real expression is three-way and your chosen route is not stated | 10 min |
| 5 | No `THIRD_PARTY_LICENSES.md` in the repository | 15 min, automated |
| 6 | `pyinstaller-hooks-contrib` hook licences not audited | 30 min |
| 7 | No SBOM | 30 min |
| 8 | SAM-vs-AGPL combined-work question unanswered in writing | Ask the institution |
| 9 | `sam3.1_multiplex.pt` redistribution question unanswered in writing | Ask the institution |

### Generating the machine-readable list

Run this in the project's own virtual environment, with **all** extras
installed, so nothing is missed:

```bash
pip install ".[gui,model,dev]"
pip install pip-licenses

pip-licenses \
    --format=markdown \
    --with-urls \
    --with-license-file \
    --with-authors \
    --output-file THIRD_PARTY_LICENSES.md
```

⚠ **`pip-licenses` is necessary but not sufficient.** It reads Python package
metadata only. It cannot see:

* bundled native libraries (FFmpeg inside OpenCV, LibRaw inside rawpy, the whole
  `imagecodecs/licenses/` folder),
* model weights,
* fonts, icons or codecs,
* code copied by hand from a blog or a Stack Overflow answer.

Those four must be audited by hand, once, and the result written down.

### SBOM

```bash
pip install cyclonedx-bom
cyclonedx-py environment -o fungicapture-sbom.json
```

Commit the SBOM with each release. It is becoming a standard expectation for
research software, and generating it takes a minute.

---

## 7. Summary table — the full picture on one page

| Component | SPDX | Distributed? | Obligation |
|---|---|---|---|
| **FungiCapture itself** | `AGPL-3.0-or-later` | Yes | Publish complete source |
| numpy, scipy, pandas, scikit-image, scikit-learn, torch, torchvision | `BSD-3-Clause` | Yes | Copyright notice |
| matplotlib | Matplotlib License | Yes | Notice |
| Pillow | `MIT-CMU` | Yes | Notice |
| platformdirs, tomli, tomli-w | `MIT` | Yes | Notice |
| imagecodecs | `BSD-3-Clause` + bundled codecs | Yes | Notice + ship `imagecodecs/licenses/` + IJG notice |
| rawpy | `MIT` | Yes | Notice |
| └ LibRaw | `LGPL-2.1-only OR CDDL-1.0` | Yes (native) | Dynamic linking + notice |
| opencv-python-headless | `MIT` / OpenCV `Apache-2.0` | Yes | Notice |
| └ FFmpeg | `LGPL-2.1` | Yes (bundled) | Dynamic linking + notice |
| PySide6 | `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only` | Yes (GUI) | Choose LGPL-3.0-only; one-folder build; replaceability notice |
| ultralytics | `AGPL-3.0-only` | Optional extra | **Forces AGPL on the combined work** |
| PyInstaller | `GPL-2.0-or-later WITH Bootloader-exception` | Bootloader only | Exception permits non-GPL apps; audit hooks |
| pytest, pytest-cov, pip-licenses | `MIT` | No (dev) | None |
| hypothesis | `MPL-2.0` | No (dev) | None |
| SAM 3 weights | Meta SAM License | **No — user downloads** | Acknowledge in publications; never call it open source |
| `sam3.1_multiplex.pt` | Derivative of SAM 3 | ⚠ **Undecided** | Resolve before release |

---

## Sources checked on 14 September 2026

- [rawpy on PyPI](https://pypi.org/project/rawpy/) — licence expression: MIT
- [PySide6 on PyPI](https://pypi.org/project/PySide6/) — `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`
- [opencv-python-headless on PyPI](https://pypi.org/project/opencv-python-headless/) — bundled FFmpeg under LGPLv2.1
- [imagecodecs on PyPI](https://pypi.org/project/imagecodecs/) — BSD-3-Clause plus bundled codec licences
- [Ultralytics AGPL-3.0 licence terms](https://www.ultralytics.com/legal/agpl-3-0-software-license)
- [LibRaw](https://www.libraw.org/)
- [PyInstaller licence](https://pyinstaller.org/en/stable/license.html)

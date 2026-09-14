# Licensing — read before releasing

**Version:** 2.0 · 14 September 2026 · supersedes v1.0 of 21 August 2026

Two dependencies constrain what licence FungiCapture can carry. One is settled.
One is not.

> **Where to look.** This file covers the two decisions and the release
> checklist. The **complete dependency-by-dependency table**, with SPDX
> expressions, bundled native libraries and the obligation each creates, is in
> **`docs/DEPENDENCIES_AND_LICENSING.md`**. The user-facing summary is
> `LICENSE.md`. The binding legal text is `LICENSE`.
>
> **Corrections in v2.0:** `rawpy` was listed as LGPL — it is **MIT**; the
> LGPL/CDDL applies to LibRaw, the native library it wraps. PySide6 was listed
> as LGPL-3.0 only — the real expression is three-way. `imagecodecs` and the
> FFmpeg bundled inside `opencv-python-headless` were missing entirely. The
> claim that not bundling SAM weights is legally *required* has been softened to
> what the licence actually says.

> **This is not legal advice.** It is an engineer's reading of published licence
> texts, written so you can take a short list of specific questions to your
> supervisor and to TU Wien's technology-transfer office.

---

## 1. FungiCapture is AGPL-3.0. This was forced, not chosen.

The SAM 3 backend imports `SAM3SemanticPredictor` from the `ultralytics`
package. **Ultralytics is licensed AGPL-3.0.**

Ultralytics' published position is that anyone distributing software depending
on it must release the complete source of the whole derivative work under
AGPL-3.0, or buy a paid Enterprise licence.

**Status: settled and implemented.**

* `pyproject.toml` declares `license = { text = "AGPL-3.0-or-later" }`.
* `LICENSE` contains the full AGPL-3.0 text (34 kB). ✅ *This was a placeholder
  in v1.0 of this document; it is now the real text.*

**Consequences:**

* FungiCapture cannot be MIT or BSD while it depends on `ultralytics`.
* AGPL-3.0 is entirely normal for an academic tool. It means anyone who builds
  on FungiCapture must also open their source.
* If someone at a conference asks about using it inside a closed commercial
  product, the honest answer is: they would need an Ultralytics Enterprise
  licence for the segmentation backend. Be ready for that question.

**One nuance to state accurately.** Ultralytics' position is a *licensor's
interpretation*, not a court ruling. How far AGPL reaches into a work that
imports a library depends on architecture and on how the work is distributed.
You have taken the conservative route, which is right for an academic tool.
Describe it in the paper as **"the conservative compliance interpretation"**,
not as settled law.

### 1.1 The architecture makes a future escape shorter than v1.0 assumed

This is worth understanding, because it was built deliberately.

`ultralytics` and `torch` are **not core dependencies**. They sit in the
optional `[model]` extra, and `core/segment.py` ships a **classical Otsu
fallback backend** that needs no model at all. From `pyproject.toml`:

```python
# core/ dependencies only. The GUI and the model backend are extras, so the
# science can be installed and run headless on a compute cluster.
```

So a complete, working installation — feature extraction, colour calibration,
phenotype scoring, CLI — exists that never installs any AGPL-licensed code.

This does **not** let you relicense today. But it means only one code path
inside `core/segment.py` touches Ultralytics. Replacing it with the HuggingFace
`transformers` SAM 3 implementation (Apache-2.0) would remove the AGPL trigger
entirely. The **weights** would still be restricted — see §2 — so this is a
partial escape at best.

**Recommendation, unchanged:** ship v1.0 as AGPL-3.0. Note the `transformers`
route as future work. **Do not delay the release over this.**

---

## 2. ⚠ Unresolved: the fine-tuned weights

The SAM 3 weights are **not** Apache-licensed. They are covered by Meta's custom
**"SAM License"** (19 November 2025), a restricted research and community
licence — **not an OSI-approved open-source licence. Never call it "open
source."**

What it requires:

* **Gated access.** A user must request access on the Hugging Face model page
  before downloading. Ultralytics' documentation confirms the weights are not
  fetched automatically.
* **The licence travels with the file.** Any distribution of the weights, *or of
  any derivative work of them*, must carry a copy of the SAM licence.
* **Publication acknowledgement.** Research published using SAM materials must
  acknowledge them:

  > Colony segmentation used the Segment Anything Model 3 (SAM 3), released by
  > Meta AI under the SAM License.

* **Prohibited uses** include military, weapons, espionage and nuclear
  applications.

### 2.1 What this means for the application — implemented

The download-on-first-run design is correct and is implemented.
`core/segment.py` contains:

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

The app sends the user to Meta's own gated page so they accept the licence
themselves. **Do not mirror the weights on a GitHub release.**

> **⚠ Wording corrected in v2.0.** Version 1.0 said not bundling the weights was
> "effectively required". That overstates the licence. Meta's licence *permits*
> redistribution and derivative works provided they remain under its terms and
> the licence text accompanies them. Not bundling is a **prudent risk-control
> decision, recommended pending institutional legal review** — not a universal
> legal prohibition.

### 2.2 The open question — `sam3.1_multiplex.pt`

Your fine-tuned weights are a **derivative work of SAM 3**. Whether, and under
what terms, you may publish them is governed by the derivative-works clause of
the SAM licence, not by your own choice of licence.

**Two questions to put in writing to your supervisor and tech-transfer office:**

1. Under the derivative-works clause of the SAM License, may we publish
   `sam3.1_multiplex.pt`, and under what terms?
2. If SAM-derived weights are conveyed alongside AGPL-3.0-licensed Ultralytics
   code as one combined work, are the two sets of conditions compatible? The SAM
   licence adds field-of-use restrictions — military, weapons, espionage,
   nuclear — that AGPL-3.0 does not permit to be added to AGPL-covered code.

**The second question is the sharper one.** It is not a conference-corridor
question.

Until it is settled, the safe position is: **publish the code, the prompt sets
and the fine-tuning procedure so others can reproduce it, without
redistributing the weights file itself.**

### 2.3 Recommended release architecture

* Do not bundle SAM 3 weights until legal compatibility is reviewed.
* Keep model acquisition separate from the application installer. ✅ *done*
* Display the SAM licence before activating the backend. ✅ *done*
* Store model provenance and a cryptographic checksum. ✅ *`file_checksum()` in
  `core/segment.py`*
* Mark fine-tuned weights as derivatives of SAM materials.
* Ship the exact Meta licence text with any permitted derivative distribution.
* Obtain written institutional guidance on the SAM–AGPL combination.

---

## 3. Other dependencies — summary

Full detail in **`docs/DEPENDENCIES_AND_LICENSING.md`**. The short version:

| Package | SPDX | Ships safely? |
|---|---|---|
| NumPy, SciPy, pandas, scikit-learn, scikit-image, PyTorch | `BSD-3-Clause` | ✅ notice only |
| matplotlib | Matplotlib License (PSF-derived) | ✅ notice only |
| Pillow | `MIT-CMU` | ✅ notice only |
| platformdirs, tomli, tomli-w | `MIT` | ✅ notice only |
| **imagecodecs** | `BSD-3-Clause` + bundled codec libraries | ✅ — **but ship `imagecodecs/licenses/` and the Independent JPEG Group notice.** *Missing from v1.0* |
| **rawpy** | **`MIT`** | ✅ — *v1.0 wrongly listed this as LGPL* |
| └ **LibRaw** (native) | `LGPL-2.1-only OR CDDL-1.0` | ✅ dynamically linked. Keep the notice |
| **opencv-python-headless** | wrapper `MIT`; OpenCV `Apache-2.0` | ✅ |
| └ **FFmpeg** (bundled in the wheel) | `LGPL-2.1` | ✅ dynamically linked. **Keep the notice.** *Missing from v1.0* |
| **PySide6** | `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only` | ✅ — **select LGPL-3.0-only and say so.** Build one-folder so Qt stays replaceable |
| **PyInstaller** | `GPL-2.0-or-later WITH Bootloader-exception` | ✅ — the exception permits packaging non-GPL apps. ⚠ audit `pyinstaller-hooks-contrib` hooks separately |
| **ultralytics** | **`AGPL-3.0-only`** | ⚠ see §1 |
| **SAM 3 weights** | **Meta SAM License** | ⚠ see §2 — not distributed |

> ⚠ **A note on the opencv split.** `ultralytics` depends on `opencv-python`
> (the desktop build), which shares the same `cv2` folder as
> `opencv-python-headless` and whose bundled Qt plugins hijack PySide6's. The
> desktop build also ships **Qt 5 under LGPL-3.0**, which the headless build
> does not. So forcing headless is both a stability fix *and* one fewer LGPL
> obligation. `core/setup_env.py::install_segmentation` handles this; if you
> install extras by hand, repeat it:
>
> ```bash
> pip uninstall -y opencv-python
> pip install --force-reinstall --no-deps opencv-python-headless
> ```

---

## 4. Release checklist

**Done:**

- [x] Full AGPL-3.0 text in `LICENSE`
- [x] First-run download points at Meta's gated page; weights not mirrored
- [x] SAM licence note shown before the backend activates
- [x] Model checksum recorded
- [x] `LICENSE.md` user-facing summary written
- [x] `docs/DEPENDENCIES_AND_LICENSING.md` written

**Outstanding:**

- [ ] `THIRD_PARTY_LICENSES.md` generated **with all extras installed** and
      committed
- [ ] SAM licence text included in the repository
- [ ] `imagecodecs/licenses/` and the IJG notice shipped
- [ ] FFmpeg (LGPL-2.1) notice shipped
- [ ] LibRaw notice shipped
- [ ] PySide6 route recorded as LGPL-3.0-only; one-folder build confirmed in
      `packaging/fungicapture.spec`
- [ ] `pyinstaller-hooks-contrib` hook licences audited
- [ ] SBOM generated (`cyclonedx-py environment -o fungicapture-sbom.json`)
- [ ] SAM acknowledgement sentence in the paper's methods
- [ ] **Derivative-works question on `sam3.1_multiplex.pt` answered in writing**
- [ ] **SAM–AGPL combined-work compatibility answered in writing**
- [ ] `CITATION.cff` added so GitHub shows a "Cite this repository" button
- [ ] All novelty and licence language approved by supervisor and
      technology-transfer office

### Generating the third-party list

```bash
pip install ".[gui,model,dev]"     # all extras, so nothing is missed
pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-license-file --with-authors \
             --output-file THIRD_PARTY_LICENSES.md
```

Re-run on every release so it cannot go stale.

⚠ **`pip-licenses` is necessary but not sufficient.** It reads Python package
metadata only. It cannot see bundled native libraries (FFmpeg inside OpenCV,
LibRaw inside rawpy, the whole `imagecodecs/licenses/` folder), model weights,
fonts, codecs, or code copied by hand. Those four categories must be audited
manually, once, and the result written down.

---

## 5. How to describe this in the paper

Rename any heading like "licence acknowledgements required by law" to
**"provisional licence-compliance checklist — not legal advice."**

In the methods, state:

> FungiCapture is released under the GNU Affero General Public License v3.0 or
> later. Colony segmentation used the Segment Anything Model 3 (SAM 3), released
> by Meta AI under the SAM License; model weights are not redistributed with the
> application.

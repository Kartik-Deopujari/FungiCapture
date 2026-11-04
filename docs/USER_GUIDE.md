# FungiCapture — Setup & User Guide

FungiCapture turns photographs of fungal colonies on agar plates into numbers:
shape, texture, and colour (including melanization), and finally GWAS-ready
phenotype scores. GWAS = genome-wide association study, a method that links
measured traits to genes.

This guide covers installing it, running it through the window, running it from
the command line, what every output file is, and how to recover when something
goes wrong.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Better segmentation: the SAM 3 model (optional)](#3-better-segmentation-the-sam-3-model-optional)
4. [First launch: choose an imaging mode](#4-first-launch-choose-an-imaging-mode)
5. [The naming convention (read this before you photograph)](#5-the-naming-convention-read-this-before-you-photograph)
6. [The five-tab workflow](#6-the-five-tab-workflow)
7. [Colour calibration](#7-colour-calibration)
8. [What lands on disk](#8-what-lands-on-disk)
9. [Command line](#9-command-line)
10. [Running on a laptop or a shared server](#10-running-on-a-laptop-or-a-shared-server)
11. [Start in the middle / re-use old results](#11-start-in-the-middle--re-use-old-results)
12. [Current limitations & practical tips](#12-current-limitations--practical-tips)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What you need

- **A computer** running Windows, macOS, or Linux.
- **Python 3.10 or newer** (3.11+ recommended) — unless you use a standalone
  build, which carries Python inside it.
- **An internet connection for the first run only.** The launcher downloads what
  it needs once, then starts in seconds after that.
- **Your plate photographs**, named by the convention in
  [section 5](#5-the-naming-convention-read-this-before-you-photograph).
- *(Optional)* a graphics card (GPU) — it makes the SAM 3 model faster, but the
  tool runs without one.

---

## 2. Install

### The easy path: the double-click launchers

The project folder contains SETUP and START files for each system.

**First time on a computer**, run the SETUP file once. It installs everything,
including (on Linux) the system graphics libraries the window needs, and checks
that a window really opens:

| System | Run once |
|---|---|
| Windows | double-click `SETUP FungiCapture (Windows).bat` |
| macOS | double-click `SETUP FungiCapture (Mac).command` |
| Linux | double-click `SETUP FungiCapture (Linux).sh` |

**Every time after that**, open the program with the START file for your system
(`START FungiCapture (Windows).bat`, `… (Mac).command`, or `… (Linux).sh`).

The first run takes about 3–10 minutes and needs the internet. You will see a
lot of text scroll past — that is normal. Leave it until the window opens.

Platform notes:
- **Windows**: if you see "Windows protected your PC", click *More info* →
  *Run anyway*. This only means the file is unsigned, not that anything is wrong.
- **macOS**: the first time, right-click the file → *Open* → *Open*.
- **Linux**: if double-click does nothing, open a terminal in the folder and run
  `bash "START FungiCapture (Linux).sh"`.

### The no-install path: standalone builds

A standalone build carries everything inside it — no Python, no downloading.

- **Linux**: unpack `FungiCapture-0.1.0-linux-x86_64.tar.gz` and run
  `Start FungiCapture.sh`.
- **Windows / macOS**: a standalone has to be built *on* that kind of machine
  (the packaging tool cannot cross-build). On Windows, double-click
  `BUILD WINDOWS EXE.bat` (10–20 minutes, once); on macOS run
  `bash packaging/build.sh`. Or build via the GitHub Actions "Test and build"
  workflow and download the result.

### The developer path: pip

If you work from the source tree and have Python set up:

```bash
pip install -e .            # core pipeline only (headless-capable)
pip install -e ".[gui]"     # add the desktop window (PySide6)
pip install -e ".[model]"   # add the SAM 3 model backend (ultralytics + torch)
pip install -e ".[all]"     # everything, plus the test tools
```

Then launch with `fungicapture gui`, or use the command line
([section 9](#9-command-line)).

> **Licence note.** The SAM 3 backend depends on `ultralytics`, which is
> AGPL-3.0 — that is why FungiCapture itself is AGPL-3.0. If you publish results
> made with SAM 3, Meta's SAM License also asks you to acknowledge it in your
> methods.

### If it says "Python is not installed"

1. Go to <https://www.python.org/downloads/> and get Python 3.11 or newer.
2. **Windows only:** on the first installer screen, tick **Add Python to PATH**
   before clicking Install.
3. Install, then start FungiCapture again.

On Linux, Python is usually already there. If not:
`sudo apt install python3 python3-venv python3-pip` (Debian/Ubuntu) or
`sudo dnf install python3 python3-pip` (Fedora).

---

## 3. Better segmentation: the SAM 3 model (optional)

Segmentation = finding the exact outline of each colony. FungiCapture works
immediately with a **built-in classical segmenter** (colour contrast + a
threshold). It is good for compact colonies with a clear edge, but weaker on the
faint, fuzzy filamentous margin.

For much better outlines, use the **SAM 3** model (an AI segmentation model that
takes text prompts, so it can be told to find both the dense core and the faint
margin):

1. Open the **2 Segment** tab.
2. Press **Segmentation setup…**.
3. The window shows your graphics hardware, what is installed, and what will
   actually run. Press **Install segmentation model**. It downloads the version
   matching your graphics driver (about 2 GB).

The weights are **not bundled** — they are under Meta's gated SAM License. The
app points you to Meta's page; you accept the licence and the file is cached on
your machine.

The Segment tab always shows, in one line, **which segmenter is really being
used**, so it can never quietly fall back without telling you.

---

## 4. First launch: choose an imaging mode

When the window opens it asks one question: **what does one photograph
contain?** Your answer sets how the Plates tab cuts your photos.

| Mode | One photo holds | Layout you place |
|---|---|---|
| **Single dish** | one round plate, one or a few colonies | one box; add a box per colony |
| **Dish batch** | a folder of round plates, one strain each | boxes, reused for every plate |
| **Gridded plate** | one large plate, many colonies in a grid | a grid |
| **Plate batch** | many large plates, same layout | a grid, set once ← most common |

You can also set the plate size (rows × columns) and the growth-medium label
here. You can change these later.

---

## 5. The naming convention (read this before you photograph)

FungiCapture reads identifiers straight out of filenames. Name each exported
colony — and, where you can, each source photo — as:

```
ACCESSION_MEDIA_WELL
```

- **ACCESSION** — the strain/sample id, e.g. `AMF270`.
- **MEDIA** — the growth medium, 2–4 letters, e.g. `SDA`, `PDA`, `MEX`.
- **WELL** — a well position, a letter A–H then two digits, e.g. `D06`.

Example: **`AMF270_SDA_D06.png`**.

When FungiCapture cuts a plate, it writes each colony as
`STRAIN_MEDIA_WELL.png` automatically, so its own crops always follow the rule.
The convention matters most for the **medium**: a colony with no recognisable
medium in its name is grouped as `UNKNOWN` at the scoring step. If you give the
tool a plate map (strain names per well — see [6.1](#61--1-plates)), it fills the
strain part for you.

PNG, JPG, TIFF, and BMP are all read. Camera **RAW** files (`.nef`, `.cr2`,
`.dng`, and friends) are decoded at full resolution.

---

## 6. The five-tab workflow

Work left to right. Every tab has its **own input and output folders** at the
top, so you can run any step on its own (see [section 11](#11-start-in-the-middle--re-use-old-results)).

```
1 Plates  →  2 Segment  →  3 Features  →  4 Validate  →  5 Phenotypes
 cut         outline         measure        check          score
```

There is also a **Run everything** action that does steps 1–4 in order with the
layout you placed on the Plates tab.

### 6.1 — 1 Plates

Turn photographs into one image per colony.

1. At the top, set **Photographs** (your input folder) and **Save colonies to**
   (defaults to the project's `crops/` folder). The line underneath tells you
   how many photographs were found.
2. Pick a photo from the list on the left. It loads in the background, so the
   window stays responsive even for a big RAW file.
3. Place the layout on the image:
   - **Gridded modes** — drag the grid lines so each cell sits over one colony.
     Auto-fit centres the grid on the colonies it detects; amber lines mark
     wells it was unsure about, so your eye goes straight to those.
   - **Dish modes** — start from one box and press **Add box** for each colony;
     drag and resize the boxes. Delete a selected box with Delete/Backspace.
   - Rubber-band select several boxes/lines at once; Ctrl/Shift-click to add to a
     selection; Ctrl+A selects all; Ctrl+wheel zooms.
4. *(Optional)* open the **plate map editor** to type which strain sits in which
   well. Names are validated as you type, so a typo is caught here rather than
   surfacing three steps later as an `unknown` colony.
5. In batch modes, **Apply layout to all plates** reuses this layout for every
   photo (it re-centres on each plate's own colonies).
6. Press **Export colony crops**. In batch modes this cuts every photo (each
   into its own sub-folder); in single modes it cuts just the plate on screen.

### 6.2 — 2 Segment

Find the colony outline in every crop, and write a black-and-white mask plus a
quick-look overlay.

1. Set **Colony images** (defaults to `crops/`) and the mask output (`masks/`).
2. Choose the backend: **auto** (SAM 3 if installed, else classical), or force
   one. The line at the top shows which segmenter is actually running.
3. Press start. For SAM 3, prompts for the dense core and the faint margin are
   run separately and combined, so a faint fringe is kept rather than lost.

### 6.3 — 3 Features

Measure every colony and write one big table.

Each colony produces a feature vector (a row of numbers) covering **shape**
(area, circularity, solidity…), **brightness**, **texture** (GLCM and LBP — two
standard ways to quantify how rough or patterned a surface is), and **colour**
in RGB, HSV, and CIE-Lab, including melanization and **radial zonation** (whether
colour changes from the centre outwards). Every feature is measured for both the
colony **interior** and a **boundary ring** around it.

Output: `vectors/all_colony_features.csv` (one row per colony), and, unless you
turn it off, one small CSV per colony under `vectors/per_image_vectors/`.

### 6.4 — 4 Validate

Look at the results and mark each colony **pass** or **fail**.

- Page through thumbnails (e.g. 24 at a time) to scan a plate at a glance, or
  inspect one colony with its mask overlay and measured numbers.
- Mark bad segmentations (contamination, a merged pair, a mis-cut well) as
  **fail**. Failed colonies are excluded from scoring by default — and the count
  of how many were excluded is reported, never hidden.
- Export a **QC report PDF**: a full report, a failures-only report, or a contact
  sheet.

### 6.5 — 5 Phenotypes

Turn the feature table into GWAS-ready scores (optional).

The step cleans the table, standardises each feature, and runs **PCA** (principal
component analysis — a way to compress many correlated measurements into a few
summary axes). By default it scores **per growth medium**, so a medium batch
effect does not masquerade as biology.

Outputs land in `phenotypes/`, including the per-medium score tables and a
`pca_diagnostics.pdf`.

> See [section 12](#12-current-limitations--practical-tips) for important caveats
> about small groups and the combined-media file.

---

## 7. Colour calibration

Colour numbers are only comparable between photos if the lighting was. Three
methods are built in and stack:

1. **EXIF drift check** (always on) — reads camera settings from the photo and
   warns when two batches are not comparable. It corrects nothing; it tells you
   when to distrust a comparison.
2. **Agar background normalisation** (always on) — finds bare agar and scales
   each colour channel so the agar reads the same everywhere. Removes most lamp
   drift on its own.
3. **Colour reference card** (optional, most accurate) — put a card in the corner
   of the photo. FungiCapture ships **no card values of its own**: you upload a
   definition of the card you own (JSON or CSV listing each patch's reference
   colour), and any card works. If your card has no printed frame, mark its
   corners on the Plates tab.

---

## 8. What lands on disk

A project is a folder. After a full run it looks like this:

```
my_project/
├── crops/                      one image per colony (lossless PNG)
│   └── <plate>/                in batch modes, one sub-folder per plate
├── masks/                      one *_mask.png and *_overlay.png per colony
├── vectors/
│   ├── all_colony_features.csv the master feature table (one row per colony)
│   └── per_image_vectors/      one small CSV per colony
├── qc/                         validation verdicts + the QC report PDF
└── phenotypes/                 per-medium score tables + pca_diagnostics.pdf
```

Every output row also carries provenance: the software version, a **params
hash** (a short code identifying the exact settings used), and a timestamp. Two
result files with different params hashes were made with different settings and
must not be pooled.

---

## 9. Command line

The command line is for large batches on a server, scripting, or reproducing a
published result. Both the window and the command line call the same core code,
so they cannot drift apart.

```bash
# Create a project
fungicapture new PROJECT --mode plate_batch --images DIR --rows 8 --cols 12 --media SDA

# Run the pipeline (all steps by default)
fungicapture run PROJECT [--steps crop,segment,features,phenotypes] \
                         [--backend auto|sam3|classical] \
                         [--plate-map MAP.csv] [--workers N] [-q]

# Write a QC report PDF
fungicapture report PROJECT --mode full|failures|contact_sheet [-o OUT.pdf]

# Show a project's settings, params hash, and run history
fungicapture info PROJECT

# Open the window (optionally on a project)
fungicapture gui [PROJECT]
```

Notes:
- `--mode` is one of `single_dish`, `dish_batch`, `gridded_plate`, `plate_batch`.
- `--steps` lets you run part of the pipeline, e.g. `--steps segment,features`.
- `--workers N` sets how many CPU cores to use (see
  [section 10](#10-running-on-a-laptop-or-a-shared-server)).
- `fungicapture info` prints the per-step timing and run history, so you can see
  where the time went.

---

## 10. Running on a laptop or a shared server

Cropping, classical segmentation, and feature measurement run in parallel across
CPU cores. By default FungiCapture picks a sensible number of workers for your
machine. You can override it:

- On the command line: `fungicapture run PROJECT --workers 4`.
- As an environment variable (useful in a cluster job script):
  `FUNGICAPTURE_WORKERS=4`.

Guidance:
- **Big photographs + few cores + limited RAM** (a typical laptop, or a
  container with a small memory limit): set `--workers` low (2–4). Each worker
  decodes one full-resolution photo at a time, so more workers means more memory
  in use at once.
- **A server with many cores**: the default caps itself; raise `--workers` if you
  want more and have the RAM for it.
- **The SAM 3 model always runs one image at a time** regardless of `--workers`
  — running copies of the model in parallel is unstable, so only the surrounding
  work is parallelised.

---

## 11. Start in the middle / re-use old results

You do **not** have to run the whole pipeline. Every tab names its own input and
output folders.

- Already have colony crops from an earlier run or another program? Open **2
  Segment**, point **Colony images** at that folder, and start there.
- Want to re-score an existing feature table with different settings? Point the
  Phenotypes step at an existing `all_colony_features.csv` and run only that
  step (`--steps phenotypes` on the command line).

---

## 12. Current limitations & practical tips

These are known, honest caveats as of this version. They came out of an internal
review and are worth knowing before you trust a number.

- **16-bit images are handled correctly.** Standard JPEG/PNG, camera RAW, and
  16-bit scanner/microscope TIFFs are all scaled the same, absolute way now, so
  the brightness and colour measurements agree and the melanization score is
  consistent across image types.
- **Small groups give exploratory scores.** If a growth medium has fewer
  colonies than measured features, the PCA axes are not uniquely determined and
  the "variance explained" is optimistic. The scoring now flags this: such a
  group is marked `pca_reliable = 0` in the score table and carries a warning.
  Treat those scores as exploratory, and still use the **Validate** tab to catch
  bad colonies by eye. (The automatic outlier flag now works at small sample
  sizes — a genuinely extreme colony is marked `is_outlier = 1` — but it flags,
  it does not delete, so the final say is yours.)
- **Use the per-medium phenotype files, not the combined one, for GWAS.** The
  combined `phenotypes_all_media.*` file stacks scores from separate per-medium
  analyses whose axes do not mean the same thing, so it is only safe for a quick
  look, not as a single trait across media.
- **A colour card needs enough patches.** The correction has four parameters per
  channel, so a card with only a handful of patches cannot be quality-checked —
  a bad read would otherwise fit exactly and pass. The tool now refuses a card
  with fewer than six usable patches (it falls back to background normalisation
  and says so), and judges a larger card's read by a held-out error rather than a
  self-fit. Use a card with plenty of patches (a 24-patch ColorChecker is ideal)
  and keep it free of glare.

---

## 13. Troubleshooting

- **The window is slow to open a folder of huge photos.** The first image is
  decoded in the background now, so the window should stay responsive. Very large
  folders on a network/USB drive can still be slow to scan — copy them to a local
  disk if so.
- **"Using SAM 3" but then every colony errors.** You have the weights but not
  the model package. Install it with the **Segmentation setup…** button, or
  `pip install ".[model]"`.
- **Out of memory / the run gets killed on a small machine.** Lower `--workers`
  (or `FUNGICAPTURE_WORKERS`), and prefer smaller images. See
  [section 10](#10-running-on-a-laptop-or-a-shared-server).
- **Scoring says "too few, or too alike."** The batch was too small or too
  uniform for PCA. The feature table is still written — only the scoring was
  skipped.
- **Colours look wrong across batches.** Check the EXIF drift warning, make sure
  agar background normalisation has bare agar to find, and use a colour card for
  the most accurate correction (see [section 7](#7-colour-calibration)).
- **Something crashed during setup.** The launcher writes `setup_log.txt` in the
  project folder. Send that file to whoever gave you FungiCapture.

---

*FungiCapture is research software (AGPL-3.0), in active development. The GWAS
scoring approach follows Zhang et al. (2018), "PCA-Based Multiple-Trait GWAS
Analysis", Animals 8(12):239.*

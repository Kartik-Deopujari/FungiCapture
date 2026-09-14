# FungiCapture — User Guide

Everything you need to go from a stack of plate photographs to GWAS-ready
numbers, and a description of **every feature** in the application.

You do not need to know anything about programming to use FungiCapture.

**Contents**

1. [Install](#1-install)
2. [The segmentation model (optional)](#2-the-segmentation-model-optional)
3. [Start a project](#3-start-a-project)
4. [The colour-calibration question](#4-the-colour-calibration-question)
5. [The workspace](#5-the-workspace)
6. [Tab 1 — Plates](#6-tab-1--plates)
7. [Tab 2 — Segment](#7-tab-2--segment)
8. [Tab 3 — Features](#8-tab-3--features)
9. [Tab 4 — Validate](#9-tab-4--validate)
10. [Tab 5 — Phenotypes](#10-tab-5--phenotypes)
11. [Run everything at once](#11-run-everything-at-once)
12. [The command line](#12-the-command-line)
13. [Reproducibility](#13-reproducibility)
14. [Troubleshooting](#14-troubleshooting)
15. [Licence and citation](#15-licence-and-citation)

---

## 1. Install

There are three ways in. Most users want the first.

### A. Double-click launchers (recommended)

Download the project folder, then, **once per computer**, run the SETUP file for
your system. It installs everything needed (on Linux, also the system graphics
libraries) and checks that a window really opens.

| System | First time (once) | Every time after |
|---|---|---|
| Windows | double-click `SETUP FungiCapture (Windows).bat` | `START FungiCapture (Windows).bat` |
| macOS | double-click `SETUP FungiCapture (Mac).command` | `START FungiCapture (Mac).command` |
| Linux | double-click `SETUP FungiCapture (Linux).sh` | `START FungiCapture (Linux).sh` |

The first run downloads what it needs (a few minutes, needs internet). After
that it opens in seconds. The START launchers also self-install on first run, so
on Windows and macOS you can skip SETUP — but on a fresh Linux machine SETUP is
the smoothest path, because only it installs the system libraries the window
needs.

**First-launch security prompts** (they appear because the app is not
code-signed, not because anything is wrong):

- **Windows** — "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS** — if it refuses to open, **right-click the file → Open → Open**.
- **Linux** — if double-clicking does nothing, open a terminal in the folder and
  run `bash "START FungiCapture (Linux).sh"`.

**If it says Python is missing:** install Python 3.11 or newer from
<https://www.python.org/downloads/>. On Windows, tick **"Add Python to PATH"** on
the first installer screen. On Linux: `sudo apt install python3 python3-venv
python3-pip` (Debian/Ubuntu) or `sudo dnf install python3 python3-pip` (Fedora).

### B. Standalone build (no install at all)

A standalone build carries everything inside it — no Python, no download.

- **Linux** — unpack `FungiCapture-*-linux-x86_64.tar.gz` and run
  `Start FungiCapture.sh`.
- **Windows / macOS** — a standalone must be built on that platform: run
  `BUILD WINDOWS EXE.bat` on a Windows PC, or `bash packaging/build.sh` on a Mac.
  You can also build via GitHub Actions (**Actions → Test and build → Run
  workflow**) and download the result.

### C. From source (developers)

```bash
git clone https://github.com/<your-user>/fungicapture
cd fungicapture
pip install -e ".[gui,dev]"     # interface + tests
pip install -e ".[model]"       # optional: SAM 3 backend
fungicapture gui
```

The core has no GUI or deep-learning dependency, so it installs and runs
headless on a compute cluster.

### Try it without your own photos

The folder `examples/sample_dataset/plates` holds three practice plates. When the
app asks for an images folder, point it there and set the plate size to **4 rows
× 6 columns**.

---

## 2. The segmentation model (optional)

FungiCapture works **straight away** using a built-in *classical* segmenter — no
download, no GPU. It handles compact colonies well but is weaker on faint,
filamentous edges.

For the much better **SAM 3** model, open the **2 Segment** tab and press
**Segmentation setup…** (see [Tab 2](#7-tab-2--segment)). FungiCapture cannot
download the weights for you: Meta's licence requires you to request access and
accept it yourself. Once you have the file, the app caches it and you never do
this again.

The Segment tab always shows, in one line, which segmenter is actually being
used — so it can never quietly fall back without telling you.

---

## 3. Start a project

The first screen asks one question: **what does one photograph contain?** Pick a
plate **shape** (a card) and, if you have a whole folder to process, tick
**Batch process**. Those two choices map to four modes:

| Mode | One photo holds | Layout you place |
|---|---|---|
| **Single dish** | one round plate | one or more boxes (see below) |
| **Dish batch** | a folder of round plates | one or more boxes, reused for all |
| **Gridded plate** | one large plate, many colonies | a grid |
| **Plate batch** | many large plates, same layout | a grid, set once |

Then press **Continue ▶**, give the project a name and folder, point it at your
photographs, and — for the gridded modes — set the plate size (rows × columns).

Everything downstream branches on this choice, so it is asked once, up front.

---

## 4. The colour-calibration question

You are then asked whether your photographs contain a colour **reference card**.

**This matters more than it looks.** Colour numbers — including the melanization
score — are only comparable between photographs if the lighting was the same.
Over a months-long experiment it never is: lamps age, white balance drifts,
someone changes the ISO.

FungiCapture always does two things about this automatically: it **normalises
against the bare agar** in each image, and it **reads the camera settings and
warns you** when they change mid-batch. Those get you most of the way.

A colour card gets you the rest. Put one in the corner of every plate
photograph, then in this dialog:

- Tick **Process images with colour reference card**.
- **Card definition** — upload a `.json` or `.csv` listing each patch and its true
  colour. FungiCapture ships **no card values of its own**, so any card works.
- **Card position** — `auto`, or a corner (`top-left`, `top-right`,
  `bottom-left`, `bottom-right`) if auto-detection struggles.

> Automatic detection needs the card to have a **printed frame** around its
> patches. On test scenes, card correction cuts colony colour error from ΔE 12.8
> to ΔE 1.0.

You can **Skip for now** and add a card later in the project's settings.

---

## 5. The workspace

Once a project is open you see the five-step workspace.

- **Header** — the project name and mode; **Run everything ▶** (does all steps in
  one go, see [section 11](#11-run-everything-at-once)); **Home** (back to the
  start screen).
- **Five tabs**, meant to be worked left to right: **1 Plates → 2 Segment →
  3 Features → 4 Validate → 5 Phenotypes**.
- **Folders on every tab.** Each tab has its own input/output folders at the top,
  so **you do not have to start at the beginning.** Already have crops from
  another tool? Open **2 Segment**, point *Colony images* at them, and start
  there. Point *Feature table* at an old `all_colony_features.csv` to re-score it
  without measuring again.
- **Status bar** — a progress bar and a **Cancel** button while a job runs, and a
  notes button afterwards that opens any warnings from the last run.
- **Menus** — **File** (New / open project, Open project folder, Save project,
  Quit) and **Help** (About; Licensing and citation).

Your folder choices and settings are saved into the project automatically, so
they survive a restart.

---

## 6. Tab 1 — Plates

Cut each photograph into one image per colony.

**Folders:** *Photographs* (input) and *Save colonies to* (output). The status
line reports how many photographs were found.

**Pick a photograph** from the list on the left; it appears on the canvas.
Zoom with the **Fit / + / −** buttons or **Ctrl + mouse-wheel**; scroll to pan.
**Show well labels** toggles the `A01`, `A02`… overlays.

The layout tools depend on the mode.

### Round-plate modes (Single dish, Dish batch)

These start with **one large box** covering the plate — the common case of one
colony per dish. If a dish holds several colonies:

- **Add box** — drops in another box; drag it over the next colony and resize it.
- **Remove selected box** — deletes the selected box (or press **Delete**).
- Click a box to select it; drag its body to move, drag an edge/corner to resize.
- Boxes are named `A01`, `A02`, `A03`… in the order added; each becomes one
  colony crop.

In **Dish batch**, the boxes you place are reused for every photograph in the
folder, and each plate's crops go into their own sub-folder.

### Large-plate modes (Gridded plate, Plate batch)

These show a regular **grid** you adjust:

- **Grid lines** (default) — drag any line to move it; neighbouring cells follow.
- **Boxes** — switch here for a crooked plate or uneven spacing; each cell
  becomes an independent box you can move and resize on its own.
- **A01 on the right (flip columns)** — for plates photographed from the back.
- Select several lines/boxes by dragging a rubber-band over them, or **Ctrl+A**
  to select all; move or resize them together.

### Common controls

- **Apply layout to all plates** (batch modes only) — use this exact layout for
  every photograph on export. Nothing is moved automatically, so line the plates
  up the same way when you photograph them.
- **Reset layout** — back to the starting grid (large-plate modes) or single box
  (dish modes).
- **Undo (Ctrl+Z)** — steps back through layout changes.

### Plate map — which strain is where

Press **Edit plate map…**:

- **Growth medium** — two-to-four letters (e.g. `SDA`, `PDA`, `MEX`). It becomes
  part of every filename: `STRAIN_MEDIUM_WELL.png`.
- Type a **Strain** and optional **Note** per well. Problems (duplicates, blanks)
  are flagged live as you type.
- **Fill down column…** copies a value down a column — plates are usually filled
  in a pattern, and typing 96 names by hand invites mistakes.
- **Import CSV… / Export CSV…** to reuse a map you already have.

In the dish modes the strain comes from each photograph's **filename** instead of
a well grid.

### Export

- **Skip wells that touch the image edge** — off by default. When off, an
  edge-overrun well is still exported and you are warned; the original tool
  dropped these silently, losing whole outer rows.
- **Export colony crops** — writes one PNG per colony. The button says *(this
  plate)* in single modes and *(all plates)* in batch modes.

---

## 7. Tab 2 — Segment

Find the colony outline in every crop.

**Folders:** *Colony images* (input crops) and *Save masks to* (output).

**Settings:**

- **Status + Segmentation setup…** — one line stating what will actually run.
  The setup window diagnoses your hardware, shows what is installed and missing,
  and can **install PyTorch + Ultralytics** (about 2 GB) with an option for the
  smaller processor-only build.
- **Run on** — `auto` (GPU when usable, else processor), or force `cuda` / `mps`
  / `cpu`.
- **Backend** — `auto` (SAM 3 when the weights are present, else classical, and
  it tells you which it used), or force `sam3` / `classical`.
- **SAM 3 weights + Install weights…** — shows whether the weights are found.
  *Install weights…* points you at Meta's gated page; once you have the `.pt`
  file, select it and it is cached.
- **Confidence** — detection threshold (default `0.25`). Lower catches fainter
  structure but risks halo pixels.
- **Save overlay images for quick checking** — writes an outline-on-crop preview
  next to each mask.
- **Prompts** — plain-language descriptions of what to find, **one per line**; a
  **blank line starts a new group**. By default there are two groups — one for
  the dense colony body, one for the faint filamentous margin — merged with OR,
  favouring a caught fringe over a few extra pixels. Edit freely.

Press **Run segmentation**. The log names the backend used and any warnings.

---

## 8. Tab 3 — Features

Measure every colony: about **400 numbers each**, computed on both the colony
**interior** and the **ring** just outside its edge.

**Folders:** *Colony images*, *Masks* (matched to crops by name, e.g.
`AMF270_SDA_D06_mask.png` ↔ `AMF270_SDA_D06.png`), and *Save results to* (writes
`all_colony_features.csv`). The status line shows how many colony/mask pairs are
ready.

**Feature families** (all on by default; untick to skip):

- **Shape** — area, circularity, solidity, and more.
- **Brightness statistics.**
- **GLCM texture** and **LBP texture.**
- **Colour and melanization** — RGB, HSV and CIE-Lab, named melanization indices,
  and radial zonation (whether colour changes from the centre outwards).

**Other settings:**

- **Include colour histograms** — many extra columns; kept out of the PCA.
- **Perimeter method** — leave on `crofton` (a perfect circle scores circularity
  0.99). `legacy_8` reproduces the original scripts (which scored the same circle
  0.62) and `legacy_4` its four-connected variant — for reproducing old results
  only.
- **Ring fraction** — the boundary ring's width as a fraction of each colony's own
  radius (default `0.10`; larger samples further into the halo).

Press **Extract features**.

---

## 9. Tab 4 — Validate

Look at every colony, mark it, and export a PDF.

**Folders:** *Colony images*, *Masks*, *Feature table* (fills in the numbers), and
*Save reports to*.

- **Show** filter — `all`, `unreviewed`, `pass`, or `fail`.
- **Colony list** — click through the colonies; each panel shows the mask, the
  ring, the texture inputs, the colour analysis, and the numbers side by side.
- **Verdict** — **✓ Pass** / **✗ Fail**, with a **note** ("Why?") kept with the
  record. A failed colony is excluded from scoring **with your reason attached**,
  not silently dropped or kept.
- **◀ Previous / Next ▶** — move through the list (marking a verdict also advances).
- **❔ What do these features mean?** — a searchable glossary of every measured
  feature.

**When judging a colony, look for five things:**

1. Does the mask follow the colony edge, fuzzy margin included?
2. Does the ring sit on agar, not on a neighbour?
3. Is the lightness (L\*) map dark where the colony looks dark to you?
4. Do the colour swatches match what you see?
5. Does the radial profile show rings only where the colony really has them?

**Export PDF report** — three kinds:

- **Full report** — one page per colony plus a summary, for supplementary material.
- **Failures only** — short enough to email to a collaborator.
- **Contact sheet** — 24 thumbnails a page, for scanning a plate at a glance.

Every report carries the settings that produced it — a QC document without that
is not evidence.

---

## 10. Tab 5 — Phenotypes

Optional. Turn the ~400 correlated features into a few GWAS-ready scores.

Running a separate GWAS on 400 features is a multiple-testing disaster, and any
single feature is a noisy view of the biology. PCA combines them into
**pseudo-traits**; running the association test on **PC1** has more power than on
any single feature when they share a causal variant.

**Folders:** *Feature table* (the `all_colony_features.csv` to score — point it at
an old one to re-score without re-measuring) and *Save scores to*.

**Settings:**

- **Score each growth medium separately** (default on) — colonies on different
  media are not directly comparable; pooling would put a medium effect into PC1
  and mask the genetics.
- **Exclude colonies marked 'fail' in the Validate tab** (default on).
- **Phenotype PCs to export** — how many principal components to write (default
  `2`).
- **Minimum non-missing** — drop feature columns with fewer than this fraction of
  real values before imputing (default `0.90`).
- **Outlier p-value** — Mahalanobis-distance threshold for flagging multivariate
  outliers (default `0.001`). Outliers are **flagged, never deleted**.

Press **Compute GWAS phenotypes**. The log reports, per medium, how many colonies
were scored, how much variance PC1 explains, the outlier count, and the top
features driving PC1.

**Outputs** land in `phenotypes/`:

| File | What it is |
|---|---|
| `phenotypes_<medium>.tsv` | Ready for GEMMA, PLINK or GAPIT |
| `phenotypes_all_media.tsv` | Every medium together |
| `pca_loadings_<medium>.tsv` | Which features drive each PC |
| `pca_diagnostics.pdf` | Scree plot, PC1 vs PC2, top loadings, PC1 distribution |
| `features_dropped_in_qc.tsv` | What was dropped and why |

**Two things to know before you use these:**

- Outliers are flagged in the `is_outlier` column, never removed. Whether to
  exclude a strain is your call, not the software's.
- Apply a **minor-allele-frequency filter of at least 0.05** downstream. The power
  advantage of this approach is greatest above about 0.2 and fades for rare
  alleles.

---

## 11. Run everything at once

The **Run everything ▶** button in the header does crop → segment → features →
phenotypes in one pass, using the layout you have placed on the Plates tab and
each tab's current settings. It stops early if a step produces nothing (every
later step would fail for the same reason). Use the tabs when you want to check
or adjust between steps; use Run everything for a settled workflow.

---

## 12. The command line

For a cluster, a script, or to reproduce a result exactly:

```bash
# Create a project
fungicapture new PROJECT --mode plate_batch --images DIR \
              --rows 8 --cols 12 --media SDA
#   --mode  single_dish | dish_batch | gridded_plate | plate_batch
#   --name NAME   --rows N   --cols N   --media LABEL

# Run the pipeline (all steps, or a subset)
fungicapture run PROJECT [--steps crop,segment,features,phenotypes] \
              [--plate-map MAP.csv] [--backend auto|sam3|classical] \
              [--no-auto-centre] [--skip-edge-wells] [-q]

# Write a QC PDF
fungicapture report PROJECT --mode full|failures|contact_sheet [-o OUT.pdf]

# Show a project's settings, params hash and run history
fungicapture info PROJECT

# Launch the graphical application (optionally on a project)
fungicapture gui [PROJECT]
```

A five-minute end-to-end demo on the bundled sample data:

```bash
fungicapture new /tmp/demo --mode plate_batch \
              --images examples/sample_dataset/plates --rows 4 --cols 6 --media SDA
fungicapture run /tmp/demo --plate-map examples/sample_dataset/plate_map.csv
fungicapture report /tmp/demo --mode contact_sheet
```

---

## 13. Reproducibility

Every output carries a **params hash**. If two result files have different hashes
they were made with different settings and **must not be pooled**. `fungicapture
info` prints the hash and the full run history.

Every setting that affects a number lives in one frozen object, read by both the
feature extractor and the QC panel — so a QC picture can never describe a
different ring width from the numbers printed beside it.

---

## 14. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Segmentation seems stuck on CPU / "classical" | SAM 3 not installed. Open **2 Segment → Segmentation setup…** |
| Masks cover the whole crop | Background too similar to the colony. Try the **sam3** backend, or lower **Confidence** |
| Colony edge / fuzzy margin is missed | Add or edit a **prompt group** for the filamentous margin; try **sam3** |
| Every colony named `unknown` | No plate map, or unrecognised CSV columns. Open **Edit plate map…** |
| Colonies grouped as `UNKNOWN` medium | Filenames must look like `AMF270_SDA_D06.png` |
| Colour values differ between batches | Camera settings changed. Check the run notes for an EXIF-drift warning; use a colour card |
| A dish has several colonies but only one crop | On the Plates tab press **Add box** and place one box per colony |
| PC1 explains very little variance | The features may not share a dominant axis. Read the scree plot before trusting PC1 |
| Something went wrong during setup | The launcher writes `setup_log.txt`; send that file to whoever gave you FungiCapture |

---

## 15. Licence and citation

FungiCapture is released under the **GNU AGPL-3.0-or-later** (see the `LICENSE`
file and `docs/LICENSING.md`). This is required, not chosen: the tool depends on
Ultralytics, which is AGPL-3.0. Anyone who builds on FungiCapture must also open
their source.

If you publish results made with FungiCapture, cite the tool and the methods it
uses — `docs/REFERENCES.md` lists every algorithm and library with its source. If
you used SAM 3, Meta's licence **requires** you to acknowledge it:

> Colony segmentation used the Segment Anything Model 3 (SAM 3), released by
> Meta AI under the SAM License.

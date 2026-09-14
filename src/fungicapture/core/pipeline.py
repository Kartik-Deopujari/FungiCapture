"""
The pipeline: running the whole thing.

Each step is a plain function taking a Project and returning a StepResult. The
graphical interface calls them from a worker thread; the command-line runner
calls them directly. Neither contains any pipeline logic of its own, so the two
front ends can never drift apart in behaviour.

Every step:

* reports progress and can be cancelled
* collects errors per item instead of aborting the whole run
* writes a run record into the project, so what happened is auditable
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import Calibrator
from .detect import centre_boxes_on_colonies, detect_dish, fit_grid
from .features.extract import analysis_columns, extract_from_files, normalise_stem
from .features.regions import load_rgb
from .plate import (
    BoxGeometry,
    GridGeometry,
    crop_filename,
    crop_normalised,
    iter_box_crops,
    iter_freeform_box_crops,
    iter_grid_crops,
)
from .platemap import PlateMap
from .project import ImagingMode, Project
from .segment import make_segmenter, save_mask, save_overlay, segment_image

# Image types a step will pick up when scanning a folder. Kept here rather
# than hard-coding ".png" so a step pointed at somebody else's crops - which
# may well be JPEG or TIFF - still finds them.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# progress(done, total, message) -> False to cancel
ProgressFn = Callable[[int, int, str], bool]


def _noop_progress(done: int, total: int, message: str) -> bool:
    return True


@dataclass
class StepResult:
    """What one pipeline step did."""

    step: str
    n_inputs: int = 0
    n_outputs: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    cancelled: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    input_noun: str = "item"
    output_noun: str = "item"
    """What the step consumed and produced. The cropping step turns 3 plates
    into 72 colonies, so 'succeeded 72 of 3' would be nonsense - these let each
    step describe itself honestly."""

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @staticmethod
    def _plural(count: int, noun: str) -> str:
        if count == 1:
            return f"1 {noun}"
        # 'colony' -> 'colonies', not 'colonys'
        if noun.endswith("y") and noun[-2:-1] not in "aeiou":
            return f"{count} {noun[:-1]}ies"
        return f"{count} {noun}s"

    def summary(self) -> str:
        inputs = self._plural(self.n_inputs, self.input_noun)
        outputs = self._plural(self.n_outputs, self.output_noun)
        if self.cancelled:
            return f"{self.step}: cancelled after {outputs} from {inputs}"
        if self.input_noun == self.output_noun:
            text = f"{self.step}: {self.n_outputs} of {inputs} succeeded"
        else:
            text = f"{self.step}: {inputs} -> {outputs}"
        if self.errors:
            text += f", {self.n_errors} failed"
        return text


# --------------------------------------------------------------------------
# Step 1 - cropping
# --------------------------------------------------------------------------


def crop_plates(
    project: Project,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    grid: GridGeometry | None = None,
    boxes: BoxGeometry | None = None,
    plate_map: PlateMap | None = None,
    auto_centre: bool = True,
    skip_edge_wells: bool = False,
    image_path: str | Path | None = None,
    progress: ProgressFn = _noop_progress,
) -> StepResult:
    """
    Turn plate photographs into one image per colony.

    Which engine runs depends on the project's imaging mode. The dish modes
    find one colony per photograph; the grid modes cut a lattice.

    ``image_path`` names a single photograph to crop. In the **single** modes
    (Gridded plate, Single dish) this is required in practice: those modes crop
    exactly the one plate the user is looking at, and every plate would
    otherwise write its wells to the same filenames and silently overwrite the
    last one - so the folder ended up holding some *other* plate's colonies, not
    the one on screen. The **batch** modes ignore it and crop every photograph,
    each into its own sub-folder.
    """
    started = datetime.now(timezone.utc)
    result = StepResult(step="crop", input_noun="plate", output_noun="colony crop")

    images_root = Path(input_dir).expanduser() if input_dir else project.image_dir
    crops_root = Path(output_dir).expanduser() if output_dir else project.crops_dir
    result.data["input_dir"] = str(images_root) if images_root else ""
    result.data["output_dir"] = str(crops_root)

    if images_root is None:
        result.warnings.append(
            "No photographs folder is set. Choose one at the top of this tab."
        )
        return result

    try:
        images = project.find_images(images_root)
    except FileNotFoundError as error:
        result.warnings.append(str(error))
        return result

    if not images:
        result.warnings.append(
            f"No images found in {images_root}. Check the folder."
        )
        return result

    # Single modes crop exactly one plate: the one the user has selected. Batch
    # modes crop the whole folder. Restricting here is what stops a single-plate
    # export from cutting every image in the folder into one another's files.
    if not project.mode.is_batch:
        chosen = None
        if image_path is not None:
            wanted = Path(image_path).expanduser().resolve()
            chosen = next(
                (p for p in images if p.resolve() == wanted), None
            )
        if chosen is None:
            # No specific plate given (or it is not in this folder): fall back
            # to the first, rather than silently cropping all of them.
            chosen = images[0]
            if image_path is not None:
                result.warnings.append(
                    f"The selected plate {Path(image_path).name} was not found "
                    f"in {images_root}; cropped {chosen.name} instead."
                )
        images = [chosen]

    result.n_inputs = len(images)

    result.warnings.extend(project.normalise_media_label())
    crops_root.mkdir(parents=True, exist_ok=True)
    layout = project.layout
    dish_mode = not project.mode.needs_grid

    # Progress is counted in colonies, not plates. The old code used the number
    # of plate photographs as the total, so exporting one plate showed "1 of 1"
    # while it cut 96 wells - which looked like the counter was wrong. The total
    # now reflects the colonies being exported.
    if dish_mode:
        # Dish modes crop the hand-placed boxes when a layout is given, and fall
        # back to finding one colony automatically when it is not.
        per_plate = len(boxes.boxes) if boxes is not None and boxes.boxes else 1
    else:
        per_plate = layout.rows * layout.cols
    total_units = max(1, len(images) * per_plate)
    state = {"done": 0, "cancelled": False}

    def bump(message: str) -> bool:
        state["done"] += 1
        if not progress(min(state["done"], total_units), total_units, message):
            state["cancelled"] = True
            return False
        return True

    for image_path in images:
        if state["cancelled"]:
            result.cancelled = True
            break
        # A message before the slow decode, without advancing the count.
        progress(min(state["done"], total_units), total_units,
                 f"Cropping {image_path.name}")
        try:
            rgb = load_rgb(image_path)

            if dish_mode:
                if boxes is not None and boxes.boxes:
                    written = _crop_dish_boxes(
                        project, rgb, image_path, boxes, plate_map,
                        result, crops_root, bump,
                    )
                else:
                    written = _crop_single_dish(
                        project, rgb, image_path, plate_map, result, crops_root
                    )
                    bump(f"Cropped {image_path.name}")
            else:
                written = _crop_gridded(
                    project, rgb, image_path, layout, grid, boxes,
                    plate_map, auto_centre, skip_edge_wells, result, crops_root,
                    bump,
                )
            result.outputs.extend(written)
            result.n_outputs += len(written)
        except Exception as error:
            result.errors.append((str(image_path), str(error)))
        if state["cancelled"]:
            result.cancelled = True
            break

    project.log_run(
        "crop", started, result.n_inputs, result.n_outputs, result.n_errors,
        notes=f"mode={project.mode.value}",
    )
    return result


def _crop_single_dish(
    project: Project,
    rgb: np.ndarray,
    image_path: Path,
    plate_map: PlateMap | None,
    result: StepResult,
    crops_root: Path,
) -> list[Path]:
    """Dish engine: find the rim, find the one colony, crop around it."""
    import cv2

    dish = detect_dish(rgb)
    for warning in dish.warnings:
        result.warnings.append(f"{image_path.name}: {warning}")

    if dish.colony is None:
        return []

    x1, y1, x2, y2 = dish.colony.bbox
    pad_x, pad_y = 0.20 * (x2 - x1), 0.20 * (y2 - y1)
    crop, _, _ = crop_normalised(
        rgb,
        (max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
         min(1.0, x2 + pad_x), min(1.0, y2 + pad_y)),
    )
    if crop is None:
        return []

    strain = (
        plate_map.strain_for(image_path.name, image_path.stem)
        if plate_map
        else image_path.stem
    )
    name = crop_filename("A01", strain, project.media_label)
    destination = crops_root / name
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode crop for {image_path.name}")
    buffer.tofile(str(destination))
    return [destination]


def _crop_dish_boxes(
    project: Project,
    rgb: np.ndarray,
    image_path: Path,
    boxes: BoxGeometry,
    plate_map: PlateMap | None,
    result: StepResult,
    crops_root: Path,
    bump: Callable[[str], bool] | None = None,
) -> list[Path]:
    """Dish engine, box variant: cut every hand-placed box.

    One plate photograph carries one strain (named from its filename), so every
    box on it is a colony of that strain and shares the strain name; the box
    position (``A01``, ``A02``, …) keeps the filenames distinct.
    """
    import cv2

    strain = (
        plate_map.strain_for(image_path.name, image_path.stem)
        if plate_map
        else image_path.stem
    )

    # A batch of dishes writes each plate to its own sub-folder, so two plates
    # that both hold an ``A01`` colony do not overwrite each other.
    subdirectory = crops_root
    if project.mode.is_batch:
        subdirectory = crops_root / image_path.stem
        subdirectory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for well in iter_freeform_box_crops(rgb, boxes):
        destination = subdirectory / crop_filename(
            well.well_id, strain, project.media_label
        )
        ok, buffer = cv2.imencode(
            ".png", cv2.cvtColor(well.image, cv2.COLOR_RGB2BGR)
        )
        if ok:
            buffer.tofile(str(destination))
            written.append(destination)
        if bump is not None and not bump(
            f"Cropping colony {well.well_id} on {image_path.name}"
        ):
            break
    return written


def _crop_gridded(
    project: Project,
    rgb: np.ndarray,
    image_path: Path,
    layout,
    grid: GridGeometry | None,
    boxes: BoxGeometry | None,
    plate_map: PlateMap | None,
    auto_centre: bool,
    skip_edge_wells: bool,
    result: StepResult,
    crops_root: Path,
    bump: Callable[[str], bool] | None = None,
) -> list[Path]:
    """Grid engine: fit or reuse a layout, then cut every well."""
    import cv2

    plate_grid = grid
    plate_boxes = boxes

    if plate_grid is None and plate_boxes is None:
        fit = fit_grid(rgb, layout.rows, layout.cols, margin=layout.margin)
        plate_grid = GridGeometry(fit.row_lines, fit.col_lines)
        for warning in fit.warnings:
            result.warnings.append(f"{image_path.name}: {warning}")
        if fit.confidence < 0.5:
            result.warnings.append(
                f"{image_path.name}: low grid confidence ({fit.confidence:.2f}) - "
                "check this plate by eye."
            )
        if auto_centre:
            plate_boxes = BoxGeometry.from_grid(plate_grid, layout.rows, layout.cols)
            centre_boxes_on_colonies(plate_boxes, fit.detections)
    elif auto_centre and plate_boxes is not None:
        # A layout reused from another plate: re-centre it on this plate's own
        # colonies, which corrects small shifts between photographs.
        fit = fit_grid(rgb, layout.rows, layout.cols, margin=layout.margin)
        centre_boxes_on_colonies(plate_boxes, fit.detections)

    subdirectory = crops_root
    if project.mode is ImagingMode.PLATE_BATCH:
        subdirectory = crops_root / image_path.stem
        subdirectory.mkdir(parents=True, exist_ok=True)

    crops = (
        iter_box_crops(rgb, plate_boxes, layout, skip_edge_wells=skip_edge_wells)
        if plate_boxes is not None
        else iter_grid_crops(rgb, plate_grid, layout, skip_edge_wells=skip_edge_wells)
    )

    written: list[Path] = []
    edge_wells: list[str] = []
    for well in crops:
        if well.touches_edge:
            edge_wells.append(well.well_id)
        strain = plate_map.strain_for(well.well_id) if plate_map else None
        destination = subdirectory / crop_filename(
            well.well_id, strain, project.media_label
        )
        ok, buffer = cv2.imencode(
            ".png", cv2.cvtColor(well.image, cv2.COLOR_RGB2BGR)
        )
        if ok:
            buffer.tofile(str(destination))
            written.append(destination)
        if bump is not None and not bump(
            f"Cropping {well.well_id} on {image_path.name}"
        ):
            break

    if edge_wells:
        result.warnings.append(
            f"{image_path.name}: {len(edge_wells)} well(s) run off the edge of the "
            f"photograph ({', '.join(edge_wells[:6])}"
            f"{'...' if len(edge_wells) > 6 else ''}). They were still exported."
        )
    return written


# --------------------------------------------------------------------------
# Step 2 - segmentation
# --------------------------------------------------------------------------


def segment_crops(
    project: Project,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    backend: str = "auto",
    weights: str | Path | None = None,
    save_overlays: bool = True,
    progress: ProgressFn = _noop_progress,
) -> StepResult:
    """
    Find the colony outline in every crop.

    ``input_dir`` and ``output_dir`` let this step run on its own. If you
    already have colony crops - from an earlier run, or from another tool -
    point ``input_dir`` at that folder and start here. Nothing requires the
    cropping step to have run first.
    """
    started = datetime.now(timezone.utc)
    result = StepResult(step="segment", input_noun="crop", output_noun="mask")

    crops_root = Path(input_dir).expanduser() if input_dir else project.crops_dir
    masks_root = Path(output_dir).expanduser() if output_dir else project.masks_dir
    result.data["input_dir"] = str(crops_root)
    result.data["output_dir"] = str(masks_root)

    if not crops_root.exists():
        result.warnings.append(
            f"The crops folder does not exist: {crops_root}\n"
            "Either run the cropping step, or point this step at a folder that "
            "already holds colony images."
        )
        return result

    crops = sorted(
        p
        for p in crops_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and "_mask" not in p.stem
        and "_overlay" not in p.stem
    )
    result.n_inputs = len(crops)
    _warn_about_duplicate_crop_names(crops, project, result)
    if not crops:
        result.warnings.append(
            f"No images found in {crops_root}. Either run the cropping step, "
            "or point this step at a folder that already holds colony images."
        )
        return result

    segmenter, messages = make_segmenter(
        project.params.segmentation, backend=backend, weights=weights
    )
    result.warnings.extend(messages)
    result.data["backend"] = segmenter.name

    masks_root.mkdir(parents=True, exist_ok=True)

    for index, crop_path in enumerate(crops, start=1):
        if not progress(index - 1, len(crops), f"Segmenting {crop_path.name}"):
            result.cancelled = True
            break
        try:
            segmentation = segment_image(segmenter, crop_path)
            for warning in segmentation.warnings:
                result.warnings.append(f"{crop_path.name}: {warning}")

            # Mirror the crop's folder structure into the mask folder.
            # Without this, two plates whose wells share a filename - which
            # happens whenever a plate map is reused across plates - write to
            # the same mask path, and the second plate silently overwrites the
            # first. Every colony of the first plate would then be measured
            # against another plate's mask.
            relative = crop_path.relative_to(crops_root)
            mask_dir = masks_root / relative.parent
            mask_dir.mkdir(parents=True, exist_ok=True)

            mask_path = mask_dir / f"{crop_path.stem}_mask.png"
            save_mask(segmentation.mask, mask_path)
            result.outputs.append(mask_path)
            result.n_outputs += 1

            if save_overlays:
                save_overlay(
                    load_rgb(crop_path),
                    segmentation.mask,
                    mask_dir / f"{crop_path.stem}_overlay.png",
                )
        except Exception as error:
            result.errors.append((str(crop_path), str(error)))

    project.log_run(
        "segment", started, result.n_inputs, result.n_outputs, result.n_errors,
        notes=f"backend={result.data.get('backend')}",
    )
    return result


# --------------------------------------------------------------------------
# Step 3 - features
# --------------------------------------------------------------------------


def colony_key(project: Project, crop_path: Path, crops_root: Path | None = None) -> str:
    """
    A key that identifies one colony uniquely across the whole project.

    The filename alone is not enough. In a plate batch the same plate map is
    reused for every plate, so well A01 produces an identically named crop on
    all 40 plates. Including the plate folder makes the key unique, which
    matters wherever colonies are matched up: mask pairing, QC verdicts, and
    joining feature rows.
    """
    root = Path(crops_root) if crops_root else project.crops_dir
    try:
        relative = crop_path.relative_to(root)
    except ValueError:
        relative = Path(crop_path.name)
    base = normalise_stem(relative.stem)
    parent = relative.parent.as_posix()
    return base if parent in (".", "") else f"{parent}/{base}"


def pair_crops_and_masks(
    project: Project,
    crops_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
) -> list[tuple[Path, Path]]:
    """
    Match every crop to its mask.

    Matching is by **relative path**, not by filename alone, so a crop is never
    paired with an identically named mask belonging to a different plate.

    Both folders can be given explicitly, so the feature step can run against
    crops and masks that were produced elsewhere.
    """
    crops_root = Path(crops_dir).expanduser() if crops_dir else project.crops_dir
    masks_root = Path(masks_dir).expanduser() if masks_dir else project.masks_dir

    if not crops_root.exists() or not masks_root.exists():
        return []

    masks: dict[str, Path] = {}
    for path in masks_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "_overlay" in path.stem:
            continue
        relative = path.relative_to(masks_root)
        key = normalise_stem(relative.stem)
        parent = relative.parent.as_posix()
        masks[key if parent in (".", "") else f"{parent}/{key}"] = path

    pairs: list[tuple[Path, Path]] = []
    for crop in sorted(crops_root.rglob("*")):
        if not crop.is_file() or crop.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "_mask" in crop.stem or "_overlay" in crop.stem:
            continue
        mask = masks.get(colony_key(project, crop, crops_root))
        if mask is not None:
            pairs.append((crop, mask))
    return pairs


def _warn_about_duplicate_crop_names(
    crops: list[Path], project: Project, result: StepResult
) -> None:
    """Tell the user when crop filenames repeat across plates.

    Not an error - the pipeline now keeps them apart by folder - but it means
    the exported names are ambiguous, which is worth knowing before those names
    end up in a paper.
    """
    seen: dict[str, int] = {}
    for path in crops:
        seen[path.name] = seen.get(path.name, 0) + 1
    repeated = {name: count for name, count in seen.items() if count > 1}
    if repeated:
        example = next(iter(repeated))
        result.warnings.append(
            f"{len(repeated)} crop filename(s) occur on more than one plate "
            f"(e.g. '{example}' appears {repeated[example]} times). They are "
            "kept apart by plate folder, but give each plate its own plate map "
            "if you want unique names."
        )


def extract_features(
    project: Project,
    *,
    crops_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_histograms: bool = True,
    write_per_image: bool = True,
    progress: ProgressFn = _noop_progress,
) -> StepResult:
    """Measure every colony and write the feature table."""
    started = datetime.now(timezone.utc)
    result = StepResult(step="features", input_noun="colony", output_noun="feature vector")

    crops_root = Path(crops_dir).expanduser() if crops_dir else project.crops_dir
    masks_root = Path(masks_dir).expanduser() if masks_dir else project.masks_dir
    vectors_root = Path(output_dir).expanduser() if output_dir else project.vectors_dir
    result.data["input_dir"] = str(crops_root)
    result.data["masks_dir"] = str(masks_root)
    result.data["output_dir"] = str(vectors_root)

    pairs = pair_crops_and_masks(project, crops_root, masks_root)
    result.n_inputs = len(pairs)
    if not pairs:
        result.warnings.append(
            f"No crop/mask pairs found.\n"
            f"  crops:  {crops_root}\n"
            f"  masks:  {masks_root}\n"
            "A crop and its mask are matched by name, so a mask should be "
            "named after its crop (for example AMF270_SDA_D06_mask.png for "
            "AMF270_SDA_D06.png). Either run the earlier steps, or point this "
            "step at folders that already hold matching pairs."
        )
        return result

    calibrator = Calibrator(project.params.calibration)
    result.warnings.extend(calibrator.load_errors)
    if calibrator.describe():
        result.data["calibration"] = calibrator.describe()

    vectors_root.mkdir(parents=True, exist_ok=True)
    per_image_root = vectors_root / "per_image_vectors"
    if write_per_image:
        per_image_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, (crop_path, mask_path) in enumerate(pairs, start=1):
        if not progress(index - 1, len(pairs), f"Measuring {crop_path.name}"):
            result.cancelled = True
            break
        try:
            features = extract_from_files(
                crop_path, mask_path, project.params,
                include_histograms=include_histograms,
                calibrator=calibrator,
            )
            for warning in features.warnings:
                result.warnings.append(f"{crop_path.name}: {warning}")

            row = features.as_row()
            # base_key from the filename alone repeats across plates, so
            # overwrite it with the project-unique key and record which plate
            # the colony came from. Everything downstream - QC verdicts,
            # feature joins, exclusion lists - keys on this.
            row["base_key"] = colony_key(project, crop_path, crops_root)
            try:
                relative_parent = crop_path.relative_to(crops_root).parent
            except ValueError:
                relative_parent = Path(".")
            row["plate_id"] = (
                "" if relative_parent.as_posix() in (".", "")
                else relative_parent.as_posix()
            )
            row.update(project.provenance())
            rows.append(row)
            result.n_outputs += 1

            if write_per_image:
                pd.DataFrame([row]).to_csv(
                    per_image_root / f"{crop_path.stem}_features.csv",
                    index=False,
                )
        except Exception as error:
            result.errors.append((str(crop_path), str(error)))

    if rows:
        frame = pd.DataFrame(rows)
        # A colony with no medium cannot be grouped in the phenotype step, and
        # the cause is almost always a filename that does not follow the
        # ACCESSION_MEDIA_WELL convention. Say so now rather than letting every
        # colony land in an 'UNKNOWN' medium group later.
        missing_media = int(frame["media"].isna().sum()) if "media" in frame else 0
        if missing_media:
            result.warnings.append(
                f"{missing_media} colony(ies) have no growth medium in their "
                "filename, so they will be grouped as UNKNOWN when scoring. "
                "Filenames should look like AMF270_SDA_D06.png."
            )

        master = vectors_root / "all_colony_features.csv"
        frame.to_csv(master, index=False)
        result.outputs.append(master)
        result.data["n_columns"] = len(frame.columns)

    result.warnings.extend(calibrator.batch_warnings())

    if result.errors:
        errors_path = vectors_root / "feature_errors.csv"
        pd.DataFrame(result.errors, columns=["file", "error"]).to_csv(
            errors_path, index=False
        )
        result.outputs.append(errors_path)

    project.log_run(
        "features", started, result.n_inputs, result.n_outputs, result.n_errors,
        notes=calibrator.describe(),
    )
    return result


# --------------------------------------------------------------------------
# Step 4 - phenotypes
# --------------------------------------------------------------------------


def score_phenotypes(
    project: Project,
    *,
    features_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    settings=None,
    exclude_failed: bool = True,
    progress: ProgressFn = _noop_progress,
) -> StepResult:
    """
    Turn the feature table into GWAS-ready scores.

    Colonies marked 'fail' in the Validate tab are excluded by default, and how
    many were excluded is reported rather than left implicit.
    """
    from .phenotype import (
        PhenotypeSettings,
        compute_phenotypes,
        write_diagnostics_pdf,
        write_phenotypes,
    )
    from .validate import VerdictStore

    started = datetime.now(timezone.utc)
    result = StepResult(step="phenotypes", input_noun="colony", output_noun="score")

    master = (
        Path(features_csv).expanduser()
        if features_csv
        else project.vectors_dir / "all_colony_features.csv"
    )
    # Accept a folder as well as a file - a user who picks the vectors folder
    # rather than the CSV inside it has done nothing wrong.
    if master.is_dir():
        master = master / "all_colony_features.csv"
    out_root = (
        Path(output_dir).expanduser() if output_dir else project.phenotypes_dir
    )
    result.data["input_file"] = str(master)
    result.data["output_dir"] = str(out_root)

    if not master.exists():
        result.warnings.append(
            f"No feature table found at {master}\n"
            "Either run the feature step, or point this step at an existing "
            "all_colony_features.csv."
        )
        return result

    frame = pd.read_csv(master)
    result.n_inputs = len(frame)
    progress(0, 3, "Loading features")

    if exclude_failed:
        store = VerdictStore(project.qc_dir / VerdictStore.FILENAME)
        failed = store.failed_keys()
        if failed and "base_key" in frame.columns:
            before = len(frame)
            frame = frame[~frame["base_key"].isin(failed)].reset_index(drop=True)
            removed = before - len(frame)
            if removed:
                result.warnings.append(
                    f"{removed} colony(ies) excluded because they were marked "
                    "'fail' during validation."
                )

    if frame.empty:
        result.warnings.append("Every colony was excluded - nothing to score.")
        return result

    candidates = analysis_columns(list(frame.columns), project.params)
    progress(1, 3, "Running PCA")

    phenotypes = compute_phenotypes(
        frame, candidates, settings or PhenotypeSettings()
    )
    result.warnings.extend(phenotypes.warnings)

    progress(2, 3, "Writing results")
    out_root.mkdir(parents=True, exist_ok=True)
    written = write_phenotypes(phenotypes, out_root)
    result.outputs.extend(written)

    try:
        pdf = write_diagnostics_pdf(
            phenotypes, out_root / "pca_diagnostics.pdf"
        )
        result.outputs.append(pdf)
    except Exception as error:
        result.warnings.append(f"Could not write the diagnostics PDF: {error}")

    result.n_outputs = len(phenotypes.combined)
    result.data["result"] = phenotypes
    progress(3, 3, "Done")

    project.log_run(
        "phenotypes", started, result.n_inputs, result.n_outputs, 0,
        notes=f"{len(phenotypes.per_media)} media group(s)",
    )
    return result


# --------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------

ALL_STEPS = ("crop", "segment", "features", "phenotypes")


def run_pipeline(
    project: Project,
    steps: Iterable[str] = ALL_STEPS,
    *,
    progress: ProgressFn = _noop_progress,
    **kwargs: Any,
) -> list[StepResult]:
    """
    Run several steps in order.

    Stops early if a step produces nothing, because every later step would then
    fail for the same reason and burying the user in errors helps nobody.
    """
    steps = [s for s in steps if s in ALL_STEPS]
    results: list[StepResult] = []

    functions = {
        "crop": crop_plates,
        "segment": segment_crops,
        "features": extract_features,
        "phenotypes": score_phenotypes,
    }
    accepted = {
        "crop": {
            "grid", "boxes", "plate_map", "auto_centre", "skip_edge_wells",
            "input_dir", "output_dir", "image_path",
        },
        "segment": {
            "backend", "weights", "save_overlays", "input_dir", "output_dir",
        },
        "features": {
            "include_histograms", "write_per_image",
            "crops_dir", "masks_dir", "output_dir",
        },
        "phenotypes": {
            "settings", "exclude_failed", "features_csv", "output_dir",
        },
    }

    for step in steps:
        step_kwargs = {k: v for k, v in kwargs.items() if k in accepted[step]}
        result = functions[step](project, progress=progress, **step_kwargs)
        results.append(result)
        if result.cancelled:
            break
        if result.n_outputs == 0:
            result.warnings.append(
                f"Stopping: the '{step}' step produced nothing, so later steps "
                "would fail too."
            )
            break
    return results

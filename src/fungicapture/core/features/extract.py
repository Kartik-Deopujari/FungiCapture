"""
The feature extractor: one colony in, one feature vector out.

This is the orchestrator. It builds the regions, calls each feature family in
turn, and returns a flat dictionary that becomes one row of the results table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..params import AnalysisParams
from . import colour as colour_mod
from . import morphology as morph_mod
from . import texture as texture_mod
from .intensity import intensity_features
from .regions import build_regions, load_gray, load_mask, load_rgb

# --------------------------------------------------------------------------
# Filename metadata
# --------------------------------------------------------------------------

# Canonical colony ID: ACCESSION_MEDIA_WELL, e.g. AMF270_SDA_D06
SAMPLE_ID_RE = re.compile(
    r"([A-Za-z0-9]+)"      # accession
    r"_([A-Z]{2,4})"       # media token
    r"_([A-H]\d{2})",      # well
    re.IGNORECASE,
)

# Suffixes the pipeline itself appends. Stripped to recover the base key that
# links an image, its mask and its feature CSV.
DERIVED_SUFFIXES: tuple[str, ...] = (
    "_sam3_semantic_overlay_features",
    "_semantic_overlay_features",
    "_overlay_features",
    "_feature_qc",
    "_features",
    "_sam3_semantic_overlay",
    "_sam3_semantic_mask",
    "_semantic_mask",
    "_core_colony_mask",
    "_fine_filaments_mask",
    "_overlay-2",
    "_overlay",
    "_mask",
    "_image",
)


def normalise_stem(stem: str) -> str:
    """
    Strip every pipeline-added suffix to recover the base colony key.

    Repeats until nothing changes, because names can carry more than one
    suffix, for example ``AMF270_SDA_D06_sam3_semantic_overlay_features``.
    The original script applied each suffix once in a single pass, which left
    stacked suffixes half-stripped.
    """
    stem = Path(stem).stem
    changed = True
    while changed:
        changed = False
        for suffix in DERIVED_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def parse_metadata(path: Path, media_tokens: tuple[str, ...] = ("MEX", "PDA", "SDA")) -> dict[str, Any]:
    """
    Pull identifiers out of a filename.

    Prefers the strict ``ACCESSION_MEDIA_WELL`` pattern. Falls back to
    token-scanning so a file that does not follow the convention still gets an
    ID rather than being dropped - losing a colony silently is worse than
    giving it an imperfect name.
    """
    stem = path.stem
    base = normalise_stem(stem)

    match = SAMPLE_ID_RE.search(base)
    if match:
        accession = match.group(1).upper()
        media = match.group(2).upper()
        well = match.group(3).upper()
        return {
            "filename": path.name,
            "stem": stem,
            "base_key": base,
            "sample_id": f"{accession}_{media}_{well}",
            "accession": accession,
            "media": media,
            "plate_index": well,
        }

    parts = base.split("_")
    media = next((p.upper() for p in parts if p.upper() in media_tokens), None)
    well = next(
        (p.upper() for p in parts if re.fullmatch(r"[A-H]\d{2}", p.upper())), None
    )
    return {
        "filename": path.name,
        "stem": stem,
        "base_key": base,
        "sample_id": base,
        "accession": parts[0].upper() if parts else base,
        "media": media,
        "plate_index": well,
    }


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class ColonyFeatures:
    """One colony's complete measurement, plus how it was produced."""

    values: dict[str, Any]
    ring_width_px: int
    n_inside_px: int
    n_ring_px: int
    warnings: list[str]

    def as_row(self) -> dict[str, Any]:
        return dict(self.values)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_from_arrays(
    gray: np.ndarray,
    mask: np.ndarray,
    params: AnalysisParams,
    *,
    rgb: np.ndarray | None = None,
    include_histograms: bool = True,
) -> ColonyFeatures:
    """
    Measure one colony from arrays already in memory.

    Kept separate from the file-reading path so it can be tested on synthetic
    shapes whose true area and circularity are known exactly.
    """
    gray = np.asarray(gray)
    mask = np.asarray(mask, dtype=bool)
    warnings: list[str] = []

    if gray.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask disagree on size: {gray.shape[:2]} vs {mask.shape[:2]}"
        )

    regions = build_regions(mask, params.ring)
    values: dict[str, Any] = {"ring_width_px": regions.ring_width_px}

    if not mask.any():
        warnings.append("Mask is empty - no colony found.")

    centroid: tuple[float, float] | None = None

    if params.compute_morphology:
        morph = morph_mod.morphology_features(mask, params.morphology)
        values.update(morph)
        if not np.isnan(morph.get("centroid_row", np.nan)):
            centroid = (morph["centroid_row"], morph["centroid_col"])

    region_map = regions.as_dict()

    if params.compute_intensity:
        for name, region in region_map.items():
            values.update(intensity_features(gray[region], name))

    if params.compute_glcm:
        for name, region in region_map.items():
            values.update(
                texture_mod.glcm_features(gray, region, name, params.glcm)
            )

    if params.compute_lbp:
        for name, region in region_map.items():
            values.update(texture_mod.lbp_features(gray, region, name, params.lbp))

    if params.compute_colour:
        if rgb is None:
            warnings.append(
                "Colour features requested but no colour image was supplied - skipped."
            )
        else:
            values.update(
                colour_mod.colour_features(
                    rgb,
                    region_map,
                    params.colour,
                    centroid=centroid,
                    include_histograms=include_histograms,
                )
            )

    if regions.ring.sum() == 0 and mask.any():
        warnings.append("Ring region is empty - colony may fill the whole crop.")

    return ColonyFeatures(
        values=values,
        ring_width_px=regions.ring_width_px,
        n_inside_px=int(mask.sum()),
        n_ring_px=int(regions.ring.sum()),
        warnings=warnings,
    )


def extract_from_files(
    image_path: str | Path,
    mask_path: str | Path,
    params: AnalysisParams,
    *,
    include_histograms: bool = True,
    calibrator=None,
) -> ColonyFeatures:
    """
    Measure one colony from an image file and its mask file.

    Parameters
    ----------
    calibrator
        Optional object with ``apply(rgb, image_path) -> (rgb, info)``, from
        ``fungicapture.core.calibration``. When present, colour correction is
        applied before any colour feature is computed, and the calibration
        details are recorded in the output row.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    gray = load_gray(image_path)
    mask = load_mask(
        mask_path,
        target_shape=gray.shape[:2],
        largest_component_only=params.segmentation.use_largest_component,
    )

    rgb = None
    calibration_info: dict[str, Any] = {}
    if params.compute_colour:
        rgb = load_rgb(image_path)
        if calibrator is not None:
            rgb, calibration_info = calibrator.apply(rgb, image_path)

    result = extract_from_arrays(
        gray,
        mask,
        params,
        rgb=rgb,
        include_histograms=include_histograms,
    )

    metadata = parse_metadata(image_path)
    merged: dict[str, Any] = {}
    merged.update(metadata)
    merged["image_path"] = str(image_path)
    merged["mask_path"] = str(mask_path)
    merged.update(calibration_info)
    merged.update(result.values)
    result.values = merged
    return result


# --------------------------------------------------------------------------
# Column ordering
# --------------------------------------------------------------------------

METADATA_COLUMNS: tuple[str, ...] = (
    "filename",
    "stem",
    "base_key",
    "plate_id",
    "sample_id",
    "accession",
    "media",
    "plate_index",
    "image_path",
    "mask_path",
    # Provenance, written into every row but never a phenotype.
    "fungicapture_version",
    "project_name",
    "imaging_mode",
    "params_hash",
    "lab_illuminant",
    "generated_utc",
    "colour_calibration_method",
    "colour_calibration_warnings",
    "colour_card_found",
)


def analysis_columns(
    columns: list[str] | tuple[str, ...], params: AnalysisParams
) -> list[str]:
    """
    The subset of columns that should enter PCA.

    Drops identifiers, path columns, and any family listed in
    ``params.exclude_from_pca`` - by default the LBP bins and colour
    histograms, which are high-dimensional and would otherwise swamp the
    interpretable features.
    """
    excluded = set(METADATA_COLUMNS)
    out: list[str] = []
    for name in columns:
        if name in excluded:
            continue
        if any(token in name for token in params.exclude_from_pca):
            continue
        out.append(name)
    return out

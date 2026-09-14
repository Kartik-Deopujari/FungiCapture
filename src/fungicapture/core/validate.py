"""
Visual QC: showing a human what the numbers were measured on.

A feature table cannot tell you that the mask caught a shadow instead of the
colony. Only a picture can. This module builds the panel that puts the mask,
the ring, the texture inputs and the colour analysis side by side with the
numbers they produced.

Also holds the pass/fail verdicts, which are stored in the project so a failed
colony is excluded from the phenotype step **with a recorded reason** rather
than quietly deleted.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .params import AnalysisParams

# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    """A human's judgement on one colony."""

    key: str
    status: str = "unreviewed"
    """``pass``, ``fail`` or ``unreviewed``."""
    note: str = ""
    reviewer: str = ""
    reviewed_at: str = ""

    @property
    def is_fail(self) -> bool:
        return self.status == "fail"


class VerdictStore:
    """
    Pass/fail decisions, saved beside the project.

    Kept in a small JSON file rather than in the feature CSV so a re-run of the
    feature step never wipes the review work.
    """

    FILENAME = "qc_verdicts.json"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.verdicts: dict[str, Verdict] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key, raw in data.get("verdicts", {}).items():
            self.verdicts[key] = Verdict(
                key=key,
                status=raw.get("status", "unreviewed"),
                note=raw.get("note", ""),
                reviewer=raw.get("reviewer", ""),
                reviewed_at=raw.get("reviewed_at", ""),
            )

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "verdicts": {k: asdict(v) for k, v in self.verdicts.items()}
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    def set(self, key: str, status: str, note: str = "", reviewer: str = "") -> Verdict:
        from datetime import datetime, timezone

        verdict = Verdict(
            key=key,
            status=status,
            note=note,
            reviewer=reviewer,
            reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.verdicts[key] = verdict
        return verdict

    def get(self, key: str) -> Verdict:
        return self.verdicts.get(key, Verdict(key=key))

    def status_counts(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "unreviewed": 0}
        for verdict in self.verdicts.values():
            counts[verdict.status] = counts.get(verdict.status, 0) + 1
        return counts

    def failed_keys(self) -> set[str]:
        return {k for k, v in self.verdicts.items() if v.is_fail}


# --------------------------------------------------------------------------
# Panel data
# --------------------------------------------------------------------------


@dataclass
class PanelData:
    """Everything one QC panel needs, gathered before any drawing happens."""

    key: str
    rgb: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    ring: np.ndarray
    ring_width_px: int
    values: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def build_panel_data(
    image_path: str | Path,
    mask_path: str | Path,
    values: dict[str, Any],
    params: AnalysisParams,
    calibrator=None,
) -> PanelData:
    """Load and prepare everything for one colony's QC panel."""
    from .features.extract import normalise_stem
    from .features.regions import build_regions, load_gray, load_mask, load_rgb

    image_path = Path(image_path)
    gray = load_gray(image_path)
    rgb = load_rgb(image_path)
    if calibrator is not None:
        rgb, _ = calibrator.apply(rgb, image_path)

    mask = load_mask(
        mask_path,
        target_shape=gray.shape[:2],
        largest_component_only=params.segmentation.use_largest_component,
    )
    regions = build_regions(mask, params.ring)

    return PanelData(
        key=normalise_stem(image_path.stem),
        rgb=rgb,
        gray=gray,
        mask=regions.inside,
        ring=regions.ring,
        ring_width_px=regions.ring_width_px,
        values=values,
    )


# --------------------------------------------------------------------------
# Panel figure
# --------------------------------------------------------------------------

_NUMBER_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Shape",
        (
            "area_px", "perimeter_px", "circularity", "roughness",
            "eccentricity", "solidity", "extent", "equivalent_diameter_area",
            "ring_width_px",
        ),
    ),
    (
        "Brightness (inside / ring)",
        ("inside_mean", "inside_std", "inside_skew", "ring_mean", "ring_std"),
    ),
    (
        "Texture",
        (
            "inside_glcm_contrast_mean", "inside_glcm_homogeneity_mean",
            "inside_glcm_correlation_mean", "inside_lbp_entropy",
            "ring_glcm_contrast_mean", "ring_lbp_entropy",
        ),
    ),
    (
        "Colour and melanization",
        (
            "inside_MI_lightness", "inside_MI_browning", "inside_MI_darkfraction",
            "inside_lab_L_mean", "inside_lab_a_mean", "inside_lab_b_mean",
            "inside_hsv_S_mean", "ring_MI_lightness",
        ),
    ),
    (
        "Zonation",
        (
            "inside_zonation_L_range", "inside_zonation_L_slope",
            "inside_zonation_n_reversals", "inside_sector_L_sd",
        ),
    ),
)


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NA"
        number = float(value)
        if abs(number) >= 10000:
            return f"{number:,.0f}"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def number_block(values: dict[str, Any]) -> str:
    """The text panel of measured numbers, grouped so it can be scanned."""
    lines: list[str] = []
    for heading, keys in _NUMBER_BLOCKS:
        present = [k for k in keys if k in values]
        if not present:
            continue
        if lines:
            lines.append("")
        lines.append(heading)
        lines.append("-" * len(heading))
        for key in present:
            label = key.replace("inside_", "in.").replace("ring_", "rg.")
            lines.append(f"{label:<28s} {_format_number(values[key]):>12s}")
    return "\n".join(lines)


def make_panel(
    data: PanelData,
    params: AnalysisParams,
    *,
    figsize: tuple[float, float] = (17.0, 9.5),
    title_extra: str = "",
):
    """
    Build the QC figure for one colony.

    Layout, left to right:

    * the original image with the mask, the ring, the bounding box, the centre
      and the fitted ellipse axes drawn on - this is where a bad segmentation
      shows itself immediately
    * the grayscale actually used for texture
    * the quantised patches fed to GLCM, inside and ring
    * the LBP code map
    * the colour panel: Lab lightness map, the radial zonation curve, and the
      dominant colour swatches
    * the numbers

    Returns a matplotlib Figure. The caller decides whether to show it in the
    interface or write it into a PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from skimage import color as skcolor

    from .features.morphology import largest_region
    from .features.texture import lbp_map, quantise_region

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(
        3, 5, width_ratios=[1.5, 1, 1, 1.1, 1.55], height_ratios=[1, 1, 1]
    )

    # ---- 1. original with regions ----
    axis = figure.add_subplot(grid[:, 0])
    axis.imshow(data.rgb)
    mask_layer = np.zeros((*data.mask.shape, 4), np.float32)
    mask_layer[data.mask] = (0.0, 1.0, 0.0, 0.25)
    ring_layer = np.zeros((*data.ring.shape, 4), np.float32)
    ring_layer[data.ring] = (0.0, 1.0, 1.0, 0.30)
    axis.imshow(mask_layer)
    axis.imshow(ring_layer)

    region = largest_region(data.mask)
    if region is not None:
        min_row, min_col, max_row, max_col = region.bbox
        axis.add_patch(
            patches.Rectangle(
                (min_col, min_row), max_col - min_col, max_row - min_row,
                fill=False, edgecolor="yellow", lw=1.4, ls="--",
            )
        )
        cy, cx = region.centroid
        axis.plot(cx, cy, "r+", markersize=13, markeredgewidth=2)
        _draw_axes(axis, region)
    axis.set_title(
        f"Colony (green) and ring (cyan, {data.ring_width_px} px)", fontsize=10
    )
    axis.axis("off")

    # ---- 2. grayscale ----
    axis = figure.add_subplot(grid[0, 1])
    axis.imshow(data.gray, cmap="gray")
    axis.set_title("Grayscale used for texture", fontsize=9)
    axis.axis("off")

    # ---- 3. GLCM patches ----
    inside_patch = quantise_region(data.gray, data.mask, params.glcm.levels)
    axis = figure.add_subplot(grid[0, 2])
    if inside_patch is not None:
        axis.imshow(inside_patch, cmap="gray", vmin=0, vmax=params.glcm.levels - 1)
    axis.set_title(f"Inside, {params.glcm.levels} levels (GLCM)", fontsize=9)
    axis.axis("off")

    ring_patch = quantise_region(data.gray, data.ring, params.glcm.levels)
    axis = figure.add_subplot(grid[1, 1])
    if ring_patch is not None:
        axis.imshow(ring_patch, cmap="gray", vmin=0, vmax=params.glcm.levels - 1)
    axis.set_title("Ring, quantised (GLCM)", fontsize=9)
    axis.axis("off")

    # ---- 4. LBP ----
    axis = figure.add_subplot(grid[1, 2])
    codes = lbp_map(data.gray, params.lbp)
    shown = np.where(data.mask, codes, np.nan)
    image = axis.imshow(shown, cmap="magma")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("LBP codes inside colony", fontsize=9)
    axis.axis("off")

    # ---- 5. colour ----
    rgb01 = np.clip(np.asarray(data.rgb, float) / 255.0, 0, 1)
    lab = skcolor.rgb2lab(rgb01[..., :3])

    axis = figure.add_subplot(grid[0, 3])
    lightness = np.where(data.mask, lab[..., 0], np.nan)
    image = axis.imshow(lightness, cmap="cividis", vmin=0, vmax=100)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("L* lightness (low = melanised)", fontsize=9)
    axis.axis("off")

    axis = figure.add_subplot(grid[1, 3])
    _plot_zonation(axis, data.values, params)

    axis = figure.add_subplot(grid[2, 3])
    _plot_dominant_colours(axis, data.values, params)

    # ---- 6. numbers ----
    axis = figure.add_subplot(grid[:, 4])
    axis.axis("off")
    text = number_block(data.values)
    # Size the font so the whole block fits the column height at any figure
    # size. Fixing it at 8 pt let the lower feature groups run off the bottom of
    # the page on the smaller (on-screen) panel, which is why features looked
    # cut off. The numbers span all three rows, so ~86% of the figure height is
    # available.
    n_lines = max(1, text.count("\n") + 1)
    available_pts = figsize[1] * 72 * 0.86
    font_size = max(5.5, min(8.5, available_pts / (n_lines * 1.42)))
    axis.text(
        0.0, 1.0, text or "Run the feature step to fill in the numbers.",
        va="top", ha="left", fontsize=font_size, family="monospace",
        linespacing=1.25,
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    axis = figure.add_subplot(grid[2, 1:3])
    axis.axis("off")
    notes = list(data.warnings)
    calibration = data.values.get("colour_calibration_method")
    if calibration:
        found = data.values.get("colour_card_found")
        residual = data.values.get("colour_card_residual_dE")
        detail = f"colour calibration: {calibration}"
        if calibration == "card":
            detail += f" (card found={bool(found)}, residual dE={_format_number(residual, 2)})"
        notes.append(detail)
    if notes:
        axis.text(
            0.0, 1.0, "\n".join(f"- {n}" for n in notes),
            va="top", ha="left", fontsize=8, color="#8a5a00", wrap=True,
        )

    figure.suptitle(f"{data.key}   {title_extra}".strip(), fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _draw_axes(axis, region) -> None:
    """Draw the fitted ellipse's major and minor axes."""
    cy, cx = region.centroid
    orientation = region.orientation
    try:
        major = region.axis_major_length / 2.0
        minor = region.axis_minor_length / 2.0
    except AttributeError:  # pragma: no cover
        major = region.major_axis_length / 2.0
        minor = region.minor_axis_length / 2.0

    dx_major = math.cos(orientation) * major
    dy_major = -math.sin(orientation) * major
    dx_minor = math.sin(orientation) * minor
    dy_minor = math.cos(orientation) * minor

    axis.plot([cx - dx_major, cx + dx_major], [cy - dy_major, cy + dy_major],
              color="lime", lw=1.8)
    axis.plot([cx - dx_minor, cx + dx_minor], [cy - dy_minor, cy + dy_minor],
              color="magenta", lw=1.8)


def _plot_zonation(axis, values: dict[str, Any], params: AnalysisParams) -> None:
    """Lightness from colony centre outwards - the ring-structure check."""
    n_rings = params.colour.zonation_n_rings
    ys = [values.get(f"inside_zone{i}_L", math.nan) for i in range(n_rings)]
    xs = np.linspace(0, 1, n_rings)
    valid = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float)) and not math.isnan(y)]

    if valid:
        axis.plot([v[0] for v in valid], [v[1] for v in valid], "o-", color="#4477aa")
        reversals = values.get("inside_zonation_n_reversals", math.nan)
        axis.set_title(
            f"Radial L* profile (reversals: {_format_number(reversals, 0)})",
            fontsize=9,
        )
    else:
        axis.set_title("Radial L* profile (not computed)", fontsize=9)
    axis.set_xlabel("centre → edge", fontsize=8)
    axis.set_ylabel("L*", fontsize=8)
    axis.tick_params(labelsize=7)


def _plot_dominant_colours(axis, values: dict[str, Any], params: AnalysisParams) -> None:
    """The main colours present, as swatches with their pixel shares."""
    from skimage import color as skcolor

    axis.axis("off")
    axis.set_title("Dominant colours inside colony", fontsize=9)

    left = 0.0
    drawn = False
    for i in range(params.colour.dominant_k):
        fraction = values.get(f"inside_dom{i}_frac", math.nan)
        L = values.get(f"inside_dom{i}_L", math.nan)
        a = values.get(f"inside_dom{i}_a", math.nan)
        b = values.get(f"inside_dom{i}_b", math.nan)
        if any(
            not isinstance(v, (int, float)) or math.isnan(v)
            for v in (fraction, L, a, b)
        ):
            continue
        rgb = skcolor.lab2rgb(np.array([[[L, a, b]]])).reshape(3)
        from matplotlib import patches

        axis.add_patch(
            patches.Rectangle(
                (left, 0.25), fraction, 0.5,
                facecolor=np.clip(rgb, 0, 1), edgecolor="black", lw=0.5,
            )
        )
        if fraction > 0.12:
            axis.text(
                left + fraction / 2, 0.5, f"{fraction:.0%}",
                ha="center", va="center", fontsize=8,
                color="white" if L < 55 else "black",
            )
        left += fraction
        drawn = True

    if not drawn:
        axis.text(0.5, 0.5, "not computed", ha="center", va="center", fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def save_panel(figure, path: str | Path, dpi: int = 150) -> Path:
    """Write a panel to disk and release it."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path

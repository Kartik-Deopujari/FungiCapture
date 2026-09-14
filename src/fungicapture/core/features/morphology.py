"""
Shape features: how big the colony is and what shape it has.

All of these come from ``skimage.measure.regionprops`` on the binary mask.
None of them look at pixel colour or brightness - a white colony and a black
colony of the same outline give identical numbers here.
"""

from __future__ import annotations

import math

import numpy as np
from skimage import measure

# Every key this module can produce, in a fixed order. Downstream code uses
# this so a colony that fails still gets the right columns, filled with NaN,
# and the CSV never changes shape between rows.
MORPHOLOGY_KEYS: tuple[str, ...] = (
    "area_px",
    "perimeter_px",
    "roughness",
    "circularity",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "equivalent_diameter_area",
    "orientation_rad",
    "centroid_row",
    "centroid_col",
)


def estimate_perimeter(region_image: np.ndarray, method: str = "crofton") -> float:
    """
    Boundary length of a pixelated shape.

    A pixel boundary travels in stair-steps, so it is always longer than the
    smooth outline it stands for. The three estimators differ in how they
    correct for that, and they disagree substantially. On a synthetic disc of
    radius 60, where the true perimeter is 377 pixels:

    ==========  =========  ============  =================
    method      perimeter  circularity   error
    ==========  =========  ============  =================
    crofton         374.6        1.010    under 1 percent
    legacy_4        392.3        0.921    8 percent low
    legacy_8        472.0        0.636    36 percent low
    ==========  =========  ============  =================

    ``legacy_8`` was the original pipeline's setting. See
    ``MorphologyParams.perimeter_method`` for why its bias is not harmless.
    """
    if method == "crofton":
        return float(measure.perimeter_crofton(region_image, directions=4))
    if method == "legacy_4":
        return float(measure.perimeter(region_image, neighborhood=4))
    if method == "legacy_8":
        return float(measure.perimeter(region_image, neighborhood=8))
    raise ValueError(
        f"Unknown perimeter method {method!r}. "
        "Expected 'crofton', 'legacy_4' or 'legacy_8'."
    )


def _axis_lengths(region) -> tuple[float, float]:
    """Ellipse axis lengths, working across scikit-image versions.

    The properties were renamed from ``major_axis_length`` to
    ``axis_major_length`` in 0.26 and the old names are being removed.
    """
    try:
        return float(region.axis_major_length), float(region.axis_minor_length)
    except AttributeError:  # pragma: no cover - older scikit-image
        return float(region.major_axis_length), float(region.minor_axis_length)


def _equivalent_diameter(region) -> float:
    """Diameter of a circle with the same area, across scikit-image versions."""
    for attribute in ("equivalent_diameter_area", "equivalent_diameter"):
        if hasattr(region, attribute):
            return float(getattr(region, attribute))
    return float(2.0 * math.sqrt(region.area / math.pi))  # pragma: no cover


def morphology_features(
    mask: np.ndarray, params=None
) -> dict[str, float]:
    """
    Measure the shape of the colony.

    Parameters
    ----------
    mask
        Boolean colony mask.
    params
        A ``MorphologyParams``. Defaults to the Crofton perimeter estimator.

    Returns
    -------
    dict
        Keys as in ``MORPHOLOGY_KEYS``.

    Notes on each measure
    ---------------------
    area_px
        Number of colony pixels. The primary growth proxy.
    perimeter_px
        Boundary length, by the estimator named in
        ``params.perimeter_method`` (Crofton by default). **State the method in
        your paper**: perimeter, and therefore circularity and roughness,
        differ by tens of percent between estimators, so a number is only
        comparable to another computed the same way.
    circularity = 4*pi*A / P^2
        1.0 for a perfect circle, lower for an irregular outline. A colony with
        a filamentous margin scores low.
    roughness = P^2 / (4*pi*A)
        The reciprocal of circularity. Kept because it is the more intuitive
        direction for "how ragged is the edge".
    eccentricity
        0 for a circle, approaching 1 for a long thin ellipse.
    solidity = A / A_convex_hull
        How much of its own convex hull the colony fills. Low means lobed or
        concave.
    extent = A / A_bounding_box
        How much of its bounding rectangle the colony fills.
    orientation_rad, centroid_row, centroid_col
        Not phenotypes. They are recorded because the QC panel draws the
        ellipse axes and centre mark from them, and because radial colour
        zonation needs the centroid.
    """
    if params is None:
        from ..params import MorphologyParams

        params = MorphologyParams()

    mask = np.asarray(mask, dtype=bool)
    out: dict[str, float] = {k: math.nan for k in MORPHOLOGY_KEYS}

    if not mask.any():
        out["area_px"] = 0.0
        return out

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        out["area_px"] = 0.0
        return out

    region = max(props, key=lambda r: r.area)

    area = float(region.area)
    # region.image is the mask cropped to its bounding box. Using it rather
    # than the full frame keeps the perimeter identical regardless of how much
    # empty space surrounds the colony in the crop.
    perimeter = estimate_perimeter(region.image, params.perimeter_method)

    out["area_px"] = area
    out["perimeter_px"] = perimeter
    out["circularity"] = (
        float(4.0 * math.pi * area / (perimeter**2)) if perimeter > 0 else math.nan
    )
    out["roughness"] = (
        float(perimeter**2 / (4.0 * math.pi * area)) if area > 0 else math.nan
    )
    out["eccentricity"] = float(region.eccentricity)
    out["solidity"] = float(region.solidity)
    out["extent"] = float(region.extent)
    major, minor = _axis_lengths(region)
    out["major_axis_length"] = major
    out["minor_axis_length"] = minor
    out["equivalent_diameter_area"] = _equivalent_diameter(region)
    out["orientation_rad"] = float(region.orientation)
    centroid_row, centroid_col = region.centroid
    out["centroid_row"] = float(centroid_row)
    out["centroid_col"] = float(centroid_col)
    return out


def largest_region(mask: np.ndarray):
    """The biggest connected blob as a regionprops object, or None.

    Shared by the QC panel, which needs the bounding box, centroid and ellipse
    axes to draw its overlay.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    props = measure.regionprops(measure.label(mask))
    return max(props, key=lambda r: r.area) if props else None

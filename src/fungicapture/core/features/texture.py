"""
Texture features: GLCM (Haralick) and LBP (Ojala).

Intensity statistics say *how bright* the pixels are. Texture says how they are
*arranged*. Two colonies can have identical mean brightness while one is smooth
and the other is wrinkled or powdery, and only texture separates them.
"""

from __future__ import annotations

import math

import numpy as np
from skimage import feature

from ..params import GLCMParams, LBPParams
from .regions import bounding_patch

# --------------------------------------------------------------------------
# GLCM
# --------------------------------------------------------------------------


def quantise_region(
    gray: np.ndarray, region: np.ndarray, levels: int
) -> np.ndarray | None:
    """
    Crop to the region and reduce it to ``levels`` grey levels.

    Two deliberate choices here, both of which matter scientifically:

    1. **Local rescaling.** The region is stretched to its own min and max
       before binning. A pale colony and a dark colony then both use the full
       range of bins, so GLCM measures the *pattern* rather than the overall
       brightness - brightness is already covered by the intensity features.
       Without this, a dark colony would compress into a few bins and its
       texture would look artificially uniform.

    2. **Median fill outside the region.** Pixels outside the mask are set to
       the region median, not to zero. Zero would put a hard black edge right
       at the colony boundary, and GLCM would report that artificial cliff as
       genuine high-contrast texture.
    """
    if region.sum() == 0:
        return None

    values = gray[region]
    lo, hi = float(values.min()), float(values.max())

    scaled = gray.astype(np.float32)
    if hi > lo:
        scaled = (scaled - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(scaled)

    scaled = np.clip(np.floor(scaled * (levels - 1)), 0, levels - 1).astype(np.uint8)
    patch = bounding_patch(scaled, region)
    return None if patch is None else patch.astype(np.uint8)


def glcm_features(
    gray: np.ndarray,
    region: np.ndarray,
    prefix: str,
    params: GLCMParams,
) -> dict[str, float]:
    """
    Gray-Level Co-occurrence Matrix texture (Haralick, Shanmugam & Dinstein 1973).

    The GLCM counts how often a pixel of grey level *i* appears at a given
    distance and direction from a pixel of level *j*. Texture properties are
    then summary numbers read off that matrix.

    For each property two values are reported:

    ``_mean``
        Averaged over all distance/angle combinations. Because the four angles
        cover 0, 45, 90 and 135 degrees, this is close to rotation-invariant -
        which matters, since a colony has no fixed orientation on the plate.
    ``_std``
        Spread across those same combinations. A high value means the texture
        is directional: it looks different along one axis than another, as in a
        colony with radial streaking.

    Returns NaN for every property when the region is too small (fewer than 8
    pixels), rather than returning a number computed from almost nothing.
    """
    out: dict[str, float] = {}
    for prop in params.properties:
        out[f"{prefix}_glcm_{prop}_mean"] = math.nan
        out[f"{prefix}_glcm_{prop}_std"] = math.nan

    if region.sum() < 8:
        return out

    patch = quantise_region(gray, region, params.levels)
    if patch is None or patch.size == 0:
        return out

    matrix = feature.graycomatrix(
        patch,
        distances=list(params.distances),
        angles=list(params.angles_rad),
        levels=params.levels,
        symmetric=True,
        normed=True,
    )
    for prop in params.properties:
        values = feature.graycoprops(matrix, prop)
        out[f"{prefix}_glcm_{prop}_mean"] = float(np.mean(values))
        out[f"{prefix}_glcm_{prop}_std"] = float(np.std(values))
    return out


def glcm_keys(prefix: str, params: GLCMParams) -> tuple[str, ...]:
    keys: list[str] = []
    for prop in params.properties:
        keys.append(f"{prefix}_glcm_{prop}_mean")
        keys.append(f"{prefix}_glcm_{prop}_std")
    return tuple(keys)


# --------------------------------------------------------------------------
# LBP
# --------------------------------------------------------------------------


def lbp_features(
    gray: np.ndarray,
    region: np.ndarray,
    prefix: str,
    params: LBPParams,
) -> dict[str, float]:
    """
    Local Binary Pattern texture (Ojala, Pietikainen & Maenpaa 2002).

    Each pixel is compared with P neighbours on a circle of radius R. Every
    neighbour brighter than the centre contributes a 1, darker contributes a 0,
    and the resulting bit pattern is that pixel's texture code.

    The ``uniform`` method keeps patterns with at most two 0->1 or 1->0
    transitions as their own bins - these are the edges, corners and flat
    patches that carry most of the information - and collapses everything else
    into one bin. That gives P + 2 bins instead of 2^P, which for P = 16 is 18
    instead of 65536.

    Outputs
    -------
    ``_lbp_bin_0`` .. ``_lbp_bin_{P+1}``
        The normalised histogram. High-dimensional, so these are written to the
        CSV but excluded from PCA by default.
    ``_lbp_entropy``
        Shannon entropy of that histogram, in bits. One number summarising
        texture complexity: a smooth uniform surface gives low entropy, a
        varied or granular one gives high entropy. This is the LBP feature that
        normally enters the analysis.
    """
    n_bins = params.n_bins
    out: dict[str, float] = {f"{prefix}_lbp_bin_{i}": math.nan for i in range(n_bins)}
    out[f"{prefix}_lbp_entropy"] = math.nan

    if region.sum() < n_bins:
        return out

    codes = feature.local_binary_pattern(
        gray, P=params.n_points, R=params.radius, method=params.method
    )
    values = codes[region]
    histogram, _ = np.histogram(
        values, bins=np.arange(0, n_bins + 1), density=True
    )
    for i, v in enumerate(histogram):
        out[f"{prefix}_lbp_bin_{i}"] = float(v)

    positive = histogram[histogram > 0]
    out[f"{prefix}_lbp_entropy"] = (
        float(-(positive * np.log2(positive)).sum()) if positive.size else 0.0
    )
    return out


def lbp_keys(prefix: str, params: LBPParams) -> tuple[str, ...]:
    keys = [f"{prefix}_lbp_bin_{i}" for i in range(params.n_bins)]
    keys.append(f"{prefix}_lbp_entropy")
    return tuple(keys)


def lbp_map(gray: np.ndarray, params: LBPParams) -> np.ndarray:
    """The full per-pixel LBP code image, for the QC panel to display."""
    return feature.local_binary_pattern(
        gray, P=params.n_points, R=params.radius, method=params.method
    )

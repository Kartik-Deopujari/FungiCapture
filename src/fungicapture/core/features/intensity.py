"""
Intensity statistics: how bright the pixels are, and how that brightness is
distributed.

Computed on the 8-bit grayscale image, separately for the colony interior and
the boundary ring.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis, skew

INTENSITY_STATS: tuple[str, ...] = (
    "n_pixels",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "p05",
    "p95",
    "skew",
    "kurtosis",
)


def intensity_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Summarise a set of pixel values.

    Parameters
    ----------
    values
        1-D array of pixel values from one region.
    prefix
        Column-name prefix, ``"inside"`` or ``"ring"``.

    What the higher moments mean here
    ---------------------------------
    skew
        Asymmetry of the brightness distribution. Positive means a tail towards
        bright pixels - for example a mostly dark colony with a pale centre.
    kurtosis
        How much of the variance sits in rare extreme values. Reported as
        **excess** kurtosis (Fisher), so a normal distribution gives 0.

    Both use the bias-corrected G1/G2 estimators (``bias=False``), as described
    by Joanes and Gill (1998). Cite that if a reviewer asks which convention
    you used - there are several and they disagree.

    Guards
    ------
    Skew needs at least 3 values and kurtosis at least 4. Both are undefined
    when every pixel has the same value, because the denominator is the
    standard deviation. In those cases 0.0 is returned rather than NaN: a
    perfectly uniform region genuinely has no asymmetry and no excess
    tail-weight, so 0 is the honest answer and it keeps the column numeric.
    """
    values = np.asarray(values, dtype=float).ravel()
    out = {f"{prefix}_{k}": math.nan for k in INTENSITY_STATS}
    out[f"{prefix}_n_pixels"] = float(values.size)

    if values.size == 0:
        return out

    spread = float(np.std(values))
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values, ddof=0))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p05"] = float(np.percentile(values, 5))
    out[f"{prefix}_p95"] = float(np.percentile(values, 95))
    out[f"{prefix}_skew"] = (
        float(skew(values, bias=False)) if values.size > 2 and spread > 0 else 0.0
    )
    out[f"{prefix}_kurtosis"] = (
        float(kurtosis(values, bias=False)) if values.size > 3 and spread > 0 else 0.0
    )
    return out


def intensity_keys(prefix: str) -> tuple[str, ...]:
    """Column names this module produces for one region."""
    return tuple(f"{prefix}_{k}" for k in INTENSITY_STATS)

"""
Colour and melanization features.

Why this module is the point of the whole rebuild
-------------------------------------------------
The original pipeline converted every image to grayscale in
``load_gray_image`` and then measured only brightness and texture. For
*Aureobasidium pullulans* - a black yeast whose defining phenotype is melanin
production - that throws away the trait of interest before measurement starts.

Worse, the grayscale conversion is not neutral. ``rgb2gray`` uses ITU-R BT.709
weights (0.2125 R, 0.7154 G, 0.0721 B), so a green-tinted colony and a
brown-pigmented one of equal darkness give different grey values for reasons
that have nothing to do with melanin.

Everything here is computed from the original colour crop instead.

Four levels
-----------
1. **Colour-space statistics** - RGB, HSV and CIE-Lab, per region.
2. **Melanization indices** - named, defensible darkness and browning scores.
3. **Histograms and dominant colours** - shape of the colour distribution.
4. **Radial zonation and sectoring** - concentric rings and uneven growth.

A warning that applies to all of it
-----------------------------------
Colour numbers are only comparable between images if the lighting and camera
settings were the same. Run the calibration in
``fungicapture.core.calibration`` before trusting any cross-batch comparison.
"""

from __future__ import annotations

import math

import numpy as np
from skimage import color

from ..params import ColourParams

# Channel names per colour space. These become part of the column names, so
# they must never change once data has been published.
SPACE_CHANNELS: dict[str, tuple[str, ...]] = {
    "rgb": ("R", "G", "B"),
    "hsv": ("H", "S", "V"),
    "lab": ("L", "a", "b"),
}

COLOUR_STATS: tuple[str, ...] = (
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


# --------------------------------------------------------------------------
# Colour space conversion
# --------------------------------------------------------------------------


def to_spaces(
    rgb: np.ndarray, params: ColourParams
) -> dict[str, np.ndarray]:
    """
    Convert an 8-bit RGB image into every requested colour space.

    What each space is for
    ----------------------
    rgb
        Raw sensor-like values, 0-255. Easy to relate to what you see, but it
        mixes "how bright" and "which colour" together, so it drifts badly with
        lighting.
    hsv
        Hue (which colour, 0-1 around the wheel), Saturation (how strong),
        Value (how bright). Separating hue from brightness makes it far more
        stable across lighting than raw RGB.
    lab
        CIE L*a*b*. L* is lightness 0-100, a* runs green to red, b* runs blue
        to yellow. It is *perceptually uniform*: an equal numeric step looks
        like an equal colour change to the eye. This is the space used for
        pigment work in food and materials science, and it is where the
        melanization indices are defined.

    The reference white is recorded in ``params.lab_illuminant`` (default D65,
    2 degree observer) and written into every output file, because L* values
    computed against a different white are not comparable.
    """
    rgb_float = np.asarray(rgb, dtype=np.float64)
    if rgb_float.max() > 1.0:
        rgb_float = rgb_float / 255.0
    rgb_float = np.clip(rgb_float, 0.0, 1.0)

    out: dict[str, np.ndarray] = {}
    for space in params.spaces:
        if space == "rgb":
            out["rgb"] = rgb_float * 255.0
        elif space == "hsv":
            out["hsv"] = color.rgb2hsv(rgb_float)
        elif space == "lab":
            out["lab"] = color.rgb2lab(
                rgb_float,
                illuminant=params.lab_illuminant,
                observer=params.lab_observer,
            )
        else:
            raise ValueError(f"Unknown colour space: {space!r}")
    return out


# --------------------------------------------------------------------------
# Level 1 - colour space statistics
# --------------------------------------------------------------------------


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Nine summary numbers for one channel in one region."""
    from scipy.stats import kurtosis, skew

    values = np.asarray(values, dtype=float).ravel()
    out = {f"{prefix}_{s}": math.nan for s in COLOUR_STATS}
    if values.size == 0:
        return out

    # Skew and kurtosis divide by the standard deviation cubed and to the
    # fourth. When a region is *nearly* uniform - a flat synthetic patch, or a
    # colony filling one quantisation step - that divisor is close to zero and
    # the result is numerical noise, not biology. Requiring the spread to be
    # meaningful relative to the values themselves keeps those cases at 0.0,
    # which is the honest answer for a region with no distribution shape.
    mean_value = float(np.mean(values))
    spread = float(np.std(values))
    shape_is_meaningful = spread > max(1e-8, 1e-6 * abs(mean_value))

    out[f"{prefix}_mean"] = mean_value
    out[f"{prefix}_std"] = float(np.std(values, ddof=0))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p05"] = float(np.percentile(values, 5))
    out[f"{prefix}_p95"] = float(np.percentile(values, 95))
    out[f"{prefix}_skew"] = (
        float(skew(values, bias=False))
        if values.size > 2 and shape_is_meaningful
        else 0.0
    )
    out[f"{prefix}_kurtosis"] = (
        float(kurtosis(values, bias=False))
        if values.size > 3 and shape_is_meaningful
        else 0.0
    )
    return out


def colour_statistics(
    spaces: dict[str, np.ndarray],
    region: np.ndarray,
    region_name: str,
    params: ColourParams,
) -> dict[str, float]:
    """
    Level 1: mean, spread and distribution shape for every channel.

    Column names look like ``inside_lab_L_mean`` or ``ring_hsv_S_p95``.
    """
    out: dict[str, float] = {}
    for space in params.spaces:
        image = spaces[space]
        for index, channel in enumerate(SPACE_CHANNELS[space]):
            prefix = f"{region_name}_{space}_{channel}"
            values = image[..., index][region] if region.any() else np.array([])
            out.update(_stats(values, prefix))
    return out


# --------------------------------------------------------------------------
# Level 2 - melanization
# --------------------------------------------------------------------------


def melanization_indices(
    lab: np.ndarray,
    region: np.ndarray,
    region_name: str,
    params: ColourParams,
) -> dict[str, float]:
    """
    Level 2: three named melanization scores.

    Fungal melanin makes a colony **dark** and shifts it towards **brown**. In
    Lab space that is low L* with a positive b*.

    ``MI_lightness = 100 - mean(L*)``
        A simple darkness score on the same 0-100 scale as L*, flipped so that
        **higher means more melanin**. 0 is a perfectly white colony, 100 a
        perfectly black one. This is the index to use if you want one number.

    ``MI_browning = mean(b*) / (mean(L*) + eps)``
        Yellow-brown content per unit of lightness. This separates a genuinely
        brown colony from one that is merely dark: a grey colony has b* near 0
        and scores low, while a tan colony scores high even if it is not very
        dark. The epsilon guard prevents a division blow-up on a near-black
        colony.

    ``MI_darkfraction``
        The fraction of colony pixels darker than
        ``params.melanin_dark_L`` (default L* < 35). This catches *partial*
        melanization that a mean would hide - a colony with a black centre and
        a pale rim has a middling mean but a clear dark fraction.

    All three are FungiCapture definitions, not standards. State the formulas
    and the threshold in your methods section.
    """
    out = {
        f"{region_name}_MI_lightness": math.nan,
        f"{region_name}_MI_browning": math.nan,
        f"{region_name}_MI_darkfraction": math.nan,
    }
    if not region.any():
        return out

    L = lab[..., 0][region]
    b = lab[..., 2][region]
    if L.size == 0:
        return out

    mean_L = float(np.mean(L))
    mean_b = float(np.mean(b))

    out[f"{region_name}_MI_lightness"] = 100.0 - mean_L
    out[f"{region_name}_MI_browning"] = mean_b / (mean_L + 1e-6)
    out[f"{region_name}_MI_darkfraction"] = float(
        np.mean(L < params.melanin_dark_L)
    )
    return out


# --------------------------------------------------------------------------
# Level 3 - histograms and dominant colours
# --------------------------------------------------------------------------

# Fixed ranges per channel, so a bin means the same thing in every image. If
# each image used its own min and max, bin 3 of one colony would not be
# comparable to bin 3 of another.
CHANNEL_RANGES: dict[str, tuple[float, float]] = {
    "R": (0.0, 255.0),
    "G": (0.0, 255.0),
    "B": (0.0, 255.0),
    "H": (0.0, 1.0),
    "S": (0.0, 1.0),
    "V": (0.0, 1.0),
    "L": (0.0, 100.0),
    "a": (-128.0, 127.0),
    "b": (-128.0, 127.0),
}

_CHANNEL_SPACE = {
    "R": "rgb", "G": "rgb", "B": "rgb",
    "H": "hsv", "S": "hsv", "V": "hsv",
    "L": "lab", "a": "lab", "b": "lab",
}


def colour_histograms(
    spaces: dict[str, np.ndarray],
    region: np.ndarray,
    region_name: str,
    params: ColourParams,
) -> dict[str, float]:
    """
    Level 3a: binned colour distributions.

    High-dimensional (6 channels x 16 bins x 2 regions by default), so these
    are written to the CSV but excluded from PCA unless you ask for them.
    They are there for anyone who wants to model the full distribution shape
    rather than its summary statistics.
    """
    out: dict[str, float] = {}
    n_bins = params.histogram_bins

    for channel in params.histogram_channels:
        space = _CHANNEL_SPACE.get(channel)
        if space is None:
            raise ValueError(
                f"Unknown histogram channel {channel!r}; expected one of "
                f"{sorted(_CHANNEL_SPACE)} (channel names are case-sensitive)."
            )
        if space not in spaces:
            continue
        index = SPACE_CHANNELS[space].index(channel)
        prefix = f"{region_name}_{space}_{channel}_hist"

        if not region.any():
            for i in range(n_bins):
                out[f"{prefix}_{i}"] = math.nan
            continue

        values = spaces[space][..., index][region]
        lo, hi = CHANNEL_RANGES[channel]
        histogram, _ = np.histogram(values, bins=n_bins, range=(lo, hi), density=False)
        total = histogram.sum()
        normalised = histogram / total if total > 0 else histogram.astype(float)
        for i, v in enumerate(normalised):
            out[f"{prefix}_{i}"] = float(v)
    return out


def dominant_colours(
    lab: np.ndarray,
    region: np.ndarray,
    region_name: str,
    params: ColourParams,
) -> dict[str, float]:
    """
    Level 3b: the k main colours present, by k-means in Lab space.

    Why this matters: a mean hides structure. A colony that is half black and
    half cream has the same mean L* as a uniformly grey one, but they are
    completely different phenotypes. Three cluster centres plus their pixel
    fractions separate them.

    Clusters are sorted by decreasing pixel fraction, so ``dom0`` is always the
    most common colour. Without sorting, k-means label order is arbitrary and
    the columns would not mean the same thing from one colony to the next.

    Clustering runs in Lab because distance there matches perceived colour
    difference, so the clusters correspond to colours a person would call
    distinct.
    """
    k = params.dominant_k
    out: dict[str, float] = {}
    for i in range(k):
        for suffix in ("L", "a", "b", "frac"):
            out[f"{region_name}_dom{i}_{suffix}"] = math.nan

    if not region.any():
        return out

    pixels = lab[region]
    if pixels.shape[0] < k:
        return out

    # Deterministic subsample: a fixed-seed generator, so the same colony gives
    # the same answer on every run and on every machine.
    if pixels.shape[0] > params.dominant_max_pixels:
        rng = np.random.default_rng(params.random_state)
        choice = rng.choice(
            pixels.shape[0], size=params.dominant_max_pixels, replace=False
        )
        pixels = pixels[choice]

    from sklearn.cluster import KMeans

    # A very uniform colony may hold fewer distinct colours than k. Asking for
    # more clusters than there are colours makes scikit-learn warn and return
    # empty clusters, so ask only for what is there. The unused dom slots stay
    # NaN, which is the honest answer: that colony has no third colour.
    distinct = min(k, len(np.unique(np.round(pixels, 2), axis=0)))
    if distinct < 1:
        return out

    model = KMeans(
        n_clusters=distinct, random_state=params.random_state, n_init=10
    ).fit(pixels)
    labels = model.labels_
    fractions = np.bincount(labels, minlength=distinct) / labels.size

    order = np.argsort(-fractions)
    for rank, cluster in enumerate(order):
        centre = model.cluster_centers_[cluster]
        out[f"{region_name}_dom{rank}_L"] = float(centre[0])
        out[f"{region_name}_dom{rank}_a"] = float(centre[1])
        out[f"{region_name}_dom{rank}_b"] = float(centre[2])
        out[f"{region_name}_dom{rank}_frac"] = float(fractions[cluster])
    return out


# --------------------------------------------------------------------------
# Level 4 - radial zonation and sectoring
# --------------------------------------------------------------------------


def radial_zonation(
    lab: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
    params: ColourParams,
    region_name: str = "inside",
) -> dict[str, float]:
    """
    Level 4a: does the colour change from the centre of the colony outwards?

    This catches concentric rings and centre-to-edge gradients, which
    *Aureobasidium* does show and which no summary statistic can detect.

    Method
    ------
    1. A Euclidean distance transform gives every colony pixel its distance to
       the nearest background pixel. That is depth *into* the colony.
    2. Depth is inverted and normalised so 0 is the centre and 1 is the edge.
       Using the distance transform rather than straight-line distance from the
       centroid means the bands follow the real colony outline, so a lobed or
       irregular colony is still divided sensibly.
    3. The colony is split into ``n_rings`` bands of equal normalised depth and
       mean L*, a*, b* and saturation are recorded per band.

    Summary numbers
    ---------------
    ``zonation_L_range``
        Largest minus smallest band lightness. Near zero means a flat, evenly
        coloured colony.
    ``zonation_L_slope``
        Slope of a straight line fitted to band lightness against normalised
        radius. Negative means the colony gets **darker** towards the edge.
    ``zonation_n_reversals``
        How many times the lightness trend changes direction across the bands.
        A monotonic gradient gives 0. **Two or more suggests genuine
        concentric ring structure** rather than a simple gradient - this is the
        number to look at for zonate colonies.
    """
    from scipy import ndimage

    n_rings = params.zonation_n_rings
    out: dict[str, float] = {}
    for i in range(n_rings):
        for suffix in ("L", "a", "b", "S"):
            out[f"{region_name}_zone{i}_{suffix}"] = math.nan
    out[f"{region_name}_zonation_L_range"] = math.nan
    out[f"{region_name}_zonation_L_slope"] = math.nan
    out[f"{region_name}_zonation_n_reversals"] = math.nan

    if not mask.any():
        return out

    distance = ndimage.distance_transform_edt(mask)
    peak = float(distance.max())
    if peak <= 0:
        return out

    # 0 at the deepest interior point, 1 at the boundary.
    normalised = 1.0 - (distance / peak)
    values = normalised[mask]

    L = lab[..., 0][mask]
    a = lab[..., 1][mask]
    b = lab[..., 2][mask]
    S = hsv[..., 1][mask]

    edges = np.linspace(0.0, 1.0, n_rings + 1)
    band_L: list[float] = []
    band_centres: list[float] = []

    for i in range(n_rings):
        lo, hi = edges[i], edges[i + 1]
        selected = (values >= lo) & (values < hi) if i < n_rings - 1 else (values >= lo)
        if not selected.any():
            band_L.append(math.nan)
            continue
        out[f"{region_name}_zone{i}_L"] = float(np.mean(L[selected]))
        out[f"{region_name}_zone{i}_a"] = float(np.mean(a[selected]))
        out[f"{region_name}_zone{i}_b"] = float(np.mean(b[selected]))
        out[f"{region_name}_zone{i}_S"] = float(np.mean(S[selected]))
        band_L.append(float(np.mean(L[selected])))
        band_centres.append(0.5 * (lo + hi))

    filled = np.array([v for v in band_L if not math.isnan(v)], dtype=float)
    if filled.size >= 2:
        out[f"{region_name}_zonation_L_range"] = float(filled.max() - filled.min())
        centres = np.array(band_centres[: filled.size], dtype=float)
        slope, _ = np.polyfit(centres, filled, 1)
        out[f"{region_name}_zonation_L_slope"] = float(slope)

        # Floating-point summation order makes band means differ at the ULP
        # level (~1e-13 on an L* of order 100) even on a perfectly flat colony.
        # Without a tolerance those sub-noise wiggles are counted as genuine
        # lightness-trend reversals, so a flat colony would falsely report ring
        # structure. Treat any change below a threshold far smaller than a real
        # lightness step (< 1e-6 L*, well below 8-bit quantisation) as no change.
        differences = np.diff(filled)
        flatness_tol = 1e-6
        differences[np.abs(differences) < flatness_tol] = 0.0
        signs = np.sign(differences)
        signs = signs[signs != 0]
        reversals = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
        out[f"{region_name}_zonation_n_reversals"] = float(reversals)
    return out


def angular_sectoring(
    lab: np.ndarray,
    mask: np.ndarray,
    centroid: tuple[float, float],
    params: ColourParams,
    region_name: str = "inside",
) -> dict[str, float]:
    """
    Level 4b: did the colony grow evenly in all directions?

    The colony is split into equal angular sectors around its centroid and the
    mean lightness of each is compared. A colony that has thrown a pale sector
    - a common sign of a morphological switch or a sectoring mutant - shows a
    high spread across sectors while its overall mean looks unremarkable.

    ``sector_L_sd``
        Standard deviation of mean L* across sectors. Higher means more uneven.
    ``sector_L_range``
        Largest minus smallest sector mean.
    """
    n_sectors = params.sector_n_sectors
    out = {
        f"{region_name}_sector_L_sd": math.nan,
        f"{region_name}_sector_L_range": math.nan,
    }
    if not mask.any():
        return out

    rows, cols = np.nonzero(mask)
    centre_row, centre_col = centroid
    angles = np.arctan2(rows - centre_row, cols - centre_col)  # -pi .. pi
    bins = np.floor((angles + math.pi) / (2 * math.pi) * n_sectors).astype(int)
    bins = np.clip(bins, 0, n_sectors - 1)

    L = lab[..., 0][mask]
    means = [
        float(np.mean(L[bins == s])) for s in range(n_sectors) if np.any(bins == s)
    ]
    if len(means) >= 2:
        array = np.array(means, dtype=float)
        out[f"{region_name}_sector_L_sd"] = float(np.std(array, ddof=0))
        out[f"{region_name}_sector_L_range"] = float(array.max() - array.min())
    return out


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def colour_features(
    rgb: np.ndarray,
    regions: dict[str, np.ndarray],
    params: ColourParams,
    *,
    centroid: tuple[float, float] | None = None,
    include_histograms: bool = True,
) -> dict[str, float]:
    """
    Every colour feature for one colony.

    Parameters
    ----------
    rgb
        The colour crop, ideally already colour-calibrated.
    regions
        ``{"inside": mask, "ring": ring}``.
    centroid
        Colony centre as ``(row, col)``, from the morphology features. Needed
        for angular sectoring; that feature is skipped if it is absent.
    """
    spaces = to_spaces(rgb, params)
    out: dict[str, float] = {}

    for name, region in regions.items():
        region = np.asarray(region, dtype=bool)
        out.update(colour_statistics(spaces, region, name, params))
        if "lab" in spaces:
            out.update(melanization_indices(spaces["lab"], region, name, params))
            out.update(dominant_colours(spaces["lab"], region, name, params))
        if include_histograms:
            out.update(colour_histograms(spaces, region, name, params))

    # Zonation and sectoring describe internal structure, so they only make
    # sense for the colony body - a thin ring has no inside to zone.
    inside = np.asarray(regions.get("inside", np.zeros((1, 1), bool)), dtype=bool)
    if "lab" in spaces and "hsv" in spaces:
        out.update(radial_zonation(spaces["lab"], spaces["hsv"], inside, params))
    if "lab" in spaces and centroid is not None:
        out.update(angular_sectoring(spaces["lab"], inside, centroid, params))
    return out

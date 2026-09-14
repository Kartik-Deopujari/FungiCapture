"""
Plain-language descriptions for every feature FungiCapture measures.

The feature table has several hundred columns, but they are built from a small
number of families (a colour space times a channel times a statistic, and so
on). Rather than hand-write hundreds of entries, ``describe_feature`` takes any
column name apart and composes a short description from its pieces. That way a
newly added feature is explained automatically, and the guide the interface
shows can never fall out of step with what is actually measured.

Nothing here imports the heavy scientific stack, so the guide dialog opens
instantly.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Vocabulary of the pieces a feature name is built from
# --------------------------------------------------------------------------

_REGION = {
    "inside": "colony interior",
    "ring": "boundary ring just outside the colony",
}

_SPACE = {
    "rgb": "RGB",
    "hsv": "HSV",
    "lab": "CIE L*a*b*",
}

# Keyed by lowercase channel letter, because names are lowercased before lookup.
_CHANNEL = {
    ("rgb", "r"): "red channel",
    ("rgb", "g"): "green channel",
    ("rgb", "b"): "blue channel",
    ("hsv", "h"): "hue",
    ("hsv", "s"): "saturation",
    ("hsv", "v"): "value / brightness",
    ("lab", "l"): "L* lightness",
    ("lab", "a"): "a* (green→red)",
    ("lab", "b"): "b* (blue→yellow)",
}

_STAT = {
    "mean": "average",
    "std": "spread (standard deviation)",
    "min": "minimum",
    "max": "maximum",
    "median": "median",
    "skew": "skew (asymmetry of the distribution)",
    "kurtosis": "kurtosis (how heavy-tailed the distribution is)",
    "n_pixels": "number of pixels measured",
}

_GLCM = {
    "contrast": "contrast — local intensity difference, high on coarse texture",
    "dissimilarity": "dissimilarity — like contrast but linear",
    "homogeneity": "homogeneity — high on smooth, even texture",
    "energy": "energy — high when a few grey patterns dominate",
    "correlation": "correlation — how predictable a pixel is from its neighbour",
    "asm": "angular second moment — texture uniformity (energy squared)",
}

# Exact-match descriptions for the one-off features that are not part of a
# regular family.
_EXACT = {
    # ---- morphology / shape ----
    "area_px": "Colony area, in pixels.",
    "perimeter_px": "Length of the colony outline, in pixels.",
    "circularity": "4·π·area / perimeter². 1.0 is a perfect circle; lower means a more irregular or lobed outline.",
    "roughness": "perimeter² / (4·π·area). The inverse of circularity — higher means a rougher, more crenelated margin.",
    "eccentricity": "How elongated the colony is: 0 is a circle, approaching 1 is a long ellipse.",
    "solidity": "Area divided by the area of its convex hull. Below 1 means dents, bays or a ragged edge.",
    "extent": "Area divided by the area of its bounding box. Low for a star-like or sprawling shape.",
    "major_axis_length": "Length of the long axis of the best-fit ellipse, in pixels.",
    "minor_axis_length": "Length of the short axis of the best-fit ellipse, in pixels.",
    "equivalent_diameter_area": "Diameter of a circle with the same area as the colony, in pixels.",
    "orientation_rad": "Angle of the colony's long axis, in radians.",
    "centroid_row": "Row (y) of the colony centre in the crop.",
    "centroid_col": "Column (x) of the colony centre in the crop.",
    "ring_width_px": "Width of the boundary ring band, in pixels. Scales with colony size.",
    # ---- melanization indices ----
    "inside_mi_lightness": "Melanin index = 100 − mean L*. Higher means a darker, more melanised interior.",
    "ring_mi_lightness": "Melanin index (100 − mean L*) for the ring: darkening at the colony margin.",
    "inside_mi_browning": "mean b* / mean L*. A browning index — yellow-brown pigment relative to lightness.",
    "ring_mi_browning": "Browning index (mean b* / mean L*) for the ring.",
    "inside_mi_darkfraction": "Fraction of interior pixels darker than the melanin L* threshold.",
    "ring_mi_darkfraction": "Fraction of ring pixels darker than the melanin L* threshold.",
    # ---- zonation and sectoring ----
    "inside_zonation_l_range": "Largest minus smallest mean L* across the concentric zones — how much lightness changes from centre to edge.",
    "inside_zonation_l_slope": "Trend in mean L* from centre to edge; negative means the margin is darker than the middle.",
    "inside_zonation_n_reversals": "How many times the centre-to-edge lightness profile changes direction — a count of concentric rings/bands.",
    "inside_sector_l_sd": "Spread of mean L* between angular sectors — high when one side of the colony is darker (sectoring/asymmetry).",
    "inside_sector_l_range": "Lightest minus darkest sector mean L* — the strength of angular asymmetry.",
    # ---- colour calibration / provenance ----
    "colour_calibration_method": "How colour was standardised: 'card', 'background' or none.",
    "colour_card_found": "Whether a colour reference card was located in the photograph.",
    "colour_card_residual_de": "Colour error (ΔE) left after card correction; lower is a better calibration.",
    "colour_bg_gain_r": "Red gain applied by agar-background normalisation.",
    "colour_bg_gain_g": "Green gain applied by agar-background normalisation.",
    "colour_bg_gain_b": "Blue gain applied by agar-background normalisation.",
    "colour_calibration_warnings": "Any problems noted while calibrating colour on this image.",
    "inside_lbp_entropy": "Entropy of the local binary pattern codes inside the colony — micro-texture disorder.",
    "ring_lbp_entropy": "Entropy of the local binary pattern codes in the ring — micro-texture disorder at the margin.",
}

# Identifier / bookkeeping columns, described briefly so the guide is complete.
_METADATA = {
    "filename", "stem", "base_key", "sample_id", "accession", "media",
    "plate_index", "plate_id", "image_path", "mask_path",
    "fungicapture_version", "project_name", "imaging_mode", "params_hash",
    "lab_illuminant", "ring_fraction", "glcm_levels", "lbp_points",
    "lbp_radius", "generated_utc",
}


def _percentile(token: str) -> str | None:
    match = re.fullmatch(r"p(\d{1,3})", token)
    if match:
        return f"{int(match.group(1))}th percentile"
    return None


def describe_feature(name: str) -> str:
    """A short, plain-language description of one feature column."""
    key = name.lower()

    if key in _EXACT:
        return _EXACT[key]
    if key in _METADATA:
        return "Identifier / run information (not a measurement)."

    region_prefix = ""
    for region, phrase in _REGION.items():
        if key.startswith(region + "_"):
            region_prefix = f"{phrase}: "
            key_body = key[len(region) + 1 :]
            break
    else:
        key_body = key

    # GLCM texture: <prop>_<mean|std>
    glcm = re.fullmatch(r"glcm_([a-z]+)_(mean|std)", key_body)
    if glcm:
        prop, stat = glcm.group(1), glcm.group(2)
        detail = _GLCM.get(prop, f"{prop} (grey-level co-occurrence texture)")
        tail = "" if stat == "mean" else " — variation across directions"
        return f"{region_prefix}GLCM texture, {detail}{tail}."

    # LBP histogram bin
    lbp = re.fullmatch(r"lbp_bin_(\d+)", key_body)
    if lbp:
        return (
            f"{region_prefix}fraction of pixels with local-binary-pattern code "
            f"{lbp.group(1)} — one micro-texture pattern."
        )

    # Colour histogram bin: <space>_<channel>_hist_<n>
    hist = re.fullmatch(r"([a-z]+)_([a-zA-Z])_hist_(\d+)", key_body)
    if hist:
        space, channel, idx = hist.group(1), hist.group(2), hist.group(3)
        chan = _CHANNEL.get((space, channel), f"{channel} channel")
        return (
            f"{region_prefix}share of pixels in histogram bin {idx} of the "
            f"{chan} ({_SPACE.get(space, space)})."
        )

    # Dominant colour: dom<k>_<L|a|b|frac>
    dom = re.fullmatch(r"dom(\d+)_(l|a|b|frac)", key_body)
    if dom:
        k, comp = dom.group(1), dom.group(2)
        rank = f"#{int(k) + 1}"
        if comp == "frac":
            return f"{region_prefix}fraction of pixels belonging to dominant colour {rank} (k-means)."
        comp_name = {"l": "L* lightness", "a": "a* (green→red)", "b": "b* (blue→yellow)"}[comp]
        return f"{region_prefix}{comp_name} of dominant colour {rank} (k-means cluster centre)."

    # Radial zone: zone<k>_<L|a|b|S>
    zone = re.fullmatch(r"zone(\d+)_(l|a|b|s)", key_body)
    if zone:
        k, comp = zone.group(1), zone.group(2)
        comp_name = {
            "l": "mean L* lightness", "a": "mean a*", "b": "mean b*",
            "s": "mean saturation",
        }[comp]
        return (
            f"{region_prefix}{comp_name} in concentric zone {int(k) + 1} "
            "(centre = 1), describing centre-to-edge colour change."
        )

    # Colour statistic: <space>_<channel>_<stat|pNN>
    colour = re.fullmatch(r"([a-z]+)_([a-zA-Z])_([a-z0-9]+)", key_body)
    if colour and colour.group(1) in _SPACE:
        space, channel, stat = colour.group(1), colour.group(2), colour.group(3)
        chan = _CHANNEL.get((space, channel), f"{channel} channel")
        stat_name = _STAT.get(stat) or _percentile(stat) or stat
        return f"{region_prefix}{stat_name} of the {chan} ({_SPACE.get(space, space)})."

    # Plain grayscale-intensity statistic: <stat|pNN>
    stat_name = _STAT.get(key_body) or _percentile(key_body)
    if stat_name:
        return f"{region_prefix}{stat_name} of grayscale intensity."

    return "Measured feature (no description available yet)."


# --------------------------------------------------------------------------
# Grouping for display
# --------------------------------------------------------------------------

# Order and matchers for the sections shown in the guide. Each entry is
# (section title, predicate on the lowercased name).
_SECTIONS: tuple[tuple[str, re.Pattern[str] | None], ...] = (
    ("Identity & run information", re.compile(r"")),  # filled by _METADATA test
    ("Colour calibration", re.compile(r"^colour_")),
    ("Shape & size", re.compile(
        r"^(area_px|perimeter_px|circularity|roughness|eccentricity|solidity|"
        r"extent|major_axis_length|minor_axis_length|equivalent_diameter_area|"
        r"orientation_rad|centroid_row|centroid_col|ring_width_px)$")),
    ("Brightness (grayscale)", re.compile(
        r"^(inside|ring)_(n_pixels|mean|std|min|max|median|p\d+|skew|kurtosis)$")),
    ("Texture — GLCM", re.compile(r"_glcm_")),
    ("Texture — LBP", re.compile(r"_lbp_")),
    ("Colour — summary statistics", re.compile(r"_(rgb|hsv|lab)_[a-zA-Z]_(?!hist_)")),
    ("Colour — histograms", re.compile(r"_hist_\d+$")),
    ("Colour — dominant colours", re.compile(r"_dom\d+_")),
    ("Melanization indices", re.compile(r"_mi_")),
    ("Radial zonation & sectoring", re.compile(r"_(zone\d+_|zonation_|sector_)")),
)


# A representative feature from each family, shown by the guide before the
# feature step has run (when the real column list is not yet available).
REPRESENTATIVE_FEATURES: tuple[str, ...] = (
    "area_px", "perimeter_px", "circularity", "roughness", "eccentricity",
    "solidity", "extent", "equivalent_diameter_area", "ring_width_px",
    "inside_mean", "inside_std", "inside_p05", "inside_p95", "inside_skew",
    "ring_mean", "ring_std",
    "inside_glcm_contrast_mean", "inside_glcm_homogeneity_mean",
    "inside_glcm_correlation_mean", "ring_glcm_contrast_mean",
    "inside_lbp_bin_0", "inside_lbp_entropy", "ring_lbp_entropy",
    "inside_rgb_R_mean", "inside_hsv_S_mean", "inside_hsv_H_mean",
    "inside_lab_L_mean", "inside_lab_a_mean", "inside_lab_b_mean",
    "inside_lab_L_hist_0", "inside_lab_L_median", "ring_lab_L_mean",
    "inside_dom0_L", "inside_dom0_frac",
    "inside_MI_lightness", "inside_MI_browning", "inside_MI_darkfraction",
    "ring_MI_lightness",
    "inside_zone0_L", "inside_zonation_L_range", "inside_zonation_L_slope",
    "inside_zonation_n_reversals", "inside_sector_L_sd", "inside_sector_L_range",
    "colour_calibration_method", "colour_card_residual_dE",
)


def group_features(columns) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Sort feature columns into titled sections, each a list of (name, description).

    Every column lands in exactly one section, so the guide covers all of them.
    """
    groups: dict[str, list[tuple[str, str]]] = {title: [] for title, _ in _SECTIONS}
    other: list[tuple[str, str]] = []

    for name in columns:
        key = name.lower()
        if key in _METADATA:
            groups["Identity & run information"].append((name, describe_feature(name)))
            continue
        placed = False
        for title, pattern in _SECTIONS:
            if title == "Identity & run information":
                continue
            if pattern is not None and pattern.search(key):
                groups[title].append((name, describe_feature(name)))
                placed = True
                break
        if not placed:
            other.append((name, describe_feature(name)))

    result = [(title, groups[title]) for title, _ in _SECTIONS if groups[title]]
    if other:
        result.append(("Other", other))
    return result

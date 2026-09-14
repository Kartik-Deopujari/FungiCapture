"""
Finding things automatically: Petri dishes, colonies, and where the grid goes.

This module holds the two engines named on the home screen.

**Dish engine** (modes 1 and 2)
    Find the dish rim, mask everything outside it, find the one colony inside.

**Grid engine** (modes 3 and 4)
    Find all the colonies on a plate, then fit a grid to their centres.

Both use fast classical computer vision rather than the segmentation model.
Running SAM 3 on a full plate photograph takes 10-30 seconds on a CPU, which
would make the application unpleasant to use for a job that takes under a
second this way. The model is available as an explicit "refine" step for the
plates where this fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from skimage import color, filters, measure, transform

from .features.regions import remove_small_holes, remove_small_objects

# Working width for detection. Colony finding does not need full resolution,
# and downscaling a 32 MB photograph first is what keeps this under a second.
DETECT_WIDTH = 1500


@dataclass
class Detection:
    """One detected colony, in normalised image coordinates."""

    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    """``(x1, y1, x2, y2)``, normalised."""
    area_frac: float
    """Area as a fraction of the whole image."""
    circularity: float
    confidence: float
    """0 to 1. Low means the app should ask a human to look."""


@dataclass
class DishResult:
    """A detected Petri dish and the colony inside it."""

    centre: tuple[float, float]
    radius: float
    """Normalised by image width."""
    found: bool
    colony: Detection | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GridFit:
    """A grid fitted to detected colonies."""

    row_lines: list[float]
    col_lines: list[float]
    detections: list[Detection]
    n_expected: int
    n_found: int
    confidence: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n_missing(self) -> int:
        return max(0, self.n_expected - self.n_found)


# --------------------------------------------------------------------------
# Shared preparation
# --------------------------------------------------------------------------


def _downscale(image: np.ndarray, width: int = DETECT_WIDTH) -> tuple[np.ndarray, float]:
    """Shrink for speed. Returns the image and the scale factor applied."""
    height, original_width = image.shape[:2]
    if original_width <= width:
        return image, 1.0
    scale = width / original_width
    resized = transform.resize(
        image,
        (int(round(height * scale)), width),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(image.dtype)
    return resized, scale


def colony_contrast(rgb: np.ndarray) -> np.ndarray:
    """
    A single-channel image where colonies stand out from agar.

    Colonies differ from agar in **colour** more reliably than in brightness -
    a pale colony on a pale plate has almost no brightness contrast but a clear
    colour difference. So this works in Lab and combines:

    * the a* and b* channels, which carry the colour difference, and
    * the L* channel, which catches a dark melanised colony that happens to be
      colour-neutral.

    Each part is normalised to its own spread before combining, so neither
    dominates just because it has larger units.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    lab = color.rgb2lab(np.clip(rgb[..., :3], 0, 1))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Distance from the median colour of the image, which the agar dominates.
    chroma = np.hypot(a - np.median(a), b - np.median(b))
    lightness = np.abs(L - np.median(L))

    def unit(x):
        spread = np.percentile(x, 99) - np.percentile(x, 1)
        return x / spread if spread > 1e-9 else np.zeros_like(x)

    combined = unit(chroma) + 0.6 * unit(lightness)
    peak = combined.max()
    return combined / peak if peak > 0 else combined


# --------------------------------------------------------------------------
# Colony detection
# --------------------------------------------------------------------------


def detect_colonies(
    rgb: np.ndarray,
    *,
    min_area_frac: float = 2e-5,
    max_area_frac: float = 0.25,
    min_circularity: float = 0.25,
    max_results: int = 4096,
    restrict_to: np.ndarray | None = None,
) -> list[Detection]:
    """
    Find every colony-like blob in an image.

    The filters are deliberately loose. A missed colony costs the user manual
    work; an extra false blob is dropped harmlessly by the grid fit, which only
    keeps blobs that line up with others. Being generous here is the safer
    error.
    """
    small, _ = _downscale(rgb)
    contrast = colony_contrast(small)

    if restrict_to is not None:
        restricted, _ = _downscale(restrict_to.astype(np.float32))
        contrast = contrast * (restricted > 0.5)

    try:
        threshold = filters.threshold_otsu(contrast)
    except ValueError:  # pragma: no cover - flat image
        return []

    mask = contrast > threshold
    height, width = mask.shape
    total = float(height * width)

    # Clean up: close small gaps inside colonies, drop specks.
    mask = remove_small_holes(mask, int(total * 1e-4) + 1)
    mask = remove_small_objects(mask, max(4, int(total * min_area_frac)))

    detections: list[Detection] = []
    for region in measure.regionprops(measure.label(mask)):
        area_frac = region.area / total
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        perimeter = measure.perimeter_crofton(region.image, directions=4)
        circularity = (
            4 * math.pi * region.area / (perimeter**2) if perimeter > 0 else 0.0
        )
        if circularity < min_circularity:
            continue

        cy, cx = region.centroid
        min_row, min_col, max_row, max_col = region.bbox
        detections.append(
            Detection(
                cx=cx / width,
                cy=cy / height,
                bbox=(
                    min_col / width,
                    min_row / height,
                    max_col / width,
                    max_row / height,
                ),
                area_frac=area_frac,
                circularity=float(min(circularity, 1.5)),
                confidence=float(min(1.0, circularity / 0.85)),
            )
        )

    detections.sort(key=lambda d: -d.area_frac)
    return detections[:max_results]


# --------------------------------------------------------------------------
# Dish engine - modes 1 and 2
# --------------------------------------------------------------------------


def detect_dish(rgb: np.ndarray, *, margin: float = 0.02) -> DishResult:
    """
    Find the Petri dish rim, then the single colony inside it.

    The rim is found with a Hough circle transform over a range of plausible
    radii. If that fails - a square dish, a rim lost in shadow, a tightly
    cropped photo - the whole frame is used instead and a warning is recorded,
    rather than the image being rejected.
    """
    small, _ = _downscale(rgb)
    gray = color.rgb2gray(small[..., :3]) if small.ndim == 3 else small
    height, width = gray.shape
    warnings: list[str] = []

    centre = (0.5, 0.5)
    radius = 0.5 - margin
    found = False

    try:
        edges = filters.sobel(gray)
        edges = edges > np.percentile(edges, 97)
        radii = np.arange(int(0.25 * width), int(0.52 * width), max(2, width // 150))
        if radii.size:
            accumulator = transform.hough_circle(edges, radii)
            peaks = transform.hough_circle_peaks(
                accumulator, radii, total_num_peaks=1
            )
            if len(peaks[0]):
                _, cx, cy, r = (p[0] for p in peaks)
                centre = (float(cx) / width, float(cy) / height)
                radius = float(r) / width
                found = True
    except Exception as error:  # pragma: no cover - defensive
        warnings.append(f"Dish rim detection failed ({error}); using the whole image.")

    if not found:
        warnings.append("No dish rim found; using the whole image instead.")

    # Mask to just inside the rim, so the rim itself is not mistaken for a
    # colony. The 0.94 inset drops the rim's own shadow.
    yy, xx = np.mgrid[0:height, 0:width]
    inside = (
        (xx / width - centre[0]) ** 2 + (yy / height - centre[1]) ** 2
    ) <= (radius * 0.94) ** 2

    colonies = detect_colonies(small, restrict_to=inside, min_area_frac=1e-4)
    colony = colonies[0] if colonies else None
    if colony is None:
        warnings.append("No colony found inside the dish.")
    elif len(colonies) > 1:
        warnings.append(
            f"{len(colonies)} colony-like objects found; kept the largest. "
            "Check this dish."
        )

    return DishResult(
        centre=centre, radius=radius, found=found, colony=colony, warnings=warnings
    )


# --------------------------------------------------------------------------
# Grid engine - modes 3 and 4
# --------------------------------------------------------------------------


def _cluster_1d(values: np.ndarray, k: int) -> np.ndarray:
    """
    Sort ``values`` into ``k`` groups along one axis.

    Uses evenly spaced starting centres rather than random ones, then a few
    rounds of nearest-centre assignment. Plate rows and columns are close to
    evenly spaced by construction, so this converges immediately and - unlike
    random initialisation - gives the same answer every run.
    """
    if values.size == 0:
        return np.zeros(0, dtype=int)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.zeros(values.size, dtype=int)

    centres = np.linspace(lo, hi, k)
    labels = np.zeros(values.size, dtype=int)
    for _ in range(25):
        labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        moved = False
        for index in range(k):
            members = values[labels == index]
            if members.size:
                new_centre = float(members.mean())
                if abs(new_centre - centres[index]) > 1e-9:
                    centres[index] = new_centre
                    moved = True
        centres.sort()
        if not moved:
            break
    return labels


def fit_grid(
    rgb: np.ndarray,
    rows: int,
    cols: int,
    *,
    margin: float = 0.02,
    detections: list[Detection] | None = None,
) -> GridFit:
    """
    Detect the colonies on a plate and fit a grid to their positions.

    How it works
    ------------
    1. Detect every colony-like blob.
    2. Cluster their x-centres into ``cols`` groups and y-centres into ``rows``
       groups. Because a plate is laid out on a lattice, colonies in the same
       column really do share an x-position.
    3. Place a boundary line midway between neighbouring cluster centres, and
       extrapolate the two outer edges.

    Safety
    ------
    Never silently drops a well. If fewer colonies are found than the plate
    should hold, the count is reported and the fit falls back to a regular grid
    when the evidence is too thin to trust. ``confidence`` tells the interface
    which plates to flag for a human.
    """
    warnings: list[str] = []
    expected = rows * cols

    if detections is None:
        detections = detect_colonies(rgb)

    found = len(detections)
    regular_rows = np.linspace(margin, 1 - margin, rows + 1).tolist()
    regular_cols = np.linspace(margin, 1 - margin, cols + 1).tolist()

    # Below half the expected colonies there is not enough signal to fit a
    # lattice; a regular grid the user can drag is more honest than a fit to
    # noise.
    if found < max(4, expected * 0.5):
        warnings.append(
            f"Only {found} colonies found on a {rows}x{cols} plate "
            f"({expected} expected). Using an evenly spaced grid - please check it."
        )
        return GridFit(
            regular_rows, regular_cols, detections, expected, found, 0.2, warnings
        )

    xs = np.array([d.cx for d in detections])
    ys = np.array([d.cy for d in detections])

    def boundaries(values: np.ndarray, k: int, fallback: list[float]) -> tuple[list[float], float]:
        labels = _cluster_1d(values, k)
        centres = []
        occupied = 0
        for index in range(k):
            members = values[labels == index]
            if members.size:
                centres.append(float(members.mean()))
                occupied += 1
        if len(centres) < 2:
            return fallback, 0.0
        centres.sort()

        edges: list[float] = []
        gaps = np.diff(centres)
        typical = float(np.median(gaps))
        edges.append(max(0.0, centres[0] - typical / 2))
        for i in range(len(centres) - 1):
            edges.append(0.5 * (centres[i] + centres[i + 1]))
        edges.append(min(1.0, centres[-1] + typical / 2))

        # Even spacing is the sign of a good fit: a real plate lattice has
        # near-identical gaps, so high variability means the clustering latched
        # onto noise.
        regularity = 1.0 - min(1.0, float(np.std(gaps) / (typical + 1e-9)))
        return edges, regularity * (occupied / k)

    col_lines, col_quality = boundaries(xs, cols, regular_cols)
    row_lines, row_quality = boundaries(ys, rows, regular_rows)

    if len(col_lines) != cols + 1:
        col_lines = regular_cols
        warnings.append("Column fit failed; using even spacing for columns.")
    if len(row_lines) != rows + 1:
        row_lines = regular_rows
        warnings.append("Row fit failed; using even spacing for rows.")

    coverage = min(1.0, found / expected)
    confidence = float(min(col_quality, row_quality) * coverage)

    if found < expected:
        warnings.append(
            f"{expected - found} well(s) appear empty "
            f"({found} colonies found, {expected} wells). "
            "They are kept in the layout and will export as empty crops."
        )
    elif found > expected * 1.15:
        warnings.append(
            f"{found} blobs found for {expected} wells - some may be debris or "
            "split colonies."
        )

    return GridFit(
        row_lines, col_lines, detections, expected, found, confidence, warnings
    )


def centre_boxes_on_colonies(
    boxes,
    detections: list[Detection],
    *,
    padding: float = 0.20,
    max_shift: float = 0.5,
):
    """
    Move each box onto the colony nearest its centre, and size it to fit.

    ``padding`` grows the box beyond the colony's own bounding box, so the
    boundary ring and any faint halo are included in the crop rather than being
    cut off at the colony edge.

    ``max_shift`` is a guard: a box is only moved if a colony lies within that
    fraction of the box size. Without it, an empty well would steal its
    neighbour's colony and two wells would export the same image.

    Returns the number of boxes that were moved.
    """
    if not detections:
        return 0

    centres = np.array([[d.cx, d.cy] for d in detections])
    used: set[int] = set()
    moved = 0

    for index, (x1, y1, x2, y2) in enumerate(list(boxes.boxes)):
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        width, height = x2 - x1, y2 - y1
        limit = max_shift * max(width, height)

        distances = np.hypot(centres[:, 0] - cx, centres[:, 1] - cy)
        for candidate in np.argsort(distances):
            if candidate in used:
                continue
            if distances[candidate] > limit:
                break
            detection = detections[candidate]
            bx1, by1, bx2, by2 = detection.bbox
            pad_x = padding * (bx2 - bx1)
            pad_y = padding * (by2 - by1)
            boxes.set_box(
                index,
                (
                    max(0.0, bx1 - pad_x),
                    max(0.0, by1 - pad_y),
                    min(1.0, bx2 + pad_x),
                    min(1.0, by2 + pad_y),
                ),
            )
            used.add(int(candidate))
            moved += 1
            break
    return moved

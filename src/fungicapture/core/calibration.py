"""
Colour calibration: making colour numbers comparable between photographs.

The problem
-----------
A melanization score is only meaningful if two colonies photographed on
different days can be compared. Lamps age, white balance drifts, someone
changes the ISO. Over a months-long GWAS this is guaranteed to happen.

Three methods, which stack rather than compete
-----------------------------------------------
**C - EXIF capture and drift warning.** Always on. Corrects nothing, but
records the camera settings and tells you when two batches are not comparable.
Cheap, and knowing when to distrust a number is worth as much as correcting it.

**B - Agar background normalisation.** Always on. Finds bare agar, and scales
each channel so the agar reads the same everywhere. Needs no change to how you
photograph, and removes most lamp drift on its own.

**A - Colour reference card.** Optional, highest accuracy. A card in the corner
of the photograph is located, its patches are read, and a correction is fitted
that maps measured patch colours onto their true values.

A runs on top of B when a card is present.

No manufacturer's card is hard-coded. The user supplies a definition file
describing their own card - see ``CardDefinition``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skimage import color

from .params import CalibrationParams

# --------------------------------------------------------------------------
# Card definition - supplied by the user
# --------------------------------------------------------------------------


@dataclass
class CardDefinition:
    """
    A user-supplied colour reference card.

    FungiCapture ships no patch values of its own. You upload a definition of
    whatever card you own, which means any card works - a commercial
    ColorChecker, a calibrated print, or one you made and measured yourself.

    Accepted formats
    ----------------
    **JSON**::

        {
          "name": "My ColorChecker",
          "rows": 4,
          "cols": 6,
          "value_space": "lab",
          "patches": [
            {"name": "dark skin", "values": [37.99, 13.56, 14.06]},
            ...
          ]
        }

    **CSV** with a header row::

        name,L,a,b
        dark skin,37.99,13.56,14.06
        ...

    or with ``R,G,B`` columns and ``value_space`` inferred as sRGB 0-255.

    Patches must be listed in reading order: left to right, top to bottom.
    """

    name: str
    rows: int
    cols: int
    patch_names: list[str]
    reference_lab: np.ndarray
    """Shape ``(n_patches, 3)``, in CIE-Lab."""

    source_path: Path | None = None

    @property
    def n_patches(self) -> int:
        return len(self.patch_names)

    @classmethod
    def load(cls, path: str | Path) -> CardDefinition:
        """Read a card definition from JSON or CSV."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Card definition not found: {path}")

        if path.suffix.lower() == ".json":
            return cls._from_json(path)
        if path.suffix.lower() in (".csv", ".tsv", ".txt"):
            return cls._from_csv(path)
        raise ValueError(
            f"Unsupported card definition format: {path.suffix}. Use .json or .csv."
        )

    @classmethod
    def _from_json(cls, path: Path) -> CardDefinition:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        patches = data.get("patches") or []
        if not patches:
            raise ValueError(f"{path.name} lists no patches.")

        space = str(data.get("value_space", "lab")).lower()
        names = [str(p.get("name", f"patch{i}")) for i, p in enumerate(patches)]
        values = np.array([p["values"] for p in patches], dtype=float)

        rows = int(data.get("rows", 0)) or 1
        cols = int(data.get("cols", 0)) or len(patches)
        if rows * cols != len(patches):
            raise ValueError(
                f"{path.name}: rows x cols = {rows * cols} but {len(patches)} "
                "patches are listed. They must match."
            )

        return cls(
            name=str(data.get("name", path.stem)),
            rows=rows,
            cols=cols,
            patch_names=names,
            reference_lab=_to_lab(values, space),
            source_path=path,
        )

    @classmethod
    def _from_csv(cls, path: Path) -> CardDefinition:
        import csv as csv_module

        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows_read = list(csv_module.DictReader(handle))
        if not rows_read:
            raise ValueError(f"{path.name} has no data rows.")

        lowered = {k.lower().strip(): k for k in rows_read[0]}
        if all(c in lowered for c in ("l", "a", "b")):
            space, columns = "lab", ("l", "a", "b")
        elif all(c in lowered for c in ("r", "g", "b")):
            space, columns = "srgb255", ("r", "g", "b")
        else:
            raise ValueError(
                f"{path.name} needs either L,a,b or R,G,B columns. "
                f"Found: {', '.join(rows_read[0])}"
            )

        name_key = lowered.get("name") or lowered.get("patch")
        names = [
            (row.get(name_key) or f"patch{i}") if name_key else f"patch{i}"
            for i, row in enumerate(rows_read)
        ]
        values = np.array(
            [[float(row[lowered[c]]) for c in columns] for row in rows_read],
            dtype=float,
        )

        count = len(names)
        # Guess a sensible grid; the user can override in JSON if it matters.
        rows_n = 4 if count % 4 == 0 else (1 if count < 4 else 2)
        cols_n = count // rows_n

        return cls(
            name=path.stem,
            rows=rows_n,
            cols=cols_n,
            patch_names=names,
            reference_lab=_to_lab(values, space),
            source_path=path,
        )

    def describe(self) -> str:
        return f"{self.name}: {self.n_patches} patches, {self.rows}x{self.cols} grid"


def _to_lab(values: np.ndarray, space: str) -> np.ndarray:
    """Convert supplied reference values into Lab."""
    space = space.lower()
    if space in ("lab", "cielab", "l*a*b*"):
        return values.astype(float)
    if space in ("srgb255", "rgb255", "rgb"):
        rgb = np.clip(values / 255.0, 0, 1).reshape(-1, 1, 3)
        return color.rgb2lab(rgb).reshape(-1, 3)
    if space in ("srgb", "rgb01"):
        rgb = np.clip(values, 0, 1).reshape(-1, 1, 3)
        return color.rgb2lab(rgb).reshape(-1, 3)
    raise ValueError(f"Unknown value_space {space!r}. Use 'lab' or 'rgb'.")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """What calibration did to one image, recorded for the output file."""

    method: str
    """``card``, ``background`` or ``none``."""

    card_found: bool = False
    card_residual_dE: float = math.nan
    """Mean colour difference across patches after fitting. High means a bad
    read - glare, blur, or the card partly hidden."""

    background_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    exif: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_columns(self) -> dict[str, Any]:
        """The block written into every feature row."""
        return {
            "colour_calibration_method": self.method,
            "colour_card_found": bool(self.card_found),
            "colour_card_residual_dE": float(self.card_residual_dE),
            "colour_bg_gain_r": float(self.background_gain[0]),
            "colour_bg_gain_g": float(self.background_gain[1]),
            "colour_bg_gain_b": float(self.background_gain[2]),
        }


# --------------------------------------------------------------------------
# Method B - agar background normalisation
# --------------------------------------------------------------------------


def estimate_background(
    rgb: np.ndarray, percentile: float = 50.0
) -> tuple[float, float, float]:
    """
    The colour of bare agar in this image.

    Agar is the most common thing in a plate photograph, so the per-channel
    median over the brighter, low-saturation pixels lands on it. Restricting to
    low saturation avoids picking up a large pigmented colony, and the
    percentile makes it robust to a stray fragment.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    hsv = color.rgb2hsv(np.clip(rgb[..., :3], 0, 1))
    saturation, value = hsv[..., 1], hsv[..., 2]

    # Bare agar: not strongly coloured, not in deep shadow.
    candidate = (saturation < np.percentile(saturation, 60)) & (
        value > np.percentile(value, 40)
    )
    if candidate.sum() < 100:
        candidate = np.ones(saturation.shape, dtype=bool)

    return tuple(
        float(np.percentile(rgb[..., c][candidate], percentile)) for c in range(3)
    )


def normalise_background(
    rgb: np.ndarray,
    target: tuple[float, float, float] | None = None,
    percentile: float = 50.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Scale each channel so the agar reads the same as in a reference image.

    ``target`` defaults to the image's own mean background level, which makes
    the correction a pure white balance: it removes a colour cast without
    changing overall exposure. Pass a fixed target to lock a whole batch to one
    reference.
    """
    measured = estimate_background(rgb, percentile)
    if target is None:
        neutral = float(np.mean(measured))
        target = (neutral, neutral, neutral)

    gains = tuple(
        float(t / m) if m > 1e-6 else 1.0 for t, m in zip(target, measured)
    )

    work = np.asarray(rgb, dtype=np.float64)
    scale = 255.0 if work.max() > 1.0 else 1.0
    work = work / scale
    for channel in range(3):
        work[..., channel] *= gains[channel]
    corrected = np.clip(work, 0, 1) * scale
    return corrected.astype(rgb.dtype if scale == 1.0 else np.uint8), gains


# --------------------------------------------------------------------------
# Method A - colour reference card
# --------------------------------------------------------------------------


def find_card_patches(
    rgb: np.ndarray, card: CardDefinition, search_region: str = "auto"
) -> np.ndarray | None:
    """
    Locate the card and read its patch colours.

    Returns an array of shape ``(n_patches, 3)`` in linear sRGB 0-1, or None if
    the card could not be found.

    Detection strategy: the card is a rigid rectangular lattice of flat colour
    patches, which is visually unlike anything else on an agar plate. Candidate
    regions are searched for a quadrilateral whose interior divides cleanly
    into ``rows x cols`` low-variance cells.

    This is a deliberately conservative detector. It would rather report
    "not found" - and let the pipeline fall back to background normalisation
    with a warning - than fit a card to something that is not one and silently
    corrupt every colour value in the image.

    Known limitation
    ----------------
    Automatic detection needs the card to have a **visible frame or border**
    around the patches, which commercial cards do. A borderless printed chart
    has no outer edge to find, only a lattice of patch edges, and detection
    fails. Measured on synthetic plates: a framed card is found and reduces
    colony colour error from dE 12.8 to dE 1.0; a borderless one is not found
    and the pipeline falls back to background normalisation with a warning.

    For a borderless card, mark the four corners once in the Plates tab and use
    ``read_card_at_corners`` instead. The corners are stored in the project and
    reused for the whole batch.
    """
    import cv2

    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]

    region_slices = {
        "top-left": (slice(0, height // 2), slice(0, width // 2)),
        "top-right": (slice(0, height // 2), slice(width // 2, width)),
        "bottom-left": (slice(height // 2, height), slice(0, width // 2)),
        "bottom-right": (slice(height // 2, height), slice(width // 2, width)),
    }
    if search_region in region_slices:
        rs, cs = region_slices[search_region]
        offset = (cs.start, rs.start)
        search = image[rs, cs]
    else:
        offset = (0, 0)
        search = image

    gray = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    frame_area = search.shape[0] * search.shape[1]

    # Pass 1: a card with a printed frame - most commercial cards have one -
    # shows up directly as a single quadrilateral contour.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = _best_quad(contours, frame_area, card)

    # Pass 2: a borderless card has no outer contour, only a lattice of patch
    # edges. Closing with a kernel wider than one patch fuses that lattice into
    # a single solid block whose outline is the card.
    if best is None:
        patch_guess = int(
            math.sqrt(frame_area * 0.02 / max(1, card.rows * card.cols))
        )
        kernel_size = max(9, patch_guess | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        fused = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(
            fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = _best_quad(contours, frame_area, card, use_min_area_rect=True)

    if best is None:
        return None

    corners = best[1] + np.array(offset, dtype=np.float32)
    return _read_patches(image, corners, card)


def _best_quad(
    contours,
    frame_area: float,
    card: CardDefinition,
    *,
    use_min_area_rect: bool = False,
) -> tuple[float, np.ndarray] | None:
    """
    Pick the largest four-cornered contour whose shape matches the card.

    The aspect-ratio test is what stops a Petri dish, a label or a shadow being
    accepted as a card. Both orientations are allowed, since the card may be
    rotated 90 degrees in the photograph.
    """
    import cv2

    target_ratio = card.cols / card.rows
    best: tuple[float, np.ndarray] | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.002 or area > frame_area * 0.6:
            continue

        if use_min_area_rect:
            # A fused lattice has ragged edges, so fit the tightest rotated
            # rectangle rather than trusting the outline itself.
            rectangle = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rectangle).astype(np.float32)
            if cv2.contourArea(points) < 1e-6:
                continue
            # Reject a blob that fills its rectangle poorly - a real card is
            # solidly rectangular.
            if area / cv2.contourArea(points) < 0.7:
                continue
        else:
            approximation = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            points = approximation.reshape(4, 2).astype(np.float32)

        corners = _order_corners(points)
        edge_w = float(np.linalg.norm(corners[1] - corners[0]))
        edge_h = float(np.linalg.norm(corners[3] - corners[0]))
        if edge_h < 1e-6 or edge_w < 1e-6:
            continue

        ratio = edge_w / edge_h
        error = min(
            abs(ratio - target_ratio) / target_ratio,
            abs((1.0 / ratio) - target_ratio) / target_ratio,
        )
        if error > 0.25:
            continue
        if best is None or area > best[0]:
            best = (area, corners)
    return best


def read_card_at_corners(
    rgb: np.ndarray, corners, card: CardDefinition
) -> np.ndarray:
    """
    Read the card's patches from four corners the user marked by hand.

    The fallback for a borderless card, and the reliable path in awkward
    lighting. ``corners`` is four ``(x, y)`` pairs in **normalised** image
    coordinates, in any order - they are sorted here.

    Because the corners are normalised, a set marked once on one plate applies
    to every other plate photographed in the same rig.
    """
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]

    points = np.array(
        [[float(x) * width, float(y) * height] for x, y in corners], dtype=np.float32
    )
    if points.shape != (4, 2):
        raise ValueError("Exactly four corners are needed to read a card.")
    return _read_patches(image, _order_corners(points), card)


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Sort four corners into top-left, top-right, bottom-right, bottom-left."""
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(total)],
            points[np.argmin(difference)],
            points[np.argmax(total)],
            points[np.argmax(difference)],
        ],
        dtype=np.float32,
    )


def _read_patches(
    image: np.ndarray, corners: np.ndarray, card: CardDefinition
) -> np.ndarray:
    """Warp the card flat and take the median colour of each patch centre."""
    import cv2

    cell = 60
    target_w, target_h = card.cols * cell, card.rows * cell
    destination = np.array(
        [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    flat = cv2.warpPerspective(image, matrix, (target_w, target_h))

    values = []
    inset = cell // 4  # sample the middle half, avoiding patch borders
    for r in range(card.rows):
        for c in range(card.cols):
            patch = flat[
                r * cell + inset : (r + 1) * cell - inset,
                c * cell + inset : (c + 1) * cell - inset,
            ]
            values.append(np.median(patch.reshape(-1, 3), axis=0))
    return np.array(values, dtype=np.float64) / 255.0


def fit_colour_correction(
    measured_rgb: np.ndarray, card: CardDefinition
) -> tuple[np.ndarray, float]:
    """
    Fit a 3x4 correction matrix mapping measured colours onto the card's
    reference values.

    Least squares on an augmented ``[R, G, B, 1]`` vector: the 3x3 part handles
    channel gain and cross-talk, the constant column handles a black-level
    offset. This is the standard linear colour-correction model - powerful
    enough for lamp and white-balance drift, and simple enough that it cannot
    invent structure that is not there.

    Returns ``(matrix, residual_dE)`` where the residual is the mean CIE76
    colour difference across patches after correction.
    """
    reference_rgb = color.lab2rgb(card.reference_lab.reshape(-1, 1, 3)).reshape(-1, 3)

    n = min(len(measured_rgb), len(reference_rgb))
    measured = measured_rgb[:n]
    reference = reference_rgb[:n]

    augmented = np.hstack([measured, np.ones((n, 1))])
    matrix, *_ = np.linalg.lstsq(augmented, reference, rcond=None)

    corrected = np.clip(augmented @ matrix, 0, 1)
    corrected_lab = color.rgb2lab(corrected.reshape(-1, 1, 3)).reshape(-1, 3)
    residual = float(
        np.mean(np.linalg.norm(corrected_lab - card.reference_lab[:n], axis=1))
    )
    return matrix, residual


def apply_correction(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a fitted correction matrix to a whole image."""
    work = np.asarray(rgb, dtype=np.float64)
    scale = 255.0 if work.max() > 1.0 else 1.0
    flat = (work[..., :3] / scale).reshape(-1, 3)
    augmented = np.hstack([flat, np.ones((flat.shape[0], 1))])
    corrected = np.clip(augmented @ matrix, 0, 1).reshape(work[..., :3].shape)
    return (corrected * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Method C - EXIF
# --------------------------------------------------------------------------

EXIF_FIELDS = {
    271: "camera_make",
    272: "camera_model",
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso",
    41987: "white_balance",
    36867: "captured",
    37386: "focal_length",
}


def read_exif(path: str | Path) -> dict[str, Any]:
    """
    Camera settings for one photograph.

    Returns an empty dictionary for formats without EXIF, such as PNG. That is
    not an error - it just means drift cannot be checked for those files, which
    the drift report says explicitly.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            raw = getattr(handle, "_getexif", lambda: None)()
        if not raw:
            return {}
        return {
            name: str(raw[tag]) for tag, name in EXIF_FIELDS.items() if tag in raw
        }
    except Exception:
        return {}


def detect_exif_drift(records: list[dict[str, Any]]) -> list[str]:
    """
    Warn when camera settings changed inside one batch.

    Corrects nothing. It tells you which comparisons to distrust, which is the
    honest thing to do when the information to correct is not there.
    """
    warnings: list[str] = []
    usable = [r for r in records if r]
    if len(usable) < 2:
        if not usable:
            warnings.append(
                "No EXIF data found (PNG and TIFF often carry none), so camera "
                "settings could not be checked for drift."
            )
        return warnings

    for field_name, label in (
        ("white_balance", "white-balance setting"),
        ("iso", "ISO"),
        ("f_number", "aperture"),
        ("exposure_time", "shutter speed"),
        ("camera_model", "camera body"),
    ):
        values = [r.get(field_name) for r in usable if r.get(field_name)]
        distinct = sorted(set(values))
        if len(distinct) > 1:
            counts = {v: values.count(v) for v in distinct}
            smallest = min(counts.values())
            warnings.append(
                f"{len(distinct)} different {label} values in this batch "
                f"({', '.join(distinct[:4])}). {smallest} image(s) differ from the "
                "majority - colour comparisons across that boundary may be unreliable."
            )
    return warnings


# --------------------------------------------------------------------------
# The calibrator
# --------------------------------------------------------------------------


class Calibrator:
    """
    Applies the calibration chain to each image.

    Usage::

        calibrator = Calibrator(project.params.calibration)
        rgb, info = calibrator.apply(rgb, image_path)

    ``info`` becomes extra columns in the feature row, so any published number
    can be traced to how it was corrected.
    """

    def __init__(self, params: CalibrationParams):
        self.params = params
        self.card: CardDefinition | None = None
        self._matrix: np.ndarray | None = None
        self._exif_records: list[dict[str, Any]] = []
        self.load_errors: list[str] = []

        if params.use_colour_card and params.card_definition_path:
            try:
                self.card = CardDefinition.load(params.card_definition_path)
            except Exception as error:
                self.load_errors.append(
                    f"Could not read the colour card definition: {error}. "
                    "Falling back to background normalisation."
                )

    # ---------------- per image ----------------

    def apply(
        self, rgb: np.ndarray, image_path: str | Path | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Correct one image and report what was done."""
        result = CalibrationResult(method="none")
        result.warnings.extend(self.load_errors)
        working = rgb

        if self.params.capture_exif and image_path is not None:
            result.exif = read_exif(image_path)
            self._exif_records.append(result.exif)

        if self.params.background_normalise:
            working, gains = normalise_background(
                working, percentile=self.params.background_percentile
            )
            result.background_gain = gains
            result.method = "background"

        if self.card is not None:
            if self.params.card_corners:
                # Corners marked by hand win over searching: the user has told
                # us exactly where the card is, so do not second-guess them.
                try:
                    measured = read_card_at_corners(
                        working, self.params.card_corners, self.card
                    )
                except Exception as error:
                    measured = None
                    result.warnings.append(f"Could not read the marked card: {error}")
            else:
                measured = find_card_patches(
                    working, self.card, self.params.card_search_region
                )

            if measured is None:
                result.warnings.append(
                    "No colour card found in this image - values are "
                    "session-relative only. If your card has no printed frame, "
                    "mark its corners in the Plates tab."
                )
            else:
                matrix, residual = fit_colour_correction(measured, self.card)
                result.card_residual_dE = residual
                if residual > self.params.card_max_residual_dE:
                    result.warnings.append(
                        f"Colour card read poorly (residual dE {residual:.1f} > "
                        f"{self.params.card_max_residual_dE}). Correction not applied "
                        "- check for glare or blur on the card."
                    )
                else:
                    working = apply_correction(working, matrix)
                    self._matrix = matrix
                    result.card_found = True
                    result.method = "card"

        columns = result.as_columns()
        columns["colour_calibration_warnings"] = "; ".join(result.warnings)
        return working, columns

    # ---------------- batch ----------------

    def batch_warnings(self) -> list[str]:
        """Drift warnings across everything processed so far (method C)."""
        return detect_exif_drift(self._exif_records)

    def describe(self) -> str:
        parts = []
        if self.card is not None:
            parts.append(f"colour card ({self.card.describe()})")
        if self.params.background_normalise:
            parts.append("agar background normalisation")
        if self.params.capture_exif:
            parts.append("EXIF drift check")
        return ", ".join(parts) if parts else "no calibration"

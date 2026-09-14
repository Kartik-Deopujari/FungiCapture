"""
Regions: the colony interior and the adaptive boundary ring.

Every feature in FungiCapture is measured on one of two regions:

* **inside** - the colony mask itself
* **ring**   - a band just outside the colony edge

The ring is where the interesting biology often sits: the diffusing pigment
halo, the filamentous margin, and secreted material. Measuring it separately
from the interior is one of the things that makes this pipeline different from
colony-size tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, io, measure, morphology, util

from ..params import RingParams


@dataclass(frozen=True)
class Regions:
    """The two measurement regions for one colony, plus how the ring was made."""

    inside: np.ndarray
    """Boolean array. True where the colony is."""

    ring: np.ndarray
    """Boolean array. True in the band outside the colony."""

    ring_width_px: int
    """The width actually used, which depends on this colony's own size."""

    @property
    def shape(self) -> tuple[int, int]:
        return self.inside.shape  # type: ignore[return-value]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"inside": self.inside, "ring": self.ring}


def load_mask(
    path,
    *,
    target_shape: tuple[int, int] | None = None,
    largest_component_only: bool = True,
) -> np.ndarray:
    """
    Read a segmentation mask and return a clean boolean array.

    Parameters
    ----------
    path
        Mask image. Any non-zero pixel counts as colony.
    target_shape
        If given and the mask does not match, the mask is resized with
        nearest-neighbour interpolation. SAM writes masks at its own internal
        resolution and they can come back a few pixels off; resizing with any
        smoothing would blur the binary edge and change the measured area.
    largest_component_only
        Keep only the biggest connected blob. Segmentation often leaves small
        specks of noise elsewhere in the crop, and those would corrupt the
        perimeter and solidity of the real colony.
    """
    raw = read_image(path)
    if raw.ndim == 3:
        raw = color.rgb2gray(raw[..., :3])
    mask = np.asarray(raw) > 0

    if target_shape is not None and mask.shape != tuple(target_shape):
        from skimage.transform import resize

        mask = (
            resize(
                mask.astype(np.float32),
                target_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            > 0.5
        )

    if largest_component_only:
        mask = keep_largest_component(mask)
    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Drop everything except the biggest connected blob."""
    if not mask.any():
        return mask
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return mask
    biggest = max(props, key=lambda r: r.area)
    return labels == biggest.label


# Camera RAW extensions. These are *not* ordinary image files: the pixels are
# undemosaiced sensor data that OpenCV, Pillow and scikit-image cannot decode.
# When handed one, those libraries silently fall back to the small embedded
# preview thumbnail (often 160x120), which is why RAW plates used to export as
# unusably low-resolution crops. RAW files must go through ``rawpy`` instead.
RAW_SUFFIXES = {
    ".nef", ".nrw",           # Nikon
    ".cr2", ".cr3", ".crw",   # Canon
    ".arw", ".sr2", ".srf",   # Sony
    ".raf",                    # Fujifilm
    ".rw2",                    # Panasonic
    ".orf",                    # Olympus
    ".pef",                    # Pentax
    ".srw",                    # Samsung
    ".dng",                    # Adobe / generic
    ".raw", ".rwl", ".dcr", ".kdc", ".mrw", ".x3f",
}


def _read_raw(path: str) -> np.ndarray:
    """Decode a camera RAW file at full resolution with ``rawpy``.

    Two routes are tried, best first:

    * ``postprocess`` demosaics the sensor data into a full-resolution RGB
      image - the highest quality result.
    * if that fails (a truncated or partially unsupported file), the embedded
      preview is used instead. On modern cameras this preview is a
      full-resolution JPEG, so it is still vastly better than the tiny
      thumbnail the generic decoders would otherwise return.

    Raising here (rather than returning a thumbnail) lets ``read_image`` report
    a real error instead of silently producing a low-resolution crop.
    """
    import rawpy

    postprocess_error: Exception | None = None
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            return np.ascontiguousarray(rgb)
    except Exception as error:
        # Full sensor decode failed (a truncated or partially unsupported
        # file). Fall back to the embedded preview - but from a *fresh* handle:
        # once ``postprocess`` fails, LibRaw leaves the handle in an error state
        # in which ``extract_thumb`` also fails, so the preview has to be read
        # from a new ``rawpy.imread``.
        postprocess_error = error

    with rawpy.imread(path) as raw:
        thumb = raw.extract_thumb()

    if thumb.format == rawpy.ThumbFormat.JPEG:
        import io as _io

        from PIL import Image

        with Image.open(_io.BytesIO(thumb.data)) as handle:
            preview = np.asarray(handle.convert("RGB"))
    else:
        # Some cameras store an uncompressed bitmap preview instead of a JPEG.
        preview = np.asarray(thumb.data)

    # Only accept the preview if it is genuinely a full-size image and not the
    # tiny 160x120 thumbnail - otherwise raising is better than silently
    # exporting low-resolution crops.
    if preview.ndim >= 2 and min(preview.shape[0], preview.shape[1]) >= 1000:
        return np.ascontiguousarray(preview)

    raise OSError(
        "the RAW sensor data could not be decoded"
        + (f" ({postprocess_error})" if postprocess_error else "")
        + " and no full-resolution embedded preview was available"
    )


def read_image(path) -> np.ndarray:
    """
    Read an image, trying more than one library.

    Camera RAW files (``.nef``, ``.cr2``, ``.dng`` and friends) are decoded
    with ``rawpy`` at full resolution. This must happen first: the generic
    decoders below "succeed" on a RAW file by returning its small embedded
    preview thumbnail, so without this branch a RAW plate would be cropped from
    a 160x120 image and every colony would be a blur.

    scikit-image is preferred for ordinary formats because it handles the
    widest range of scientific formats, but it delegates TIFF to ``tifffile``,
    which cannot decompress LZW or JPEG-in-TIFF without the optional
    ``imagecodecs`` package. Those are exactly what scanner and microscope
    software writes, so a plate photograph saved as a compressed TIFF would
    fail to open with a message about a missing codec - which tells a biologist
    nothing useful.

    OpenCV and Pillow decode those without extra packages, so they are tried in
    turn. Only if all decoders fail is an error raised, and it then names every
    reason rather than just the last one.
    """
    path = str(path)
    problems: list[str] = []

    from pathlib import Path as _Path

    if _Path(path).suffix.lower() in RAW_SUFFIXES:
        try:
            return _read_raw(path)
        except Exception as error:
            # Fall through to the generic decoders so a RAW file with an
            # unusual container still opens *somehow* - but record why the
            # proper full-resolution path did not work.
            problems.append(f"rawpy: {error}")

    try:
        return io.imread(path)
    except Exception as error:
        problems.append(f"scikit-image: {error}")

    try:
        import cv2

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 3 and image.shape[2] >= 3:
                # OpenCV reads as BGR; everything else here expects RGB.
                image = image[..., :3][..., ::-1]
            return np.ascontiguousarray(image)
        problems.append("OpenCV: could not decode the file")
    except Exception as error:
        problems.append(f"OpenCV: {error}")

    try:
        from PIL import Image

        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"))
    except Exception as error:
        problems.append(f"Pillow: {error}")

    raise OSError(
        f"Could not read the image {path}\n"
        + "\n".join(f"  {p}" for p in problems)
        + "\n\nIf this is a compressed TIFF, installing the 'imagecodecs' "
        "package usually fixes it. If this is a camera RAW file, installing "
        "the 'rawpy' package is required."
    )


def load_gray(path) -> np.ndarray:
    """
    Read an image as 8-bit grayscale, matching the original pipeline exactly.

    Note for the paper: ``rgb2gray`` uses ITU-R BT.709 luminance weights
    (0.2125 R, 0.7154 G, 0.0721 B), so green dominates the grey value. For a
    melanin-pigmented fungus that is not a neutral choice, and it is one reason
    the grayscale-only pipeline missed melanization. Colour features are
    computed from the colour image instead - see ``features.colour``.
    """
    img = read_image(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(np.float32)
        if gray.max() > 1.0:
            if np.issubdtype(img.dtype, np.integer):
                gray = gray / np.iinfo(img.dtype).max
            else:
                gray = gray / gray.max()
    return util.img_as_ubyte(gray)


def load_rgb(path) -> np.ndarray:
    """Read an image as 8-bit RGB, for the colour features."""
    img = read_image(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = (color.rgba2rgb(img) * 255).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] > 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        peak = float(img.max())
        img = (
            (img.astype(np.float64) / peak * 255).astype(np.uint8)
            if peak > 0
            else img.astype(np.uint8)
        )
    return img


def build_regions(mask: np.ndarray, params: RingParams) -> Regions:
    """
    Build the interior and ring regions for one colony.

    The ring width is derived from this colony's own area, so a small colony
    gets a thin ring and a large one a thick ring. Both then sample the same
    *relative* depth into the halo, which makes colonies of different sizes
    comparable.
    """
    mask = np.asarray(mask, dtype=bool)
    width = params.width_for_area(float(mask.sum()))
    dilated = _dilate(mask, morphology.disk(width))
    ring = np.logical_and(dilated, ~mask)
    return Regions(inside=mask, ring=ring, ring_width_px=width)


def _dilate(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Binary dilation that works across scikit-image versions.

    ``binary_dilation`` is deprecated from 0.26 in favour of ``dilation``. The
    disc footprint is symmetric, so the mirroring difference noted in the
    deprecation warning does not change the result here.
    """
    try:
        return np.asarray(morphology.dilation(mask, footprint), dtype=bool)
    except TypeError:  # pragma: no cover - older signature
        return np.asarray(morphology.binary_dilation(mask, footprint), dtype=bool)


def bounding_patch(
    array: np.ndarray,
    region: np.ndarray,
    fill: float | None = None,
) -> np.ndarray | None:
    """
    Crop ``array`` to the bounding box of ``region`` and blank everything
    outside the region.

    ``fill`` defaults to the median of the in-region values. Filling with zero
    instead would create a hard black edge at the mask boundary, and a texture
    measure would then report that artificial edge as real colony texture.
    """
    if not region.any():
        return None
    rows, cols = np.where(region)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    patch = array[r0:r1, c0:c1].copy()
    patch_mask = region[r0:r1, c0:c1]
    if fill is None:
        fill = np.median(patch[patch_mask]) if patch_mask.any() else 0
    return np.where(patch_mask, patch, fill)


# --------------------------------------------------------------------------
# scikit-image compatibility
# --------------------------------------------------------------------------


def remove_small_holes(mask: np.ndarray, size: int) -> np.ndarray:
    """Fill holes smaller than ``size``, across scikit-image versions.

    The keyword was renamed from ``area_threshold`` to ``max_size`` in 0.26.
    Calling the old name still works but prints a deprecation warning for every
    image, which in a batch of 96 colonies buries the messages that matter.
    """
    try:
        return morphology.remove_small_holes(mask, max_size=size)
    except TypeError:  # pragma: no cover - scikit-image < 0.26
        return morphology.remove_small_holes(mask, area_threshold=size)


def remove_small_objects(mask: np.ndarray, size: int) -> np.ndarray:
    """Drop objects smaller than ``size``, across scikit-image versions.

    ``min_size`` became ``max_size`` in 0.26 - see ``remove_small_holes``.
    """
    try:
        return morphology.remove_small_objects(mask, max_size=size)
    except TypeError:  # pragma: no cover - scikit-image < 0.26
        return morphology.remove_small_objects(mask, min_size=size)

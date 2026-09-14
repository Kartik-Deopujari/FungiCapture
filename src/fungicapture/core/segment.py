"""
Segmentation: finding the colony outline.

Two backends
------------
**SAM 3** - the real one. Text-promptable, so the colony core and the faint
filamentous margin are described in words rather than tuned by thresholds.
Needs the model weights, which are downloaded on first run.

**Classical** - a fallback using colour contrast and Otsu thresholding. Not as
good on faint filamentous edges, but it needs no model, no GPU and no
download. It exists so the application is usable the moment it is installed,
so the pipeline can be tested without gigabytes of weights, and so a user
without the weights is never left with a dead tool.

The backend used is recorded in every output, because the two do not produce
identical masks and results from them must not be pooled blindly.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .params import SegmentationParams

# Where the SAM 3 weights are hosted. Meta gates access behind a licence
# acceptance, so the application must send the user here rather than mirroring
# the file. See docs/LICENSING.md.
SAM3_MODEL_PAGE = "https://huggingface.co/facebook/sam3"
SAM3_LICENCE_NOTE = (
    "The SAM 3 weights are released by Meta under the SAM License, which is a "
    "restricted research licence, not an open-source one. You must request "
    "access on Meta's model page and accept the licence yourself. FungiCapture "
    "does not redistribute the weights.\n\n"
    "If you publish results made with SAM 3, the licence requires you to "
    "acknowledge it in your methods."
)


# --------------------------------------------------------------------------
# Model storage
# --------------------------------------------------------------------------


def model_cache_dir() -> Path:
    """Where downloaded weights live, per user, outside the project."""
    from platformdirs import user_cache_dir

    path = Path(user_cache_dir("FungiCapture", "FungiCapture")) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_checksum(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, recorded in every result for reproducibility.

    Two runs with different model checksums used different models, whatever the
    filenames say.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _weight_search_dirs() -> list[Path]:
    """
    Every folder ``find_model`` looks in, in order of preference.

    Kept as its own function so the search set is testable and so the exact
    list can be shown to the user when a weights file cannot be found.

    The launcher does ``cd`` into the FungiCapture folder before starting, so
    the current working directory is normally the project root - which is
    exactly where a user drops a downloaded ``sam3*.pt``. That folder was being
    missed before: the old code checked the working directory for the exact
    name ``sam3.pt`` but only *globbed* the cache directory, so a file named
    ``sam3.1_multiplex.pt`` sitting next to the launcher was never found and
    SAM 3 silently fell back to the classical segmenter.
    """
    dirs: list[Path] = [model_cache_dir(), Path.cwd()]

    # Two levels above the installed package is the project root in a source
    # checkout (``src/fungicapture/core/segment.py`` -> project root), another
    # common place for a downloaded weights file.
    try:
        dirs.append(Path(__file__).resolve().parents[3])
    except IndexError:
        pass

    # A folder set by the user or a cluster admin, e.g. a shared model store.
    env_dir = os.environ.get("FUNGICAPTURE_MODEL_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    return unique


def find_model(name: str = "sam3.pt") -> Path | None:
    """
    Look for the weights, in order of preference.

    1. ``FUNGICAPTURE_SAM3_WEIGHTS`` - a full path to one file. Lets a cluster
       point every run at one shared copy.
    2. The exact canonical name ``sam3.pt`` in any search directory.
    3. Any ``sam3*.pt`` in any search directory - which is how a fine-tuned
       variant such as ``sam3.1_multiplex.pt`` is picked up. The plain
       ``sam3.pt`` wins if both are present; otherwise the first by name.

    The directories searched are given by :func:`_weight_search_dirs`.
    """
    override = os.environ.get("FUNGICAPTURE_SAM3_WEIGHTS")
    if override and Path(override).expanduser().exists():
        return Path(override).expanduser()

    dirs = _weight_search_dirs()

    for directory in dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    for directory in dirs:
        try:
            matches.extend(sorted(directory.glob("sam3*.pt")))
        except OSError:
            continue
    if matches:
        # Prefer the canonical name, then a stable alphabetical order so the
        # choice is reproducible across runs.
        matches.sort(key=lambda p: (p.name != "sam3.pt", p.name))
        return matches[0]
    return None


def install_model(source: str | Path, name: str | None = None) -> Path:
    """
    Copy a weights file the user downloaded into the cache.

    Deliberately a copy of a local file rather than a download from a URL we
    control: the user must obtain the file from Meta's gated page themselves,
    which is what the licence requires.
    """
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"No weights file at {source}")
    destination = model_cache_dir() / (name or source.name)
    if source.resolve() != destination.resolve():
        import shutil

        shutil.copy2(source, destination)
    return destination


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------


def resolve_device(preference: str = "auto") -> str:
    """
    Pick a compute device: CUDA, then Apple Silicon (MPS), then CPU.

    The original script assumed CUDA and hard-coded FP16, which fails outright
    on a Mac or a laptop without a GPU. Detecting instead is what makes the
    application cross-platform.
    """
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_half_precision(preference: str, device: str) -> bool:
    """
    FP16 only on CUDA.

    Half precision is unsupported on CPU and unreliable on some Apple Silicon
    paths. The original's hard-coded ``half=True`` is exactly why that script
    was not portable.
    """
    if preference == "auto":
        return device == "cuda"
    return bool(preference) and preference not in ("false", "False", "0")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """One colony's mask, plus how it was produced."""

    mask: np.ndarray
    backend: str
    group_masks: dict[str, np.ndarray] = field(default_factory=dict)
    """Per prompt-group masks, kept for debugging prompt behaviour."""
    warnings: list[str] = field(default_factory=list)
    model_checksum: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.mask.any()

    @property
    def area_fraction(self) -> float:
        return float(self.mask.mean())


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class ClassicalSegmenter:
    """
    Colour-contrast segmentation with no model.

    Works on the same principle as the colony detector: a colony differs from
    agar in colour more reliably than in brightness. Good enough for a compact
    colony with a clear edge; weaker on the faint hyphal fringe, which is
    precisely what SAM 3 was brought in for.
    """

    name = "classical"

    def __init__(self, params: SegmentationParams):
        self.params = params

    def segment(self, rgb: np.ndarray) -> SegmentationResult:
        from skimage import filters

        from .detect import colony_contrast
        from .features.regions import (
            keep_largest_component,
            remove_small_holes,
            remove_small_objects,
        )

        warnings: list[str] = []
        contrast = colony_contrast(rgb)

        try:
            threshold = filters.threshold_otsu(contrast)
        except ValueError:
            return SegmentationResult(
                np.zeros(contrast.shape, bool),
                self.name,
                warnings=["Image has no contrast - nothing to segment."],
            )

        mask = contrast > threshold
        total = mask.size
        mask = remove_small_holes(mask, max(16, total // 400))
        mask = remove_small_objects(mask, max(16, total // 2000))

        if self.params.use_largest_component:
            mask = keep_largest_component(mask)

        if not mask.any():
            warnings.append("No colony found by the classical segmenter.")
        elif mask.mean() > 0.9:
            warnings.append(
                "The mask covers almost the whole crop - the colony may be "
                "touching the edges, or the background may be very similar in "
                "colour to the colony."
            )
        return SegmentationResult(mask, self.name, warnings=warnings)


class Sam3Segmenter:
    """
    Text-promptable segmentation with SAM 3.

    Prompt groups are run separately and merged with a logical OR. That favours
    recall: a doubtful edge pixel is included rather than a real hyphal fringe
    being lost. For a filamentous fungus that is the right trade - a missing
    fringe is a missing phenotype, while a few extra halo pixels shift a mean
    slightly.
    """

    name = "sam3"

    def __init__(self, params: SegmentationParams, weights: str | Path | None = None):
        self.params = params
        self.weights = Path(weights) if weights else find_model()
        if self.weights is None:
            raise ModelNotAvailable(
                "SAM 3 weights not found.\n\n" + SAM3_LICENCE_NOTE
            )
        self.device = resolve_device(params.device)
        self.half = resolve_half_precision(params.half_precision, self.device)
        self._predictor = None
        self._checksum = ""

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as error:
            raise ModelNotAvailable(
                "SAM 3 needs the 'ultralytics' package, version 8.3.237 or "
                "newer - that is the release where SAM 3 was added, so an older "
                "8.3.x installs but cannot load it.\n"
                "Install it with:  pip install 'fungicapture[model]'\n"
                "or use the 'Segmentation setup...' button, which fetches the "
                "right version for you.\n\n"
                "Note: ultralytics is AGPL-3.0, which is why FungiCapture is "
                "AGPL-3.0. See docs/LICENSING.md."
            ) from error

        self._predictor = SAM3SemanticPredictor(
            overrides=dict(
                conf=self.params.confidence,
                task="segment",
                mode="predict",
                model=str(self.weights),
                half=self.half,
                device=self.device,
                save=False,
                verbose=False,
            )
        )
        self._checksum = file_checksum(self.weights)[:16]
        return self._predictor

    def segment(self, image_path: str | Path) -> SegmentationResult:
        from .features.regions import keep_largest_component

        predictor = self._load()
        warnings: list[str] = []
        predictor.set_image(str(image_path))

        group_masks: dict[str, np.ndarray] = {}
        try:
            for group, prompts in self.params.prompt_groups.items():
                try:
                    results = predictor(text=list(prompts))
                    combined = _merge_result_masks(results)
                except Exception as error:
                    warnings.append(f"Prompt group '{group}' failed: {error}")
                    combined = None
                if combined is None:
                    warnings.append(f"Prompt group '{group}' returned no mask.")
                    continue
                group_masks[group] = combined
        finally:
            try:
                predictor.reset_image()
            except Exception:
                pass

        if not group_masks:
            from skimage import io

            shape = io.imread(image_path).shape[:2]
            return SegmentationResult(
                np.zeros(shape, bool),
                self.name,
                warnings=warnings + ["SAM 3 returned no masks for this image."],
                model_checksum=self._checksum,
            )

        # Logical OR across groups - the union of core and filaments.
        mask = np.zeros_like(next(iter(group_masks.values())), dtype=bool)
        for group_mask in group_masks.values():
            mask |= group_mask.astype(bool)

        if self.params.use_largest_component:
            mask = keep_largest_component(mask)

        return SegmentationResult(
            mask, self.name, group_masks, warnings, self._checksum
        )


class ModelNotAvailable(RuntimeError):
    """Raised when SAM 3 cannot be used, with an explanation of what to do."""


def _merge_result_masks(results) -> np.ndarray | None:
    """Union every mask in an Ultralytics result list."""
    if results is None or len(results) == 0:
        return None
    combined: np.ndarray | None = None
    for result in results:
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or masks.data.shape[0] == 0:
            continue
        array = masks.data.cpu().numpy()
        union = array.sum(axis=0) > 0
        combined = union if combined is None else (combined | union)
    return combined


# --------------------------------------------------------------------------
# Front door
# --------------------------------------------------------------------------


def make_segmenter(
    params: SegmentationParams,
    *,
    backend: str = "auto",
    weights: str | Path | None = None,
) -> tuple[Any, list[str]]:
    """
    Build the best available segmenter.

    ``backend`` may be ``"auto"``, ``"sam3"`` or ``"classical"``. With
    ``"auto"``, SAM 3 is used when the weights and the package are present, and
    the classical backend otherwise - with a message saying so, never silently.

    Returns ``(segmenter, messages)``.
    """
    messages: list[str] = []

    if backend in ("auto", "sam3"):
        try:
            segmenter = Sam3Segmenter(params, weights)
            messages.append(
                f"Using SAM 3 on {segmenter.device}"
                f"{' with FP16' if segmenter.half else ''}."
            )
            return segmenter, messages
        except ModelNotAvailable as error:
            if backend == "sam3":
                raise
            messages.append(str(error))
            messages.append(
                "Falling back to the classical segmenter. It handles compact "
                "colonies well but is weaker on faint filamentous margins."
            )

    return ClassicalSegmenter(params), messages


def segment_image(
    segmenter, image_path: str | Path, rgb: np.ndarray | None = None
) -> SegmentationResult:
    """Run whichever backend was built, with the input each one needs."""
    if isinstance(segmenter, Sam3Segmenter):
        return segmenter.segment(image_path)
    if rgb is None:
        from .features.regions import load_rgb

        rgb = load_rgb(image_path)
    return segmenter.segment(rgb)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Write a boolean mask as an 8-bit PNG of 0 and 255."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", (np.asarray(mask, bool).astype(np.uint8) * 255))
    if not ok:
        raise OSError(f"Could not encode mask for {path}")
    # tofile handles non-ASCII paths on Windows, which cv2.imwrite does not.
    buffer.tofile(str(path))
    return path


def save_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    colour: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.35,
) -> Path:
    """Write a quick-look overlay: tinted mask plus a contour line."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR)
    tint = np.zeros_like(bgr)
    tint[np.asarray(mask, bool)] = colour
    blended = cv2.addWeighted(bgr, 1.0, tint, alpha, 0)

    contours, _ = cv2.findContours(
        (np.asarray(mask, bool).astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 1)

    ok, buffer = cv2.imencode(".png", blended)
    if not ok:
        raise OSError(f"Could not encode overlay for {path}")
    buffer.tofile(str(path))
    return path

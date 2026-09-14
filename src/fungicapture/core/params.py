"""
Analysis parameters, defined in exactly one place.

Why this module exists
----------------------
In the original scripts, ``RING_FRACTION`` was written at the top of
``vectorzer_2.py`` AND again at the top of ``overlay_validator.py``. If the two
values drift apart, the QC picture stops describing the numbers it is printed
next to. That is a silent scientific error: the validator says the ring is one
width while the feature values were computed with another.

Every parameter that affects a measured number lives here, in one frozen
dataclass. The vectorizer and the validator are handed the *same instance*.
They cannot disagree.

All parameter sets are serialised into every output file, so any result can be
traced back to the exact settings that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

# --------------------------------------------------------------------------
# Region parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RingParams:
    """
    The adaptive boundary ring: a band just outside the colony whose width
    scales with the colony's own size.

    The band captures the diffusing pigment halo and the filamentous margin,
    which carry different biology from the colony interior.

    Geometry
    --------
    ``r_eq``      = sqrt(area / pi), the radius of a circle of equal area
    ``width_px``  = clamp(round(fraction * r_eq), min_px, max_px)
    ``ring``      = dilate(mask, disk(width_px)) AND NOT mask

    Scaling the width to the colony means a small colony gets a thin ring and a
    large colony a thick one, so the ring is always the same *relative* depth
    into the halo. A fixed pixel width would over-sample small colonies and
    under-sample large ones.
    """

    fraction: float = 0.10
    """Ring width as a fraction of the colony's equivalent radius."""

    min_width_px: int = 3
    """Floor. Below about 3 px a dilation band is mostly aliasing artefacts."""

    max_width_px: int = 30
    """Ceiling. Stops a very large colony from sampling half the plate."""

    def width_for_area(self, area_px: float) -> int:
        """Ring width in pixels for a colony of the given area."""
        import math

        if area_px <= 0:
            return self.min_width_px
        r_eq = math.sqrt(area_px / math.pi)
        w = int(round(self.fraction * r_eq))
        return max(self.min_width_px, min(w, self.max_width_px))


# --------------------------------------------------------------------------
# Texture parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MorphologyParams:
    """
    Shape measurement settings.

    The only real choice here is how to estimate perimeter, and it matters more
    than it looks. Perimeter feeds circularity and roughness, which are two of
    the headline phenotypes.
    """

    perimeter_method: str = "crofton"
    """How to estimate the boundary length of a pixelated shape.

    ``crofton`` (default)
        ``skimage.measure.perimeter_crofton`` with 4 directions. Uses
        integral-geometry line intersections. On a synthetic disc of radius 60
        it gives circularity 1.010 against a true value of 1.0 - an error under
        1 percent.

    ``legacy_4``
        ``skimage.measure.perimeter(neighborhood=4)``. Gives circularity 0.921
        on the same disc, an 8 percent underestimate.

    ``legacy_8``
        ``skimage.measure.perimeter(neighborhood=8)``. **This is what the
        original vectorzer_2.py used, and it is badly biased**: circularity
        0.636 on a perfect circle, a 36 percent underestimate.

    Why the old default was a problem
    ---------------------------------
    A pixelated boundary is always longer than the smooth shape it represents,
    because it travels in stair-steps. The 8-connected estimator counts those
    steps generously.

    If the resulting bias were constant it would only rescale the axis and do
    no harm. It is not constant. Measured as ``legacy_8 / crofton``:

    ===========================  ======
    shape                        ratio
    ===========================  ======
    axis-aligned square           1.05
    smooth disc                   1.26
    wobbly disc                   1.26
    diamond (45 degree edges)     1.47
    disc with thin spikes         0.95
    ===========================  ======

    The driver is the **orientation and thinness** of the boundary, not how
    ragged it is - wobbly discs sit at the same 1.26 as smooth ones. Diagonal
    edges are penalised most, and thin filaments reverse the sign so
    ``legacy_8`` under-estimates instead.

    That matters here specifically: a compact colony and a filamentous one are
    biased in *opposite directions*, and telling those two apart is the
    comparison this pipeline exists to make.

    Honest caveat
    -------------
    Crofton with 4 directions slightly under-estimates straight axis-aligned
    edges, so on a perfect square ``legacy_8`` happens to be closer to the
    truth. Colonies are curved and irregular, not rectangular, so Crofton is
    still the right default - but the claim is "better for colony-shaped
    objects", not "better always". Both tests live in
    ``tests/test_regression_vs_original.py``.

    ``legacy_8`` is kept only so the Phase 0 regression test can prove the new
    code reproduces the old numbers. Do not use it for new analysis, and do not
    pool results computed under different settings - the ``params_hash`` in
    every output file exists to catch that.
    """


@dataclass(frozen=True)
class GLCMParams:
    """
    Gray-Level Co-occurrence Matrix settings (Haralick et al. 1973).

    The GLCM counts how often a pixel of value i sits at distance d and angle
    theta from a pixel of value j. Texture properties are then read off that
    matrix.
    """

    distances: tuple[int, ...] = (1, 2, 4)
    """Pixel separations. 1 catches fine texture, 4 catches coarser structure."""

    angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    """Directions. Averaging over four angles makes the result near
    rotation-invariant, which matters because a colony has no fixed
    orientation on the plate."""

    levels: int = 32
    """Grey levels after quantisation. 256 levels would make a 256x256 matrix
    that is mostly zeros and dominated by noise; 32 is the usual compromise."""

    properties: tuple[str, ...] = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    )
    """Properties passed to ``skimage.feature.graycoprops``.

    Note: ``dissimilarity`` is not one of Haralick's original 14. It comes from
    the later remote-sensing literature. Do not attribute it to Haralick."""

    @property
    def angles_rad(self) -> tuple[float, ...]:
        import math

        return tuple(math.radians(a) for a in self.angles_deg)

    @property
    def n_matrices(self) -> int:
        """How many co-occurrence matrices are averaged over."""
        return len(self.distances) * len(self.angles_deg)


@dataclass(frozen=True)
class LBPParams:
    """
    Local Binary Pattern settings (Ojala et al. 2002).

    Each pixel is compared with P neighbours on a circle of radius R. The
    resulting bit pattern is a texture code. The 'uniform' method keeps only
    patterns with at most two 0->1 or 1->0 transitions as separate bins and
    collapses the rest into one, which cuts the histogram size sharply while
    keeping the informative edge and corner patterns.
    """

    n_points: int = 16
    """Neighbours on the circle."""

    radius: int = 2
    """Circle radius in pixels. R=2 covers about a 4 px neighbourhood, which
    suits hyphal widths in Aureobasidium colonies."""

    method: str = "uniform"

    @property
    def n_bins(self) -> int:
        """Uniform LBP yields P + 2 bins."""
        return self.n_points + 2


# --------------------------------------------------------------------------
# Colour parameters (new in FungiCapture)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourParams:
    """
    Colour and melanization settings.

    The original pipeline threw colour away in ``load_gray_image``. For a
    melanin-producing fungus that discards the phenotype of interest, so every
    colour feature here is computed on the original colour crop.
    """

    spaces: tuple[str, ...] = ("rgb", "hsv", "lab")
    """Colour spaces to compute statistics in.

    rgb - raw, matches what you see.
    hsv - separates which colour from how strong and how bright, so it is more
          robust to lighting change than raw RGB.
    lab - perceptually uniform; the space used in food and materials science
          for pigment work, so a reviewer will accept it."""

    lab_illuminant: str = "D65"
    """Reference white. scikit-image defaults to D65 with the 2 degree standard
    observer. L* values are NOT comparable across software that uses a
    different white, so this is recorded in every output file."""

    lab_observer: str = "2"

    melanin_dark_L: float = 35.0
    """L* below this counts as 'dark' for MI_darkfraction. A project setting,
    written into every result, never hard-coded at the call site."""

    histogram_bins: int = 16
    """Bins per channel for the colour histograms."""

    histogram_channels: tuple[str, ...] = ("H", "S", "V", "L", "a", "b")

    dominant_k: int = 3
    """Clusters for dominant-colour extraction. A mean hides a colony that is
    half dark and half pale; three clusters do not."""

    dominant_max_pixels: int = 20_000
    """Subsample cap before k-means, so a huge colony does not stall the run.
    Sampling is deterministic (see ``random_state``)."""

    zonation_n_rings: int = 5
    """Concentric bands from centre to edge, for radial colour zonation."""

    sector_n_sectors: int = 12
    """Angular sectors on the outermost band, for detecting uneven growth."""

    random_state: int = 42
    """Fixed so k-means and subsampling give the same answer on every run."""


# --------------------------------------------------------------------------
# Colour calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationParams:
    """
    Colour calibration settings.

    Three methods, which stack rather than compete:

    C  EXIF capture and drift warning  - always on, corrects nothing, but tells
       you when two batches are not comparable.
    B  Agar background normalisation   - always on, needs no change to your
       photography, removes most lamp drift.
    A  Colour reference card           - optional, highest accuracy. The user
       supplies a card definition file; see ``card_definition_path``.

    A runs on top of B when a card is present.
    """

    use_colour_card: bool = False
    """Set by the new-project prompt: 'Process images with colour reference
    card'."""

    card_definition_path: str | None = None
    """Path to the user-supplied colour reference card document.

    FungiCapture does not hard-code any manufacturer's patch values. The user
    uploads a definition file describing their own card - patch names, their
    reference values, and the grid layout. See
    ``fungicapture.core.calibration`` for the accepted formats."""

    card_search_region: str = "auto"
    """Where to look for the card: 'auto', or one of the image corners
    ('top-left', 'top-right', 'bottom-left', 'bottom-right')."""

    card_corners: tuple[tuple[float, float], ...] = ()
    """Four (x, y) corners of the card in normalised image coordinates, marked
    by hand in the Plates tab.

    Set this when automatic detection fails - which it does for a card with no
    printed frame, since there is then no outer edge to find. Because the
    values are normalised, corners marked once apply to every plate shot in the
    same rig. When present, these are used instead of searching."""

    card_max_residual_dE: float = 5.0
    """Mean colour difference across patches after fitting, above which the
    card read is treated as failed. A high residual means the card was blurred,
    glared or partly hidden."""

    background_normalise: bool = True
    """Method B. Kept as a switch only so it can be disabled for testing."""

    background_percentile: float = 50.0
    """Percentile of the detected bare-agar pixels used as the reference
    colour. The median is robust to a stray colony fragment."""

    capture_exif: bool = True
    """Method C."""


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationParams:
    """SAM 3 prompt-based segmentation settings."""

    prompt_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "fine_filaments": (
                "Fine cotton thread like structure",
                "Fine Fungal Filaments",
                "Blood capillaries like structures",
            ),
            "core_colony": (
                "only the dense circular central body of a single yellow fungal colony",
                "the bright compact inner disc of the colony without the fuzzy edge",
                "the solid smooth central region of the colony with no halo or filaments",
            ),
        }
    )
    """Text prompts, in groups. Each group is run separately and the resulting
    masks are merged with a logical OR, which favours recall: a doubtful edge
    pixel is included rather than a real hyphal fringe being missed.

    Fixes a bug in the original script, where a missing comma joined
    'Fine Fungal Filaments' and 'Blood capillaries like structures' into a
    single prompt by Python string concatenation."""

    confidence: float = 0.25
    use_largest_component: bool = True
    """Keep only the biggest connected blob, dropping specks of noise."""

    device: str = "auto"
    """'auto' resolves to cuda -> mps -> cpu. Override for reproducibility
    testing."""

    half_precision: str = "auto"
    """'auto' enables FP16 only on CUDA. FP16 fails on CPU and on some Apple
    Silicon paths, which is why the original script's hard-coded True was not
    portable."""

    def __post_init__(self) -> None:
        # TOML has no tuple type, so a saved-and-reloaded project returns the
        # prompt lists as Python lists. Normalise here so a project compares
        # equal to itself across a save/load cycle, and so prompt groups stay
        # immutable like every other parameter.
        normalised = {
            str(group): tuple(prompts)
            for group, prompts in dict(self.prompt_groups).items()
        }
        object.__setattr__(self, "prompt_groups", normalised)


# --------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisParams:
    """
    Every setting that affects a measured number.

    Pass one instance to the feature extractor AND to the validator. They then
    cannot disagree about ring width, quantisation level or anything else.
    """

    ring: RingParams = field(default_factory=RingParams)
    morphology: MorphologyParams = field(default_factory=MorphologyParams)
    glcm: GLCMParams = field(default_factory=GLCMParams)
    lbp: LBPParams = field(default_factory=LBPParams)
    colour: ColourParams = field(default_factory=ColourParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)

    # Which feature families to compute. Turning one off removes its columns.
    compute_morphology: bool = True
    compute_intensity: bool = True
    compute_glcm: bool = True
    compute_lbp: bool = True
    compute_colour: bool = True

    # Families written to the CSV but excluded from PCA by default, because
    # they are high-dimensional and would swamp the other features.
    exclude_from_pca: tuple[str, ...] = ("_lbp_bin_", "_hist_")

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionaries, ready for TOML or JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisParams:
        """Rebuild from a dictionary, filling anything absent with defaults.

        Unknown keys are ignored rather than raising, so a project file written
        by a newer version still opens in an older one.
        """
        sub = {
            "ring": RingParams,
            "morphology": MorphologyParams,
            "glcm": GLCMParams,
            "lbp": LBPParams,
            "colour": ColourParams,
            "calibration": CalibrationParams,
            "segmentation": SegmentationParams,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub.items():
            raw = data.get(name) or {}
            declared = klass.__dataclass_fields__
            clean = {k: v for k, v in raw.items() if k in declared}
            # TOML has no tuple type, so every tuple comes back as a list.
            # Restore tuples wherever the dataclass declares one, otherwise a
            # reloaded project would not compare equal to the one just saved.
            for k, v in list(clean.items()):
                if not isinstance(v, list):
                    continue
                annotation = str(declared[k].type)
                default = declared[k].default
                if isinstance(default, tuple) or "tuple" in annotation:
                    clean[k] = tuple(v)
            kwargs[name] = klass(**clean)

        own = {
            f
            for f in cls.__dataclass_fields__
            if f not in sub
        }
        for k in own:
            if k in data:
                v = data[k]
                if k == "exclude_from_pca" and isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
        return cls(**kwargs)

    def with_changes(self, **kwargs: Any) -> AnalysisParams:
        """A copy with some fields replaced. The original stays frozen."""
        return replace(self, **kwargs)


DEFAULT_PARAMS = AnalysisParams()
"""The defaults. Matches the behaviour of the original scripts, so the Phase 0
regression test can compare like with like."""

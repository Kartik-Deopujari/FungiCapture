"""
Regression tests for the low-severity review fixes.

Covers: masked ring GLCM (S4), the scale-robust near-constant filter (S7), the
combined-media caveat (S6), the circularity clamp, the freeform well-id collision,
the SAM 3 "package missing" fall-back (R8), and the decompression-bomb guard (G).
The GUI close-handler fix (R6) needs a running Qt event loop and is reviewed by
code only.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("cv2")
import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.morphology import dilation, disk  # noqa: E402


# --- E: freeform well-id collision ------------------------------------------
def test_freeform_well_ids_are_unique():
    from fungicapture.core.plate import freeform_well_id

    ids = [freeform_well_id(i) for i in range(1000)]
    assert len(set(ids)) == 1000              # no collisions
    assert freeform_well_id(693) != freeform_well_id(792)  # the old clash


# --- D: circularity clamp ----------------------------------------------------
def test_circularity_is_clamped_for_tiny_masks():
    from fungicapture.core.features.morphology import morphology_features
    from fungicapture.core.params import MorphologyParams

    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True  # a single pixel - Crofton perimeter under-estimates here
    out = morphology_features(mask, MorphologyParams())
    assert out["circularity"] <= 1.0
    assert out["roughness"] >= 1.0


# --- S7: scale-robust near-constant filter ----------------------------------
def test_zero_centred_feature_is_not_dropped_as_constant():
    from fungicapture.core.phenotype import PhenotypeSettings, select_features

    frame = pd.DataFrame({
        "a_star": np.linspace(-5, 5, 50),  # mean ~0 but real spread -> keep
        "flat_50": np.full(50, 50.0),      # constant -> drop
        "flat_0": np.zeros(50),            # constant at zero -> drop
    })
    kept, dropped = select_features(frame, list(frame.columns), PhenotypeSettings())
    assert "a_star" in kept
    assert "flat_50" in dropped and "flat_0" in dropped


# --- S6: combined-media caveat ----------------------------------------------
def test_combined_media_file_carries_a_caveat():
    from fungicapture.core.phenotype import PhenotypeSettings, compute_phenotypes

    rng = np.random.default_rng(0)
    rows = []
    for medium in ("SDA", "PDA"):
        for i in range(30):
            row = {f"feat{j}": rng.normal() for j in range(5)}
            row["media"] = medium
            row["sample_id"] = f"{medium}_{i}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    result = compute_phenotypes(frame, [f"feat{j}" for j in range(5)], PhenotypeSettings())
    assert any("per-medium files" in w for w in result.warnings)


# --- S4: masked ring GLCM ----------------------------------------------------
def test_ring_glcm_measures_the_ring_not_the_fill():
    from fungicapture.core.features.texture import glcm_features
    from fungicapture.core.params import GLCMParams

    size = 120
    yy, xx = np.ogrid[:size, :size]
    centre = size // 2
    inside = (yy - centre) ** 2 + (xx - centre) ** 2 <= 40 * 40
    ring = dilation(inside, disk(8)) & ~inside

    rng = np.random.default_rng(0)
    gray = np.zeros((size, size), np.uint8)
    gray[ring] = rng.integers(40, 200, ring.sum())  # a genuinely textured ring

    out = glcm_features(gray, ring, "ring", GLCMParams())
    assert all(np.isfinite(v) for v in out.values())
    # If the fill dominated (the old behaviour), the ring would read as almost
    # perfectly homogeneous (~1). A real textured ring is far from that.
    assert out["ring_glcm_homogeneity_mean"] < 0.5


# --- G: decompression-bomb guard --------------------------------------------
def test_oversized_image_is_refused(tmp_path, monkeypatch):
    import fungicapture.core.features.regions as regions

    path = tmp_path / "big.png"
    cv2.imwrite(str(path), np.zeros((2000, 2000, 3), np.uint8))  # 4 MP
    monkeypatch.setattr(regions, "_MAX_DECODE_PIXELS", 1_000_000)  # 1 MP limit
    with pytest.raises(OSError, match="megapixels"):
        regions.read_image(str(path))
    # Under the real limit it reads fine.
    monkeypatch.setattr(regions, "_MAX_DECODE_PIXELS", 300_000_000)
    assert regions.read_image(str(path)).shape[:2] == (2000, 2000)


# --- R8: SAM 3 package missing -> clean fall-back ---------------------------
def test_sam3_without_package_falls_back_to_classical(tmp_path, monkeypatch):
    import importlib.util

    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed; the fall-back path is not exercised")

    from fungicapture.core.params import SegmentationParams
    from fungicapture.core.segment import make_segmenter

    (tmp_path / "sam3.pt").write_bytes(b"not a real model")
    monkeypatch.setenv("FUNGICAPTURE_SAM3_WEIGHTS", str(tmp_path / "sam3.pt"))
    segmenter, messages = make_segmenter(SegmentationParams(), backend="auto")
    assert segmenter.name == "classical"
    assert any("ultralytics" in m for m in messages)

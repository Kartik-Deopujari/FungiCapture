"""
Regression test for the colour-scaling fix in load_rgb.

A non-8-bit image (e.g. a 16-bit scanner/microscope TIFF) must be scaled by the
dtype's full range - the same ABSOLUTE scaling load_gray uses - not stretched to
the image's own brightest pixel. The old per-image stretch made a dark colony
read dark in the intensity (grayscale) features but bright in the colour
features, corrupting the melanization phenotype.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

import cv2  # noqa: E402

from fungicapture.core.features.regions import load_gray, load_rgb  # noqa: E402


def test_16bit_is_absolute_scaled_not_peak_stretched(tmp_path):
    # A dark colony across the frame, plus one very bright pixel so the per-image
    # peak is high. Under the OLD stretch the dark region would be pulled bright.
    img = np.zeros((8, 8, 3), np.uint16)
    img[:] = (8000, 6000, 4000)
    img[0, 0] = (60000, 60000, 60000)
    path = tmp_path / "plate.tif"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    rgb = load_rgb(path)
    assert rgb.dtype == np.uint8

    # Absolute scaling: 8000/65535*255 ≈ 31, 6000 ≈ 23, 4000 ≈ 16 (±1 rounding).
    dark = rgb[4, 4].astype(int)
    assert abs(dark[0] - 31) <= 1
    assert abs(dark[1] - 23) <= 1
    assert abs(dark[2] - 16) <= 1

    # It must NOT be stretched to the per-image peak (which would push the dark
    # region far brighter, e.g. red channel 8000/60000*255 ≈ 34 ... the giveaway
    # is that the two scalings diverge as the peak rises; here assert it matches
    # the absolute value, well below any peak-stretch result for a higher peak).
    # Cross-check: the grayscale path reads the same colony as dark too.
    gray = load_gray(path)
    assert int(gray[4, 4]) < 40  # dark in both paths -> consistent


def test_8bit_is_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    path = tmp_path / "plate.png"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out = load_rgb(path)
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)  # 8-bit round-trips exactly, no rescaling

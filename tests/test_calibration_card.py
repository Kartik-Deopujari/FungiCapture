"""
Regression tests for the colour-card quality-check fix.

The correction is a linear model with four parameters per channel. With no more
patches than parameters the least-squares fit is exact, so the old training
residual read ~0 dE for ANY measurement - a bad card passed the quality gate.
The fix: judge quality by a leave-one-out residual, and refuse a card with too
few patches to be checkable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
from skimage import color  # noqa: E402

from fungicapture.core.calibration import (  # noqa: E402
    _CORRECTION_PARAMS,
    _MIN_CARD_PATCHES,
    _leave_one_out_residual_dE,
    fit_colour_correction,
)


class _Card:
    """Minimal stand-in for CardDefinition (only reference_lab/n_patches used)."""

    def __init__(self, reference_lab: np.ndarray):
        self.reference_lab = reference_lab

    @property
    def n_patches(self) -> int:
        return len(self.reference_lab)


def _lab(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return color.rgb2lab(rng.random((n, 1, 3))).reshape(-1, 3)


def test_min_patches_is_above_parameter_count():
    assert _MIN_CARD_PATCHES > _CORRECTION_PARAMS


def test_leave_one_out_is_undefined_with_too_few_patches():
    ref_lab = _lab(4, seed=0)
    ref_rgb = color.lab2rgb(ref_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    measured = np.random.default_rng(1).random((4, 3))
    # n - 1 = 3 < 4 parameters -> each fold is under-determined -> None.
    assert _leave_one_out_residual_dE(measured, ref_rgb, ref_lab) is None


def test_held_out_residual_separates_good_and_bad_reads():
    ref_lab = _lab(24, seed=2)
    ref_rgb = color.lab2rgb(ref_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    card = _Card(ref_lab)
    rng = np.random.default_rng(3)

    # A good read: the reference under a mild linear distortion plus faint noise.
    good = np.clip(ref_rgb * 0.9 + 0.03 + rng.normal(0, 0.01, (24, 3)), 0, 1)
    _, residual_good = fit_colour_correction(good, card)

    # A bad read: pure noise, unrelated to the reference.
    bad = rng.random((24, 3))
    _, residual_bad = fit_colour_correction(bad, card)

    assert residual_good < 10.0            # a real read passes a sane dE gate
    assert residual_bad > residual_good * 3  # garbage is clearly worse, not ~0

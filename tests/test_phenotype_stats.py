"""
Regression tests for the two phenotype statistics fixes.

1. The outlier test must flag a genuinely extreme colony at the small sample
   sizes real media groups have. The old Mahalanobis-from-sample-covariance test
   was mathematically bounded and flagged nothing below ~32 colonies.
2. Scoring a group with fewer colonies than features must say so (a warning and
   pca_reliable=0), because the PCA axes are then not uniquely determined and the
   variance-explained figures are optimistic.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
import pandas as pd  # noqa: E402

from fungicapture.core.phenotype import (  # noqa: E402
    PhenotypeSettings,
    compute_phenotypes,
    flag_outliers,
)


@pytest.mark.parametrize("n", [8, 15, 20, 30])
def test_outlier_test_flags_extreme_colony_at_small_n(n):
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(n, 10))
    scores[0, 0] = 50.0  # one wildly extreme colony on PC1
    is_outlier, score, p = flag_outliers(scores, n_pcs=10, alpha=0.001)
    assert bool(is_outlier[0]) is True          # the extreme colony is caught
    assert score[0] > 5                          # its robust z is large
    assert 0.0 <= p[0] <= 1.0                     # p is a genuine probability


def test_outlier_test_is_quiet_on_clean_data():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(25, 10))
    is_outlier, _, _ = flag_outliers(scores, n_pcs=10, alpha=0.001)
    assert int(is_outlier.sum()) <= 1            # no false-positive storm


def _frame(n_colonies: int, n_features: int, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    cols = {f"feat{i}": rng.normal(size=n_colonies) for i in range(n_features)}
    frame = pd.DataFrame(cols)
    frame["media"] = "SDA"
    frame["sample_id"] = [f"s{i}" for i in range(n_colonies)]
    return frame, [f"feat{i}" for i in range(n_features)]


def test_fewer_colonies_than_features_is_flagged_unreliable():
    frame, features = _frame(n_colonies=10, n_features=30)
    result = compute_phenotypes(frame, features, PhenotypeSettings())
    table = result.per_media["SDA"].table
    assert int(table["pca_reliable"].iloc[0]) == 0
    assert any("fewer colonies than features" in w for w in result.warnings)
    # The renamed outlier columns are present and the stale name is gone.
    assert {"outlier_score", "outlier_p", "is_outlier"} <= set(table.columns)
    assert "mahalanobis_d2" not in table.columns


def test_healthy_group_is_reliable_and_unwarned():
    frame, features = _frame(n_colonies=120, n_features=10)
    result = compute_phenotypes(frame, features, PhenotypeSettings())
    table = result.per_media["SDA"].table
    assert int(table["pca_reliable"].iloc[0]) == 1
    assert not any("fewer colonies than features" in w for w in result.warnings)

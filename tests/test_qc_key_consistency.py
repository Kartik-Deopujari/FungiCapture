"""
Regression test for the QC/feature/verdict key mismatch.

In a batch project the feature step writes ``base_key`` as the project-unique
colony key (which includes the plate sub-folder). The Validate tab used to look
numbers up - and record pass/fail verdicts - by the bare filename stem, so in a
batch project the panel showed "run the feature step" even after it had run, and
colonies marked 'fail' were not actually excluded from scoring. This checks that
the one key (``colony_key``) matches across the feature table, the Validate
lookup, and the fail-exclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")
import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from fungicapture.core import pipeline  # noqa: E402
from fungicapture.core.pipeline import colony_key, pair_crops_and_masks  # noqa: E402
from fungicapture.core.plate import BoxGeometry  # noqa: E402
from fungicapture.core.project import ImagingMode, PlateLayout, Project  # noqa: E402
from fungicapture.core.validate import VerdictStore  # noqa: E402

COLONIES = [(150, 120), (450, 120), (150, 280), (450, 280)]


def _make_project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for p in range(2):  # two plates -> per-plate sub-folders -> batch keys
        img = (np.full((400, 600, 3), 210, np.uint8)
               + rng.integers(-8, 8, (400, 600, 3))).clip(0, 255).astype(np.uint8)
        for (cx, cy) in COLONIES:
            img[cy - 55:cy + 55, cx - 55:cx + 55] = rng.integers(40, 130, (110, 110, 3))
        cv2.imwrite(str(images / f"PLATE{p}_SDA.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    project = Project.create(
        tmp_path / "proj", name="qc", mode=ImagingMode.DISH_BATCH,
        image_dir=images, layout=PlateLayout(rows=1, cols=1), exist_ok=True,
    )
    project.media_label = "SDA"
    project.save()
    boxes = BoxGeometry(
        rows=1, cols=1, freeform=True,
        boxes=[[(cx - 55) / 600, (cy - 55) / 400, (cx + 55) / 600, (cy + 55) / 400]
               for (cx, cy) in COLONIES],
    )
    pipeline.crop_plates(project, boxes=boxes, workers=2)
    pipeline.segment_crops(project, backend="classical", workers=2)
    pipeline.extract_features(project, workers=2)
    return project


def test_colony_key_matches_feature_table_in_a_batch(tmp_path):
    project = _make_project(tmp_path)
    frame = pd.read_csv(project.vectors_dir / "all_colony_features.csv")
    base_keys = set(frame["base_key"].astype(str))

    # Sanity: this really is the batch case (keys carry the plate sub-folder).
    assert any("/" in k for k in base_keys)

    pairs = pair_crops_and_masks(project, project.crops_dir, project.masks_dir)
    assert pairs
    for crop_path, _ in pairs:
        key = colony_key(project, crop_path, project.crops_dir)
        assert key in base_keys  # the Validate panel lookup will now hit


def test_failed_verdict_is_excluded_from_scoring(tmp_path):
    project = _make_project(tmp_path)
    pairs = pair_crops_and_masks(project, project.crops_dir, project.masks_dir)
    victim = colony_key(project, pairs[0][0], project.crops_dir)

    store = VerdictStore(project.qc_dir / VerdictStore.FILENAME)
    store.set(victim, "fail", "test")
    store.save()

    result = pipeline.score_phenotypes(project, exclude_failed=True)
    # The exclusion only happens when the verdict key matches base_key, which is
    # exactly what the fix makes true.
    assert any("excluded because they were marked" in w for w in result.warnings)

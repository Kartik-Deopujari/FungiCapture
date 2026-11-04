"""
End-to-end guard: the parallel pipeline must give the same answers as serial.

The whole pipeline is run twice on one set of synthetic plates - once with a
single worker, once with four - and the outputs are compared. This is the test
that would catch a parallelism bug: a race, a dropped item, a mangled file, or
a feature table that came out in a different order.

Skipped automatically where the scientific stack (OpenCV, scikit-image,
scikit-learn, pandas) is not installed, so a core-only checkout still collects
cleanly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# The pipeline needs the full scientific stack; skip the whole module without it.
pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fungicapture.core import pipeline  # noqa: E402
from fungicapture.core.plate import BoxGeometry  # noqa: E402
from fungicapture.core.project import ImagingMode, PlateLayout, Project  # noqa: E402

N_PLATES = 4
IMG_H, IMG_W = 600, 800
COLONIES = [(IMG_W // 4, IMG_H // 3), (3 * IMG_W // 4, IMG_H // 3),
            (IMG_W // 4, 2 * IMG_H // 3), (3 * IMG_W // 4, 2 * IMG_H // 3)]
R = 90


def _make_plate(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = (np.full((IMG_H, IMG_W, 3), 215, np.uint8)
           + rng.integers(-8, 8, (IMG_H, IMG_W, 3))).clip(0, 255).astype(np.uint8)
    for (cx, cy) in COLONIES:
        # A textured colony so the texture/colour features have real work.
        patch = rng.integers(40, 130, (2 * R, 2 * R, 3)).astype(np.uint8)
        img[cy - R:cy + R, cx - R:cx + R] = patch
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _boxes() -> BoxGeometry:
    b = [[(cx - R) / IMG_W, (cy - R) / IMG_H, (cx + R) / IMG_W, (cy + R) / IMG_H]
         for (cx, cy) in COLONIES]
    return BoxGeometry(rows=1, cols=1, boxes=b, freeform=True)


def _run(root: Path, images: Path, workers: int):
    project = Project.create(
        root, name="t", mode=ImagingMode.DISH_BATCH, image_dir=images,
        layout=PlateLayout(rows=1, cols=1), exist_ok=True,
    )
    project.media_label = "SDA"
    project.save()
    boxes = _boxes()
    results = {
        "crop": pipeline.crop_plates(project, boxes=boxes, workers=workers),
        "segment": pipeline.segment_crops(project, backend="classical", workers=workers),
        "features": pipeline.extract_features(project, workers=workers),
        "phenotypes": pipeline.score_phenotypes(project),
    }
    return project, results


def _hashes(root: Path, suffix: str) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob(f"*{suffix}"))
    }


@pytest.fixture()
def images(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("images")
    for i in range(N_PLATES):
        _make_plate(directory / f"STRAIN{i:02d}_SDA.png", seed=i)
    return directory


def test_serial_and_parallel_agree(tmp_path, images):
    proj1, res1 = _run(tmp_path / "serial", images, workers=1)
    proj4, res4 = _run(tmp_path / "parallel", images, workers=4)

    expected = N_PLATES * len(COLONIES)

    # Every step processed the same number of items, with no errors.
    for step in ("crop", "segment", "features"):
        assert res1[step].n_outputs == res4[step].n_outputs == expected, step
        assert res1[step].n_errors == 0 and res4[step].n_errors == 0, step

    # Timing and worker count are recorded (the "measure first" instrumentation).
    assert res1["crop"].data["workers"] == 1
    assert res4["crop"].data["workers"] == 4
    assert "elapsed_s" in res4["features"].data

    # Crops and masks are byte-for-byte identical - deterministic and lossless.
    crops1, crops4 = _hashes(proj1.crops_dir, ".png"), _hashes(proj4.crops_dir, ".png")
    assert crops1 == crops4 and len(crops1) == expected
    assert all(name.endswith(".png") for name in crops1)
    assert _hashes(proj1.masks_dir, "_mask.png") == _hashes(proj4.masks_dir, "_mask.png")

    # The feature table matches in content and row order. The ~1e-14 wobble from
    # scikit-learn's multi-threaded KMeans (present even serial-to-serial) is far
    # below 1e-9, so this tolerance checks "same answers", not bit-identity.
    f1 = pd.read_csv(proj1.vectors_dir / "all_colony_features.csv")
    f4 = pd.read_csv(proj4.vectors_dir / "all_colony_features.csv")
    assert list(f1["base_key"]) == list(f4["base_key"])  # order preserved

    drop = [c for c in ("generated_utc", "image_path", "mask_path") if c in f1.columns]
    f1n = f1.drop(columns=drop).sort_values("base_key").reset_index(drop=True)
    f4n = f4.drop(columns=drop).sort_values("base_key").reset_index(drop=True)
    pd.testing.assert_frame_equal(f1n, f4n, check_exact=False, rtol=1e-9, atol=1e-9)

    assert res1["phenotypes"].n_outputs > 0 and res4["phenotypes"].n_outputs > 0


def test_degenerate_batch_does_not_crash(tmp_path):
    """A tiny, uniform batch must not abort the whole run.

    When every colony is alike, phenotype quality control drops every feature
    and compute_phenotypes raises. The run must catch that and finish with a
    warning (and the feature table still written), not crash out of
    run_pipeline the way it used to.
    """
    images = tmp_path / "images"
    images.mkdir()
    for i in range(4):
        img = np.full((400, 600, 3), 200, np.uint8)
        for (cx, cy) in COLONIES[:2]:  # reuse two colony centres, scaled to 600x400
            cx = int(cx * 600 / IMG_W)
            cy = int(cy * 400 / IMG_H)
            cv2.circle(img, (cx, cy), 55, (80, 60, 40), -1)  # identical solid colonies
        cv2.imwrite(str(images / f"S{i:02d}_SDA.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    project = Project.create(
        tmp_path / "proj", name="degen", mode=ImagingMode.DISH_BATCH,
        image_dir=images, layout=PlateLayout(rows=1, cols=1), exist_ok=True,
    )
    project.media_label = "SDA"
    project.save()
    b = []
    for (cx, cy) in COLONIES[:2]:
        cx = cx * 600 / IMG_W
        cy = cy * 400 / IMG_H
        b.append([(cx - 70) / 600, (cy - 70) / 400, (cx + 70) / 600, (cy + 70) / 400])
    boxes = BoxGeometry(rows=1, cols=1, boxes=b, freeform=True)

    # The whole run must complete without raising.
    results = pipeline.run_pipeline(project, backend="classical", boxes=boxes, workers=2)
    steps = [r.step for r in results]
    assert "phenotypes" in steps  # reached the last step rather than crashing
    phenotypes = next(r for r in results if r.step == "phenotypes")
    assert phenotypes.warnings  # it explained why it produced nothing
    # The feature table is still on disk - the measured data is not lost.
    assert (project.vectors_dir / "all_colony_features.csv").exists()

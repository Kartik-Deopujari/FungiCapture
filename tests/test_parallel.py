"""
Tests for ``fungicapture.core.parallel``.

Standard-library only - no numpy, OpenCV or model weights - so this runs fast
anywhere and guards the behaviour the whole pipeline now leans on: results come
back in order, one failing item never sinks the rest, cancellation stops the
run, and progress is counted in the right units.
"""

from __future__ import annotations

import threading
import time

import pytest

from fungicapture.core import parallel as parallel_mod
from fungicapture.core.parallel import (
    limit_native_threads,
    parallel_map,
    resolve_workers,
)


def test_order_is_preserved_regardless_of_completion_order():
    # Smaller numbers sleep longer, so they finish last - yet the outcomes must
    # still come back in submission order.
    def slow(x: int) -> int:
        time.sleep((10 - x) * 0.005)
        return x * x

    run = parallel_map(range(10), slow, workers=4)
    assert [o.result for o in run.outcomes] == [i * i for i in range(10)]
    assert all(o.ok for o in run.outcomes)
    assert run.cancelled is False


def test_per_item_errors_are_captured_and_others_succeed():
    def maybe_fail(x: int) -> int:
        if x == 3:
            raise ValueError("boom")
        return x

    run = parallel_map(range(6), maybe_fail, workers=3)
    errored = [o for o in run.outcomes if o.error is not None]
    assert len(errored) == 1
    assert errored[0].item == 3
    assert isinstance(errored[0].error, ValueError)
    assert sorted(o.result for o in run.outcomes if o.ok) == [0, 1, 2, 4, 5]


def test_threads_actually_run_concurrently():
    active = 0
    peak = 0
    lock = threading.Lock()

    def watch(x: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x

    parallel_map(range(8), watch, workers=4)
    assert peak > 1  # more than one item was in flight at once


def test_cancellation_stops_the_run():
    def cancel_after_three(done: int, total: int, message: str) -> bool:
        return done < 3

    run = parallel_map(
        range(100), lambda x: x, workers=2, progress=cancel_after_three
    )
    assert run.cancelled is True
    assert len(run.outcomes) < 100


def test_progress_is_counted_in_weight_units():
    seen: list[tuple[int, int]] = []

    def progress(done: int, total: int, message: str) -> bool:
        seen.append((done, total))
        return True

    parallel_map(
        [("plateA", 96), ("plateB", 48)],
        lambda it: it[0],
        workers=2,
        weight=lambda it: it[1],
        progress=progress,
    )
    assert all(total == 144 for _, total in seen)
    assert seen[-1][0] == 144


def test_empty_input_is_a_clean_no_op():
    run = parallel_map([], lambda x: x)
    assert run.outcomes == []
    assert run.cancelled is False


def test_serial_path_matches_parallel_behaviour():
    def maybe_fail(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    run = parallel_map(range(5), maybe_fail, workers=1)
    assert [o.item for o in run.outcomes] == [0, 1, 2, 3, 4]
    assert any(o.error is not None for o in run.outcomes)


def test_resolve_workers_rules(monkeypatch):
    monkeypatch.delenv("FUNGICAPTURE_WORKERS", raising=False)
    assert resolve_workers(3) == 3                 # explicit wins
    assert resolve_workers(100, cap=5) == 5        # capped to work available
    assert resolve_workers(-5) >= 1                 # never below 1

    monkeypatch.setenv("FUNGICAPTURE_WORKERS", "2")
    assert resolve_workers(None) == 2               # env var honoured

    monkeypatch.setenv("FUNGICAPTURE_WORKERS", "nonsense")
    assert resolve_workers(None, cap=4) >= 1        # bad env var ignored, no crash


def test_keyboard_interrupt_is_not_swallowed_on_the_serial_path():
    # A per-item Exception (incl. MemoryError) is captured and the run goes on,
    # but KeyboardInterrupt / SystemExit must propagate so Ctrl+C can stop a run.
    def interrupt_on_two(x: int) -> int:
        if x == 2:
            raise KeyboardInterrupt
        return x

    with pytest.raises(KeyboardInterrupt):
        parallel_map(range(5), interrupt_on_two, workers=1)

    def oom_on_two(x: int) -> int:
        if x == 2:
            raise MemoryError("simulated")
        return x

    run = parallel_map(range(5), oom_on_two, workers=1)
    errored = [o for o in run.outcomes if o.error is not None]
    assert len(errored) == 1 and isinstance(errored[0].error, MemoryError)


def test_resolve_workers_is_affinity_and_ram_aware(monkeypatch):
    monkeypatch.delenv("FUNGICAPTURE_WORKERS", raising=False)
    assert parallel_mod._available_cpus() >= 1

    # Tight RAM: the auto worker count is capped so workers fit in memory.
    monkeypatch.setattr(parallel_mod, "_available_ram_bytes", lambda: 1_000_000_000)
    capped = resolve_workers(None, cap=100, mem_per_worker_bytes=400_000_000)
    assert capped <= 2  # 0.6 * 1 GB / 400 MB  ->  1

    # An explicit request is trusted and ignores the RAM cap.
    assert resolve_workers(8, cap=100, mem_per_worker_bytes=400_000_000) == 8

    # No RAM info -> no cap, just the CPU/affinity default (never crashes).
    monkeypatch.setattr(parallel_mod, "_available_ram_bytes", lambda: None)
    assert resolve_workers(None, cap=100, mem_per_worker_bytes=400_000_000) >= 1


def test_limit_native_threads_restores_opencv():
    cv2 = pytest.importorskip("cv2")
    before = cv2.getNumThreads()
    with limit_native_threads(True):
        assert cv2.getNumThreads() == 1
    assert cv2.getNumThreads() == before
    # Inactive (single worker): leave the native pools alone.
    with limit_native_threads(False):
        assert cv2.getNumThreads() == before

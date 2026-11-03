"""
Running independent per-item work across several CPU cores.

Every slow pipeline step is the same shape: a list of items (plates, crops,
colonies) where each item is processed on its own and nothing depends on the
order. That is the easy case for parallel work - no shared state to coordinate,
so we can hand each item to a different core and collect the answers.

Why threads, not processes
--------------------------
The heavy work here - decoding RAW/TIFF photographs, encoding PNGs, the skimage
texture maths, OpenCV, numpy - is native code that *releases Python's global
lock* while it runs. That means plain threads give real parallel speed-up for
this workload, without the cost of processes: no pickling of large images, no
re-importing the whole program on Windows, and a smaller memory footprint
(threads share one copy of the data). Processes would add exactly the kind of
fragility - spawn failures, serialization errors - that makes a tool feel buggy
across a mix of machines.

What this module deliberately does *not* do is parallelise an AI model (SAM 3)
by running copies of it. One model, used serially, is correct and stable;
many copies fight over memory and the GPU and crash. See ``segment.py``.

Cancellation and progress keep the same meaning as the old serial loops:
progress is reported as items finish, and a cancel takes effect *between*
items - an item already running is allowed to finish so no half-written file is
left behind.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# An upper bound on auto-detected worker threads. Past roughly this many, the
# bottleneck is disk and memory bandwidth, not cores, and more threads only add
# contention. A user who knows their machine can override with the env var or
# the ``--workers`` flag.
_DEFAULT_MAX_WORKERS = 8

# progress(done, total, message) -> False to cancel. ``done`` and ``total`` are
# in *weight* units (see ``parallel_map``), not always item counts.
ProgressFn = Callable[[int, int, str], bool]


def resolve_workers(requested: int | None = None, *, cap: int | None = None) -> int:
    """
    Decide how many worker threads to use.

    Order of preference:

    1. ``requested`` when it is a positive number - an explicit choice always
       wins.
    2. The ``FUNGICAPTURE_WORKERS`` environment variable, so a cluster job
       script can set it once.
    3. The machine's CPU count, capped at ``_DEFAULT_MAX_WORKERS``.

    The result is always at least 1, and never more than ``cap`` (pass the
    number of items so we never start more threads than there is work for).
    """
    workers = 0

    if requested is not None and requested > 0:
        workers = int(requested)
    else:
        env = os.environ.get("FUNGICAPTURE_WORKERS", "").strip()
        if env:
            try:
                parsed = int(env)
                if parsed > 0:
                    workers = parsed
            except ValueError:
                # A malformed env var should not crash a run; fall through to
                # auto-detection rather than failing.
                workers = 0

    if workers <= 0:
        workers = min(os.cpu_count() or 1, _DEFAULT_MAX_WORKERS)

    workers = max(1, workers)
    if cap is not None:
        workers = max(1, min(workers, cap))
    return workers


@dataclass
class ItemOutcome(Generic[T, R]):
    """What happened to one item: its result, or the error it raised."""

    index: int
    item: T
    result: R | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ParallelRun(Generic[T, R]):
    """The outcome of a whole parallel pass."""

    outcomes: list[ItemOutcome[T, R]]
    cancelled: bool = False


def parallel_map(
    items: Iterable[T],
    work: Callable[[T], R],
    *,
    workers: int | None = None,
    progress: ProgressFn | None = None,
    weight: Callable[[T], int] | None = None,
    label: Callable[[T], str] | None = None,
) -> ParallelRun[T, R]:
    """
    Run ``work(item)`` for every item, across several threads.

    Parameters
    ----------
    items
        The things to process. Consumed into a list up front so the total is
        known for progress reporting.
    work
        Called once per item, on a worker thread. Must be self-contained: it may
        read the item and shared read-only data, but must not mutate anything
        another call also touches. Any exception it raises is caught and
        attached to that item's outcome, so one bad item never stops the rest -
        this mirrors the old per-item ``try/except`` in each step.
    workers
        How many threads. ``None`` auto-detects (see :func:`resolve_workers`).
    progress
        Called **on the calling thread** as each item finishes, with cumulative
        weight done, total weight, and a message. Returning ``False`` cancels:
        no new items are started, items already running finish, and the run
        returns with ``cancelled=True``.
    weight
        Progress units one item is worth (default 1 each). Cropping uses this to
        count progress in colonies rather than plates, so a single plate that
        cuts 96 wells does not show "1 of 1".
    label
        A short message for an item, shown via ``progress``.

    Returns
    -------
    ParallelRun
        ``outcomes`` holds one :class:`ItemOutcome` per item that was processed,
        **in the original item order** (not completion order), so downstream
        output - feature rows, the warnings list - is deterministic regardless
        of which thread finished first. If the run was cancelled, outcomes for
        items that never started are absent.
    """
    item_list = list(items)
    total_items = len(item_list)
    if total_items == 0:
        return ParallelRun(outcomes=[], cancelled=False)

    def item_weight(it: T) -> int:
        return weight(it) if weight is not None else 1

    def item_label(it: T) -> str:
        return label(it) if label is not None else ""

    total_weight = sum(item_weight(it) for it in item_list)
    n = resolve_workers(workers, cap=total_items)

    outcomes: list[ItemOutcome[T, R] | None] = [None] * total_items
    done_weight = 0
    cancelled = False

    def record(index: int) -> None:
        """Run one item inline and store its outcome. Used by the serial path."""
        it = item_list[index]
        outcome: ItemOutcome[T, R] = ItemOutcome(index=index, item=it)
        try:
            outcome.result = work(it)
        except BaseException as error:  # noqa: BLE001 - captured per item on purpose
            outcome.error = error
        outcomes[index] = outcome

    # --- Serial fast path -------------------------------------------------
    # One worker means no threads at all: identical behaviour to the old loop,
    # which keeps the simple single-core case simple and easy to debug.
    if n <= 1:
        for index in range(total_items):
            it = item_list[index]
            if progress is not None and not progress(
                min(done_weight, total_weight), total_weight, item_label(it)
            ):
                cancelled = True
                break
            record(index)
            done_weight += item_weight(it)
            if progress is not None:
                progress(min(done_weight, total_weight), total_weight, item_label(it))
        return _collect(outcomes, cancelled)

    # --- Parallel path ----------------------------------------------------
    # At most ``n`` items are ever in flight: as each finishes we submit the
    # next. That bounds memory (we are not decoding every photograph at once)
    # and makes cancellation responsive - on cancel we simply stop submitting.
    executor = ThreadPoolExecutor(max_workers=n)
    try:
        pending: dict[Any, int] = {}
        next_index = 0

        def refill() -> None:
            nonlocal next_index
            while next_index < total_items and len(pending) < n:
                future = executor.submit(work, item_list[next_index])
                pending[future] = next_index
                next_index += 1

        refill()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                it = item_list[index]
                outcome = ItemOutcome(index=index, item=it)
                try:
                    outcome.result = future.result()
                except BaseException as error:  # noqa: BLE001
                    outcome.error = error
                outcomes[index] = outcome
                done_weight += item_weight(it)
                if progress is not None and not progress(
                    min(done_weight, total_weight), total_weight, item_label(it)
                ):
                    cancelled = True
            if cancelled:
                break
            refill()
    finally:
        # cancel_futures stops queued-but-not-started work; wait=True lets the
        # few still running finish so they never leave a half-written file.
        executor.shutdown(wait=True, cancel_futures=True)

    return _collect(outcomes, cancelled)


def _collect(
    outcomes: list[ItemOutcome[T, R] | None], cancelled: bool
) -> ParallelRun[T, R]:
    """Drop the slots for items that never started, keeping original order."""
    return ParallelRun(
        outcomes=[o for o in outcomes if o is not None],
        cancelled=cancelled,
    )

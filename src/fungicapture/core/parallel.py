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

import contextlib
import os
from collections.abc import Callable, Iterable, Iterator
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

# When auto-sizing by memory, use at most this fraction of the memory that is
# actually available, leaving headroom for the rest of the program and the OS.
_RAM_SAFETY_FRACTION = 0.6

# progress(done, total, message) -> False to cancel. ``done`` and ``total`` are
# in *weight* units (see ``parallel_map``), not always item counts.
ProgressFn = Callable[[int, int, str], bool]


def _available_cpus() -> int:
    """Cores this process may actually use, not the host's core count.

    On Linux ``sched_getaffinity`` reflects CPU affinity and the cgroup cpuset a
    container is pinned to, so a 2-core slice of a 64-core node reads as 2 - the
    plain ``os.cpu_count()`` would wrongly read 64 and oversubscribe. Falls back
    to ``cpu_count`` where affinity is unavailable (macOS, Windows).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _available_ram_bytes() -> int | None:
    """Best-effort free memory in bytes, container-aware, or None if unknown.

    Takes the smaller of the host's MemAvailable and, when present, the cgroup
    memory limit minus current usage - so a container with a small memory limit
    on a large host is sized by the limit, not the host. Any read failure yields
    None, which simply disables the memory cap (never a crash).
    """
    candidates: list[int] = []
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v2
        with open("/sys/fs/cgroup/memory.max") as handle:
            raw = handle.read().strip()
        if raw and raw != "max":
            limit = int(raw)
            used = 0
            try:
                with open("/sys/fs/cgroup/memory.current") as handle:
                    used = int(handle.read().strip())
            except (OSError, ValueError):
                used = 0
            candidates.append(max(0, limit - used))
    except (OSError, ValueError):
        pass
    return min(candidates) if candidates else None


def resolve_workers(
    requested: int | None = None,
    *,
    cap: int | None = None,
    mem_per_worker_bytes: int | None = None,
) -> int:
    """
    Decide how many worker threads to use.

    Order of preference:

    1. ``requested`` when it is a positive number - an explicit choice always
       wins, and is honoured as-is (only clamped to ``cap``).
    2. The ``FUNGICAPTURE_WORKERS`` environment variable, so a cluster job
       script can set it once. Also treated as an explicit choice.
    3. Auto: the cores this process may actually use (affinity/cgroup aware),
       capped at ``_DEFAULT_MAX_WORKERS``, and - when ``mem_per_worker_bytes`` is
       given - further capped so the workers together fit in available RAM.

    The result is always at least 1, and never more than ``cap`` (pass the
    number of items so we never start more threads than there is work for). The
    RAM cap applies only to the auto path: if the user or a job script asked for
    a specific number, we trust it.
    """
    workers = 0
    explicit = False

    if requested is not None and requested > 0:
        workers = int(requested)
        explicit = True
    else:
        env = os.environ.get("FUNGICAPTURE_WORKERS", "").strip()
        if env:
            try:
                parsed = int(env)
                if parsed > 0:
                    workers = parsed
                    explicit = True
            except ValueError:
                # A malformed env var should not crash a run; fall through to
                # auto-detection rather than failing.
                workers = 0

    if not explicit:
        workers = min(_available_cpus(), _DEFAULT_MAX_WORKERS)
        if mem_per_worker_bytes and mem_per_worker_bytes > 0:
            available = _available_ram_bytes()
            if available is not None:
                ram_workers = int(available * _RAM_SAFETY_FRACTION // mem_per_worker_bytes)
                workers = min(workers, max(1, ram_workers))

    workers = max(1, workers)
    if cap is not None:
        workers = max(1, min(workers, cap))
    return workers


@contextlib.contextmanager
def limit_native_threads(active: bool = True) -> Iterator[None]:
    """
    Pin the native maths libraries to one thread each for the duration.

    numpy/BLAS, OpenMP (used by LibRaw) and OpenCV each run their own internal
    thread pool. When we already run several items on our own worker threads,
    those inner pools multiply out - N workers x ~4 native threads on an 8-core
    machine - and oversubscribe the CPU, which measurably *loses* speed. Limiting
    the inner pools to one thread while our pool owns the parallelism avoids that.

    Only active when ``active`` is true (i.e. we are actually running more than
    one worker); with a single worker the inner pools should use every core. Best
    effort: if ``threadpoolctl`` or OpenCV is absent, the missing part is skipped.
    """
    if not active:
        yield
        return

    stack = contextlib.ExitStack()
    cv2_previous: int | None = None
    try:
        import cv2

        cv2_previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
    except Exception:
        cv2_previous = None
    try:
        from threadpoolctl import threadpool_limits

        stack.enter_context(threadpool_limits(limits=1))
    except Exception:
        pass
    try:
        yield
    finally:
        stack.close()
        if cv2_previous is not None:
            try:
                import cv2

                cv2.setNumThreads(cv2_previous)
            except Exception:
                pass


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
        except Exception as error:  # noqa: BLE001 - captured per item on purpose
            # Exception (not BaseException): a per-item failure, including
            # MemoryError, is recorded and the run continues. KeyboardInterrupt
            # and SystemExit are NOT Exceptions, so they propagate - Ctrl+C can
            # still stop the run on the serial path, where work() runs here on
            # the calling thread.
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
                except Exception as error:  # noqa: BLE001 - per-item failure, incl. MemoryError
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

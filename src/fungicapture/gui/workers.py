"""
Background workers.

Segmenting 96 colonies takes minutes. Run on the main thread it would freeze
the window and macOS would show the spinning wheel, so every long job runs in a
``QThread``.

The rule: only ``core`` code runs in the worker, and only signals cross back to
the interface. Qt widgets must never be touched from a worker thread - doing so
crashes rather than misbehaving, and the crash usually appears somewhere else
entirely.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class PipelineWorker(QObject):
    """Runs one callable that takes a ``progress`` keyword argument."""

    progressed = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, function: Callable[..., Any], **kwargs: Any):
        super().__init__()
        self._function = function
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the job to stop at its next checkpoint.

        Cooperative rather than forced: killing a thread mid-write would leave
        half-written files behind.
        """
        self._cancelled = True

    def _progress(self, done: int, total: int, message: str = "") -> bool:
        self.progressed.emit(done, total, message)
        return not self._cancelled

    def run(self) -> None:
        try:
            result = self._function(progress=self._progress, **self._kwargs)
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class _MainThreadRelay(QObject):
    """
    Receives the worker's signals on the MAIN thread and forwards them.

    The worker lives on the background thread, so its signals are emitted there.
    This relay is created in (and never moved from) the main thread, so the
    worker->relay connections are queued and the relay's slots run on the main
    thread - where it is safe to touch Qt widgets. The relay then calls the
    caller's callbacks, so those run on the main thread too.

    This is what makes the callbacks safe even when they are plain functions or
    lambdas. A signal connected directly to a context-less callable uses a
    *direct* connection and runs on the worker thread; a callback that then
    touched a widget (as the segmentation-setup window's progress callback did)
    crashed the whole application. Routing everything through this QObject
    removes that trap for every caller.
    """

    def __init__(self, on_progress, on_finished, on_failed):
        super().__init__()
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_failed = on_failed

    @Slot(int, int, str)
    def progress(self, done: int, total: int, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total, message)

    @Slot(object)
    def finished(self, result: Any) -> None:
        if self._on_finished is not None:
            self._on_finished(result)

    @Slot(str, str)
    def failed(self, message: str, trace: str) -> None:
        if self._on_failed is not None:
            self._on_failed(message, trace)


class WorkerHandle:
    """
    Keeps a worker and its thread alive for as long as the job runs.

    Without holding these references, Python garbage-collects the thread object
    mid-run and the application dies with no traceback. This class exists
    entirely to prevent that. It also holds the main-thread relay, for the same
    lifetime reason.
    """

    def __init__(
        self,
        worker: PipelineWorker,
        thread: QThread,
        relay: _MainThreadRelay | None = None,
    ):
        self.worker = worker
        self.thread = thread
        self.relay = relay

    def cancel(self) -> None:
        self.worker.cancel()

    def is_running(self) -> bool:
        return self.thread.isRunning()

    def wait(self, milliseconds: int = 5000) -> bool:
        return self.thread.wait(milliseconds)


def run_in_background(
    function: Callable[..., Any],
    on_progress: Callable[[int, int, str], None] | None = None,
    on_finished: Callable[[Any], None] | None = None,
    on_failed: Callable[[str, str], None] | None = None,
    **kwargs: Any,
) -> WorkerHandle:
    """
    Start a job on a worker thread and wire up its signals.

    Returns a handle the caller must keep - see ``WorkerHandle``.
    """
    thread = QThread()
    worker = PipelineWorker(function, **kwargs)
    worker.moveToThread(thread)

    # The relay stays on the main thread (created here, never moved), so every
    # callback is delivered on the main thread - safe to touch Qt widgets from,
    # whether the callback is a bound method or a lambda.
    relay = _MainThreadRelay(on_progress, on_finished, on_failed)

    thread.started.connect(worker.run)
    worker.progressed.connect(relay.progress)
    worker.finished.connect(relay.finished)
    worker.failed.connect(relay.failed)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)

    thread.start()
    return WorkerHandle(worker, thread, relay)

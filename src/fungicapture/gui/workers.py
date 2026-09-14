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

from PySide6.QtCore import QObject, QThread, Signal


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


class WorkerHandle:
    """
    Keeps a worker and its thread alive for as long as the job runs.

    Without holding these references, Python garbage-collects the thread object
    mid-run and the application dies with no traceback. This class exists
    entirely to prevent that.
    """

    def __init__(self, worker: PipelineWorker, thread: QThread):
        self.worker = worker
        self.thread = thread

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

    thread.started.connect(worker.run)
    if on_progress:
        worker.progressed.connect(on_progress)
    if on_finished:
        worker.finished.connect(on_finished)
    if on_failed:
        worker.failed.connect(on_failed)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)

    thread.start()
    return WorkerHandle(worker, thread)

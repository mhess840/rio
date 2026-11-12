"""Bounded background writes for rollout diagnostics."""

import copy
import math
import threading
from collections.abc import Callable
from queue import Full, Queue


class AsyncDiagnostics:
    """Write immutable snapshots in order without blocking the control loop."""

    def __init__(self, max_pending: int = 1024):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._jobs: Queue = Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="rollout-diagnostics", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                function, args, kwargs = job
                with self._lock:
                    failed = self._error is not None
                if not failed:
                    function(*args, **kwargs)
            except BaseException as error:
                with self._lock:
                    self._error = self._error or error
            finally:
                self._jobs.task_done()

    def check(self) -> None:
        """Propagate worker failures to the caller before another control command."""
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Rollout diagnostic write failed") from error
        if not self._closed and not self._worker.is_alive():
            raise RuntimeError("Rollout diagnostic worker exited unexpectedly")

    def submit(self, function: Callable, *args, **kwargs) -> None:
        """Queue a copied snapshot or fail immediately when the writer falls behind."""
        self.check()
        if self._closed:
            raise RuntimeError("Rollout diagnostics are closed")
        snapshot_args, snapshot_kwargs = copy.deepcopy((args, kwargs))
        try:
            self._jobs.put_nowait((function, snapshot_args, snapshot_kwargs))
        except Full as error:
            raise RuntimeError("Rollout diagnostic backlog exceeded its bounded capacity") from error

    def close(self, timeout: float = 5.0) -> None:
        """Flush accepted snapshots after motion stops, with a bounded wait."""
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("Diagnostic close timeout must be finite and positive")
        if not self._closed:
            try:
                self._jobs.put(None, timeout=timeout)
            except Full as error:
                raise TimeoutError("Rollout diagnostics did not accept the shutdown request") from error
            self._closed = True
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("Rollout diagnostics did not finish before shutdown timeout")
        self.check()

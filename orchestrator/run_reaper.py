"""Stale GPU run-row reaper (US-021).

A GPU run row is finalized by ops/gpu_pipeline.py itself — but a SIGKILL,
reboot, or OOM of the transient unit skips the pipeline's finally block and
leaves the row stuck at status 'running' forever. A stranded row bricks its
Slack thread: check_research_status reports a corpse and start_research
refuses to launch again.

``GpuRunReaper`` runs as a background thread inside the orchestrator app
(same pattern as ApprovalsBridge): each tick it finds runs with
status='running', backend='gpu' whose transient unit (rdq-gpu-run-<ts>) is no
longer active, waits one grace period (so a pipeline mid-finalize is never
raced), then posts a note in the run's thread and marks the row 'failed'.
The note posts BEFORE the status flip (poller completion convention): a
transient Slack failure leaves the row 'running' and the whole reap retries
on the next tick.

Grace tracking is in-memory on purpose — a restart just restarts the grace
clock, which only ever delays a reap, never skips one. The global run lock
needs no attention here: a reaped unit's lock reads stale and self-cleans on
the next acquire (ops/run_lock.py).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from ops import run_lock
from orchestrator.state import Run, StateStore

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS = 900.0
DEFAULT_INTERVAL_SECONDS = 300.0


class SlackPoster(Protocol):
    """The one slack_sdk WebClient method the reaper needs."""

    def chat_postMessage(self, **kwargs: Any) -> Any: ...  # noqa: N802 - slack_sdk casing


def _default_unit_active(thread_ts: str) -> bool:
    return run_lock.unit_is_active(run_lock.unit_name(thread_ts))


def reap_note(run: Run) -> str:
    unit = run_lock.unit_name(run.thread_ts)
    return (
        f":headstone: research run reaped — its pipeline unit `{unit}` is no"
        " longer active and never finalized this run (killed, rebooted, or"
        " crashed before its cleanup ran). The run row is now marked failed;"
        f" check `journalctl --user -u {unit}` for what happened."
        " start_research can launch a fresh run in this thread."
    )


class GpuRunReaper:
    """Periodically finalize GPU run rows whose pipeline unit died.

    ``unit_active`` maps a thread_ts to whether its transient pipeline unit
    is still active (injectable for tests; app.py passes the real
    GpuBackend.unit_active). ``clock`` is a monotonic-seconds source.
    """

    def __init__(
        self,
        store: StateStore,
        slack: SlackPoster,
        channel_id: str,
        *,
        unit_active: Callable[[str], bool] = _default_unit_active,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._slack = slack
        self._channel_id = channel_id
        self._unit_active = unit_active
        self._grace = grace_seconds
        self._interval = interval_seconds
        self._clock = clock
        # thread_ts -> monotonic time the unit was first seen dead
        self._first_dead: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle (ApprovalsBridge pattern) --------------------------------

    def start(self) -> None:
        """Start the daemon reaper thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-run-reaper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — the reaper thread must never die
                logger.exception("run reaper tick failed")
            self._stop.wait(self._interval)

    # -- one pass ------------------------------------------------------------

    def tick(self) -> list[str]:
        """One reap pass; returns the thread_ts of every run reaped."""
        now = self._clock()
        try:
            candidates = [
                run for run in self._store.list_runs(status="running") if run.backend == "gpu"
            ]
        except Exception:  # noqa: BLE001 — a DB hiccup retries next tick
            logger.exception("run reaper could not list running runs")
            return []
        # Forget threads that stopped being reap candidates (finalized by the
        # pipeline after all, or reaped) so a future run restarts its grace.
        live = {run.thread_ts for run in candidates}
        for thread_ts in list(self._first_dead):
            if thread_ts not in live:
                del self._first_dead[thread_ts]

        reaped: list[str] = []
        for run in candidates:
            try:
                if self._unit_active(run.thread_ts):
                    self._first_dead.pop(run.thread_ts, None)
                    continue
                first_dead = self._first_dead.setdefault(run.thread_ts, now)
                if now - first_dead < self._grace:
                    continue
                self._reap(run)
            except Exception:  # noqa: BLE001 — one bad run must not block the rest
                logger.exception("run reaper failed on thread %s (will retry)", run.thread_ts)
                continue
            self._first_dead.pop(run.thread_ts, None)
            reaped.append(run.thread_ts)
        return reaped

    def _reap(self, run: Run) -> None:
        # Note first, status flip last: the flip is what removes the run from
        # the candidate set, so a failed Slack post retries the whole reap.
        self._slack.chat_postMessage(
            channel=self._channel_id, thread_ts=run.thread_ts, text=reap_note(run)
        )
        self._store.update_run_status(run.thread_ts, "failed")
        logger.warning(
            "reaped stale GPU run for thread %s (unit %s inactive past grace)",
            run.thread_ts,
            run_lock.unit_name(run.thread_ts),
        )

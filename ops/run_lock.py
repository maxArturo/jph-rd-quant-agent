"""Global GPU run mutual exclusion (US-020).

Exactly one GPU worker exists at a time — ``gpu_worker.sh`` keys all of its
state off a single ``worker.env`` — so a second concurrent research run would
share, and at teardown DESTROY, the first run's droplet. The lock file makes
that impossible:

- ``ops/gpu_pipeline.py`` acquires the lock before doing anything and
  releases it in its ``finally`` block. A refused pipeline exits without ever
  touching the shared worker state (crucially: without running teardown).
- ``ConversationCore.start_research`` refuses to launch while another
  thread's run holds the lock; ``stop_run`` only cancels when the requesting
  thread owns it (both via ``GpuBackend.active_run_lock``).

Staleness: the lock records the owning transient unit, thread and pid. The
owner is dead when the unit is no longer active AND the pid is gone (manual
CLI runs have no real unit, so the pid check is what keeps their lock live) —
a dead owner's lock is broken automatically and the break is reported by the
caller (Slack note), so a crashed pipeline can never wedge research forever.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOCK_FILE = Path.home() / "rdq-runs" / "gpu_worker" / "run.lock"


def unit_name(thread_ts: str) -> str:
    """Transient pipeline unit for a Slack-launched run (gpu_backend convention)."""
    return "rdq-gpu-run-" + thread_ts.replace(".", "-")


@dataclass(frozen=True)
class RunLock:
    """Contents of the global run-lock file."""

    unit: str
    thread_ts: str | None = None
    pid: int | None = None
    acquired_at: str | None = None

    def describe(self) -> str:
        thread = f"thread {self.thread_ts}" if self.thread_ts else "a manual CLI run"
        return f"{thread} (unit {self.unit})"


class LockHeldError(RuntimeError):
    """A live run already holds the global lock."""

    def __init__(self, lock: RunLock) -> None:
        self.lock = lock
        super().__init__(f"GPU run lock held by {lock.describe()}")


IsActiveFn = Callable[[str], bool]


def unit_is_active(unit: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    return Path(f"/proc/{pid}").exists()


def read_lock(path: Path) -> RunLock | None:
    """The lock on disk, or None. A corrupt file reads as an unknown (and
    therefore stale — no live unit/pid) owner so it can be broken, not wedge."""
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
        return RunLock(
            unit=str(data["unit"]),
            thread_ts=data.get("thread_ts"),
            pid=data.get("pid"),
            acquired_at=data.get("acquired_at"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return RunLock(unit="unknown (corrupt lock file)")


def is_stale(lock: RunLock, is_active: IsActiveFn = unit_is_active) -> bool:
    return not is_active(lock.unit) and not _pid_alive(lock.pid)


def check_lock(
    path: Path, *, is_active: IsActiveFn = unit_is_active
) -> tuple[RunLock | None, RunLock | None]:
    """Return ``(active, broken_stale)`` — a stale lock file is removed."""
    lock = read_lock(path)
    if lock is None:
        return None, None
    if is_stale(lock, is_active):
        path.unlink(missing_ok=True)
        return None, lock
    return lock, None


def acquire_lock(
    path: Path,
    *,
    unit: str,
    thread_ts: str | None = None,
    pid: int | None = None,
    is_active: IsActiveFn = unit_is_active,
) -> RunLock | None:
    """Take the global lock; returns the stale lock broken to do so, if any.

    Raises LockHeldError when a live owner holds it. Creation is atomic
    (O_CREAT|O_EXCL), so two racing pipelines cannot both win.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    broken: RunLock | None = None
    last_seen: RunLock | None = None
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            active, stale = check_lock(path, is_active=is_active)
            if active is not None:
                raise LockHeldError(active) from None
            broken = stale or broken
            last_seen = stale or last_seen
            continue  # stale lock removed — retry the exclusive create
        with os.fdopen(fd, "w") as fh:
            json.dump(
                {
                    "unit": unit,
                    "thread_ts": thread_ts,
                    "pid": pid,
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                },
                fh,
            )
        return broken
    # Pathological: the file kept reappearing between break and create.
    raise LockHeldError(last_seen or RunLock(unit="unknown"))


def release_lock(path: Path, unit: str) -> bool:
    """Remove the lock, but only when ``unit`` still owns it (a stale-broken
    and re-acquired lock must never be deleted by the old owner)."""
    lock = read_lock(path)
    if lock is None or lock.unit != unit:
        return False
    path.unlink(missing_ok=True)
    return True

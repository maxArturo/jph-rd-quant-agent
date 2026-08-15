"""US-020: global GPU run mutual exclusion (ops/run_lock.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ops.run_lock import (
    DEFAULT_LOCK_FILE,
    LockHeldError,
    RunLock,
    acquire_lock,
    check_lock,
    is_stale,
    read_lock,
    release_lock,
    unit_name,
)


def never_active(unit: str) -> bool:
    del unit
    return False


def always_active(unit: str) -> bool:
    del unit
    return True


class TestUnitName:
    def test_matches_gpu_backend_convention(self) -> None:
        # gpu_backend's transient units are named from the same function.
        from orchestrator.gpu_backend import _unit_name

        assert unit_name("123.456") == "rdq-gpu-run-123-456"
        assert _unit_name is unit_name

    def test_default_lock_lives_in_gpu_worker_state(self) -> None:
        assert DEFAULT_LOCK_FILE.name == "run.lock"
        assert DEFAULT_LOCK_FILE.parent == Path.home() / "rdq-runs" / "gpu_worker"


class TestAcquireRelease:
    def test_acquire_writes_owner_and_returns_no_break(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        broken = acquire_lock(
            path, unit="rdq-gpu-run-1-2", thread_ts="1.2", pid=os.getpid()
        )
        assert broken is None
        lock = read_lock(path)
        assert lock is not None
        assert lock.unit == "rdq-gpu-run-1-2"
        assert lock.thread_ts == "1.2"
        assert lock.pid == os.getpid()
        assert lock.acquired_at  # stamped

    def test_second_acquire_refused_while_owner_lives(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        acquire_lock(path, unit="rdq-gpu-run-1-2", thread_ts="1.2", pid=os.getpid())
        with pytest.raises(LockHeldError) as exc:
            acquire_lock(
                path,
                unit="rdq-gpu-run-9-9",
                thread_ts="9.9",
                pid=os.getpid(),
                is_active=never_active,
            )
        assert exc.value.lock.thread_ts == "1.2"
        assert "1.2" in str(exc.value)
        # The original owner's lock is untouched.
        lock = read_lock(path)
        assert lock is not None and lock.thread_ts == "1.2"

    def test_active_unit_keeps_lock_live_without_pid(self, tmp_path: Path) -> None:
        # Owner pid gone but its unit still active (e.g. lock read from
        # another process): not stale.
        path = tmp_path / "run.lock"
        path.write_text(json.dumps({"unit": "rdq-gpu-run-1-2", "thread_ts": "1.2"}))
        with pytest.raises(LockHeldError):
            acquire_lock(path, unit="other", is_active=always_active)

    def test_release_removes_own_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        acquire_lock(path, unit="rdq-gpu-run-1-2", thread_ts="1.2", pid=os.getpid())
        assert release_lock(path, "rdq-gpu-run-1-2") is True
        assert not path.exists()

    def test_release_refuses_foreign_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        acquire_lock(path, unit="rdq-gpu-run-1-2", thread_ts="1.2", pid=os.getpid())
        assert release_lock(path, "rdq-gpu-run-9-9") is False
        assert path.exists()

    def test_release_of_missing_lock_is_noop(self, tmp_path: Path) -> None:
        assert release_lock(tmp_path / "run.lock", "any") is False


class TestStaleBreak:
    def test_dead_owner_is_broken_and_reacquired(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        # Unit inactive AND pid absent -> dead owner.
        path.write_text(
            json.dumps({"unit": "rdq-gpu-run-1-2", "thread_ts": "1.2", "pid": None})
        )
        broken = acquire_lock(
            path,
            unit="rdq-gpu-run-9-9",
            thread_ts="9.9",
            pid=os.getpid(),
            is_active=never_active,
        )
        assert broken is not None and broken.thread_ts == "1.2"
        lock = read_lock(path)
        assert lock is not None and lock.thread_ts == "9.9"

    def test_live_pid_keeps_manual_lock_live(self, tmp_path: Path) -> None:
        # Manual CLI runs have no real unit — the pid is what keeps them live.
        path = tmp_path / "run.lock"
        path.write_text(json.dumps({"unit": "manual-1", "pid": os.getpid()}))
        lock = read_lock(path)
        assert lock is not None
        assert is_stale(lock, is_active=never_active) is False

    def test_corrupt_lock_file_reads_as_breakable(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        path.write_text("{not json")
        lock = read_lock(path)
        assert lock is not None and "corrupt" in lock.unit
        active, stale = check_lock(path, is_active=never_active)
        assert active is None
        assert stale is not None
        assert not path.exists()

    def test_check_lock_reports_active_owner(self, tmp_path: Path) -> None:
        path = tmp_path / "run.lock"
        acquire_lock(path, unit="rdq-gpu-run-1-2", thread_ts="1.2", pid=os.getpid())
        active, stale = check_lock(path, is_active=never_active)
        assert stale is None
        assert active is not None and active.thread_ts == "1.2"

    def test_check_lock_empty_path(self, tmp_path: Path) -> None:
        assert check_lock(tmp_path / "run.lock", is_active=never_active) == (None, None)


class TestDescribe:
    def test_thread_owner(self) -> None:
        lock = RunLock(unit="rdq-gpu-run-1-2", thread_ts="1.2")
        assert "thread 1.2" in lock.describe()
        assert "rdq-gpu-run-1-2" in lock.describe()

    def test_manual_owner(self) -> None:
        assert "manual CLI run" in RunLock(unit="manual-77").describe()

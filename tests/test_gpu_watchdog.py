"""Offline tests for the orphaned-GPU-worker watchdog (ops/gpu_watchdog.py)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ops.gpu_watchdog as gpu_watchdog
from ops.gpu_watchdog import droplet_age_hours, read_state

NOW = datetime(2026, 8, 6, 20, 0, 0, tzinfo=timezone.utc)


def write_state(tmp_path: Path, created_at: str = "2026-08-06T10:00:00Z") -> Path:
    state = tmp_path / "worker.env"
    state.write_text(f"DROPLET_ID=12345\nDROPLET_IP=203.0.113.9\nSIZE=gpu-x\nCREATED_AT={created_at}\n")
    return state


def fake_doctl(monkeypatch, returncode: int) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="")

    monkeypatch.setattr(gpu_watchdog.subprocess, "run", run)
    monkeypatch.setattr(gpu_watchdog.shutil, "which", lambda _cmd: "/usr/bin/doctl")
    return calls


def fake_worker_sh(monkeypatch, returncodes: dict[str, int]) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def worker_sh(*args: str):
        calls.append(args)
        rc = returncodes.get(args[0], 0)
        return subprocess.CompletedProcess(args, rc, stdout="", stderr="")

    monkeypatch.setattr(gpu_watchdog, "worker_sh", worker_sh)
    return calls


class TestHelpers:
    def test_read_state(self, tmp_path: Path) -> None:
        state = read_state(write_state(tmp_path))
        assert state["DROPLET_ID"] == "12345"
        assert state["SIZE"] == "gpu-x"

    def test_droplet_age_hours(self) -> None:
        assert droplet_age_hours("2026-08-06T10:00:00Z", NOW) == 10.0
        assert droplet_age_hours("garbage", NOW) is None


class TestMain:
    def test_no_state_file_is_quiet_success(self, tmp_path: Path) -> None:
        rc = gpu_watchdog.main(["--state-file", str(tmp_path / "absent.env"), "--no-slack"])
        assert rc == 0

    def test_missing_doctl_is_loud_failure(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """No doctl on PATH -> notify + exit 1, never FileNotFoundError (US-012)."""
        state = write_state(tmp_path)
        monkeypatch.setattr(gpu_watchdog.shutil, "which", lambda _cmd: None)
        rc = gpu_watchdog.main(["--state-file", str(state), "--no-slack"])
        assert rc == 1
        assert "doctl" in capsys.readouterr().err
        assert state.exists()  # state kept so the next healthy tick re-checks

    def test_gone_droplet_cleans_stale_state(self, tmp_path: Path, monkeypatch) -> None:
        state = write_state(tmp_path)
        fake_doctl(monkeypatch, returncode=1)  # doctl get: not found
        rc = gpu_watchdog.main(["--state-file", str(state), "--no-slack"])
        assert rc == 0
        assert not state.exists()

    def test_overage_fetches_and_destroys(self, tmp_path: Path, monkeypatch) -> None:
        # Created 10h before NOW; max 2h -> destroy. Freeze "now" via CREATED_AT
        # far in the past instead of patching datetime.
        state = write_state(tmp_path, created_at="2020-01-01T00:00:00Z")
        fake_doctl(monkeypatch, returncode=0)  # droplet exists
        calls = fake_worker_sh(monkeypatch, {"destroy": 0})
        rc = gpu_watchdog.main(["--state-file", str(state), "--max-hours", "2", "--no-slack"])
        assert rc == 0
        assert ("fetch",) in calls
        assert ("destroy", "--force") in calls

    def test_failed_destroy_is_loud_failure(self, tmp_path: Path, monkeypatch) -> None:
        state = write_state(tmp_path, created_at="2020-01-01T00:00:00Z")
        fake_doctl(monkeypatch, returncode=0)
        fake_worker_sh(monkeypatch, {"destroy": 1})
        rc = gpu_watchdog.main(["--state-file", str(state), "--max-hours", "2", "--no-slack"])
        assert rc == 1

    def test_fresh_droplet_dead_run_warns_only(self, tmp_path: Path, monkeypatch, capsys) -> None:
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state = write_state(tmp_path, created_at=recent)
        fake_doctl(monkeypatch, returncode=0)
        calls = fake_worker_sh(monkeypatch, {"ssh": 1})  # tmux session dead
        rc = gpu_watchdog.main(["--state-file", str(state), "--no-slack"])
        assert rc == 0
        assert ("destroy", "--force") not in calls
        assert "NO active research run" in capsys.readouterr().err

    def test_fresh_droplet_live_run_is_silent(self, tmp_path: Path, monkeypatch, capsys) -> None:
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state = write_state(tmp_path, created_at=recent)
        fake_doctl(monkeypatch, returncode=0)
        fake_worker_sh(monkeypatch, {"ssh": 0})  # tmux session alive
        rc = gpu_watchdog.main(["--state-file", str(state), "--no-slack"])
        assert rc == 0
        assert capsys.readouterr().err == ""

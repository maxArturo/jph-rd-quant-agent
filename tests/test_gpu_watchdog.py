"""Offline tests for the orphaned-GPU-worker watchdog (ops/gpu_watchdog.py)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import ops.gpu_watchdog as gpu_watchdog
from ops.gpu_watchdog import droplet_age_hours, notify, owning_channel, read_state

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


class TestOwningChannel:
    def test_reads_channel_from_status_file(self, tmp_path: Path) -> None:
        status = tmp_path / "pipeline_status.json"
        status.write_text(json.dumps({"thread_ts": "1.2", "channel": "C_LIVE"}))
        assert owning_channel(status) == "C_LIVE"

    def test_unknown_owner_is_none(self, tmp_path: Path) -> None:
        assert owning_channel(tmp_path / "absent.json") is None
        junk = tmp_path / "junk.json"
        junk.write_text("not json")
        assert owning_channel(junk) is None
        no_channel = tmp_path / "no_channel.json"
        no_channel.write_text(json.dumps({"thread_ts": "1.2", "channel": None}))
        assert owning_channel(no_channel) is None


class RecordingWebClient:
    instances: list[RecordingWebClient] = []

    def __init__(self, token: str) -> None:
        self.token = token
        self.proxy: str | None = "preset"
        self.posts: list[dict[str, Any]] = []
        RecordingWebClient.instances.append(self)

    def chat_postMessage(self, **kwargs: Any) -> None:  # noqa: N802
        self.posts.append(kwargs)


class TestNotifyRouting:
    def test_posts_to_owning_channel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from orchestrator.config import SlackConfig

        RecordingWebClient.instances = []
        monkeypatch.setattr("slack_sdk.WebClient", RecordingWebClient)
        monkeypatch.setattr(
            "orchestrator.config.load_slack_config",
            lambda: SlackConfig(bot_token="xoxb-1", app_token="xapp-1", channel_id="C_PAPER"),
        )
        notify("alert", False, "C_LIVE")
        client = RecordingWebClient.instances[0]
        assert client.posts == [{"channel": "C_LIVE", "text": "alert"}]
        assert client.proxy is None

    def test_unknown_owner_falls_back_to_paper_notifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[str] = []
        monkeypatch.setattr(
            "execution.rebalance.slack_notifier", lambda live=False: sent.append
        )
        notify("alert", False, None)
        assert sent == ["alert"]

    def test_main_routes_alerts_to_the_owning_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dead-run warning goes to the channel recorded in the status file."""
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state = write_state(tmp_path, created_at=recent)
        status = tmp_path / "pipeline_status.json"
        status.write_text(json.dumps({"thread_ts": "1.2", "channel": "C_LIVE"}))
        fake_doctl(monkeypatch, returncode=0)
        fake_worker_sh(monkeypatch, {"ssh": 1})  # tmux session dead
        notices: list[tuple[str, bool, str | None]] = []
        monkeypatch.setattr(
            gpu_watchdog,
            "notify",
            lambda text, no_slack, channel=None: notices.append((text, no_slack, channel)),
        )

        rc = gpu_watchdog.main(["--state-file", str(state), "--status-file", str(status)])

        assert rc == 0
        assert len(notices) == 1
        assert notices[0][2] == "C_LIVE"

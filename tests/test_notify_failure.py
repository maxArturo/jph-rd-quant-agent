"""Tests for ops/notify_failure.py — the OnFailure Slack notifier (US-018)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from ops import notify_failure


class TestJournalTail:
    def test_asks_journalctl_for_the_unit_tail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="line1\nline2\n", stderr="")

        monkeypatch.setattr(notify_failure.subprocess, "run", fake_run)
        tail = notify_failure.journal_tail("rdq-rebalance.service")
        assert tail == "line1\nline2"
        (cmd,) = calls
        assert cmd[:4] == ["journalctl", "--user", "-u", "rdq-rebalance.service"]
        assert "-n" in cmd and cmd[cmd.index("-n") + 1] == "10"

    def test_journalctl_failure_becomes_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no journal"),
        )
        assert notify_failure.journal_tail("x.service") == "(journalctl failed: no journal)"

    def test_missing_journalctl_becomes_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_missing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("journalctl")

        monkeypatch.setattr(notify_failure.subprocess, "run", raise_missing)
        assert notify_failure.journal_tail("x.service").startswith("(journal unavailable:")

    def test_empty_journal_becomes_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr=""),
        )
        assert notify_failure.journal_tail("x.service") == "(journal empty)"

    def test_oversized_tail_keeps_the_newest_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        long_out = "old " * 2000 + "NEWEST"
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=long_out, stderr=""),
        )
        tail = notify_failure.journal_tail("x.service")
        assert len(tail) <= notify_failure.MAX_TAIL_CHARS + 1
        assert tail.endswith("NEWEST")
        assert tail.startswith("…")


class TestBuildMessage:
    def test_names_the_failed_unit_and_carries_the_tail(self) -> None:
        message = notify_failure.build_message("rdq-rebalance.service", "boom line")
        assert "unit rdq-rebalance.service failed" in message
        assert "```\nboom line\n```" in message
        assert "journalctl --user -u rdq-rebalance.service" in message


class TestMain:
    def test_posts_via_slack_notifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posted: list[str] = []
        monkeypatch.setattr(
            "execution.rebalance.slack_notifier", lambda: posted.append
        )
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="tail\n", stderr=""),
        )
        assert notify_failure.main(["rdq-sweep.service"]) == 0
        (message,) = posted
        assert "unit rdq-sweep.service failed" in message
        assert "tail" in message

    def test_no_slack_prints_only(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def explode() -> None:
            raise AssertionError("slack_notifier must not be called with --no-slack")

        monkeypatch.setattr("execution.rebalance.slack_notifier", explode)
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="tail\n", stderr=""),
        )
        assert notify_failure.main(["rdq-sweep.service", "--no-slack"]) == 0
        assert "unit rdq-sweep.service failed" in capsys.readouterr().err

    def test_slack_failure_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken_notifier() -> Any:
            raise RuntimeError("no tokens")

        monkeypatch.setattr("execution.rebalance.slack_notifier", broken_notifier)
        monkeypatch.setattr(
            notify_failure.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="tail\n", stderr=""),
        )
        assert notify_failure.main(["rdq-sweep.service"]) == 1
        assert "slack notice failed" in capsys.readouterr().err

    def test_lines_flag_reaches_journalctl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="t\n", stderr="")

        monkeypatch.setattr(notify_failure.subprocess, "run", fake_run)
        assert notify_failure.main(["u.service", "--lines", "25", "--no-slack"]) == 0
        (cmd,) = calls
        assert cmd[cmd.index("-n") + 1] == "25"

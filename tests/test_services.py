"""Offline tests for the systemd units (US-010, US-018) + install script."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT = REPO_ROOT / "ops" / "rdq-orchestrator.service"
RESEARCH_UNIT = REPO_ROOT / "ops" / "rdq-research.service"
INSTALL = REPO_ROOT / "ops" / "install_services.sh"


def _systemd_analyze_verify(unit: Path) -> None:
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    result = subprocess.run(
        ["systemd-analyze", "--user", "verify", str(unit)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if "Failed to connect" in result.stderr:
        pytest.skip("no systemd user manager available")
    assert result.returncode == 0, result.stdout + result.stderr


class TestOrchestratorUnit:
    def test_exists(self) -> None:
        assert UNIT.is_file()

    def test_runs_bot_under_onecli_identity(self) -> None:
        text = UNIT.read_text()
        assert "onecli run --agent rdq-orchestrator" in text
        assert "python -m orchestrator.app" in text

    def test_restart_always(self) -> None:
        assert "Restart=always" in UNIT.read_text()

    def test_slack_bypasses_onecli_proxy(self) -> None:
        """docs/decisions.md: Slack is never routed through the OneCLI proxy.

        onecli run injects HTTPS_PROXY process-wide, so the unit must exempt
        slack.com (urllib suffix-matches NO_PROXY, covering *.slack.com).
        """
        text = UNIT.read_text()
        assert 'Environment="NO_PROXY=slack.com" "no_proxy=slack.com"' in text

    def test_installable_user_unit(self) -> None:
        text = UNIT.read_text()
        assert "[Install]" in text
        assert "WantedBy=default.target" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(UNIT)


class TestResearchUnit:
    def test_exists(self) -> None:
        assert RESEARCH_UNIT.is_file()

    def test_runs_server_ui_under_onecli_identity(self) -> None:
        text = RESEARCH_UNIT.read_text()
        assert "onecli run --agent rdq-research" in text
        assert "python -m research.server_ui" in text

    def test_restart_always(self) -> None:
        assert "Restart=always" in RESEARCH_UNIT.read_text()

    def test_no_proxy_covers_loopback_and_tailnet(self) -> None:
        """AC: NO_PROXY covers 127.0.0.1, localhost, and the tailnet range —
        loopback control traffic must not transit the OneCLI proxy."""
        text = RESEARCH_UNIT.read_text()
        assert (
            'Environment="NO_PROXY=127.0.0.1,localhost,100.64.0.0/10"'
            ' "no_proxy=127.0.0.1,localhost,100.64.0.0/10"'
        ) in text

    def test_documents_no_tailscale_exposure(self) -> None:
        """AC: the unit comments must state it is NOT exposed via tailscale
        serve (PLAN.md port table: server_ui stays dark)."""
        text = RESEARCH_UNIT.read_text()
        assert "NOT exposed via tailscale serve" in text
        assert "PLAN.md" in text

    def test_state_dirs_outside_repo(self) -> None:
        text = RESEARCH_UNIT.read_text()
        assert 'Environment="UI_TRACE_FOLDER=%h/rdq-runs/server_ui/traces"' in text

    def test_installable_user_unit(self) -> None:
        text = RESEARCH_UNIT.read_text()
        assert "[Install]" in text
        assert "WantedBy=default.target" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(RESEARCH_UNIT)


class TestInstallScript:
    def test_exists_and_executable(self) -> None:
        assert INSTALL.is_file()
        assert os.access(INSTALL, os.X_OK), "script must be executable"

    def test_links_units_and_reloads(self) -> None:
        text = INSTALL.read_text()
        assert "rdq-orchestrator.service" in text
        assert "rdq-research.service" in text
        assert ".config/systemd/user" in text
        assert "daemon-reload" in text

    def test_every_listed_unit_file_exists(self) -> None:
        """UNITS entries must point at real files in ops/ (catches rename drift)."""
        in_units = False
        listed: list[str] = []
        for line in INSTALL.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("UNITS=("):
                in_units = True
                continue
            if in_units:
                if stripped == ")":
                    break
                if stripped and not stripped.startswith("#"):
                    listed.append(stripped)
        assert listed, "UNITS array should list at least one unit"
        for unit in listed:
            assert (REPO_ROOT / "ops" / unit).is_file(), f"missing unit file: {unit}"

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shellcheck_clean(self) -> None:
        result = subprocess.run(
            ["shellcheck", str(INSTALL)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

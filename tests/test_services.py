"""Offline tests for the US-010 orchestrator systemd unit + install script."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT = REPO_ROOT / "ops" / "rdq-orchestrator.service"
INSTALL = REPO_ROOT / "ops" / "install_services.sh"


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
        env = dict(os.environ)
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        result = subprocess.run(
            ["systemd-analyze", "--user", "verify", str(UNIT)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if "Failed to connect" in result.stderr:
            pytest.skip("no systemd user manager available")
        assert result.returncode == 0, result.stdout + result.stderr


class TestInstallScript:
    def test_exists_and_executable(self) -> None:
        assert INSTALL.is_file()
        assert os.access(INSTALL, os.X_OK), "script must be executable"

    def test_links_units_and_reloads(self) -> None:
        text = INSTALL.read_text()
        assert "rdq-orchestrator.service" in text
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

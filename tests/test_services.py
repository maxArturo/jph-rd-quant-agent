"""Offline tests for the systemd units (US-010, US-018, US-020, US-036) + install script."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT = REPO_ROOT / "ops" / "rdq-orchestrator.service"
RESEARCH_UNIT = REPO_ROOT / "ops" / "rdq-research.service"
REFRESH_UNIT = REPO_ROOT / "ops" / "rdq-data-refresh.service"
REFRESH_TIMER = REPO_ROOT / "ops" / "rdq-data-refresh.timer"
REBALANCE_UNIT = REPO_ROOT / "ops" / "rdq-rebalance.service"
REBALANCE_TIMER = REPO_ROOT / "ops" / "rdq-rebalance.timer"
INSTALL = REPO_ROOT / "ops" / "install_services.sh"
RUN_US_QUANT = REPO_ROOT / "ops" / "run_us_quant.sh"


def timer_schedule(timer: Path) -> tuple[str, str]:
    """(day spec, HH:MM) from the timer's OnCalendar= line."""
    match = re.search(
        r"^OnCalendar=(\S+) (\d{2}:\d{2}) America/New_York$", timer.read_text(), re.MULTILINE
    )
    assert match, f"{timer.name} needs 'OnCalendar=<days> HH:MM America/New_York'"
    return match.group(1), match.group(2)


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

    def test_us_run_env_wiring(self) -> None:
        """US-020: fin_quant runs spawned via /upload must inherit the US-market
        environment, or they silently backtest with rdagent's CN defaults."""
        text = RESEARCH_UNIT.read_text()
        assert "FACTOR_CoSTEER_DATA_FOLDER=%h/rdq-data/factor_source/us_liquid/data_folder" in text
        assert (
            "FACTOR_CoSTEER_DATA_FOLDER_DEBUG="
            "%h/rdq-data/factor_source/us_liquid/data_folder_debug"
        ) in text
        assert "APP_TPL=%h/rd-agent-q/research/app_tpl" in text
        assert (
            "QLIB_QUANT_FACTOR_HYPOTHESIS2EXPERIMENT="
            "research.us_quant.USQlibFactorHypothesis2Experiment"
        ) in text
        assert (
            "QLIB_QUANT_MODEL_HYPOTHESIS2EXPERIMENT="
            "research.us_quant.USQlibModelHypothesis2Experiment"
        ) in text
        assert "WORKSPACE_PATH=%h/rdq-runs/server_ui/workspace" in text
        # LLM backend env for spawned runs; NOT optional (no '-' prefix) so a
        # missing file fails loudly instead of falling back to OpenAI defaults.
        assert "EnvironmentFile=%h/rd-agent-q/research/.env" in text
        assert "EnvironmentFile=-" not in text

    def test_unit_dates_match_run_us_quant_defaults(self) -> None:
        """The unit duplicates wire_env's date defaults (all three prefixes);
        this catches drift between ops/run_us_quant.sh and the unit."""
        script_defaults = dict(
            re.findall(r"RDQ_((?:TRAIN|VALID|TEST)_(?:START|END)):-(\d{4}-\d{2}-\d{2})",
                       RUN_US_QUANT.read_text())
        )
        assert len(script_defaults) == 6, script_defaults
        unit_text = RESEARCH_UNIT.read_text()
        for prefix in ("QLIB_QUANT", "QLIB_FACTOR", "QLIB_MODEL"):
            for segment, date in script_defaults.items():
                assert f'"{prefix}_{segment}={date}"' in unit_text, (
                    f"{prefix}_{segment} missing or out of sync with "
                    f"run_us_quant.sh default {date}"
                )

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(RESEARCH_UNIT)


class TestRefreshUnits:
    def test_exist(self) -> None:
        assert REFRESH_UNIT.is_file()
        assert REFRESH_TIMER.is_file()

    def test_runs_refresh_under_exec_paper_identity(self) -> None:
        """AC: units run as rdq-exec-paper (FMP key injected by the proxy)."""
        text = REFRESH_UNIT.read_text()
        assert "onecli run --agent rdq-exec-paper" in text
        assert "python -m data.refresh" in text
        assert "Type=oneshot" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text

    def test_timer_weekday_preopen_new_york(self) -> None:
        """AC: explicit America/New_York handling, scheduled before market open."""
        days, hhmm = timer_schedule(REFRESH_TIMER)
        assert days == "Mon..Fri"
        assert hhmm < "09:30"
        # missed refreshes are harmless to catch up (incremental + idempotent)
        assert "Persistent=true" in REFRESH_TIMER.read_text()
        assert "WantedBy=timers.target" in REFRESH_TIMER.read_text()

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(REFRESH_UNIT)
        _systemd_analyze_verify(REFRESH_TIMER)


class TestRebalanceUnits:
    def test_exist(self) -> None:
        assert REBALANCE_UNIT.is_file()
        assert REBALANCE_TIMER.is_file()

    def test_runs_rebalance_under_exec_paper_identity(self) -> None:
        text = REBALANCE_UNIT.read_text()
        assert "onecli run --agent rdq-exec-paper" in text
        assert "python -m execution.rebalance" in text
        assert "Type=oneshot" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text

    def test_slack_bypasses_onecli_proxy(self) -> None:
        """The rebalancer posts abort notices + the daily summary to Slack,
        which must never transit the OneCLI proxy (docs/decisions.md)."""
        text = REBALANCE_UNIT.read_text()
        assert 'Environment="NO_PROXY=slack.com" "no_proxy=slack.com"' in text

    def test_timer_weekday_preopen_new_york(self) -> None:
        days, hhmm = timer_schedule(REBALANCE_TIMER)
        assert days == "Mon..Fri"
        assert hhmm < "09:30"
        # a rebalance missed while the box was down must be skipped, not
        # fired at an arbitrary later time of day
        assert "Persistent=false" in REBALANCE_TIMER.read_text()
        assert "WantedBy=timers.target" in REBALANCE_TIMER.read_text()

    def test_refresh_scheduled_before_rebalance(self) -> None:
        """AC ordering: refresh runs first so the rebalance prices off a store
        that already holds the previous session's bars."""
        _, refresh_time = timer_schedule(REFRESH_TIMER)
        _, rebalance_time = timer_schedule(REBALANCE_TIMER)
        assert refresh_time < rebalance_time

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(REBALANCE_UNIT)
        _systemd_analyze_verify(REBALANCE_TIMER)


class TestInstallScript:
    def test_exists_and_executable(self) -> None:
        assert INSTALL.is_file()
        assert os.access(INSTALL, os.X_OK), "script must be executable"

    def test_links_units_and_reloads(self) -> None:
        text = INSTALL.read_text()
        assert "rdq-orchestrator.service" in text
        assert "rdq-research.service" in text
        assert "rdq-data-refresh.service" in text
        assert "rdq-data-refresh.timer" in text
        assert "rdq-rebalance.service" in text
        assert "rdq-rebalance.timer" in text
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

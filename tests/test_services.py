"""Offline tests for the systemd units (US-010, US-018, US-020, US-036, US-041) + install script."""

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
PRED_REFRESH_UNIT = REPO_ROOT / "ops" / "rdq-pred-refresh.service"
PRED_REFRESH_TIMER = REPO_ROOT / "ops" / "rdq-pred-refresh.timer"
REBALANCE_UNIT = REPO_ROOT / "ops" / "rdq-rebalance.service"
REBALANCE_TIMER = REPO_ROOT / "ops" / "rdq-rebalance.timer"
SWEEP_UNIT = REPO_ROOT / "ops" / "rdq-sweep.service"
SWEEP_TIMER = REPO_ROOT / "ops" / "rdq-sweep.timer"
GPU_WATCHDOG_UNIT = REPO_ROOT / "ops" / "rdq-gpu-watchdog.service"
GPU_WATCHDOG_TIMER = REPO_ROOT / "ops" / "rdq-gpu-watchdog.timer"
DIVERGENCE_UNIT = REPO_ROOT / "ops" / "rdq-divergence.service"
DIVERGENCE_TIMER = REPO_ROOT / "ops" / "rdq-divergence.timer"
NOTIFY_TEMPLATE = REPO_ROOT / "ops" / "rdq-notify-failure@.service"
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

    def test_slack_and_local_control_plane_bypass_onecli_proxy(self) -> None:
        """docs/decisions.md: Slack is never routed through the OneCLI proxy.

        onecli run injects HTTP(S)_PROXY process-wide, so the unit must exempt
        slack.com (urllib suffix-matches NO_PROXY, covering *.slack.com) and
        the local control-plane hosts (rdagent server_ui :19899, OneCLI
        management/approvals :10254/:10255 — US-039).
        """
        text = UNIT.read_text()
        for var in ("NO_PROXY", "no_proxy"):
            assert f'"{var}=slack.com,127.0.0.1,localhost"' in text

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

    def test_execution_env_wiring(self) -> None:
        """US-043: no conda on this box — backtests/model training must opt
        into docker, and generated factor code must run with the venv python.
        The unit and run_us_quant.sh wire_env must stay in sync."""
        unit_text = RESEARCH_UNIT.read_text()
        assert 'Environment="MODEL_CoSTEER_ENV_TYPE=docker"' in unit_text
        assert (
            'Environment="FACTOR_CoSTEER_PYTHON_BIN=%h/rd-agent-q/.venv/bin/python"'
            in unit_text
        )
        assert 'Environment="QLIB_DOCKER_BUILD_FROM_DOCKERFILE=false"' in unit_text
        assert 'Environment="LITELLM_DROP_PARAMS=true"' in unit_text
        # MLflow >= 3.6 in local_qlib:latest refuses qlib's ./mlruns file
        # store without this opt-out (QlibDockerConf.env_dict JSON).
        assert (
            'Environment="QLIB_DOCKER_ENV_DICT='
            '{\\"MLFLOW_ALLOW_FILE_STORE\\":\\"true\\"}"'
        ) in unit_text
        script_text = RUN_US_QUANT.read_text()
        assert 'export MODEL_CoSTEER_ENV_TYPE="docker"' in script_text
        assert 'export FACTOR_CoSTEER_PYTHON_BIN="${PYTHON}"' in script_text
        assert 'export QLIB_DOCKER_BUILD_FROM_DOCKERFILE="false"' in script_text
        assert 'export LITELLM_DROP_PARAMS="true"' in script_text
        assert (
            "export QLIB_DOCKER_ENV_DICT="
            "'{\"MLFLOW_ALLOW_FILE_STORE\":\"true\"}'"
        ) in script_text

    def test_unit_dates_match_run_us_quant_defaults(self) -> None:
        """The unit duplicates wire_env's date defaults (all three prefixes);
        this catches drift between ops/run_us_quant.sh and the unit.

        Since US-008 the hardcoded TEST_END is a FALLBACK only: GPU-path runs
        get a launch-computed rolling RDQ_TEST_END from ops/gpu_pipeline.py,
        and run_us_quant.sh refuses any test end more than 90 calendar days
        behind the store calendar end (see test_stale_test_end_guard_present)."""
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

    def test_stale_test_end_guard_present(self) -> None:
        """US-008: the fallback TEST_END rots as the store rolls forward —
        run_us_quant.sh must carry the 90-day store-lag guard in both modes
        so a stale window fails loud instead of backtesting ancient data."""
        text = RUN_US_QUANT.read_text()
        assert "MAX_TEST_END_LAG_DAYS=90" in text
        assert "behind the store calendar end" in text
        assert text.count("check_test_end_lag") >= 3  # definition + both modes

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


class TestPredRefreshUnits:
    def test_exist(self) -> None:
        assert PRED_REFRESH_UNIT.is_file()
        assert PRED_REFRESH_TIMER.is_file()

    def test_runs_local_refresh_oneshot(self) -> None:
        """US-048: purely local work (docker + filesystem + SQLite read) — no
        onecli wrapper; timer-driven oneshot convention (enable the timer)."""
        text = PRED_REFRESH_UNIT.read_text()
        assert "python -m execution.pred_refresh" in text
        assert "onecli run" not in text
        assert "Type=oneshot" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text
        assert "[Install]" not in [line.strip() for line in text.splitlines()]

    def test_ordered_after_data_refresh(self) -> None:
        """US-048: the refresh trains on the store the data refresh just
        advanced — systemd ordering covers the overlap case."""
        after = re.search(r"^After=(.+)$", PRED_REFRESH_UNIT.read_text(), re.MULTILINE)
        assert after and "rdq-data-refresh.service" in after.group(1)

    def test_timer_between_data_refresh_and_rebalance(self) -> None:
        days, hhmm = timer_schedule(PRED_REFRESH_TIMER)
        assert days == "Mon..Fri"
        _, data_refresh_time = timer_schedule(REFRESH_TIMER)
        _, rebalance_time = timer_schedule(REBALANCE_TIMER)
        assert data_refresh_time < hhmm < rebalance_time
        # a missed refresh is harmless to catch up (short-circuits when fresh)
        assert "Persistent=true" in PRED_REFRESH_TIMER.read_text()
        assert "WantedBy=timers.target" in PRED_REFRESH_TIMER.read_text()

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(PRED_REFRESH_UNIT)
        _systemd_analyze_verify(PRED_REFRESH_TIMER)


class TestRebalanceUnits:
    def test_exist(self) -> None:
        assert REBALANCE_UNIT.is_file()
        assert REBALANCE_TIMER.is_file()

    def test_ordered_after_pred_refresh(self) -> None:
        """US-048: an in-flight prediction refresh must delay the rebalance
        rather than race it (After= holds a queued start until it finishes)."""
        after = re.search(r"^After=(.+)$", REBALANCE_UNIT.read_text(), re.MULTILINE)
        assert after and "rdq-pred-refresh.service" in after.group(1)

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


class TestSweepUnits:
    def test_exist(self) -> None:
        assert SWEEP_UNIT.is_file()
        assert SWEEP_TIMER.is_file()

    def test_runs_local_sweep_oneshot(self) -> None:
        """The sweep is purely local (filesystem + SQLite read) — no onecli
        wrapper, no injected credentials."""
        text = SWEEP_UNIT.read_text()
        assert "python -m ops.sweep" in text
        assert "onecli run" not in text
        assert "Type=oneshot" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text
        # timer-driven oneshot convention: enable the timer, not the service
        assert "[Install]" not in [line.strip() for line in text.splitlines()]

    def test_timer_weekly_new_york(self) -> None:
        """AC: weekly timer; scheduled off-market with explicit timezone."""
        days, hhmm = timer_schedule(SWEEP_TIMER)
        assert days == "Sun"
        assert hhmm < "09:30"
        # a missed sweep is harmless to catch up (deletes only weeks-old trees)
        assert "Persistent=true" in SWEEP_TIMER.read_text()
        assert "WantedBy=timers.target" in SWEEP_TIMER.read_text()

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(SWEEP_UNIT)
        _systemd_analyze_verify(SWEEP_TIMER)


class TestGpuWatchdogUnits:
    def test_exist(self) -> None:
        assert GPU_WATCHDOG_UNIT.is_file()
        assert GPU_WATCHDOG_TIMER.is_file()

    def test_path_reaches_doctl(self) -> None:
        """AC US-002: doctl lives in ~/.local/bin — without this PATH line the
        watchdog crashed with FileNotFoundError on every tick that had work."""
        assert (
            "Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin"
            in GPU_WATCHDOG_UNIT.read_text()
        )

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(GPU_WATCHDOG_UNIT)
        _systemd_analyze_verify(GPU_WATCHDOG_TIMER)


class TestDivergenceUnits:
    def test_exist(self) -> None:
        assert DIVERGENCE_UNIT.is_file()
        assert DIVERGENCE_TIMER.is_file()

    def test_runs_divergence_under_exec_paper_identity(self) -> None:
        """The tracker reads Alpaca portfolio history — same injected paper
        credentials as the rebalancer, and the same oneshot conventions."""
        text = DIVERGENCE_UNIT.read_text()
        assert "onecli run --agent rdq-exec-paper" in text
        assert "python -m execution.divergence" in text
        assert "Type=oneshot" in text
        assert "WorkingDirectory=%h/rd-agent-q" in text
        # timer-driven oneshot convention: enable the timer, not the service
        assert "[Install]" not in [line.strip() for line in text.splitlines()]

    def test_slack_bypasses_onecli_proxy(self) -> None:
        """Warn/halt/failure notices go to Slack, which must never transit
        the OneCLI proxy (docs/decisions.md)."""
        text = DIVERGENCE_UNIT.read_text()
        assert 'Environment="NO_PROXY=slack.com" "no_proxy=slack.com"' in text

    def test_ordered_after_rebalance(self) -> None:
        """AC US-017: a still-running rebalance must finish before the
        divergence check reads the book it just traded."""
        after = re.search(r"^After=(.+)$", DIVERGENCE_UNIT.read_text(), re.MULTILINE)
        assert after and "rdq-rebalance.service" in after.group(1)

    def test_timer_weekday_postclose_new_york(self) -> None:
        days, hhmm = timer_schedule(DIVERGENCE_TIMER)
        assert days == "Mon..Fri"
        assert hhmm == "16:30"  # post-close so today's session is in the history
        # safety monitor: a check missed while the box was down fires on boot
        assert "Persistent=true" in DIVERGENCE_TIMER.read_text()
        assert "WantedBy=timers.target" in DIVERGENCE_TIMER.read_text()

    def test_scheduled_after_rebalance_timer(self) -> None:
        """The daily order is rebalance (pre-open) then divergence (post-close)."""
        _, rebalance_time = timer_schedule(REBALANCE_TIMER)
        _, divergence_time = timer_schedule(DIVERGENCE_TIMER)
        assert rebalance_time < divergence_time

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(DIVERGENCE_UNIT)
        _systemd_analyze_verify(DIVERGENCE_TIMER)


class TestNotifyFailureUnits:
    """US-018: templated OnFailure Slack notifier wired into every rdq service."""

    def test_template_exists(self) -> None:
        assert NOTIFY_TEMPLATE.is_file()

    def test_runs_notify_failure_module_on_the_instance(self) -> None:
        text = NOTIFY_TEMPLATE.read_text()
        assert "Type=oneshot" in text
        assert "python -m ops.notify_failure %i" in text

    def test_template_never_recurses_or_installs(self) -> None:
        """A notifier that cannot notify must not re-trigger itself; it is
        only ever activated via OnFailure=, never enabled."""
        unit_lines = [
            line
            for line in NOTIFY_TEMPLATE.read_text().splitlines()
            if not line.startswith("#")
        ]
        assert not any(line.startswith("OnFailure=") for line in unit_lines)
        assert "[Install]" not in unit_lines

    def test_every_rdq_service_has_onfailure(self) -> None:
        services = sorted((REPO_ROOT / "ops").glob("rdq-*.service"))
        real_services = [unit for unit in services if "@" not in unit.name]
        assert len(real_services) >= 8, [unit.name for unit in services]
        for unit in real_services:
            assert re.search(
                r"^OnFailure=rdq-notify-failure@%n\.service$", unit.read_text(), re.MULTILINE
            ), f"{unit.name} missing OnFailure=rdq-notify-failure@%n.service"

    def test_installed_by_install_script(self) -> None:
        assert "rdq-notify-failure@.service" in INSTALL.read_text()

    @pytest.mark.skipif(
        shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed"
    )
    def test_systemd_analyze_verify(self) -> None:
        _systemd_analyze_verify(NOTIFY_TEMPLATE)


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
        assert "rdq-pred-refresh.service" in text
        assert "rdq-pred-refresh.timer" in text
        assert "rdq-rebalance.service" in text
        assert "rdq-rebalance.timer" in text
        assert "rdq-sweep.service" in text
        assert "rdq-sweep.timer" in text
        assert "rdq-divergence.service" in text
        assert "rdq-divergence.timer" in text
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

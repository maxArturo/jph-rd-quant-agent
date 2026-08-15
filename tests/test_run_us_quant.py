"""Offline tests for the US-017 US fin_quant run wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "run_us_quant.sh"


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, check=False, env=merged
    )


class TestScriptContract:
    def test_exists_and_executable(self) -> None:
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_wraps_onecli_fin_quant_with_env_wiring(self) -> None:
        text = SCRIPT.read_text()
        assert "onecli run --agent rdq-research" in text
        assert "fin_quant" in text
        assert "LOG_TRACE_PATH" in text
        assert "WORKSPACE_PATH" in text
        assert "APP_TPL" in text
        assert "FACTOR_CoSTEER_DATA_FOLDER" in text
        assert "FACTOR_CoSTEER_DATA_FOLDER_DEBUG" in text
        # Dates must fan out to all three settings prefixes (runners re-read
        # QLIB_FACTOR_*/QLIB_MODEL_*, only the scenario reads QLIB_QUANT_*).
        for prefix in ("QLIB_QUANT", "QLIB_FACTOR", "QLIB_MODEL"):
            assert prefix in text
        for var in (
            "TRAIN_START",
            "TRAIN_END",
            "VALID_START",
            "VALID_END",
            "TEST_START",
            "TEST_END",
        ):
            assert var in text
        assert "research.us_quant.USQlibFactorHypothesis2Experiment" in text
        assert "research.us_quant.USQlibModelHypothesis2Experiment" in text

    def test_header_documents_completion_criterion(self) -> None:
        header = "".join(
            line for line in SCRIPT.read_text().splitlines(keepends=True) if line.startswith("#")
        )
        assert "qlib_res.csv" in header
        assert "IC" in header
        assert "ARR" in header
        assert "MDD" in header
        assert "pred.pkl" in header

    def test_help_exits_zero_with_usage(self) -> None:
        result = run_script("--help")
        assert result.returncode == 0
        assert "--check" in result.stdout
        assert "--loop_n" in result.stdout
        assert "--all_duration" in result.stdout

    def test_unknown_arg_fails(self) -> None:
        result = run_script("--bogus")
        assert result.returncode != 0
        assert "unknown argument" in result.stderr

    def test_loop_n_requires_value(self) -> None:
        result = run_script("--loop_n")
        assert result.returncode != 0
        assert "--loop_n needs a value" in result.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shellcheck_clean(self) -> None:
        result = subprocess.run(
            ["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def make_fake_layout(tmp_path: Path, *, with_spy: bool = True) -> dict[str, str]:
    """Fake store + factor source satisfying --check's filesystem legs."""
    store = tmp_path / "us_data"
    (store / "calendars").mkdir(parents=True)
    (store / "calendars" / "day.txt").write_text("2026-01-02\n")
    (store / "instruments").mkdir()
    rows = ["AAPL\t2016-01-04\t2026-06-30"]
    if with_spy:
        rows.append("SPY\t2016-01-04\t2026-06-30")
    (store / "instruments" / "all.txt").write_text("\n".join(rows) + "\n")
    (store / "instruments" / "us_liquid.txt").write_text(rows[0] + "\n")

    source = tmp_path / "factor_source"
    for sub in ("data_folder", "data_folder_debug"):
        (source / sub).mkdir(parents=True)
        (source / sub / "daily_pv.h5").write_bytes(b"\x00")
        (source / sub / "README.md").write_text("fixture")

    return {"RDQ_QLIB_STORE": str(store), "RDQ_FACTOR_SOURCE": str(source)}


ONECLI_MISSING = shutil.which("onecli") is None


@pytest.mark.skipif(ONECLI_MISSING, reason="onecli not installed on this box")
class TestCheckMode:
    def test_check_passes_on_complete_layout(self, tmp_path: Path) -> None:
        result = run_script("--check", env=make_fake_layout(tmp_path))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK: environment ready" in result.stdout
        assert "PASS: research.us_quant hook classes import" in result.stdout

    def test_check_fails_naming_missing_factor_source(self, tmp_path: Path) -> None:
        env = make_fake_layout(tmp_path)
        env["RDQ_FACTOR_SOURCE"] = str(tmp_path / "nowhere")
        result = run_script("--check", env=env)
        assert result.returncode != 0
        assert "nowhere" in result.stderr
        assert "make_factor_source" in result.stderr

    def test_check_fails_without_spy_benchmark(self, tmp_path: Path) -> None:
        result = run_script("--check", env=make_fake_layout(tmp_path, with_spy=False))
        assert result.returncode != 0
        assert "SPY" in result.stderr

    def test_check_fails_on_unordered_dates(self, tmp_path: Path) -> None:
        env = make_fake_layout(tmp_path)
        env["RDQ_TEST_START"] = "2020-01-01"  # before valid_end default
        result = run_script("--check", env=env)
        assert result.returncode != 0
        assert "not strictly ordered" in result.stderr


class TestStoreLagGuard:
    """US-008: the hardcoded fallback TEST_END must fail loud once it trails
    the store calendar end by more than 90 calendar days. These run --check,
    which hits the guard before any onecli/rdagent legs."""

    def test_check_fails_when_test_end_over_90_days_behind_store(self, tmp_path: Path) -> None:
        env = make_fake_layout(tmp_path)
        calendar = Path(env["RDQ_QLIB_STORE"]) / "calendars" / "day.txt"
        calendar.write_text("2026-01-02\n2026-04-02\n")  # store end 91 days past test end
        env["RDQ_TEST_END"] = "2026-01-01"
        result = run_script("--check", env=env)
        assert result.returncode != 0
        assert "behind the store calendar end" in result.stderr
        assert "gpu_pipeline" in result.stderr  # message names the fresh-date path

    def test_check_passes_at_exactly_90_days_behind(self, tmp_path: Path) -> None:
        env = make_fake_layout(tmp_path)
        calendar = Path(env["RDQ_QLIB_STORE"]) / "calendars" / "day.txt"
        calendar.write_text("2026-01-02\n2026-04-01\n")  # store end exactly 90 days past
        env["RDQ_TEST_END"] = "2026-01-01"
        result = run_script("--check", env=env)
        assert "behind the store calendar end" not in result.stderr
        assert "PASS: TEST_END 2026-01-01 within 90 days" in result.stdout

    @pytest.mark.skipif(
        not (REPO_ROOT / "research" / ".env").exists(),
        reason="research/.env not present on this checkout",
    )
    def test_run_mode_hits_the_guard_before_launching(self, tmp_path: Path) -> None:
        """The guard must protect real launches (worker path), not just --check."""
        env = make_fake_layout(tmp_path)
        calendar = Path(env["RDQ_QLIB_STORE"]) / "calendars" / "day.txt"
        calendar.write_text("2026-04-02\n")
        env["RDQ_TEST_END"] = "2026-01-01"
        env["RDQ_RUN_ROOT"] = str(tmp_path / "runs")
        result = run_script("--loop_n", "1", env=env)
        assert result.returncode != 0
        assert "behind the store calendar end" in result.stderr
        assert "Launching" not in result.stdout


class TestDirectLauncher:
    """RDQ_LAUNCHER=direct — the GPU-worker path (ops/gpu_worker/), which runs
    without onecli and instead requires the OneCLI proxy env pre-exported."""

    def test_invalid_launcher_rejected(self) -> None:
        result = run_script("--check", env={"RDQ_LAUNCHER": "bogus"})
        assert result.returncode != 0
        assert "RDQ_LAUNCHER" in result.stderr

    def test_direct_check_requires_https_proxy(self) -> None:
        result = run_script(
            "--check", env={"RDQ_LAUNCHER": "direct", "HTTPS_PROXY": ""}
        )
        assert result.returncode != 0
        assert "HTTPS_PROXY" in result.stderr

    def test_direct_check_requires_ca_bundle(self) -> None:
        result = run_script(
            "--check",
            env={
                "RDQ_LAUNCHER": "direct",
                "HTTPS_PROXY": "http://x:token@127.0.0.1:9",
                "REQUESTS_CA_BUNDLE": "",
            },
        )
        assert result.returncode != 0
        assert "REQUESTS_CA_BUNDLE" in result.stderr

    def test_direct_check_probes_proxy_reachability(self, tmp_path: Path) -> None:
        """A dead tunnel must fail the check, without leaking the token
        (the URL userinfo must not appear in output)."""
        ca = tmp_path / "ca.pem"
        ca.write_text("fixture")
        result = run_script(
            "--check",
            env={
                "RDQ_LAUNCHER": "direct",
                "HTTPS_PROXY": "http://x:secrettoken@127.0.0.1:9",  # closed port
                "REQUESTS_CA_BUNDLE": str(ca),
            },
        )
        assert result.returncode != 0
        assert "not reachable" in result.stderr
        assert "secrettoken" not in result.stdout + result.stderr

    @pytest.mark.skipif(
        not (REPO_ROOT / "research" / ".env").exists(),
        reason="research/.env not present on this checkout",
    )
    def test_direct_run_fails_fast_without_proxy_env(self, tmp_path: Path) -> None:
        """Run mode must die on the proxy-env check BEFORE exec'ing rdagent."""
        result = run_script(
            "--loop_n",
            "1",
            env={
                "RDQ_LAUNCHER": "direct",
                "HTTPS_PROXY": "",
                "RDQ_RUN_ROOT": str(tmp_path),
            },
        )
        assert result.returncode != 0
        assert "HTTPS_PROXY" in result.stderr
        assert "Launching" not in result.stdout

"""Offline tests for ops/research_caps.sh (burst-compute caps generator).

The script owns the rdq-research resources.conf drop-in and sizes it to the
detected hardware; RDQ_CAPS_CORES / RDQ_CAPS_MEM_GB / RDQ_CAPS_FILE override
detection so every sizing rule is testable offline. Restart-path coverage
uses the PATH-shimmed stub-binary pattern from tests/test_health.py.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "research_caps.sh"

STUB_SYSTEMCTL = """#!/usr/bin/env bash
echo "systemctl $*" >> "$CAPS_STUB_DIR/calls"
exit 0
"""

STUB_DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$CAPS_STUB_DIR/calls"
if [[ -f "$CAPS_STUB_DIR/container_running" ]]; then echo "abc123"; fi
exit 0
"""

STUB_PGREP = """#!/usr/bin/env bash
echo "pgrep $*" >> "$CAPS_STUB_DIR/calls"
exit 1
"""


def run_script(
    *args: str,
    cores: int,
    mem_gb: int,
    caps_file: Path | None = None,
    stub_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RDQ_CAPS_CORES"] = str(cores)
    env["RDQ_CAPS_MEM_GB"] = str(mem_gb)
    if caps_file is not None:
        env["RDQ_CAPS_FILE"] = str(caps_file)
    if stub_dir is not None:
        env["CAPS_STUB_DIR"] = str(stub_dir)
        env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, check=False, env=env
    )


def make_stubs(stub_dir: Path, *, container_running: bool = False) -> None:
    for name, body in (
        ("systemctl", STUB_SYSTEMCTL),
        ("docker", STUB_DOCKER),
        ("pgrep", STUB_PGREP),
    ):
        stub = stub_dir / name
        stub.write_text(body)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    if container_running:
        (stub_dir / "container_running").touch()


def env_dict(conf: str) -> dict[str, str]:
    """Parse the QLIB_DOCKER_ENV_DICT JSON out of the generated drop-in."""
    match = re.search(r'QLIB_DOCKER_ENV_DICT=(\{.*\})"$', conf, re.MULTILINE)
    assert match, "QLIB_DOCKER_ENV_DICT line missing"
    return json.loads(match.group(1).replace('\\"', '"'))


class TestScriptContract:
    def test_exists_and_executable(self) -> None:
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_rejects_unknown_arguments(self) -> None:
        result = run_script("--bogus", cores=4, mem_gb=8)
        assert result.returncode == 2
        assert "unknown argument" in result.stderr


class TestSizingRules:
    def test_current_small_box_matches_hand_written_caps(self) -> None:
        """4 cores / 8G must reproduce the measured 2026-08-05 drop-in values."""
        conf = run_script("--print", cores=4, mem_gb=8).stdout
        assert "CPUWeight=40" in conf
        assert "MemoryHigh=3G" in conf
        assert '"OMP_NUM_THREADS=3"' in conf
        assert '"MKL_NUM_THREADS=3"' in conf
        assert '"OPENBLAS_NUM_THREADS=3"' in conf
        assert '"NUMEXPR_NUM_THREADS=3"' in conf
        assert "QLIB_DOCKER_MEM_LIMIT=4g" in conf
        assert "QLIB_DOCKER_SHM_SIZE=2g" in conf

    def test_burst_c16_scales_up(self) -> None:
        conf = run_script("--print", cores=16, mem_gb=32).stdout
        assert "CPUWeight=80" in conf
        assert "MemoryHigh=8G" in conf
        assert '"OMP_NUM_THREADS=14"' in conf
        assert "QLIB_DOCKER_MEM_LIMIT=28g" in conf
        assert "QLIB_DOCKER_SHM_SIZE=14g" in conf

    def test_floors_hold_on_a_tiny_box(self) -> None:
        conf = run_script("--print", cores=2, mem_gb=2).stdout
        assert '"OMP_NUM_THREADS=3"' in conf
        assert "QLIB_DOCKER_MEM_LIMIT=4g" in conf
        assert "QLIB_DOCKER_SHM_SIZE=2g" in conf
        assert "MemoryHigh=3G" in conf
        assert "CPUWeight=40" in conf

    def test_env_dict_is_valid_json_and_keeps_mlflow_opt_out(self) -> None:
        """ENV_DICT REPLACES the unit's value; losing MLFLOW_ALLOW_FILE_STORE
        kills every backtest in create_exp (see unit comments)."""
        for cores, mem_gb, threads in ((4, 8, "3"), (16, 32, "14"), (32, 64, "30")):
            parsed = env_dict(run_script("--print", cores=cores, mem_gb=mem_gb).stdout)
            assert parsed["MLFLOW_ALLOW_FILE_STORE"] == "true"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                assert parsed[key] == threads


class TestApply:
    def test_writes_drop_in_and_is_idempotent(self, tmp_path: Path) -> None:
        caps_file = tmp_path / "d" / "resources.conf"
        first = run_script("--no-restart", cores=4, mem_gb=8, caps_file=caps_file)
        assert first.returncode == 0
        assert "wrote" in first.stdout
        conf = caps_file.read_text()
        assert "GENERATED by ops/research_caps.sh" in conf
        assert conf.startswith("#")
        assert "[Service]" in conf

        second = run_script("--no-restart", cores=4, mem_gb=8, caps_file=caps_file)
        assert second.returncode == 0
        assert "unchanged" in second.stdout

    def test_no_restart_never_touches_systemd(self, tmp_path: Path) -> None:
        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()
        make_stubs(stub_dir)
        result = run_script(
            "--no-restart",
            cores=4,
            mem_gb=8,
            caps_file=tmp_path / "resources.conf",
            stub_dir=stub_dir,
        )
        assert result.returncode == 0
        calls = (stub_dir / "calls").read_text() if (stub_dir / "calls").exists() else ""
        assert "systemctl" not in calls

    def test_restart_refused_while_run_active(self, tmp_path: Path) -> None:
        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()
        make_stubs(stub_dir, container_running=True)
        result = run_script(
            cores=4, mem_gb=8, caps_file=tmp_path / "resources.conf", stub_dir=stub_dir
        )
        assert result.returncode == 1
        assert "active" in result.stderr
        calls = (stub_dir / "calls").read_text()
        assert "systemctl" not in calls, "must not restart over a live run"

    def test_force_restarts_over_active_run(self, tmp_path: Path) -> None:
        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()
        make_stubs(stub_dir, container_running=True)
        result = run_script(
            "--force",
            cores=4,
            mem_gb=8,
            caps_file=tmp_path / "resources.conf",
            stub_dir=stub_dir,
        )
        assert result.returncode == 0
        calls = (stub_dir / "calls").read_text()
        assert "systemctl --user daemon-reload" in calls
        assert "systemctl --user restart rdq-research.service" in calls

    def test_idle_box_restarts_cleanly(self, tmp_path: Path) -> None:
        stub_dir = tmp_path / "stubs"
        stub_dir.mkdir()
        make_stubs(stub_dir)
        result = run_script(
            cores=16, mem_gb=32, caps_file=tmp_path / "resources.conf", stub_dir=stub_dir
        )
        assert result.returncode == 0
        assert "restarted" in result.stdout
        calls = (stub_dir / "calls").read_text()
        assert "systemctl --user restart rdq-research.service" in calls

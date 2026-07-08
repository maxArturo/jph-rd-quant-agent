"""Offline tests for the US-005 vanilla fin_factor run wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "run_vanilla_factor.sh"


class TestScriptContract:
    def test_exists_and_executable(self) -> None:
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_wraps_onecli_fin_factor(self) -> None:
        text = SCRIPT.read_text()
        assert "onecli run --agent rdq-research" in text
        assert "fin_factor --loop_n 1" in text
        assert "LOG_TRACE_PATH" in text
        assert "WORKSPACE_PATH" in text

    def test_header_documents_slow_first_run_and_completion(self) -> None:
        header = "".join(
            line for line in SCRIPT.read_text().splitlines(keepends=True) if line.startswith("#")
        )
        assert "cn_data" in header
        assert "local_qlib:latest" in header
        assert "qlib_res.csv" in header
        assert "IC" in header

    def test_help_exits_zero_with_usage(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--help"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0
        assert "--check" in result.stdout

    def test_unknown_arg_fails(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--bogus"], capture_output=True, text=True, check=False
        )
        assert result.returncode != 0
        assert "unknown argument" in result.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shellcheck_clean(self) -> None:
        result = subprocess.run(
            ["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_rdagent_cli_importable() -> None:
    """Guard the pydantic-ai-slim pin: the full rdagent CLI graph must import.

    rdagent leaves pydantic-ai-slim unpinned; 2.x renames the MCP classes the
    pinned commit imports (see docs/decisions.md, 2026-07-08).
    """
    result = subprocess.run(
        [sys.executable, "-c", "from rdagent.app.cli import app"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Offline tests for the GPU burst-worker lifecycle script (ops/gpu_worker/)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "gpu_worker" / "gpu_worker.sh"
README = REPO_ROOT / "ops" / "gpu_worker" / "README.md"


def run_script(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
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

    def test_bash_syntax_clean(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shellcheck_clean(self) -> None:
        result = subprocess.run(
            ["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_help_lists_lifecycle(self) -> None:
        result = run_script("--help")
        assert result.returncode == 0
        for sub in ("provision", "bootstrap", "tunnel", "run", "status", "fetch", "destroy"):
            assert sub in result.stdout

    def test_missing_subcommand_fails(self) -> None:
        result = run_script()
        assert result.returncode != 0
        assert "missing subcommand" in result.stderr

    def test_unknown_subcommand_fails(self) -> None:
        result = run_script("frobnicate")
        assert result.returncode != 0
        assert "unknown subcommand" in result.stderr

    def test_worker_run_uses_direct_launcher(self) -> None:
        """The remote launch path must go through run_us_quant.sh's direct
        launcher — never a bare rdagent invocation that would skip wire_env."""
        text = SCRIPT.read_text()
        assert "RDQ_LAUNCHER=direct" in text
        assert "run_us_quant.sh" in text

    def test_tunnel_never_persists_token_locally(self) -> None:
        """The proxy URL (with auth token) may be written only to the worker;
        nothing should redirect it into the local state dir."""
        text = SCRIPT.read_text()
        assert "printenv HTTPS_PROXY" in text  # fetched fresh from onecli
        state_heredoc = text.split('cat > "${STATE_FILE}"')[1].split("EOF")[0]
        assert "proxy_url" not in state_heredoc


class TestStateHandling:
    def test_run_without_state_names_provision(self, tmp_path: Path) -> None:
        result = run_script("run", env={"RDQ_GPU_STATE_DIR": str(tmp_path)})
        assert result.returncode != 0
        assert "provision" in result.stderr

    def test_status_without_state_names_provision(self, tmp_path: Path) -> None:
        result = run_script("status", env={"RDQ_GPU_STATE_DIR": str(tmp_path)})
        assert result.returncode != 0
        assert "provision" in result.stderr

    def test_destroy_without_state_is_safe(self, tmp_path: Path) -> None:
        """destroy must refuse (not delete anything) when there is no state."""
        result = run_script("destroy", env={"RDQ_GPU_STATE_DIR": str(tmp_path)})
        assert result.returncode != 0
        assert "provision" in result.stderr

    def test_corrupt_state_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "worker.env").write_text("JUNK=1\n")
        result = run_script("status", env={"RDQ_GPU_STATE_DIR": str(tmp_path)})
        assert result.returncode != 0
        assert "corrupt state" in result.stderr

    def test_provision_requires_doctl(self, tmp_path: Path) -> None:
        """Without doctl (or unauthenticated), provision fails loudly before
        touching anything."""
        result = run_script(
            "provision",
            env={"RDQ_GPU_STATE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )
        if shutil.which("doctl", path="/usr/bin:/bin"):
            pytest.skip("doctl present in /usr/bin — cannot simulate absence")
        assert result.returncode != 0
        assert "doctl" in result.stderr


class TestRunbook:
    def test_readme_documents_lifecycle_and_billing(self) -> None:
        text = README.read_text()
        for required in ("provision", "bootstrap", "tunnel", "fetch", "destroy"):
            assert required in text
        assert "destroy" in text and "billing" in text.lower()
        # Promotion happens outside this flow — the runbook must say so.
        assert "Promotion" in text

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

    def test_bootstrap_apt_waits_for_dpkg_lock(self) -> None:
        # Fresh droplets race first-boot apt (cloud-init/unattended-upgrades)
        # for the dpkg lock — the 2026-08-11 run died on it. Bootstrap must
        # wait for first boot to finish and tell apt to wait for the lock.
        source = SCRIPT.read_text()
        assert "cloud-init status --wait" in source
        for line in source.splitlines():
            if "apt-get" in line and not line.lstrip().startswith("#"):
                assert "DPkg::Lock::Timeout" in line, f"apt-get without lock timeout: {line}"

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

    def test_run_test_end_rejects_non_iso_dates(self, tmp_path: Path) -> None:
        """--test-end (US-008) is validated at parse time, before any state or
        remote access — a malformed date must never reach the launch script."""
        result = run_script(
            "run", "--test-end", "07/10/2026", env={"RDQ_GPU_STATE_DIR": str(tmp_path)}
        )
        assert result.returncode != 0
        assert "YYYY-MM-DD" in result.stderr

    def test_run_exports_rdq_test_end_to_launch_script(self) -> None:
        """The generated launch script must export RDQ_TEST_END so the worker's
        run_us_quant.sh uses the launch-computed rolling window, not its
        hardcoded fallback."""
        text = SCRIPT.read_text()
        assert "export RDQ_TEST_END='${test_end}'" in text
        heredoc = text.split("cat > /root/rdq-launch.sh")[1].split("\nEOF")[0]
        assert "${test_end_line}" in heredoc

    def test_snapshot_selection_is_region_aware_and_delegated(self) -> None:
        """US-022: image selection must live in ops/gpu_snapshot.py (one
        offline-tested implementation) and must be keyed on the target region
        — snapshots are regional, and a size-plan fallback into another region
        must not select an image that isn't there."""
        text = SCRIPT.read_text()
        assert "ops.gpu_snapshot" in text
        assert '--region "${RDQ_GPU_REGION}"' in text

    def test_snapshot_name_embeds_worker_inputs_hash(self) -> None:
        """US-022: the bake names the image rdq-gpu-base-<hash>-<ts> so
        provision can tell current from stale by name alone; pruning (keep the
        newest 2) is delegated to ops.gpu_snapshot."""
        text = SCRIPT.read_text()
        assert '${SNAPSHOT_PREFIX}-${inputs_hash:+${inputs_hash}-}' in text
        assert "ops.gpu_snapshot prune" in text

    def test_sync_list_delivers_factor_source_contents(self, tmp_path: Path) -> None:
        """US-068: the bootstrap sync must deliver companion files
        (market_series.h5) to the worker — asserted against the actual rsync
        file list (sync-list = same source dir, same flags, dry-run), no live
        worker needed."""
        home = tmp_path / "home"
        for folder in ("data_folder", "data_folder_debug"):
            target = home / "rdq-data" / "factor_source" / "us_liquid" / folder
            target.mkdir(parents=True)
            (target / "daily_pv.h5").write_bytes(b"h5")
            (target / "market_series.h5").write_bytes(b"h5")
            (target / "README.md").write_text("readme\n")
        result = run_script("sync-list", env={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        lines = set(result.stdout.splitlines())
        for folder in ("data_folder", "data_folder_debug"):
            assert f"us_liquid/{folder}/market_series.h5" in lines
            assert f"us_liquid/{folder}/daily_pv.h5" in lines

    def test_bootstrap_factor_source_sync_is_unfiltered(self) -> None:
        """The bootstrap rsync and sync-list must share FACTOR_SOURCE_DIR and
        carry NO exclude filters — otherwise sync-list could pass while the
        real sync silently drops companion files."""
        text = SCRIPT.read_text()
        assert (
            'rsync_remote "${FACTOR_SOURCE_DIR}/" '
            '"root@${DROPLET_IP}:/root/rdq-data/factor_source/"' in text
        )
        sync_list = text.split("cmd_sync_list()")[1].split("\n}")[0]
        assert '"${FACTOR_SOURCE_DIR}/"' in sync_list
        assert "--exclude" not in sync_list

    def test_run_log_is_truncated_not_appended(self) -> None:
        """2026-08-17 incident: a snapshot-booted worker carries the previous
        run's log; 'tee -a' left its exit trailer for the pipeline's
        completion poll to misread as the new run finishing."""
        text = SCRIPT.read_text()
        assert "tee ${RUN_LOG}" in text
        assert "tee -a ${RUN_LOG}" not in text

    def test_snapshot_clears_run_state_before_bake(self) -> None:
        """The baked image must be pristine: no run log (stale exit trailer),
        no traces/workspaces, no launch script, and no per-agent proxy token.
        A failed cleanup must refuse to bake, not ship a polluted image."""
        text = SCRIPT.read_text()
        bake = text.split("cmd_snapshot()")[1].split("\n}")[0]
        assert "rm -rf /root/rdq-runs" in bake
        assert "${PROXY_ENV_FILE}" in bake
        assert "/root/rdq-launch.sh" in bake
        assert "refusing to bake" in bake
        # Cleanup happens BEFORE the power-off that precedes the snapshot.
        assert bake.index("rm -rf /root/rdq-runs") < bake.index("power-off")

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

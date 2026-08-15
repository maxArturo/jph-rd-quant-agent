"""Offline tests for the orchestrator's GPU backend (orchestrator/gpu_backend.py)."""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator.gpu_backend as gpu_backend
from orchestrator.gpu_backend import (
    GpuBackend,
    GpuLaunchError,
    format_gpu_status,
    locate_run_artifacts,
)
from orchestrator.rdagent_client import ArtifactNotFoundError


class RecordingRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, cmd, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr="boom")


class TestGpuBackend:
    def test_launch_builds_transient_unit(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        unit = backend.launch(
            "1234.5678", loop_n=7, universe="ai_semis", instruction="focus on volume"
        )
        assert unit == "rdq-gpu-run-1234-5678"
        cmd = runner.calls[0]
        assert cmd[0] == "systemd-run"
        assert f"--unit={unit}" in cmd
        assert "ops.gpu_pipeline" in cmd
        assert "--thread-ts" in cmd and "1234.5678" in cmd
        assert "--universe" in cmd and "ai_semis" in cmd
        assert "--instruction" in cmd and "focus on volume" in cmd
        assert "--loop_n" in cmd and "7" in cmd

    def test_launch_passes_composed_instruction_unchanged(self, tmp_path: Path) -> None:
        """US-015: the directive + memory digest composition must reach the
        pipeline byte-for-byte as ONE --instruction argv element."""
        from orchestrator.run_memory import compose_instruction

        composed = compose_instruction(
            "Test whether 12-1 momentum beats SPY\nConstraints: long-only",
            "Run-history digest (prior research runs, newest first):\n\n"
            "[2026-08-14 | completed] directive: try downside-share factors",
        )
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2", instruction=composed)
        cmd = runner.calls[0]
        assert cmd[cmd.index("--instruction") + 1] == composed

    def test_launch_injects_path_for_doctl(self, tmp_path: Path) -> None:
        """Transient units get the user manager's minimal PATH — the pipeline
        needs ~/.local/bin injected or doctl/onecli are unreachable (US-002)."""
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2")
        expected = f"--setenv=PATH={Path.home() / '.local/bin'}:/usr/local/bin:/usr/bin:/bin"
        assert expected in runner.calls[0]

    def test_launch_args_get_snapshot_auto_mode(self, tmp_path: Path) -> None:
        """US-022: Slack-launched runs get base-snapshot auto-use by default —
        the exact argv launch() builds must parse to snapshot_mode 'auto'."""
        from ops.gpu_pipeline import build_options

        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2", loop_n=3)
        cmd = runner.calls[0]
        pipeline_argv = cmd[cmd.index("ops.gpu_pipeline") + 1 :]
        options = build_options(pipeline_argv)
        assert options.snapshot_mode == "auto"
        assert options.loop_n == 3

    def test_launch_omits_default_universe(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2", universe="us_liquid")
        assert "--universe" not in runner.calls[0]

    def test_launch_failure_raises(self, tmp_path: Path) -> None:
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=RecordingRunner(1))
        with pytest.raises(GpuLaunchError, match="systemd-run failed"):
            backend.launch("1.2")

    def test_cancel_failure_is_explained(self, tmp_path: Path) -> None:
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=RecordingRunner(255))
        with pytest.raises(GpuLaunchError, match="could not reach"):
            backend.cancel()

    def test_read_status_roundtrip_and_absence(self, tmp_path: Path) -> None:
        status_file = tmp_path / "s.json"
        backend = GpuBackend(status_file=status_file, runner=RecordingRunner())
        assert backend.read_status() is None
        status_file.write_text(json.dumps({"stage": "running"}))
        assert backend.read_status() == {"stage": "running"}
        status_file.write_text("not json")
        assert backend.read_status() is None

    def test_active_run_lock_probes_owner_unit_via_runner(self, tmp_path: Path) -> None:
        """US-020: unit-active probe rides the injected runner (returncode 0 =
        active), so a live owner's lock is reported, not broken."""
        lock_file = tmp_path / "run.lock"
        lock_file.write_text(json.dumps({"unit": "rdq-gpu-run-9-9", "thread_ts": "9.9"}))
        runner = RecordingRunner(returncode=0)
        backend = GpuBackend(
            status_file=tmp_path / "s.json", runner=runner, lock_file=lock_file
        )
        active, broken = backend.active_run_lock()
        assert broken is None
        assert active is not None and active.thread_ts == "9.9"
        assert ["systemctl", "--user", "is-active", "--quiet", "rdq-gpu-run-9-9"] in runner.calls
        assert lock_file.exists()

    def test_active_run_lock_breaks_stale_lock(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "run.lock"
        # Unit inactive (returncode 1) and no pid -> dead owner.
        lock_file.write_text(json.dumps({"unit": "rdq-gpu-run-9-9", "thread_ts": "9.9"}))
        backend = GpuBackend(
            status_file=tmp_path / "s.json", runner=RecordingRunner(1), lock_file=lock_file
        )
        active, broken = backend.active_run_lock()
        assert active is None
        assert broken is not None and broken.unit == "rdq-gpu-run-9-9"
        assert not lock_file.exists()

    def test_active_run_lock_without_lock_file(self, tmp_path: Path) -> None:
        backend = GpuBackend(
            status_file=tmp_path / "s.json",
            runner=RecordingRunner(),
            lock_file=tmp_path / "run.lock",
        )
        assert backend.active_run_lock() == (None, None)


class TestFormatStatus:
    def test_running_status(self) -> None:
        text = format_gpu_status(
            {
                "stage": "running",
                "worker": "gpu-6000adax1-48gb in tor1",
                "loops": [
                    {"loop": 0, "decision": True, "metrics": {"IC": 0.02}},
                    {"loop": 1, "decision": False, "metrics": {"IC": 0.01}},
                    {"loop": 2, "decision": None},
                ],
                "exit": None,
            },
            unit_active=True,
        )
        assert "stage: running" in text
        assert "loops finished: 2 (1 SOTA)" in text
        assert "worker: gpu-6000adax1-48gb" in text

    def test_no_status_yet(self) -> None:
        assert "starting up" in format_gpu_status(None, unit_active=True)
        assert "may not have started" in format_gpu_status(None, unit_active=False)

    def test_dead_unit_mid_run_warns(self) -> None:
        text = format_gpu_status({"stage": "running", "exit": None}, unit_active=False)
        assert "WARNING" in text


def write_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


def make_fetched_trace(results_root: Path, *, with_sota: bool = True) -> Path:
    """Synthetic fetched GPU trace + workspace under a fake results root."""
    trace = results_root / "us_quant" / "log" / "2026-08-06_14-03-18"
    workspace = results_root / "us_quant" / "workspace" / "abc123"
    workspace.mkdir(parents=True)
    (workspace / "qlib_res.csv").write_text(",0\nIC,0.02\n")
    (workspace / "ret.pkl").write_bytes(b"\x00")  # existence is all locate checks
    loop = trace / "Loop_0"
    write_pickle(
        loop / "feedback" / "feedback" / "1" / "2026-08-06_15-00-00-000001.pkl",
        SimpleNamespace(decision=with_sota, reason="x"),
    )
    write_pickle(
        loop / "running" / "runner result" / "1" / "2026-08-06_14-59-00-000001.pkl",
        SimpleNamespace(
            experiment_workspace=SimpleNamespace(
                workspace_path=Path("/root/rdq-runs/us_quant/workspace/abc123")
            )
        ),
    )
    return trace


class TestLocateDispatch:
    def test_gpu_trace_resolves_candidate_workspace(self, tmp_path: Path, monkeypatch) -> None:
        results_root = tmp_path / "results"
        trace = make_fetched_trace(results_root)
        monkeypatch.setattr(gpu_backend, "GPU_RESULTS_ROOT", results_root)
        artifacts = locate_run_artifacts(trace)
        assert artifacts.workspace_path.name == "abc123"
        assert artifacts.qlib_res_csv.is_file()
        assert artifacts.ret_pkl is not None

    def test_gpu_trace_without_sota_refuses(self, tmp_path: Path, monkeypatch) -> None:
        results_root = tmp_path / "results"
        trace = make_fetched_trace(results_root, with_sota=False)
        monkeypatch.setattr(gpu_backend, "GPU_RESULTS_ROOT", results_root)
        with pytest.raises(ArtifactNotFoundError, match="no SOTA loop"):
            locate_run_artifacts(trace)

    def test_non_gpu_path_delegates_to_server_ui_locate(self, tmp_path: Path, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(gpu_backend, "locate_artifacts", lambda p: sentinel)
        assert locate_run_artifacts(tmp_path / "server_ui" / "traces" / "x") is sentinel

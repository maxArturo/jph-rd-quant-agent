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

    def test_launch_omits_default_universe(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2", universe="us_liquid")
        assert "--universe" not in runner.calls[0]

    def test_launch_passes_owning_channel(self, tmp_path: Path) -> None:
        """US-017: the run's home channel rides the pipeline command line."""
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2", channel="C_LIVE")
        cmd = runner.calls[0]
        assert "--channel" in cmd
        assert cmd[cmd.index("--channel") + 1] == "C_LIVE"

    def test_launch_omits_channel_when_unknown(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        backend = GpuBackend(status_file=tmp_path / "s.json", runner=runner)
        backend.launch("1.2")
        assert "--channel" not in runner.calls[0]

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

"""GPU burst-worker backend for the Slack orchestrator.

All research runs execute on disposable DO GPU droplets (2026-08-06 decision);
this box is the control plane. The pieces:

- ``GpuBackend.launch`` starts ``ops/gpu_pipeline.py`` as a transient systemd
  user unit (detached from the bot process, clean env — the bot's onecli
  proxy env must NOT leak into doctl/ssh).
- ``GpuBackend.cancel`` kills the remote research tmux session; the pipeline
  notices, fetches partial results, posts the summary, and tears down.
- ``GpuBackend.read_status`` reads the pipeline's status JSON — the source
  for the check_research_status conversational tool.
- ``locate_run_artifacts`` is the promotion locate that understands BOTH
  backends: fetched GPU traces (worker-absolute pickled paths need the
  prefix remap, and the candidate is the last SOTA loop, not the last loop)
  and legacy on-box trace dirs from pre-GPU runs (delegates to
  ``locate_artifacts``, which lives here since the US-028 removal of the
  old control-plane client).
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ops import run_lock
from ops.gpu_trace import loop_reports, promotion_candidate
from ops.run_lock import RunLock

logger = logging.getLogger(__name__)


class ArtifactNotFoundError(RuntimeError):
    """No finished loop with backtest artifacts could be resolved."""


@dataclass(frozen=True)
class RunArtifacts:
    """A finished loop's backtest outputs, resolved from its trace dir."""

    workspace_path: Path
    qlib_res_csv: Path
    ret_pkl: Path | None  # equity-curve DataFrame; absent on some failures
    source_pkl: Path  # the trace pkl the workspace was resolved from


def locate_artifacts(trace_path: str | Path) -> RunArtifacts:
    """Resolve a finished loop's workspace + qlib_res.csv + ret.pkl from a trace dir.

    rdagent logs each finished backtest experiment under a ``runner result``
    tag (FileStorage pkl whose object carries
    ``experiment_workspace.workspace_path``); qlib_res.csv / ret.pkl live in
    that workspace (written by the workspace's read_exp_res.py). Newest
    result wins; unreadable pkls and workspaces without qlib_res.csv are
    skipped.
    """
    trace_path = Path(trace_path).expanduser()
    if not trace_path.is_dir():
        raise ArtifactNotFoundError(f"trace directory does not exist: {trace_path}")

    candidates = sorted(
        trace_path.glob("**/runner result/**/*.pkl"),
        key=lambda p: p.name,
        reverse=True,
    )
    problems: list[str] = []
    for pkl_file in candidates:
        try:
            with pkl_file.open("rb") as handle:
                obj = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - any unpickle failure just skips this candidate
            problems.append(f"{pkl_file}: failed to unpickle ({exc})")
            continue
        workspace = getattr(getattr(obj, "experiment_workspace", None), "workspace_path", None)
        if workspace is None:
            problems.append(f"{pkl_file}: object has no experiment_workspace.workspace_path")
            continue
        workspace_path = Path(workspace)
        qlib_res_csv = workspace_path / "qlib_res.csv"
        if not qlib_res_csv.is_file():
            problems.append(f"{pkl_file}: no qlib_res.csv in workspace {workspace_path}")
            continue
        ret_pkl = workspace_path / "ret.pkl"
        return RunArtifacts(
            workspace_path=workspace_path,
            qlib_res_csv=qlib_res_csv,
            ret_pkl=ret_pkl if ret_pkl.is_file() else None,
            source_pkl=pkl_file,
        )

    detail = "; ".join(problems) if problems else "no 'runner result' pkl found"
    raise ArtifactNotFoundError(
        f"no finished loop with backtest artifacts under {trace_path}: {detail}"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_WORKER_SH = REPO_ROOT / "ops" / "gpu_worker" / "gpu_worker.sh"
PIPELINE_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

GPU_STATE_DIR = Path.home() / "rdq-runs" / "gpu_worker"
GPU_RESULTS_ROOT = GPU_STATE_DIR / "results"
STATUS_FILE = GPU_STATE_DIR / "pipeline_status.json"
WORKER_WS_PREFIX = "/root/rdq-runs/us_quant"


class GpuLaunchError(RuntimeError):
    """The pipeline unit could not be started/stopped."""


# The transient-unit naming convention lives in ops/run_lock.py (US-020) —
# the pipeline derives its lock-owner unit from the same function.
_unit_name = run_lock.unit_name


class GpuBackend:
    """Thin, testable wrapper around the pipeline lifecycle commands."""

    def __init__(
        self,
        *,
        status_file: Path = STATUS_FILE,
        repo_root: Path = REPO_ROOT,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        lock_file: Path = run_lock.DEFAULT_LOCK_FILE,
    ) -> None:
        self.status_file = status_file
        self._repo_root = repo_root
        self._run = runner or subprocess.run
        self._lock_file = lock_file

    def launch(
        self,
        thread_ts: str,
        *,
        loop_n: int = 10,
        universe: str | None = None,
        instruction: str | None = None,
    ) -> str:
        """Start the pipeline as a transient user unit; returns the unit name."""
        unit = _unit_name(thread_ts)
        command = [
            "systemd-run",
            "--user",
            "--collect",
            f"--unit={unit}",
            f"--property=WorkingDirectory={self._repo_root}",
            # doctl + onecli live in ~/.local/bin; transient units get the
            # user manager's minimal PATH otherwise.
            f"--setenv=PATH={Path.home() / '.local/bin'}:/usr/local/bin:/usr/bin:/bin",
            str(self._repo_root / ".venv" / "bin" / "python"),
            "-m",
            "ops.gpu_pipeline",
            "--loop_n",
            str(loop_n),
            "--thread-ts",
            thread_ts,
            "--status-file",
            str(self.status_file),
        ]
        if universe and universe != "us_liquid":
            command += ["--universe", universe]
        if instruction:
            command += ["--instruction", instruction]
        result = self._run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1:] or ["no output"]
            raise GpuLaunchError(f"systemd-run failed ({result.returncode}): {detail[0]}")
        return unit

    def stop_unit(self, unit: str) -> None:
        self._run(
            ["systemctl", "--user", "stop", unit], capture_output=True, text=True, check=False
        )

    def unit_active(self, thread_ts: str) -> bool:
        result = self._run(
            ["systemctl", "--user", "is-active", "--quiet", _unit_name(thread_ts)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def active_run_lock(self) -> tuple[RunLock | None, RunLock | None]:
        """US-020: ``(active, broken_stale)`` from the global run lock.

        A stale lock (owning unit inactive, pid gone) is removed here; the
        caller reports the break. The is-active probe goes through the same
        injected runner as the other systemctl calls, so tests stay hermetic.
        """

        def is_active(unit: str) -> bool:
            result = self._run(
                ["systemctl", "--user", "is-active", "--quiet", unit],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0

        return run_lock.check_lock(self._lock_file, is_active=is_active)

    def cancel(self) -> str:
        """Kill the remote research session; the pipeline finalizes on its own."""
        result = self._run(
            [str(GPU_WORKER_SH), "ssh", "tmux kill-session -t rdq-run"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            raise GpuLaunchError(
                "could not reach the GPU worker to cancel"
                f" ({detail[-1] if detail else 'no output'}) — the run may already be over"
            )
        return (
            "cancel signal sent — the pipeline will fetch partial results, post the"
            " summary, and destroy the worker"
        )

    def read_status(self) -> dict | None:
        try:
            return json.loads(self.status_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None


def format_gpu_status(status: dict | None, *, unit_active: bool) -> str:
    """Operator-facing one-screen status of the GPU pipeline."""
    if status is None:
        if unit_active:
            return "GPU pipeline is starting up (no status written yet)."
        return "No GPU pipeline status found — the run may not have started, or finished long ago."
    lines = [f"stage: {status.get('stage', '?')}"]
    if status.get("worker"):
        lines.append(f"worker: {status['worker']}")
    loops = status.get("loops") or []
    done = [loop for loop in loops if loop.get("decision") is not None]
    sota = [loop for loop in done if loop.get("decision")]
    lines.append(f"loops finished: {len(done)} ({len(sota)} SOTA)")
    for loop in done[-2:]:
        metrics = loop.get("metrics") or {}
        metric_text = " ".join(
            f"{k}={metrics[k]:.4f}" for k in ("IC", "ARR", "MDD") if k in metrics
        )
        verdict = "SOTA" if loop.get("decision") else "not adopted"
        lines.append(f"  loop {loop.get('loop')}: {verdict} {metric_text}")
    if status.get("exit") is not None:
        lines.append(f"run exited with code {status['exit']}")
    if status.get("candidate_workspace"):
        lines.append(f"promotion candidate: {Path(status['candidate_workspace']).name[:8]}")
    if not unit_active and status.get("exit") is None:
        lines.append("WARNING: pipeline unit is not running — check rdq-gpu-watchdog / journalctl")
    return "\n".join(lines)


def locate_run_artifacts(session_path: str | Path) -> RunArtifacts:
    """Promotion locate that dispatches on where the trace lives."""
    path = Path(session_path).expanduser()
    if not str(path).startswith(str(GPU_RESULTS_ROOT)):
        return locate_artifacts(session_path)

    remap = (WORKER_WS_PREFIX, str(GPU_RESULTS_ROOT / "us_quant"))
    candidate = promotion_candidate(loop_reports(path, remap))
    if candidate is None or not candidate.workspace:
        raise ArtifactNotFoundError(
            f"no SOTA loop with readable artifacts in the fetched GPU trace {path}"
        )
    workspace = Path(candidate.workspace)
    ret_pkl = workspace / "ret.pkl"
    return RunArtifacts(
        workspace_path=workspace,
        qlib_res_csv=workspace / "qlib_res.csv",
        ret_pkl=ret_pkl if ret_pkl.is_file() else None,
        source_pkl=path,  # the fetched trace dir stands in for the trace pkl
    )

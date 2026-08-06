"""Offline tests for the GPU-run trace reader (ops/gpu_trace.py)."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ops.gpu_trace import (
    LoopReport,
    latest_trace_dir,
    loop_reports,
    promotion_candidate,
    remap_path,
    run_exit_code,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


def write_metrics(workspace: Path, ic: float = 0.02, arr: float = 0.5) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "qlib_res.csv").write_text(
        ",0\n"
        f"IC,{ic}\n"
        f"1day.excess_return_with_cost.annualized_return,{arr}\n"
        "1day.excess_return_with_cost.max_drawdown,-0.15\n"
    )


def make_loop(
    trace: Path,
    loop: int,
    *,
    decision: bool | None = True,
    workspace: Path | None = None,
    hypothesis: str = "vol-normalized momentum adds orthogonal alpha",
) -> None:
    loop_dir = trace / f"Loop_{loop}"
    hyp_dir = loop_dir / "direct_exp_gen" / "hypothesis generation" / "77"
    write_pickle(
        hyp_dir / "2026-08-06_14-00-00-000001.pkl",
        SimpleNamespace(hypothesis=hypothesis, action="factor", concise_reason="orthogonal"),
    )
    if decision is not None:
        write_pickle(
            loop_dir / "feedback" / "feedback" / "77" / "2026-08-06_14-10-00-000001.pkl",
            SimpleNamespace(decision=decision, reason="beat SOTA" if decision else "diluted"),
        )
    if workspace is not None:
        write_pickle(
            loop_dir / "running" / "runner result" / "77" / "2026-08-06_14-05-00-000001.pkl",
            SimpleNamespace(experiment_workspace=SimpleNamespace(workspace_path=workspace)),
        )


class TestLoopReports:
    def test_reads_hypothesis_feedback_and_metrics(self, tmp_path: Path) -> None:
        trace = tmp_path / "trace"
        workspace = tmp_path / "ws" / "abc123"
        write_metrics(workspace, ic=0.0186)
        make_loop(trace, 0, decision=True, workspace=workspace)
        reports = loop_reports(trace)
        assert len(reports) == 1
        report = reports[0]
        assert report.hypothesis == "vol-normalized momentum adds orthogonal alpha"
        assert report.action == "factor"
        assert report.decision is True
        assert report.workspace == str(workspace)
        assert report.metrics is not None
        assert report.metrics["IC"] == 0.0186
        assert report.metrics["ARR"] == 0.5

    def test_remap_rewrites_worker_paths(self, tmp_path: Path) -> None:
        trace = tmp_path / "trace"
        local_ws = tmp_path / "fetched" / "abc123"
        write_metrics(local_ws)
        make_loop(trace, 0, workspace=Path("/root/rdq-runs/us_quant/workspace/abc123"))
        remap = ("/root/rdq-runs/us_quant/workspace", str(tmp_path / "fetched"))
        [report] = loop_reports(trace, remap)
        assert report.workspace == str(local_ws)
        assert report.metrics is not None

    def test_unreadable_pickles_degrade_to_none(self, tmp_path: Path) -> None:
        trace = tmp_path / "trace"
        bad = trace / "Loop_0" / "feedback" / "feedback" / "77" / "2026-08-06_14-10-00-000001.pkl"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"not a pickle")
        [report] = loop_reports(trace)
        assert report.decision is None
        assert report.workspace is None

    def test_loops_sorted_numerically(self, tmp_path: Path) -> None:
        trace = tmp_path / "trace"
        for loop in (10, 2, 0):
            make_loop(trace, loop, decision=False)
        assert [r.loop for r in loop_reports(trace)] == [0, 2, 10]

    def test_missing_trace_dir_is_empty(self, tmp_path: Path) -> None:
        assert loop_reports(tmp_path / "nope") == []


class TestCandidateAndHelpers:
    def test_candidate_is_last_sota_with_metrics(self, tmp_path: Path) -> None:
        ws0, ws5 = tmp_path / "ws0", tmp_path / "ws5"
        write_metrics(ws0)
        write_metrics(ws5)
        reports = [
            LoopReport(loop=0, decision=True, workspace=str(ws0), metrics={"IC": 0.01}),
            LoopReport(loop=5, decision=True, workspace=str(ws5), metrics={"IC": 0.02}),
            LoopReport(loop=9, decision=False, workspace=str(ws5), metrics={"IC": 0.01}),
        ]
        candidate = promotion_candidate(reports)
        assert candidate is not None
        assert candidate.loop == 5

    def test_no_candidate_without_sota_or_metrics(self) -> None:
        reports = [
            LoopReport(loop=0, decision=False, workspace="/x", metrics={"IC": 0.1}),
            LoopReport(loop=1, decision=True, workspace="/x", metrics=None),
        ]
        assert promotion_candidate(reports) is None

    def test_remap_path(self) -> None:
        assert remap_path("/root/a/b", ("/root/a", "/tmp/z")) == "/tmp/z/b"
        assert remap_path("/other/a", ("/root/a", "/tmp/z")) == "/other/a"
        assert remap_path("/root/a/b", None) == "/root/a/b"

    def test_run_exit_code(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        assert run_exit_code(log) is None
        log.write_text("noise\n=== run exit=0 2026-08-06T15:31:00Z ===\n")
        assert run_exit_code(log) == 0
        log.write_text("=== run exit=2 x ===\nmore\n=== run exit=1 y ===\n")
        assert run_exit_code(log) == 1

    def test_latest_trace_dir(self, tmp_path: Path) -> None:
        assert latest_trace_dir(tmp_path / "nope") is None
        (tmp_path / "2026-08-01_00-00-00").mkdir()
        (tmp_path / "2026-08-06_14-03-18").mkdir()
        latest = latest_trace_dir(tmp_path)
        assert latest is not None
        assert latest.name == "2026-08-06_14-03-18"


class TestCli:
    def test_json_output(self, tmp_path: Path) -> None:
        trace = tmp_path / "log" / "2026-08-06_14-03-18"
        workspace = tmp_path / "ws" / "abc123"
        write_metrics(workspace)
        make_loop(trace, 0, decision=True, workspace=workspace)
        run_log = tmp_path / "run.log"
        run_log.write_text("=== run exit=0 x ===\n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.gpu_trace",
                "--log-root",
                str(tmp_path / "log"),
                "--run-log",
                str(run_log),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["exit"] == 0
        assert payload["candidate_loop"] == 0
        assert payload["loops"][0]["decision"] is True

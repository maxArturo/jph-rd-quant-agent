"""Offline per-loop reader for rdagent fin_quant trace logs (GPU worker runs).

Used in two places:

- ON the worker (``python -m ops.gpu_trace --log-root /root/rdq-runs/us_quant/log
  --json``, worker venv): ops/gpu_pipeline.py polls this over SSH to post
  per-loop Slack digests while the run is live.
- On the control box over a FETCHED copy (``--remap /root/rdq-runs/us_quant:...``)
  to pick the promotion candidate: runner-result pickles store the ABSOLUTE
  workspace path of the machine that ran the loop (/root/... on a worker), so
  reading a fetched trace requires the prefix remap.

FileStorage layout (same tree ops/sweep.py parses)::

    <trace>/Loop_<n>/direct_exp_gen/hypothesis generation/<pid>/<ts>.pkl
    <trace>/Loop_<n>/running/runner result/<pid>/<ts>.pkl
    <trace>/Loop_<n>/feedback/feedback/<pid>/<ts>.pkl

Reading is conservative like the sweep: an unreadable pickle degrades that
field to None, never raises out of loop_reports().
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

_LOOP_DIR = re.compile(r"^Loop_(\d+)$")
_EXIT_LINE = re.compile(r"^=== run exit=(\d+)")

HYPOTHESIS_GLOB = "direct_exp_gen/hypothesis generation/*/*.pkl"
RUNNER_GLOB = "running/runner result/*/*.pkl"
FEEDBACK_GLOB = "feedback/feedback/*/*.pkl"


@dataclasses.dataclass
class LoopReport:
    loop: int
    hypothesis: str | None = None
    action: str | None = None
    concise_reason: str | None = None
    decision: bool | None = None
    feedback_reason: str | None = None
    workspace: str | None = None
    metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _latest_pickle(loop_dir: Path, pattern: str) -> Any | None:
    """Newest pickle under the tag (filenames are timestamps), or None."""
    candidates = sorted(loop_dir.glob(pattern))
    for path in reversed(candidates):
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001 — unreadable pkl degrades to None
            continue
    return None


def remap_path(raw: str, remap: tuple[str, str] | None) -> str:
    if remap and raw.startswith(remap[0]):
        return remap[1] + raw[len(remap[0]) :]
    return raw


def _workspace_metrics(workspace: str | None) -> dict[str, float] | None:
    if not workspace:
        return None
    csv = Path(workspace) / "qlib_res.csv"
    if not csv.is_file():
        return None
    from orchestrator.summary import METRIC_SPECS, load_metrics  # lazy: pulls pandas

    try:
        raw = load_metrics(csv)
    except Exception:  # noqa: BLE001 — metrics degrade, never break a digest
        return None
    # load_metrics returns qlib's raw csv keys; map to the operator-facing
    # labels (IC/ARR/MDD/...) the same way summary.format_summary does.
    labelled: dict[str, float] = {}
    for label, csv_keys, _style in METRIC_SPECS:
        for key in csv_keys:
            if key in raw:
                labelled[label] = raw[key]
                break
    return labelled or None


def loop_reports(trace_dir: Path, remap: tuple[str, str] | None = None) -> list[LoopReport]:
    reports: list[LoopReport] = []
    loop_dirs: list[tuple[int, Path]] = []
    for child in trace_dir.iterdir() if trace_dir.is_dir() else []:
        match = _LOOP_DIR.match(child.name)
        if match and child.is_dir():
            loop_dirs.append((int(match.group(1)), child))
    for index, loop_dir in sorted(loop_dirs):
        report = LoopReport(loop=index)
        hypothesis = _latest_pickle(loop_dir, HYPOTHESIS_GLOB)
        if hypothesis is not None:
            report.hypothesis = getattr(hypothesis, "hypothesis", None)
            report.action = getattr(hypothesis, "action", None)
            report.concise_reason = getattr(hypothesis, "concise_reason", None)
        feedback = _latest_pickle(loop_dir, FEEDBACK_GLOB)
        if feedback is not None:
            decision = getattr(feedback, "decision", None)
            report.decision = bool(decision) if decision is not None else None
            reason = getattr(feedback, "reason", None)
            report.feedback_reason = str(reason) if reason else None
        runner = _latest_pickle(loop_dir, RUNNER_GLOB)
        workspace = getattr(getattr(runner, "experiment_workspace", None), "workspace_path", None)
        if workspace is not None:
            report.workspace = remap_path(str(workspace), remap)
            report.metrics = _workspace_metrics(report.workspace)
        reports.append(report)
    return reports


def latest_trace_dir(log_root: Path) -> Path | None:
    """Newest timestamped trace dir under a run_us_quant.sh LOG root."""
    candidates = [p for p in log_root.iterdir() if p.is_dir()] if log_root.is_dir() else []
    return max(candidates, key=lambda p: p.name, default=None)


def run_exit_code(run_log: Path) -> int | None:
    """Exit code from the launch wrapper's '=== run exit=N ===' trailer."""
    if not run_log.is_file():
        return None
    exit_code: int | None = None
    for line in run_log.read_text(errors="replace").splitlines():
        match = _EXIT_LINE.match(line.strip())
        if match:
            exit_code = int(match.group(1))
    return exit_code


def promotion_candidate(reports: list[LoopReport]) -> LoopReport | None:
    """Last SOTA loop (decision=True) whose workspace has readable metrics.

    Mirrors the operator rule for control-box runs ("stop at a SOTA result,
    then promote"): feedback decisions are the SOTA signal, qlib_res.csv is
    the artifact gate (same gate rdagent_client.locate_artifacts applies).
    """
    for report in reversed(reports):
        if report.decision and report.workspace and report.metrics:
            return report
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root", type=Path, help="run_us_quant.sh LOG root (picks newest trace)"
    )
    parser.add_argument("--trace", type=Path, help="explicit trace dir (overrides --log-root)")
    parser.add_argument("--remap", help="workspace path prefix remap FROM:TO (fetched copies)")
    parser.add_argument("--run-log", type=Path, help="gpu-run.log to extract the run exit code")
    args = parser.parse_args(argv)

    trace_dir = args.trace
    if trace_dir is None:
        if args.log_root is None:
            parser.error("one of --trace / --log-root is required")
        trace_dir = latest_trace_dir(args.log_root)
    remap = None
    if args.remap:
        source, _, target = args.remap.partition(":")
        if not source or not target:
            parser.error("--remap must be FROM:TO")
        remap = (source, target)

    reports = loop_reports(trace_dir, remap) if trace_dir else []
    candidate = promotion_candidate(reports)
    payload = {
        "trace_dir": str(trace_dir) if trace_dir else None,
        "loops": [r.to_dict() for r in reports],
        "candidate_loop": candidate.loop if candidate else None,
        "exit": run_exit_code(args.run_log) if args.run_log else None,
    }
    json.dump(payload, sys.stdout)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GPU-worker research pipeline: one command from droplet to torn-down droplet.

    python -m ops.gpu_pipeline --loop_n 10

Stages (all driven from the control box, shelling out to
ops/gpu_worker/gpu_worker.sh so the lifecycle mechanics live in one place):

1. provision  — tries each (size, region) in RDQ_GPU_SIZE_PLAN until one has
   stock (GPU availability fluctuates; a sold-out size 422s otherwise).
2. bootstrap + tunnel + check.
3. run --loop_n N, then poll the worker (ops.gpu_trace over SSH) every
   --poll seconds; each completed loop is posted to Slack as a digest
   (hypothesis, SOTA verdict, IC/ARR/MDD).
4. On run exit: fetch results, pick the promotion candidate (last SOTA loop),
   post the final summary with the exact promote command, and DESTROY the
   droplet (billing guard: also destroys on pipeline failure and on the
   --max-hours abort; --keep-worker opts out).

Slack posting: root message to SLACK_CHANNEL_ID (repo-root .env), then a
thread per run — same channel the orchestrator uses, but these runs live
outside server_ui, so thread replies do NOT drive approve/stop/promote; the
loop auto-runs to its budget. Stop early with:
  ops/gpu_worker/gpu_worker.sh ssh tmux kill-session -t rdq-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_WORKER = REPO_ROOT / "ops" / "gpu_worker" / "gpu_worker.sh"

WORKER_REPO = "/root/rd-agent-q"
WORKER_PY = f"{WORKER_REPO}/.venv/bin/python"
WORKER_LOG_ROOT = "/root/rdq-runs/us_quant/log"
WORKER_RUN_LOG = "/root/rdq-runs/gpu-run.log"
WORKER_WS_PREFIX = "/root/rdq-runs/us_quant"

DEFAULT_SIZE_PLAN = "gpu-4000adax1-20gb:tor1,gpu-6000adax1-48gb:tor1,gpu-l40sx1-48gb:tor1"
PRICE_PER_HOUR = {
    "gpu-4000adax1-20gb": 0.76,
    "gpu-6000adax1-48gb": 1.57,
    "gpu-l40sx1-48gb": 1.57,
    "gpu-h100x1-80gb": 4.41,
}
NO_STOCK_MARKER = "not currently available"


@dataclass
class PipelineOptions:
    loop_n: int = 10
    all_duration: str | None = None
    poll_seconds: int = 120
    max_hours: float = 24.0
    keep_worker: bool = False
    reuse_worker: bool = False
    no_slack: bool = False
    size_plan: list[tuple[str, str]] = field(default_factory=list)


def parse_size_plan(raw: str) -> list[tuple[str, str]]:
    """"size:region,size:region" -> [(size, region), ...]; rejects junk."""
    plan: list[tuple[str, str]] = []
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        size, sep, region = entry.partition(":")
        if not sep or not size or not region:
            raise ValueError(f"size plan entry must be SIZE:REGION, got {entry!r}")
        plan.append((size, region))
    if not plan:
        raise ValueError("size plan is empty")
    return plan


class SlackThread:
    """Posts a root message then threaded replies; stderr fallback throughout."""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._client = None
        self._channel: str | None = None
        self._thread_ts: str | None = None
        if enabled:
            try:
                from slack_sdk import WebClient

                from orchestrator.config import load_slack_config

                config = load_slack_config()
                self._client = WebClient(token=config.bot_token)
                # slack_sdk loads HTTPS_PROXY and ignores NO_PROXY (see
                # orchestrator/app.py main()); loopback/Slack must not proxy.
                self._client.proxy = None
                self._channel = config.channel_id
            except Exception as exc:  # noqa: BLE001 — never block the run on Slack
                print(f"slack disabled ({exc}); falling back to stderr", file=sys.stderr)
                self._client = None

    def post(self, text: str) -> None:
        print(f"[slack] {text}", file=sys.stderr)
        if self._client is None or self._channel is None:
            return
        try:
            response = self._client.chat_postMessage(
                channel=self._channel, text=text, thread_ts=self._thread_ts
            )
            if self._thread_ts is None:
                self._thread_ts = response["ts"]
        except Exception as exc:  # noqa: BLE001
            print(f"slack post failed ({exc})", file=sys.stderr)


def worker_sh(
    *args: str, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    import os

    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        [str(GPU_WORKER), *args], capture_output=True, text=True, check=False, env=merged
    )
    if check and result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"gpu_worker.sh {args[0]} failed: {' | '.join(tail) or 'no output'}")
    return result


def provision_with_fallback(plan: list[tuple[str, str]], slack: SlackThread) -> tuple[str, str]:
    last_error = ""
    for size, region in plan:
        result = worker_sh(
            "provision", env={"RDQ_GPU_SIZE": size, "RDQ_GPU_REGION": region}, check=False
        )
        if result.returncode == 0:
            return size, region
        last_error = result.stderr.strip()
        if NO_STOCK_MARKER in last_error:
            slack.post(f":hourglass: {size} has no stock in {region} — trying the next size")
            continue
        break
    detail = last_error.splitlines()[-1] if last_error else "unknown"
    raise RuntimeError(f"provision failed: {detail}")


def worker_trace_status() -> dict:
    """Run ops.gpu_trace on the worker; {} when the trace isn't parseable yet."""
    result = worker_sh(
        "ssh",
        f"cd {WORKER_REPO} && {WORKER_PY} -m ops.gpu_trace"
        f" --log-root {WORKER_LOG_ROOT} --run-log {WORKER_RUN_LOG}",
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


def format_loop_digest(loop: dict) -> str:
    decision = loop.get("decision")
    if decision is None:
        verdict = ":hourglass: no verdict"
    elif decision:
        verdict = ":white_check_mark: new SOTA"
    else:
        verdict = ":heavy_multiplication_x: not adopted"
    metrics = loop.get("metrics") or {}
    parts = [f"{key} {metrics[key]:.4f}" for key in ("IC", "ARR", "MDD") if key in metrics]
    metric_text = " · ".join(parts) if parts else "no backtest artifacts"
    hypothesis = (loop.get("hypothesis") or "(no hypothesis recorded)")[:400]
    workspace = loop.get("workspace") or ""
    ws_tag = Path(workspace).name[:8] if workspace else "n/a"
    return (
        f"*Loop {loop.get('loop')}* [{loop.get('action') or '?'}] — {verdict}\n"
        f"> {hypothesis}\n"
        f"{metric_text} · workspace `{ws_tag}`"
    )


def reportable(loop: dict) -> bool:
    """A loop is worth posting once its feedback verdict exists."""
    return loop.get("decision") is not None


def format_final_summary(
    status: dict, exit_code: int | None, elapsed_hours: float, size: str
) -> str:
    loops = status.get("loops") or []
    done = [loop for loop in loops if reportable(loop)]
    sota = [loop for loop in done if loop.get("decision")]
    candidate_loop = status.get("candidate_loop")
    candidate = next((loop for loop in loops if loop.get("loop") == candidate_loop), None)
    price = PRICE_PER_HOUR.get(size)
    cost = f" · ~${elapsed_hours * price:.2f} droplet" if price else ""
    lines = [
        f":checkered_flag: GPU run finished (exit {exit_code}) — "
        f"{len(done)} loops, {len(sota)} SOTA, {elapsed_hours:.1f}h{cost}",
    ]
    if candidate and candidate.get("workspace"):
        metrics = candidate.get("metrics") or {}
        parts = [f"{k} {metrics[k]:.4f}" for k in ("IC", "ARR", "MDD") if k in metrics]
        lines.append(
            f"Promotion candidate: loop {candidate_loop}, workspace "
            f"`{Path(candidate['workspace']).name}` ({' · '.join(parts)})"
        )
        lines.append(
            "Promote with: `.venv/bin/python -m ops.promote_fetched --workspace "
            f"{candidate['workspace']}` (run from ~/rd-agent-q)"
        )
    else:
        lines.append("No SOTA loop with artifacts — nothing to promote from this run.")
    return "\n".join(lines)


def run_pipeline(options: PipelineOptions) -> int:
    slack = SlackThread(enabled=not options.no_slack)
    size, region = "(reused)", "(reused)"
    started = time.monotonic()
    try:
        if options.reuse_worker:
            slack.post(":recycle: reusing the existing GPU worker")
        else:
            size, region = provision_with_fallback(options.size_plan, slack)
            slack.post(
                f":rocket: GPU worker up: {size} in {region} "
                f"(~${PRICE_PER_HOUR.get(size, 0):.2f}/hr) — bootstrapping"
            )
            worker_sh("bootstrap")
        worker_sh("tunnel")
        worker_sh("check")
        run_args = ["run", "--loop_n", str(options.loop_n)]
        if options.all_duration:
            run_args += ["--all_duration", options.all_duration]
        worker_sh(*run_args)
        slack.post(
            f":microscope: research loop launched — budget {options.loop_n} hypotheses; "
            "per-loop digests will follow in this thread"
        )

        posted: set[int] = set()
        status: dict = {}
        exit_code: int | None = None
        while True:
            time.sleep(options.poll_seconds)
            elapsed_hours = (time.monotonic() - started) / 3600
            if elapsed_hours > options.max_hours:
                slack.post(
                    f":octagonal_sign: max runtime {options.max_hours}h exceeded — "
                    "killing the run and tearing down"
                )
                worker_sh("ssh", "tmux kill-session -t rdq-run || true", check=False)
                break
            # A dead tunnel starves the run of LLM auth — self-heal it.
            tunnel = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", "rdq-gpu-tunnel"], check=False
            )
            if tunnel.returncode != 0:
                slack.post(":warning: proxy tunnel dropped — restarting it")
                worker_sh("tunnel", check=False)
            status = worker_trace_status() or status
            for loop in status.get("loops") or []:
                index = loop.get("loop")
                if reportable(loop) and index not in posted:
                    slack.post(format_loop_digest(loop))
                    posted.add(index)
            exit_code = status.get("exit")
            if exit_code is not None:
                break

        worker_sh("fetch")
        results_root = Path.home() / "rdq-runs" / "gpu_worker" / "results"
        from ops.gpu_trace import latest_trace_dir, loop_reports, promotion_candidate

        trace_dir = latest_trace_dir(results_root / "us_quant" / "log")
        remap = (WORKER_WS_PREFIX, str(results_root / "us_quant"))
        reports = loop_reports(trace_dir, remap) if trace_dir else []
        candidate = promotion_candidate(reports)
        final_status = {
            "loops": [r.to_dict() for r in reports],
            "candidate_loop": candidate.loop if candidate else None,
        }
        elapsed_hours = (time.monotonic() - started) / 3600
        slack.post(format_final_summary(final_status, exit_code, elapsed_hours, size))
        return 0 if exit_code == 0 else 1
    except Exception as exc:  # noqa: BLE001 — report, tear down, re-raise as exit code
        slack.post(f":x: GPU pipeline failed: {exc}")
        return 1
    finally:
        if options.keep_worker:
            slack.post(":warning: --keep-worker set — droplet still BILLING; destroy manually")
        else:
            teardown = worker_sh("destroy", "--force", check=False)
            if teardown.returncode == 0:
                slack.post(":wastebasket: worker destroyed — billing stopped")
            elif "no worker state" not in teardown.stderr:
                slack.post(
                    ":rotating_light: DESTROY FAILED — droplet may still be billing! "
                    "Run: ops/gpu_worker/gpu_worker.sh destroy --force"
                )


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop_n", type=int, default=10)
    parser.add_argument("--all_duration", default=None, help="rdagent wall-clock budget, e.g. 12h")
    parser.add_argument("--poll", type=int, default=120, dest="poll_seconds")
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--keep-worker", action="store_true")
    parser.add_argument("--reuse-worker", action="store_true", help="skip provision/bootstrap")
    parser.add_argument("--no-slack", action="store_true")
    parser.add_argument(
        "--size-plan",
        default=os.environ.get("RDQ_GPU_SIZE_PLAN", DEFAULT_SIZE_PLAN),
        help="comma-separated SIZE:REGION fallback order",
    )
    args = parser.parse_args(argv)
    options = PipelineOptions(
        loop_n=args.loop_n,
        all_duration=args.all_duration,
        poll_seconds=args.poll_seconds,
        max_hours=args.max_hours,
        keep_worker=args.keep_worker,
        reuse_worker=args.reuse_worker,
        no_slack=args.no_slack,
        size_plan=parse_size_plan(args.size_plan),
    )
    return run_pipeline(options)


if __name__ == "__main__":
    raise SystemExit(main())

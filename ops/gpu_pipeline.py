"""GPU-worker research pipeline: one command from droplet to torn-down droplet.

    python -m ops.gpu_pipeline --loop_n 10

Stages (all driven from the control box, shelling out to
ops/gpu_worker/gpu_worker.sh so the lifecycle mechanics live in one place):

0. dates (US-008) — RDQ_TEST_END rolls with the data store: the store
   calendar end minus --confirm-days (default 42) trading days. The reserved
   slice (TEST_END, store end] is the confirmation window the hypothesis
   search never sees — the gate's out-of-sample leg. The resolved instrument
   list is hashed at launch (promotion_gate.hash_instruments) as the gate's
   universe-parity input. Both land in the status file and the run-start
   Slack message. Computed BEFORE provisioning so a broken store costs $0.
1. provision  — tries each (size, region) in RDQ_GPU_SIZE_PLAN until one has
   stock (GPU availability fluctuates; a sold-out size 422s otherwise).
2. bootstrap + tunnel + check.
3. run --loop_n N --test-end <computed>, then poll the worker (ops.gpu_trace
   over SSH) every --poll seconds; each completed loop is posted to Slack as
   a digest (hypothesis, SOTA verdict, IC/ARR/MDD).
4. On run exit: fetch results, pick the promotion candidate (last SOTA loop),
   post the final summary — candidate metrics + factor list + backtest window
   and the incumbent (promoted) strategy's own numbers, with a warning when
   the windows differ — with the exact promote command, and DESTROY the
   droplet (billing guard: also destroys on pipeline failure and on the
   --max-hours abort; --keep-worker opts out).
5. gate + auto-promotion (US-011) — after the final summary the candidate is
   run through ops/promotion_gate.evaluate_gate (parity from the launch-
   recorded instrument hash, confirmation evidence over the reserved window
   via ops/confirm_window). On PASS the candidate is promoted through the
   SAME write path as ops.promote_fetched (snapshot first — a snapshot
   failure blocks promotion, never warn-and-promote), source=auto_gate, the
   verdict stored in promotion_history. A gate fail or gate error NEVER
   fails the run; promotion_gate.auto_promote=false is the report-only
   kill-switch.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ops.promotion_gate import GateVerdict
    from orchestrator.state import PromotedStrategy

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_WORKER = REPO_ROOT / "ops" / "gpu_worker" / "gpu_worker.sh"

WORKER_REPO = "/root/rd-agent-q"
WORKER_PY = f"{WORKER_REPO}/.venv/bin/python"
WORKER_LOG_ROOT = "/root/rdq-runs/us_quant/log"
WORKER_RUN_LOG = "/root/rdq-runs/gpu-run.log"
WORKER_WS_PREFIX = "/root/rdq-runs/us_quant"

DEFAULT_SIZE_PLAN = "gpu-4000adax1-20gb:tor1,gpu-6000adax1-48gb:tor1,gpu-l40sx1-48gb:tor1"
DEFAULT_CONFIRM_DAYS = 42
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
    # Orchestrator integration (all optional — manual CLI runs skip them):
    thread_ts: str | None = None  # post digests into this Slack thread + finalize its run row
    universe: str | None = None  # confirmed custom universe (artifacts must be materialized)
    instruction: str | None = None  # research directive, seeded into the loop's plan
    status_file: Path | None = None  # live JSON the bot's check_research_status reads
    snapshot: bool = False  # bake an rdq-gpu-base image after check (fast future boots)
    no_notion: bool = False  # skip the plain-language Notion write-up
    confirm_days: int = DEFAULT_CONFIRM_DAYS  # trading days reserved past TEST_END (US-008)


@dataclass(frozen=True)
class RunDates:
    """Launch-computed run dates (US-008): the test window rolls with the
    store instead of rotting on a hardcoded TEST_END, and the trailing
    ``confirm_days`` trading days are reserved for the promotion gate's
    confirmation window — the search never sees them."""

    test_end: str  # RDQ_TEST_END shipped to the worker
    confirm_start: str  # first reserved trading day (the one after test_end)
    store_end: str  # store calendar end == last reserved day
    confirm_days: int


def qlib_store_path() -> Path:
    import os

    return Path(os.environ.get("RDQ_QLIB_STORE", "~/.qlib/qlib_data/us_data")).expanduser()


def compute_run_dates(
    store: Path | None = None, confirm_days: int = DEFAULT_CONFIRM_DAYS
) -> RunDates:
    """TEST_END = store calendar end minus confirm_days TRADING days."""
    if confirm_days < 1:
        raise ValueError(f"confirm_days must be >= 1, got {confirm_days}")
    from execution.signal import _read_calendar

    store = store if store is not None else qlib_store_path()
    days = sorted(_read_calendar(store / "calendars" / "day.txt"))
    if len(days) <= confirm_days:
        raise RuntimeError(
            f"store calendar has only {len(days)} trading days — cannot reserve a "
            f"{confirm_days}-day confirmation window (is the store built?)"
        )
    return RunDates(
        test_end=days[-(confirm_days + 1)].isoformat(),
        confirm_start=days[-confirm_days].isoformat(),
        store_end=days[-1].isoformat(),
        confirm_days=confirm_days,
    )


def resolve_instrument_hash(universe: str, store: Path | None = None) -> str:
    """Hash the resolved instrument list — the gate's universe-parity input.

    Read from the control-box store the worker is bootstrapped from; the hash
    is not derivable from a workspace later (conf names the market, not the
    resolved list), so it MUST be recorded at launch.
    """
    store = store if store is not None else qlib_store_path()
    path = store / "instruments" / f"{universe}.txt"
    if not path.is_file():
        raise RuntimeError(
            f"instrument list for universe '{universe}' not found at {path} — "
            "materialize it with data/make_universe.py before launching"
        )
    from ops.promotion_gate import hash_instruments

    symbols = (line.split("\t", 1)[0] for line in path.read_text().splitlines())
    return hash_instruments(symbols)


class StatusFile:
    """Best-effort live status JSON — the check_research_status tool's source."""

    def __init__(self, path: Path | None, thread_ts: str | None) -> None:
        self._path = path
        self._data: dict = {"thread_ts": thread_ts, "stage": "starting", "exit": None}
        self.update()

    def update(self, **fields) -> None:
        self._data.update(fields)
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data))
            tmp.replace(self._path)
        except OSError as exc:  # noqa: PERF203 — status is advisory, never fatal
            print(f"status file write failed ({exc})", file=sys.stderr)


def parse_size_plan(raw: str) -> list[tuple[str, str]]:
    """ "size:region,size:region" -> [(size, region), ...]; rejects junk."""
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
    """Posts a root message then threaded replies; stderr fallback throughout.

    With ``thread_ts`` given, everything goes into that existing thread (the
    orchestrator conversation that started the run) instead of a new one.
    """

    def __init__(self, enabled: bool, thread_ts: str | None = None) -> None:
        self._enabled = enabled
        self._client = None
        self._channel: str | None = None
        self._thread_ts: str | None = thread_ts
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

    def upload(self, png: bytes, *, filename: str, title: str) -> None:
        print(f"[slack] (upload {filename}, {len(png)} bytes)", file=sys.stderr)
        if self._client is None or self._channel is None:
            return
        # files_upload_v2 is multi-request and 504s transiently (seen on the
        # first live run, 2026-08-06) — retry before degrading to text-only.
        for attempt in range(3):
            try:
                self._client.files_upload_v2(
                    channel=self._channel,
                    thread_ts=self._thread_ts,
                    filename=filename,
                    title=title,
                    file=png,
                )
                return
            except Exception as exc:  # noqa: BLE001 — the chart is supplementary
                print(f"slack upload failed (attempt {attempt + 1}/3: {exc})", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        self.post(f":warning: could not upload {title} after 3 attempts — see pipeline logs")


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


def incumbent_report(db_path: Path | None = None) -> dict | None:
    """The promoted strategy's own backtest numbers — the baseline a candidate
    must beat (2026-08-12: a promotion stalled because these never reached the
    digests or the write-up). None when nothing is promoted; metrics/window
    each degrade to None independently.
    """
    try:
        from execution.promoted import load_promoted_strategy

        promoted = load_promoted_strategy(db_path) if db_path else load_promoted_strategy()
    except Exception:  # noqa: BLE001 — nothing promoted yet is a normal state
        return None
    from ops.gpu_trace import workspace_metrics, workspace_window

    workspace = str(Path(promoted.workspace_path).expanduser())
    return {
        "workspace": workspace,
        "metrics": workspace_metrics(workspace),
        "window": workspace_window(workspace),
    }


def _metric_parts(metrics: dict) -> str:
    return " · ".join(f"{k} {metrics[k]:.4f}" for k in ("IC", "ARR", "MDD") if k in metrics)


def _incumbent_lines(incumbent: dict | None, candidate_window: list | None) -> list[str]:
    """Baseline lines for the final summary — the numbers a promotion call needs."""
    if not incumbent:
        return ["No promoted strategy on record — nothing to compare the candidate against."]
    tag = Path(incumbent.get("workspace") or "").name[:8] or "unknown"
    metrics = incumbent.get("metrics")
    if not metrics:
        return [
            f":warning: promoted workspace `{tag}` has no readable qlib_res.csv — "
            "incumbent baseline unavailable"
        ]
    window = incumbent.get("window")
    if window and candidate_window and window == candidate_window:
        note = f" — same backtest window ({window[0]} → {window[1]})"
    elif window and candidate_window:
        note = (
            f" — :warning: windows differ (incumbent {window[0]} → {window[1]}, "
            f"candidate {candidate_window[0]} → {candidate_window[1]}); "
            "metrics are not directly comparable"
        )
    else:
        note = ""
    return [f"Incumbent (promoted `{tag}`): {_metric_parts(metrics)}{note}"]


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
        window = status.get("candidate_window")
        window_text = f", backtest {window[0]} → {window[1]}" if window else ""
        lines.append(
            f"Promotion candidate: loop {candidate_loop}, workspace "
            f"`{Path(candidate['workspace']).name}` ({_metric_parts(metrics)}{window_text})"
        )
        factors = status.get("candidate_factors")
        if factors:
            lines.append(f"Candidate factors: {', '.join(factors)}")
        lines.extend(_incumbent_lines(status.get("incumbent"), window))
        lines.append(
            "Promote with: `.venv/bin/python -m ops.promote_fetched --workspace "
            f"{candidate['workspace']}` (run from ~/rd-agent-q)"
        )
    else:
        lines.append("No SOTA loop with artifacts — nothing to promote from this run.")
    return "\n".join(lines)


def post_comparison_chart(slack: SlackThread, candidate_workspace: str) -> None:
    """Candidate vs currently-promoted equity curves into the thread."""
    from orchestrator.summary import SummaryError, render_comparison_curve

    candidate_ret = Path(candidate_workspace) / "ret.pkl"
    if not candidate_ret.is_file():
        slack.post(":warning: candidate workspace has no ret.pkl — skipping the comparison chart")
        return
    promoted_ret = None
    promoted_label = "promoted"
    try:
        from execution.promoted import load_promoted_strategy

        promoted = load_promoted_strategy()
        path = Path(promoted.workspace_path) / "ret.pkl"
        if path.is_file():
            promoted_ret = path
            promoted_label = f"promoted ({Path(promoted.workspace_path).name[:8]}, live paper)"
    except Exception:  # noqa: BLE001 — nothing promoted yet is a normal state
        pass
    try:
        png = render_comparison_curve(
            candidate_ret,
            promoted_ret,
            candidate_label=f"candidate ({Path(candidate_workspace).name[:8]})",
            promoted_label=promoted_label,
        )
    except SummaryError as exc:
        slack.post(f":warning: comparison chart failed: {exc}")
        return
    slack.upload(png, filename="candidate_vs_promoted.png", title="Candidate vs promoted")


def notion_writeup(options: PipelineOptions, final_status: dict, candidate: dict) -> str | None:
    """Run ops.notion_summary under the orchestrator identity; returns the URL."""
    import datetime

    loops = final_status.get("loops") or []
    context = {
        "run_date": datetime.date.today().isoformat(),
        "universe": options.universe or "us_liquid",
        "directive": options.instruction,
        "loops_total": len([loop for loop in loops if loop.get("decision") is not None]),
        "sota_count": len([loop for loop in loops if loop.get("decision")]),
        "candidate": {
            "loop": candidate.get("loop"),
            "hypothesis": candidate.get("hypothesis"),
            "feedback_reason": candidate.get("feedback_reason"),
            "metrics": candidate.get("metrics") or {},
            "factors": final_status.get("candidate_factors"),
            "window": final_status.get("candidate_window"),
        },
        "incumbent": final_status.get("incumbent"),
    }
    context_path = Path.home() / "rdq-runs" / "gpu_worker" / "notion_context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context))
    result = subprocess.run(
        [
            "onecli",
            "run",
            "--agent",
            "rdq-orchestrator",
            "--",
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ops.notion_summary",
            "--context",
            str(context_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("http")]
    if result.returncode == 0 and lines:
        return lines[-1]
    tail = (result.stderr or result.stdout).strip().splitlines()[-1:]
    print(f"notion write-up failed: {' '.join(tail)}", file=sys.stderr)
    return None


def build_run_args(options: PipelineOptions, dates: RunDates) -> list[str]:
    """gpu_worker.sh run argv — always carries the launch-computed TEST_END."""
    run_args = ["run", "--loop_n", str(options.loop_n), "--test-end", dates.test_end]
    if options.all_duration:
        run_args += ["--all_duration", options.all_duration]
    if options.universe:
        run_args += ["--universe", options.universe]
    if options.instruction:
        run_args += ["--instruction", options.instruction]
    return run_args


def format_run_start(options: PipelineOptions, dates: RunDates, instrument_hash: str) -> str:
    """Run-start Slack message — states the rolling window the operator can't
    otherwise see (TEST_END no longer lives in any file)."""
    return (
        f":microscope: research loop launched — budget {options.loop_n} hypotheses"
        f"{', universe ' + options.universe if options.universe else ''}"
        f"{', directive-seeded' if options.instruction else ''}\n"
        f"Test window ends {dates.test_end} (rolls with the store); confirmation window "
        f"{dates.confirm_start} → {dates.store_end} ({dates.confirm_days} trading days "
        f"reserved, unseen by the search) · universe hash `{instrument_hash}`\n"
        "Per-loop digests will follow in this thread"
    )


ROLLBACK_COMMAND = ".venv/bin/python -m ops.rollback_promotion --yes"


def _promoted_row(db_path: Path) -> PromotedStrategy | None:
    """Current promoted row, or None — never creates the DB (read-only caller,
    same guard as sweep.py: StateStore(path) would create a fresh file)."""
    db_path = db_path.expanduser()
    if not db_path.is_file():
        return None
    from orchestrator.state import StateStore

    return StateStore(db_path).get_promoted_strategy()


def _recorded_instrument_hash(config: dict) -> str | None:
    """The incumbent's universe hash, from the ticker list its promotion pinned.

    A workspace can't yield its own hash (the conf names the market, not the
    resolved list) — the candidate's comes from launch (US-008), the
    incumbent's only from its promotion record. hash_instruments canonicalizes
    (sorted/deduped), so the two are comparable.
    """
    tickers = config.get("universe_tickers") or []
    from ops.promotion_gate import hash_instruments

    try:
        return hash_instruments(tickers)
    except ValueError:  # no tickers pinned at promotion time
        return None


def _manual_promote_command(candidate_workspace: str) -> str:
    return (
        f"`.venv/bin/python -m ops.promote_fetched --workspace {candidate_workspace} --yes` "
        "(run from ~/rd-agent-q)"
    )


def _auto_promote(
    candidate_workspace: str,
    verdict: GateVerdict,
    db_path: Path,
    store_path: Path | None,
    thread_ts: str | None = None,
) -> tuple[bool, str]:
    """Promote through the promote_fetched write path; (promoted?, Slack line).

    Any failure — validation or the pred-refresh snapshot — happens BEFORE the
    pointer flip (promote_workspace orders it that way), so the run finalizes
    unpromoted with the old strategy untouched.
    """
    from ops.promote_fetched import promote_workspace

    kwargs: dict = {"db_path": db_path}
    if store_path is not None:
        kwargs["store_path"] = store_path
    try:
        result = promote_workspace(
            Path(candidate_workspace),
            source="auto_gate",
            gate_verdict=verdict.to_dict(),
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — blocks the promotion, never the run
        return False, (
            f":rotating_light: gate PASSED but promotion failed before the pointer flip "
            f"({exc}) — run finalizes UNPROMOTED; the current strategy is untouched. "
            f"Fix and promote manually: {_manual_promote_command(candidate_workspace)}"
        )
    # US-012: auto-promotions leave the same Notion Decision Log record as the
    # CLI and conversational paths (best-effort — never unwinds the promotion).
    try:
        from ops import promotion_decision

        promotion_decision.record_via_onecli(
            promotion_decision.build_payload(
                result, source="auto_gate", gate_verdict=verdict.to_dict(), thread_ts=thread_ts
            )
        )
    except Exception as exc:  # noqa: BLE001 — bookkeeping must never fail the run
        print(f"decision log write failed ({exc})", file=sys.stderr)
    replaced = (
        f"replacing `{Path(result.replaced_workspace).name[:8]}`"
        if result.replaced_workspace
        else "(first promotion)"
    )
    return True, (
        f":trophy: auto-promoted `{Path(result.workspace).name[:8]}` {replaced} — "
        f"market {result.market}, topk {result.topk}/drop {result.n_drop}. "
        f"Roll back with `{ROLLBACK_COMMAND}` (run from ~/rd-agent-q). "
        "Verify with `python -m execution.pred_refresh --no-slack`."
    )


def gate_and_promote(
    candidate_workspace: str,
    dates: RunDates,
    instrument_hash: str,
    slack: SlackThread,
    status_file: StatusFile,
    *,
    db_path: Path | None = None,
    store_path: Path | None = None,
    config_path: Path | None = None,
    confirm_kwargs: dict | None = None,
    thread_ts: str | None = None,
) -> bool:
    """US-011: gate the candidate against the incumbent; auto-promote on pass.

    NEVER raises — a gate fail, a gate error, or a snapshot failure all
    finalize the run unpromoted with the verdict (or the error) posted to the
    thread. Returns whether a promotion happened. Parity inputs come from
    launch (``instrument_hash``, US-008); the confirmation window is the
    launch-reserved slice in ``dates``.
    """
    import datetime as dt

    from orchestrator.state import DEFAULT_DB_PATH

    db = db_path if db_path is not None else Path(DEFAULT_DB_PATH)
    status_file.update(stage="gate")
    try:
        from ops.promotion_gate import (
            DEFAULT_CONFIG_PATH,
            evaluate_gate,
            load_confirmation_evidence,
            load_gate_config,
            load_metric_bundle,
        )

        gate_config = load_gate_config(config_path if config_path else DEFAULT_CONFIG_PATH)
        candidate = load_metric_bundle(candidate_workspace, instrument_hash=instrument_hash)
        incumbent_row = _promoted_row(db)
        if incumbent_row is None:
            verdict = evaluate_gate(candidate, None, gate_config)
        else:
            incumbent = load_metric_bundle(
                incumbent_row.workspace_path,
                instrument_hash=_recorded_instrument_hash(incumbent_row.config),
            )
            evidence = load_confirmation_evidence(
                candidate_workspace,
                incumbent_row.workspace_path,
                dt.date.fromisoformat(dates.confirm_start),
                dt.date.fromisoformat(dates.store_end),
                **(confirm_kwargs or {}),
            )
            verdict = evaluate_gate(candidate, incumbent, gate_config, evidence)
    except Exception as exc:  # noqa: BLE001 — a gate error must never fail the run
        slack.post(
            f":warning: promotion gate errored ({exc}) — run finalizes unpromoted; "
            f"evaluate manually if warranted: {_manual_promote_command(candidate_workspace)}"
        )
        status_file.update(stage="finished", gate={"error": str(exc)}, auto_promoted=False)
        return False

    promoted = False
    if verdict.passed and gate_config.auto_promote:
        promoted, outcome = _auto_promote(
            candidate_workspace, verdict, db, store_path, thread_ts
        )
    elif verdict.passed:
        outcome = (
            ":pause_button: gate PASSED but promotion_gate.auto_promote is false — report-only "
            f"mode; promote manually with {_manual_promote_command(candidate_workspace)}"
        )
    else:
        failing = [criterion.name for criterion in verdict.criteria if not criterion.passed]
        if not verdict.parity_ok:
            failing.insert(0, "parity")
        outcome = f":no_entry_sign: not promoted — failing criteria: {', '.join(failing)}"
    slack.post(f"{verdict.slack_text()}\n{outcome}")
    status_file.update(stage="finished", gate=verdict.to_dict(), auto_promoted=promoted)
    return promoted


def finalize_run_row(thread_ts: str, trace_dir: Path | None, exit_code: int | None) -> None:
    """Point the orchestrator's run row at the fetched trace and close it out."""
    try:
        from orchestrator.state import DEFAULT_DB_PATH, StateStore

        if not Path(DEFAULT_DB_PATH).is_file():
            print("no orchestrator state.sqlite — run row not finalized", file=sys.stderr)
            return
        store = StateStore(DEFAULT_DB_PATH)
        if store.get_run(thread_ts) is None:
            return
        if trace_dir is not None:
            store.update_run_session_path(thread_ts, str(trace_dir))
        status = "completed" if exit_code == 0 else ("stopped" if exit_code is None else "failed")
        store.update_run_status(thread_ts, status)
    except Exception as exc:  # noqa: BLE001 — never lose teardown over bookkeeping
        print(f"run row finalization failed ({exc})", file=sys.stderr)


def run_pipeline(options: PipelineOptions) -> int:
    slack = SlackThread(enabled=not options.no_slack, thread_ts=options.thread_ts)
    status_file = StatusFile(options.status_file, options.thread_ts)
    size, region = "(reused)", "(reused)"
    started = time.monotonic()
    trace_dir = None
    exit_code: int | None = None
    try:
        # US-008: dates + universe hash come first — a stale store or missing
        # instrument list must fail before any droplet starts billing.
        dates = compute_run_dates(confirm_days=options.confirm_days)
        instrument_hash = resolve_instrument_hash(options.universe or "us_liquid")
        status_file.update(
            test_end=dates.test_end,
            confirmation_window=[dates.confirm_start, dates.store_end],
            confirm_days=dates.confirm_days,
            instrument_hash=instrument_hash,
        )
        if options.reuse_worker:
            slack.post(":recycle: reusing the existing GPU worker")
        else:
            status_file.update(stage="provisioning")
            size, region = provision_with_fallback(options.size_plan, slack)
            slack.post(
                f":rocket: GPU worker up: {size} in {region} "
                f"(~${PRICE_PER_HOUR.get(size, 0):.2f}/hr) — bootstrapping"
            )
            status_file.update(stage="bootstrapping", worker=f"{size} in {region}")
            worker_sh("bootstrap")
        status_file.update(stage="tunnel")
        worker_sh("tunnel")
        worker_sh("check")
        if options.snapshot:
            slack.post(":camera: baking the worker into a base snapshot (future boots ~3 min)")
            status_file.update(stage="snapshot")
            worker_sh("snapshot")
        worker_sh(*build_run_args(options, dates))
        status_file.update(stage="running")
        slack.post(format_run_start(options, dates, instrument_hash))

        posted: set[int] = set()
        status: dict = {}
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
            status_file.update(stage="running", loops=status.get("loops"), exit=exit_code)
            if exit_code is not None:
                break
            # Killed run (stop_run / crash): the tmux session dies without
            # writing the exit trailer — finalize as 'stopped'.
            session = worker_sh("ssh", "tmux has-session -t rdq-run 2>/dev/null", check=False)
            if session.returncode != 0:
                slack.post(
                    ":octagonal_sign: research session ended without an exit marker — finalizing"
                )
                break

        status_file.update(stage="fetching")
        worker_sh("fetch")
        results_root = Path.home() / "rdq-runs" / "gpu_worker" / "results"
        from ops.gpu_trace import (
            latest_trace_dir,
            loop_reports,
            promotion_candidate,
            workspace_factors,
            workspace_window,
        )

        trace_dir = latest_trace_dir(results_root / "us_quant" / "log")
        remap = (WORKER_WS_PREFIX, str(results_root / "us_quant"))
        reports = loop_reports(trace_dir, remap) if trace_dir else []
        candidate = promotion_candidate(reports)
        final_status = {
            "loops": [r.to_dict() for r in reports],
            "candidate_loop": candidate.loop if candidate else None,
            "candidate_window": workspace_window(candidate.workspace) if candidate else None,
            "candidate_factors": workspace_factors(candidate.workspace) if candidate else None,
            "incumbent": incumbent_report(),
        }
        elapsed_hours = (time.monotonic() - started) / 3600
        slack.post(format_final_summary(final_status, exit_code, elapsed_hours, size))
        status_file.update(
            stage="finished",
            loops=final_status["loops"],
            exit=exit_code,
            candidate_workspace=candidate.workspace if candidate else None,
            candidate_window=final_status["candidate_window"],
            candidate_factors=final_status["candidate_factors"],
            incumbent=final_status["incumbent"],
        )
        if candidate is not None and candidate.workspace:
            post_comparison_chart(slack, candidate.workspace)
            if not options.no_notion:
                url = notion_writeup(options, final_status, candidate.to_dict())
                if url:
                    slack.post(
                        f":memo: Plain-language write-up (result + investing approach): {url}"
                    )
                else:
                    slack.post(":warning: Notion write-up failed — see pipeline logs")
            # US-011: gate + auto-promotion — last, so the chart/write-up
            # still describe the pre-promotion world. Never raises.
            gate_and_promote(
                candidate.workspace,
                dates,
                instrument_hash,
                slack,
                status_file,
                thread_ts=options.thread_ts,
            )
        return 0 if exit_code == 0 else 1
    except Exception as exc:  # noqa: BLE001 — report, tear down, re-raise as exit code
        slack.post(f":x: GPU pipeline failed: {exc}")
        status_file.update(stage="failed", error=str(exc))
        # A pipeline failure closes the run row as 'failed' regardless of how
        # far the research itself got (finalize maps non-zero/non-None -> failed).
        exit_code = 1
        return 1
    finally:
        if options.thread_ts:
            finalize_run_row(options.thread_ts, trace_dir, exit_code)
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
    parser.add_argument("--thread-ts", default=None, help="orchestrator thread to post into")
    parser.add_argument("--universe", default=None, help="confirmed custom universe name")
    parser.add_argument(
        "--instruction", default=None, help="research directive (seeded into the loop)"
    )
    parser.add_argument("--status-file", type=Path, default=None, help="live status JSON path")
    parser.add_argument(
        "--confirm-days",
        type=int,
        default=int(os.environ.get("RDQ_CONFIRM_DAYS", DEFAULT_CONFIRM_DAYS)),
        help="trading days reserved past TEST_END as the confirmation window",
    )
    parser.add_argument("--snapshot", action="store_true", help="bake a base image after check")
    parser.add_argument("--no-notion", action="store_true", help="skip the Notion write-up")
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
        thread_ts=args.thread_ts,
        universe=args.universe,
        instruction=args.instruction,
        status_file=args.status_file,
        snapshot=args.snapshot,
        no_notion=args.no_notion,
        confirm_days=args.confirm_days,
    )
    return run_pipeline(options)


if __name__ == "__main__":
    raise SystemExit(main())

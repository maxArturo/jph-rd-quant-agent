"""Promote a FETCHED GPU-worker workspace into the control box's live layout.

    .venv/bin/python -m ops.promote_fetched \
        --workspace ~/rdq-runs/gpu_worker/results/us_quant/workspace/<hash>

Why this exists: the Slack-thread promotion flow (US-044) reaches a workspace
only through the thread's `runs` row and the trace's pickled ABSOLUTE
workspace paths — a run executed on a GPU worker has neither (no thread row;
pickles say /root/...). This command produces the SAME records the normal
flow writes (orchestrator/promotion.py confirm_promotion):

1. conf_pred_refresh.yaml + pred_refresh.env + pred_refresh_params.pkl (the
   backtested run's weights, US-049) inside the workspace
   (execution.pred_refresh.snapshot_pred_refresh).
2. The promoted_strategy row in orchestrator/state.sqlite with the
   conf-derived universe (US-023: market comes from the workspace conf,
   never from a label) and the conf's real topk/n_drop.
3. A Slack notice to the channel.

Audit trail (US-012): every CLI promotion appends a promotion_history row
(source 'cli') and writes a Notion Decision Log row via ops.promotion_decision
(the write hops through `onecli run --agent rdq-orchestrator` for the Notion
bearer; on failure the manual reminder prints instead). The promotion gate
(ops/promotion_gate.py) runs in ADVISORY mode first — same evaluation as the
GPU pipeline's auto-gate, with the candidate's universe hash taken from the
CURRENT store instrument list and the confirmation window derived from the
store calendar (gpu_pipeline.compute_run_dates). A failing or unavailable
verdict blocks --yes unless --force, and --force is recorded in the history
row's gate_verdict ("forced": true).

Safety rails:
- refuses if state.sqlite doesn't already exist (never creates the DB);
- prints the CURRENT promoted strategy and requires --yes to replace it;
- a failing gate verdict requires an explicit --force (recorded);
- warns that re-snapshotting overwrites an operator-pinned market in an
  existing conf_pred_refresh.yaml (e.g. the us_liquid_promoted_30 freeze);
- verifies the market's instruments file exists on THIS box.

Verify afterwards:  .venv/bin/python -m execution.pred_refresh --no-slack
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from execution.pred_refresh import snapshot_pred_refresh
from execution.rebalance import DEFAULT_STORE_PATH
from execution.signal import SignalError, load_market, load_strategy_params
from ops.gpu_trace import workspace_metrics
from orchestrator.state import DEFAULT_DB_PATH, StateStore

if TYPE_CHECKING:
    from ops.promotion_gate import GateVerdict

PRED_REFRESH_CONF = "conf_pred_refresh.yaml"


class PromoteFetchedError(RuntimeError):
    """Validation failure — nothing was written."""


@dataclass(frozen=True)
class PromotionResult:
    """What ``promote_workspace`` wrote — everything a caller needs to report."""

    workspace: str
    market: str
    topk: int
    n_drop: int
    metrics: dict[str, float]
    tickers: list[str] | None
    replaced_workspace: str | None
    promoted_at: str
    snapshot_files: tuple[str, str, str]


def promote_workspace(
    workspace: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    store_path: Path = DEFAULT_STORE_PATH,
    source: str = "cli",
    gate_verdict: dict[str, Any] | None = None,
    session_path: str | None = None,
    thread_ts: str | None = None,
) -> PromotionResult:
    """THE promotion write path: validate → snapshot → pointer flip + history.

    Shared by the CLI below and the GPU pipeline's auto-promotion (US-011) so
    every route leaves identical records. Ordering is load-bearing: the
    pred-refresh snapshot runs BEFORE the pointer flip, so a snapshot failure
    raises with NOTHING written — a workspace that cannot re-predict must
    never become the live strategy (do not copy orchestrator/promotion.py's
    warn-and-promote-anyway behavior here).
    """
    workspace = workspace.expanduser().resolve()
    candidate = validate_workspace(workspace)
    market = candidate["market"]
    tickers = read_tickers(store_path.expanduser() / "instruments", market)
    db_path = db_path.expanduser()
    if not db_path.is_file():
        raise PromoteFetchedError(
            f"{db_path} does not exist — run this from the deployed checkout "
            "(~/rd-agent-q), never let it create a fresh DB"
        )
    store = StateStore(db_path)
    current = store.get_promoted_strategy()
    conf_path, env_path, params_path = snapshot_pred_refresh(workspace)
    promoted = store.set_promoted_strategy(
        str(workspace),
        {
            "universe": market,
            "universe_tickers": tickers,
            "topk": candidate["topk"],
            "n_drop": candidate["n_drop"],
            "thread_ts": thread_ts,
            "session_path": session_path,
        },
        source=source,
        gate_verdict=gate_verdict,
    )
    return PromotionResult(
        workspace=str(workspace),
        market=market,
        topk=candidate["topk"],
        n_drop=candidate["n_drop"],
        metrics=dict(candidate["metrics"]),
        tickers=tickers,
        replaced_workspace=current.workspace_path if current else None,
        promoted_at=promoted.promoted_at,
        snapshot_files=(conf_path.name, env_path.name, params_path.name),
    )


def evaluate_advisory_gate(
    workspace: Path,
    tickers: list[str] | None,
    store: StateStore,
    *,
    store_path: Path = DEFAULT_STORE_PATH,
    config_path: Path | None = None,
    snapshot_missing: bool = False,
) -> tuple[GateVerdict | None, str | None]:
    """US-012: the auto-gate's evaluation, assembled for a CLI promotion.

    Returns (verdict, None) or (None, error) — never raises. Differences from
    the pipeline's launch-time gate, both inherent to promoting after the
    fact: the candidate's universe hash comes from the CURRENT store
    instrument list (the launch-recorded hash lives in that run's
    pipeline_status.json, which a bare workspace path can't reach), and the
    confirmation window is re-derived from the store calendar
    (compute_run_dates — same confirm_days default as launch).

    ``snapshot_missing=True`` (the --yes path) snapshots the candidate first
    when its pred-refresh files are absent: confirmation needs them, and
    promote_workspace would write the identical files moments later anyway.
    Dry-run keeps its nothing-written promise and lets confirmation degrade.
    """
    try:
        import datetime as dt

        from ops.gpu_pipeline import compute_run_dates
        from ops.promotion_gate import (
            DEFAULT_CONFIG_PATH,
            align_overlap,
            evaluate_gate,
            hash_instruments,
            load_confirmation_evidence,
            load_gate_config,
            load_metric_bundle,
        )

        gate_config = load_gate_config(config_path if config_path else DEFAULT_CONFIG_PATH)
        candidate = load_metric_bundle(
            workspace, instrument_hash=hash_instruments(tickers) if tickers else None
        )
        incumbent_row = store.get_promoted_strategy()
        if incumbent_row is None:
            return evaluate_gate(candidate, None, gate_config), None
        incumbent_tickers = incumbent_row.config.get("universe_tickers") or []
        incumbent = load_metric_bundle(
            incumbent_row.workspace_path,
            instrument_hash=hash_instruments(incumbent_tickers) if incumbent_tickers else None,
        )
        if snapshot_missing and not (workspace / PRED_REFRESH_CONF).is_file():
            snapshot_pred_refresh(workspace)
        resolved_store = store_path.expanduser()
        dates = compute_run_dates(store=resolved_store)
        evidence = load_confirmation_evidence(
            workspace,
            incumbent_row.workspace_path,
            dt.date.fromisoformat(dates.confirm_start),
            dt.date.fromisoformat(dates.store_end),
            store_path=resolved_store,
        )
        overlap = align_overlap(candidate, incumbent, gate_config.min_overlap_days)
        return evaluate_gate(candidate, incumbent, gate_config, evidence, overlap), None
    except Exception as exc:  # noqa: BLE001 — advisory: the verdict degrades, main() decides
        return None, str(exc)


def read_tickers(instruments_dir: Path, market: str) -> list[str] | None:
    path = instruments_dir / f"{market}.txt"
    if not path.is_file():
        return None
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    tickers = sorted({line.split("\t")[0].strip() for line in rows})
    return tickers or None


def validate_workspace(workspace: Path) -> dict:
    """All the read-only checks; returns the candidate config + metrics."""
    if not workspace.is_dir():
        raise PromoteFetchedError(f"workspace is not a directory: {workspace}")
    if not (workspace / "qlib_res.csv").is_file():
        raise PromoteFetchedError(
            f"no qlib_res.csv in {workspace} — not a completed backtest workspace"
        )
    if not list(workspace.glob("mlruns/*/*/artifacts/pred.pkl")):
        raise PromoteFetchedError(f"no mlruns/**/pred.pkl under {workspace} — predictions missing")
    if not list((workspace / "logs").glob("docker_execution_*.log")):
        raise PromoteFetchedError(
            f"no logs/docker_execution_*.log under {workspace} — "
            "the pred-refresh snapshot cannot recover its jinja context"
        )
    try:
        market = load_market(workspace)
        params = load_strategy_params(workspace)
    except SignalError as exc:
        raise PromoteFetchedError(f"workspace conf is unusable: {exc}") from exc
    metrics = workspace_metrics(str(workspace)) or {}
    return {"market": market, "topk": params.topk, "n_drop": params.n_drop, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="orchestrator state.sqlite"
    )
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE_PATH, help="qlib us_data store"
    )
    parser.add_argument(
        "--session-path", default=None, help="informational trace dir for the record"
    )
    parser.add_argument("--yes", action="store_true", help="actually write (default is dry-run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="promote despite a failing/unavailable gate verdict (recorded in history)",
    )
    parser.add_argument("--no-slack", action="store_true")
    parser.add_argument(
        "--no-notion", action="store_true", help="skip the Notion Decision Log write"
    )
    args = parser.parse_args(argv)

    workspace = args.workspace.expanduser().resolve()
    try:
        candidate = validate_workspace(workspace)
    except PromoteFetchedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    market = candidate["market"]
    tickers = read_tickers(args.store.expanduser() / "instruments", market)
    if tickers is None:
        print(
            f"WARNING: no instruments/{market}.txt in the local store — the pinned ticker "
            "list will be empty and the rebalancer's universe check degrades to advisory-skip",
            file=sys.stderr,
        )

    if not args.db.expanduser().is_file():
        print(
            f"REFUSED: {args.db} does not exist — run this from the deployed checkout "
            "(~/rd-agent-q), never let it create a fresh DB",
            file=sys.stderr,
        )
        return 1
    store = StateStore(args.db.expanduser())
    current = store.get_promoted_strategy()

    metrics = candidate["metrics"]
    metric_text = " · ".join(
        f"{key} {metrics[key]:.4f}" for key in ("IC", "ARR", "MDD") if metrics.get(key) is not None
    )
    print(f"candidate: {workspace}")
    print(f"  market={market} topk={candidate['topk']} n_drop={candidate['n_drop']}")
    print(f"  metrics: {metric_text or 'n/a'}")
    print(f"  tickers pinned: {len(tickers) if tickers else 0}")
    if current:
        print(f"current promoted: {current.workspace_path}")
        print(f"  config: {json.dumps(current.config)[:300]}")
        if (Path(current.workspace_path) / PRED_REFRESH_CONF).is_file():
            print(
                "  NOTE: promoting replaces the promoted row; the OLD workspace keeps its "
                "snapshot, and THIS workspace gets a fresh conf_pred_refresh.yaml — any "
                "operator-pinned market (e.g. a frozen *_promoted_* universe) must be "
                "re-applied to the NEW snapshot afterwards."
            )

    # US-012: same gate the pipeline auto-runs, advisory here. On the --yes
    # path a missing candidate snapshot is written up-front (confirmation
    # needs it; promote_workspace would write the same files right after).
    verdict, gate_error = evaluate_advisory_gate(
        workspace,
        tickers,
        store,
        store_path=args.store.expanduser(),
        snapshot_missing=args.yes,
    )
    if verdict is not None:
        print(verdict.slack_text())
    else:
        print(f"gate verdict unavailable: {gate_error}", file=sys.stderr)
    gate_passed = verdict is not None and verdict.passed

    if not args.yes:
        if not gate_passed:
            print("gate verdict is not a PASS — promoting will additionally need --force")
        print("dry-run (no --yes): nothing written")
        return 0

    if not gate_passed and not args.force:
        print(
            "REFUSED: gate verdict is FAIL/unavailable — re-run with --force to promote "
            "anyway (the override is recorded in promotion_history)",
            file=sys.stderr,
        )
        return 1
    forced = bool(args.force and not gate_passed)
    gate_record: dict[str, Any] = (
        verdict.to_dict() if verdict is not None else {"error": gate_error}
    )
    gate_record["forced"] = forced

    try:
        result = promote_workspace(
            workspace,
            db_path=args.db,
            store_path=args.store,
            source="cli",
            gate_verdict=gate_record,
            session_path=args.session_path,
        )
    except PromoteFetchedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"snapshot written: {', '.join(result.snapshot_files)}")
    print(f"promoted_strategy row set at {result.promoted_at}")

    decision_recorded = False
    if not args.no_notion:
        from ops import promotion_decision

        decision_recorded = promotion_decision.record_via_onecli(
            promotion_decision.build_payload(
                result,
                source="cli",
                gate_verdict=verdict.to_dict() if verdict is not None else None,
                gate_error=gate_error,
                forced=forced,
            )
        )
        print(
            "Notion Decision Log row written."
            if decision_recorded
            else "Decision Log write FAILED — add the entry manually in Notion.",
            file=sys.stdout if decision_recorded else sys.stderr,
        )

    notice = (
        f":trophy: Promoted GPU-run workspace `{workspace.name[:8]}` "
        f"(market {market}, topk {candidate['topk']}/drop {candidate['n_drop']}"
        f"{', ' + metric_text if metric_text else ''})"
        f"{' — gate verdict OVERRIDDEN with --force' if forced else ''}. "
        "Verify with `python -m execution.pred_refresh --no-slack`."
        + ("" if decision_recorded else " Reminder: add a Decision Log entry in Notion.")
    )
    if args.no_slack:
        print(notice, file=sys.stderr)
    else:
        try:
            from execution.rebalance import slack_notifier

            slack_notifier()(notice)
        except Exception as exc:  # noqa: BLE001 — the promotion already happened
            print(f"slack notice failed ({exc}): {notice}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

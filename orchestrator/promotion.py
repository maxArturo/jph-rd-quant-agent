"""Strategy promotion flow: only deliberate choices ever trade (US-033).

A finished run's summary (completed, or deliberately stopped by the operator
— see PROMOTABLE_STATUSES) carries a Promote button. Clicking it posts a
confirmation that restates the universe, the TopkDropoutStrategy params
(topk/n_drop read from the workspace's own qlib conf — the same values the
rebalancer's signal extraction will trade), and the headline backtest
metrics. Only the Confirm click promotes: it pins the workspace path + config
into THE single ``promoted_strategy`` SQLite row (replacing any previous
strategy, with a Slack notice naming what was replaced) and records a
Decision Log row in Notion.

The candidate is re-derived from the run row + artifacts on every click
(nothing is cached in memory or in button values beyond the thread_ts), so
buttons keep working across orchestrator restarts.

Confirming also snapshots everything the automated morning prediction
refresh needs into the workspace (conf_pred_refresh.yaml + pred_refresh.env +
pred_refresh_params.pkl, US-048/049 — see execution/pred_refresh.py) while
the run's logs and artifacts still exist. A
snapshot failure warns in-thread but never blocks the promotion: the manual
refresh procedure remains available, and the rebalancer's stale-pred abort
is the backstop.

The other half of the story lives in execution/promoted.py: the rebalancer
entrypoint refuses to run when no promoted strategy exists.

Live promotion (US-009) lives here too: ``LivePromotion`` pins the
INDEPENDENT live slot (``promoted_strategy_live``) either from the
paper-promoted strategy (copied in full) or directly from any promotable
run, reusing ``PromotionFlow.candidate_from_run`` for provenance so paper
and live promotion can never disagree about what a run trades. The US-010
Slack tools are thin wrappers over it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution import allocation, pred_refresh, signal
from execution.rebalance import DEFAULT_STORE_PATH
from orchestrator import summary
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.rdagent_client import ArtifactNotFoundError, RunArtifacts, locate_artifacts
from orchestrator.state import PromotedStrategy, Run, StateStore

logger = logging.getLogger(__name__)

# slack_bolt's Say or any equivalent accepting (text=..., blocks=..., thread_ts=...).
SayFn = Callable[..., Any]

# Block Kit action ids (app.py registers a Bolt listener per id).
# Button values carry the owning thread_ts — the run row is the durable state.
ACTION_PROMOTE = "run_promote"
ACTION_PROMOTE_CONFIRM = "promote_confirm"
ACTION_PROMOTE_CANCEL = "promote_cancel"

# runs.universe is always set by start_research; None only on pre-US-020 rows.
FALLBACK_UNIVERSE = "us_liquid"

# Run statuses whose artifacts may be promoted. 'stopped' is deliberate:
# orchestrator-started runs are unbounded (they never complete on their own),
# so the operator ends every successful one by stopping it at a SOTA result —
# refusing 'stopped' would make promotion unreachable from Slack. 'failed'
# and 'running' stay refused (no coherent final artifacts to pin).
PROMOTABLE_STATUSES = frozenset({"completed", "stopped"})

# Slack section blocks cap text at 3000 chars.
_MAX_SECTION_TEXT = 2900


class PromotionError(RuntimeError):
    """A promotion refusal with an operator-actionable message."""


@dataclass(frozen=True)
class PromotionCandidate:
    """Everything the confirmation restates and the promotion pins.

    ``universe`` is derived from the workspace conf's ``market:`` line — the
    ground truth that bounds pred.pkl — with the same rigor as topk/n_drop
    (2026-08-05 incident: a run labeled 'ai_deployers' had backtested
    us_liquid all along). ``universe_label`` keeps the run row's label so the
    confirmation can call out a mismatch; ``universe_tickers`` come from the
    store's instruments file for the derived market (None when unreadable —
    the rebalancer's divergence check skips then).
    """

    run: Run
    workspace: Path
    universe: str
    params: signal.StrategyParams
    metrics: dict[str, float]
    sharpe: float | None
    universe_label: str | None = None
    universe_tickers: tuple[str, ...] | None = None

    @property
    def label_mismatch(self) -> bool:
        return self.universe_label is not None and self.universe_label != self.universe

    @property
    def config(self) -> dict[str, Any]:
        """The strategy config pinned into promoted_strategy (what trades)."""
        tickers = self.universe_tickers
        return {
            "universe": self.universe,
            "universe_tickers": None if tickers is None else list(tickers),
            "topk": self.params.topk,
            "n_drop": self.params.n_drop,
            "thread_ts": self.run.thread_ts,
            "session_path": self.run.session_path,
        }


def _mismatch_warning(candidate: PromotionCandidate) -> str:
    return (
        f":warning: The run was labeled `{candidate.universe_label}`, but the workspace's"
        f" backtest actually ran `market: {candidate.universe}` — the universe wiring gap"
        f" (US-023). What trades is `{candidate.universe}`."
    )


def _section(text: str) -> dict[str, Any]:
    if len(text) > _MAX_SECTION_TEXT:
        text = text[: _MAX_SECTION_TEXT - 1] + "…"
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _button(label: str, action_id: str, value: str, style: str | None = None) -> dict[str, Any]:
    element: dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": label},
        "action_id": action_id,
        "value": value,
    }
    if style is not None:
        element["style"] = style
    return element


def promotion_offer_blocks(thread_ts: str, summary_text: str) -> list[dict[str, Any]]:
    """The finished-run summary with its Promote button (posted by the poller)."""
    return [
        _section(summary_text),
        {
            "type": "actions",
            "block_id": f"promote_offer_{thread_ts}",
            "elements": [
                _button("Promote to paper trading", ACTION_PROMOTE, thread_ts, style="primary")
            ],
        },
    ]


def confirmation_text(candidate: PromotionCandidate, previous: PromotedStrategy | None) -> str:
    """Restate exactly what a Confirm click will make the rebalancer trade."""
    universe_line = f"• *Universe:* `{candidate.universe}`"
    if candidate.universe_tickers is not None:
        universe_line += f" ({len(candidate.universe_tickers)} tickers)"
    lines = [
        ":rocket: *Confirm promotion to paper trading*",
        "The nightly rebalancer will trade this strategy:",
        universe_line,
        f"• *Strategy:* TopkDropoutStrategy — topk={candidate.params.topk},"
        f" n_drop={candidate.params.n_drop}",
        f"• *Workspace:* `{candidate.workspace}`",
        summary.format_summary(candidate.metrics, candidate.sharpe),
    ]
    if candidate.label_mismatch:
        lines.insert(1, _mismatch_warning(candidate))
    if candidate.universe_tickers is None:
        lines.append(
            f":warning: No instruments file found for `{candidate.universe}` — the"
            " rebalancer's universe-divergence check will be skipped for this strategy."
        )
    if previous is not None:
        lines.append(
            f":warning: This replaces the currently promoted strategy"
            f" (workspace `{previous.workspace_path}`, promoted {previous.promoted_at})."
        )
    return "\n".join(lines)


def confirmation_blocks(
    candidate: PromotionCandidate, previous: PromotedStrategy | None
) -> list[dict[str, Any]]:
    thread_ts = candidate.run.thread_ts
    return [
        _section(confirmation_text(candidate, previous)),
        {
            "type": "actions",
            "block_id": f"promote_confirm_{thread_ts}",
            "elements": [
                _button("Confirm promotion", ACTION_PROMOTE_CONFIRM, thread_ts, style="primary"),
                _button("Cancel", ACTION_PROMOTE_CANCEL, thread_ts),
            ],
        },
    ]


class PromotionFlow:
    """Handles the Promote / Confirm / Cancel button clicks (Bolt listeners).

    Share one instance per process. Every method re-derives the candidate
    from SQLite + the workspace artifacts, posts refusals in-thread, and
    never raises into Bolt. ``locate`` and ``load_params`` are injectable for
    tests (defaults resolve the real run artifacts and the workspace's own
    qlib conf).
    """

    def __init__(
        self,
        store: StateStore,
        recorder: NotionRecorder | None = None,
        locate: Callable[[str | Path], RunArtifacts] = locate_artifacts,
        load_params: Callable[[Path], signal.StrategyParams] = signal.load_strategy_params,
        snapshot: Callable[[Path], object] = pred_refresh.snapshot_pred_refresh,
        load_market: Callable[[Path], str] = signal.load_market,
        instruments_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._recorder = recorder
        self._locate = locate
        self._load_params = load_params
        self._snapshot = snapshot
        self._load_market = load_market
        self._instruments_dir = (
            instruments_dir
            if instruments_dir is not None
            else DEFAULT_STORE_PATH.expanduser() / "instruments"
        )

    # -- button handlers ------------------------------------------------------

    def request_promotion(self, thread_ts: str, say: SayFn) -> None:
        """Promote click: post the confirmation restating what would trade."""
        try:
            candidate = self._candidate(thread_ts)
        except PromotionError as exc:
            say(text=f":no_entry: Cannot promote: {exc}", thread_ts=thread_ts)
            return
        previous = self._store.get_promoted_strategy()
        say(
            text=confirmation_text(candidate, previous),
            blocks=confirmation_blocks(candidate, previous),
            thread_ts=thread_ts,
        )

    def confirm_promotion(self, thread_ts: str, say: SayFn) -> None:
        """Confirm click: pin the strategy, notify, and write the Decision Log."""
        try:
            candidate = self._candidate(thread_ts)
        except PromotionError as exc:
            say(text=f":no_entry: Cannot promote: {exc}", thread_ts=thread_ts)
            return
        previous = self._store.get_promoted_strategy()
        promoted = self._store.set_promoted_strategy(str(candidate.workspace), candidate.config)
        lines = [
            ":rocket: *Strategy promoted to paper trading.*",
            f"• Universe `{candidate.universe}`, topk={candidate.params.topk},"
            f" n_drop={candidate.params.n_drop}",
            f"• Workspace `{promoted.workspace_path}`",
            "The nightly rebalancer will trade this strategy from its next run.",
        ]
        if candidate.label_mismatch:
            lines.append(_mismatch_warning(candidate))
        if previous is not None:
            lines.append(
                f":arrows_counterclockwise: Replaced the previously promoted strategy"
                f" (workspace `{previous.workspace_path}`, promoted {previous.promoted_at})."
            )
        # US-048: capture everything the morning prediction refresh needs while
        # the run's logs still exist. Warn-don't-block: the manual refresh
        # procedure still works and the rebalancer's stale-pred abort backstops.
        try:
            self._snapshot(candidate.workspace)
        except Exception as exc:  # noqa: BLE001 - any failure becomes a warning
            logger.warning(
                "pred-refresh snapshot failed for %s: %s", candidate.workspace, exc
            )
            lines.append(
                f":warning: Pred-refresh snapshot failed ({exc}) — the automated morning"
                " prediction refresh cannot run for this strategy until"
                f" {pred_refresh.SNAPSHOT_CONF_NAME}, {pred_refresh.SNAPSHOT_ENV_NAME}"
                f" and {pred_refresh.SNAPSHOT_PARAMS_NAME}"
                " exist in the workspace (tasks/us-048-automated-pred-refresh.md)."
            )
        say(text="\n".join(lines), thread_ts=thread_ts)
        logger.info(
            "promoted strategy from thread %s (workspace %s, replaced=%s)",
            thread_ts,
            promoted.workspace_path,
            previous is not None,
        )
        if self._recorder is not None:
            self._recorder.record_decision(
                title=self._decision_title(candidate),
                decision_type="promotion",
                details=self._decision_details(candidate, previous),
                thread_ts=thread_ts,
            )
            self._recorder.record_idea_status(thread_ts, "promoted")

    def cancel_promotion(self, thread_ts: str, say: SayFn) -> None:
        say(
            text=":leftwards_arrow_with_hook: Promotion cancelled — nothing was changed.",
            thread_ts=thread_ts,
        )

    # -- internals -------------------------------------------------------------

    def _candidate(self, thread_ts: str) -> PromotionCandidate:
        """Re-derive the promotable strategy for a thread, or refuse loudly."""
        run = self._store.get_run(thread_ts)
        if run is None:
            raise PromotionError("this thread has no research run to promote")
        if run.status not in PROMOTABLE_STATUSES:
            raise PromotionError(
                f"the run is '{run.status}' — only a completed or operator-stopped"
                " run can be promoted"
            )
        return self.candidate_from_run(run)

    def candidate_from_run(self, run: Run) -> PromotionCandidate:
        """Derive a run's full provenance (workspace, params, conf-derived
        universe, pinned tickers, headline metrics). The single derivation
        path shared by paper promotion and ``LivePromotion`` — callers have
        already checked the run's status against PROMOTABLE_STATUSES."""
        try:
            artifacts = self._locate(run.session_path)
        except (ArtifactNotFoundError, OSError) as exc:
            raise PromotionError(f"run artifacts are unavailable ({exc})") from exc
        try:
            params = self._load_params(Path(artifacts.workspace_path))
        except signal.SignalError as exc:
            raise PromotionError(
                f"cannot determine the strategy's topk/n_drop from the workspace"
                f" config ({exc}) — refusing to promote a strategy the rebalancer"
                f" could not reproduce"
            ) from exc
        # The universe gets the same rigor as topk/n_drop: the conf's market
        # line bounds pred.pkl, so it is what actually trades — never the
        # run-row label (2026-08-05 ai_deployers incident).
        try:
            market = self._load_market(Path(artifacts.workspace_path))
        except signal.SignalError as exc:
            raise PromotionError(
                f"cannot determine the universe the workspace backtested"
                f" (no readable market line: {exc}) — refusing to promote a"
                f" strategy whose traded universe is unknown"
            ) from exc
        # Metrics degrade to n/a rather than blocking: the operator already saw
        # (and is acting on) the completion summary built from the same files.
        metrics: dict[str, float] = {}
        try:
            metrics = summary.load_metrics(artifacts.qlib_res_csv)
        except summary.SummaryError as exc:
            logger.warning(
                "promotion confirmation without metrics for %s: %s", run.thread_ts, exc
            )
        sharpe: float | None = None
        if artifacts.ret_pkl is not None:
            try:
                sharpe = summary.compute_sharpe(artifacts.ret_pkl)
            except summary.SummaryError:
                sharpe = None
        return PromotionCandidate(
            run=run,
            workspace=Path(artifacts.workspace_path),
            universe=market,
            params=params,
            metrics=metrics,
            sharpe=sharpe,
            universe_label=run.universe,
            universe_tickers=self._universe_tickers(market),
        )

    def _universe_tickers(self, market: str) -> tuple[str, ...] | None:
        """Symbols in the store's instruments file for *market*, or None.

        Read at promote time so the pinned list is the membership the operator
        confirmed; unreadable file degrades to None (the confirmation warns,
        and the rebalancer's divergence check skips).
        """
        path = self._instruments_dir / f"{market}.txt"
        try:
            symbols = {
                line.split("\t")[0].strip().upper()
                for line in path.read_text().splitlines()
                if line.strip()
            }
        except OSError as exc:
            logger.warning("cannot read instruments file %s: %s", path, exc)
            return None
        return tuple(sorted(symbols)) if symbols else None

    def _decision_title(self, candidate: PromotionCandidate) -> str:
        directive = self._store.get_directive(candidate.run.thread_ts)
        subject = directive.objective if directive is not None else candidate.universe
        return f"Promote '{subject}' to paper trading"

    def _decision_details(
        self, candidate: PromotionCandidate, previous: PromotedStrategy | None
    ) -> str:
        lines = [
            f"Workspace: {candidate.workspace}",
            f"Universe: {candidate.universe}",
            f"TopkDropoutStrategy: topk={candidate.params.topk},"
            f" n_drop={candidate.params.n_drop}",
            f"Thread TS: {candidate.run.thread_ts}",
        ]
        if candidate.label_mismatch:
            lines.insert(
                2,
                f"Universe label on the run row was '{candidate.universe_label}'"
                " (workspace conf market wins)",
            )
        for label, keys, _style in summary.METRIC_SPECS:
            value = next((candidate.metrics[k] for k in keys if k in candidate.metrics), None)
            if value is not None:
                lines.append(f"{label}: {value:.4f}")
        if candidate.sharpe is not None:
            lines.append(f"Sharpe: {candidate.sharpe:.4f}")
        if previous is not None:
            lines.append(
                f"Replaced: {previous.workspace_path} (promoted {previous.promoted_at})"
            )
        return "\n".join(lines)


# --- Live promotion backend (US-009) ------------------------------------------

# Decision Log Type select values (Notion auto-creates new select options).
LIVE_DECISION_PROMOTE = "promote_live"
LIVE_DECISION_DEMOTE = "demote_live"

# A run reference shorter than this never fuzzy-matches session paths —
# tiny fragments would match half the store by accident.
_MIN_REFERENCE_FRAGMENT = 6


class LivePromotionError(RuntimeError):
    """A live promotion/demotion refusal with an operator-actionable message."""


def _run_line(run: Run) -> str:
    return f"• thread {run.thread_ts} — {run.universe or '?'} ({run.status})"


@dataclass(frozen=True)
class LivePromotionResult:
    """What ``LivePromotion.promote`` pinned — everything the US-010 armed
    summary restates. ``promoted`` is the live slot row exactly as written
    (its config carries the paper keys + live_equity_allocation_pct)."""

    promoted: PromotedStrategy
    source: str  # 'run' (direct promotion) or 'paper' (copy of the paper slot)
    source_thread_ts: str | None
    metrics: dict[str, float]
    sharpe: float | None
    universe_label: str | None
    warnings: tuple[str, ...]
    replaced: PromotedStrategy | None

    @property
    def universe(self) -> str:
        return str(self.promoted.config["universe"])

    @property
    def universe_tickers(self) -> tuple[str, ...] | None:
        tickers = self.promoted.config.get("universe_tickers")
        return None if tickers is None else tuple(tickers)

    @property
    def allocation_pct(self) -> float:
        return float(self.promoted.config["live_equity_allocation_pct"])

    @property
    def label_mismatch(self) -> bool:
        return self.universe_label is not None and self.universe_label != self.universe


class LivePromotion:
    """Promote-to-live backend: resolve source → snapshot → pin → Decision Log.

    Candidate resolution never guesses (US-009):

    1. an explicitly named run — a thread timestamp, or a fragment (>= 6
       chars) of the run's session path; ambiguity and no-match both refuse
       with a listing of what was found;
    2. otherwise the current thread's run, when the thread has one;
    3. otherwise the current paper-promoted strategy, copied IN FULL
       (universe provenance included — nothing is re-derived).

    Direct-run candidates go through ``PromotionFlow.candidate_from_run``
    (workspace conf ``market:`` line via execution.signal.load_market,
    tickers from the store instruments file) — the same provenance path as
    paper promotion. Success writes the live slot and a Decision Log row;
    ``demote`` clears the slot and records its own row. All refusals raise
    ``LivePromotionError`` and write nothing.
    """

    def __init__(
        self,
        store: StateStore,
        flow: PromotionFlow,
        recorder: NotionRecorder | None = None,
        load_allocation: Callable[
            [], allocation.LiveAllocation
        ] = allocation.load_live_allocation,
        snapshot: Callable[[Path], object] = pred_refresh.snapshot_pred_refresh,
    ) -> None:
        self._store = store
        self._flow = flow
        self._recorder = recorder
        self._load_allocation = load_allocation
        self._snapshot = snapshot

    # -- public API -------------------------------------------------------------

    def promote(
        self,
        reference: str | None = None,
        thread_ts: str | None = None,
        trigger_permalink: str | None = None,
    ) -> LivePromotionResult:
        """Pin the live slot from a named run, the thread's run, or the paper slot."""
        try:
            alloc = self._load_allocation()
        except allocation.AllocationConfigError as exc:
            raise LivePromotionError(f"live allocation config is unusable: {exc}") from exc

        run: Run | None = None
        if reference is not None and reference.strip():
            run = self._resolve_reference(reference)
        elif thread_ts is not None:
            run = self._store.get_run(thread_ts)

        if run is not None:
            candidate = self._run_candidate(run)
            workspace = candidate.workspace
            config = dict(candidate.config)
            metrics: dict[str, float] = candidate.metrics
            sharpe = candidate.sharpe
            source = "run"
            source_thread: str | None = run.thread_ts
            universe_label = candidate.universe_label
        else:
            paper = self._store.get_promoted_strategy()
            if paper is None:
                raise LivePromotionError(
                    "nothing to promote: no run was named, this thread has no"
                    " research run, and no paper-promoted strategy exists to copy"
                )
            workspace = Path(paper.workspace_path).expanduser()
            if not workspace.is_dir():
                raise LivePromotionError(
                    f"the paper-promoted workspace is missing on disk: {workspace}"
                    " — promote a run whose workspace still exists"
                )
            # Copied IN FULL: the paper slot's provenance (universe,
            # universe_tickers, topk/n_drop, thread_ts, session_path) is what
            # the operator already confirmed — never re-derive it here.
            config = dict(paper.config)
            metrics = self._workspace_metrics(workspace)
            sharpe = None
            source = "paper"
            thread = config.get("thread_ts")
            source_thread = thread if isinstance(thread, str) else None
            universe_label = None

        config["live_equity_allocation_pct"] = alloc.live_equity_allocation_pct
        warnings = self._ensure_snapshot(workspace)
        replaced = self._store.get_promoted_strategy_live()
        promoted = self._store.set_promoted_strategy_live(str(workspace), config)
        result = LivePromotionResult(
            promoted=promoted,
            source=source,
            source_thread_ts=source_thread,
            metrics=metrics,
            sharpe=sharpe,
            universe_label=universe_label,
            warnings=warnings,
            replaced=replaced,
        )
        logger.info(
            "promoted to LIVE from %s (workspace %s, allocation %s%%, replaced=%s)",
            source,
            promoted.workspace_path,
            alloc.live_equity_allocation_pct,
            replaced is not None,
        )
        self._record_promote_decision(result, trigger_permalink)
        return result

    def demote(self, trigger_permalink: str | None = None) -> PromotedStrategy:
        """Clear the live slot; returns what was demoted (refuses when empty)."""
        current = self._store.get_promoted_strategy_live()
        if current is None:
            raise LivePromotionError(
                "the live slot is already empty — nothing to demote"
            )
        self._store.clear_promoted_strategy_live()
        logger.info("demoted live strategy (workspace %s)", current.workspace_path)
        if self._recorder is not None:
            lines = [
                f"Workspace: {current.workspace_path}",
                f"Universe: {current.config.get('universe')}",
                "The live slot is now empty; the next live rebalance will abort"
                " with no promoted strategy.",
            ]
            if trigger_permalink is not None:
                lines.append(f"Triggered by: {trigger_permalink}")
            thread = current.config.get("thread_ts")
            self._recorder.record_decision(
                title=f"Demote live strategy ({Path(current.workspace_path).name})",
                decision_type=LIVE_DECISION_DEMOTE,
                details="\n".join(lines),
                thread_ts=thread if isinstance(thread, str) else None,
            )
        return current

    # -- internals ---------------------------------------------------------------

    def _resolve_reference(self, reference: str) -> Run:
        """One run for an explicit reference, or a refusal listing what matched."""
        ref = reference.strip().strip("`*_")
        runs = self._store.list_runs()
        matches = [run for run in runs if self._matches(ref, run)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = "\n".join(_run_line(run) for run in matches)
            raise LivePromotionError(
                f"'{reference}' is ambiguous — it matches {len(matches)} runs;"
                f" name one by its thread timestamp:\n{listing}"
            )
        promotable = [run for run in runs if run.status in PROMOTABLE_STATUSES]
        listing = "\n".join(_run_line(run) for run in promotable[-10:]) or "(none)"
        raise LivePromotionError(
            f"no run matches '{reference}' (searched {len(runs)} runs by thread"
            f" timestamp and session path). Promotable runs:\n{listing}"
        )

    @staticmethod
    def _matches(ref: str, run: Run) -> bool:
        if not ref:
            return False
        if ref == run.thread_ts:
            return True
        return len(ref) >= _MIN_REFERENCE_FRAGMENT and ref in run.session_path

    def _run_candidate(self, run: Run) -> PromotionCandidate:
        if run.status not in PROMOTABLE_STATUSES:
            raise LivePromotionError(
                f"run {run.thread_ts} is '{run.status}' — only a completed or"
                " operator-stopped run can be promoted to live"
            )
        try:
            return self._flow.candidate_from_run(run)
        except PromotionError as exc:
            raise LivePromotionError(str(exc)) from exc

    @staticmethod
    def _workspace_metrics(workspace: Path) -> dict[str, float]:
        """Headline metrics for a paper-slot copy; degrade to {} like paper does."""
        try:
            return summary.load_metrics(workspace / "qlib_res.csv")
        except (summary.SummaryError, OSError) as exc:
            logger.warning("live promotion without metrics for %s: %s", workspace, exc)
            return {}

    def _ensure_snapshot(self, workspace: Path) -> tuple[str, ...]:
        """Create the pred-refresh snapshot when incomplete; refuse on failure.

        A COMPLETE snapshot is never touched: its conf may carry an
        operator-pinned market (the frozen *_promoted_* universe pattern) and
        live must keep trading exactly what the operator pinned. Regenerating
        an incomplete set overwrites an existing conf, so that is warned
        about — never silent (the ops/promote_fetched.py rule). Unlike paper
        promotion's warn-don't-block, a snapshot failure REFUSES here: live
        must not arm on a workspace the morning refresh cannot keep fresh.
        """
        conf = workspace / pred_refresh.SNAPSHOT_CONF_NAME
        env = workspace / pred_refresh.SNAPSHOT_ENV_NAME
        params = workspace / pred_refresh.SNAPSHOT_PARAMS_NAME
        if conf.is_file() and env.is_file() and params.is_file():
            return ()
        warnings: list[str] = []
        if conf.is_file():
            warnings.append(
                f"an existing {pred_refresh.SNAPSHOT_CONF_NAME} was regenerated to"
                " complete the pred-refresh snapshot — re-apply any operator-pinned"
                " market (e.g. a frozen *_promoted_* universe) before the next refresh"
            )
        try:
            self._snapshot(workspace)
        except Exception as exc:  # noqa: BLE001 - any failure is a refusal here
            raise LivePromotionError(
                f"pred-refresh snapshot failed ({exc}) — refusing to arm live"
                " trading on a workspace the morning prediction refresh cannot"
                " keep fresh"
            ) from exc
        return tuple(warnings)

    def _record_promote_decision(
        self, result: LivePromotionResult, trigger_permalink: str | None
    ) -> None:
        if self._recorder is None:
            return
        config = result.promoted.config
        subject: str | None = None
        if result.source_thread_ts is not None:
            directive = self._store.get_directive(result.source_thread_ts)
            if directive is not None:
                subject = directive.objective
        tickers = result.universe_tickers
        universe_line = f"Universe: {result.universe}" + (
            f" ({len(tickers)} tickers pinned)"
            if tickers is not None
            else " (no tickers pinned)"
        )
        source_line = (
            "Source: paper-promoted strategy (copied in full)"
            if result.source == "paper"
            else f"Source: direct run promotion (thread {result.source_thread_ts})"
        )
        lines = [
            f"Workspace: {result.promoted.workspace_path}",
            source_line,
            universe_line,
            f"Allocation: {result.allocation_pct}% of live equity",
            f"TopkDropoutStrategy: topk={config.get('topk')}, n_drop={config.get('n_drop')}",
        ]
        if result.label_mismatch:
            lines.append(
                f"Universe label on the run row was '{result.universe_label}'"
                " (workspace conf market wins)"
            )
        for label, keys, _style in summary.METRIC_SPECS:
            value = next((result.metrics[k] for k in keys if k in result.metrics), None)
            if value is not None:
                lines.append(f"{label}: {value:.4f}")
        if result.sharpe is not None:
            lines.append(f"Sharpe: {result.sharpe:.4f}")
        if result.replaced is not None:
            lines.append(
                f"Replaced: {result.replaced.workspace_path}"
                f" (promoted {result.replaced.promoted_at})"
            )
        if trigger_permalink is not None:
            lines.append(f"Triggered by: {trigger_permalink}")
        self._recorder.record_decision(
            title=f"Promote '{subject or result.universe}' to LIVE trading",
            decision_type=LIVE_DECISION_PROMOTE,
            details="\n".join(lines),
            thread_ts=result.source_thread_ts,
        )

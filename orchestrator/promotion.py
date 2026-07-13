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

The other half of the story lives in execution/promoted.py: the rebalancer
entrypoint refuses to run when no promoted strategy exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution import signal
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
    """Everything the confirmation restates and the promotion pins."""

    run: Run
    workspace: Path
    universe: str
    params: signal.StrategyParams
    metrics: dict[str, float]
    sharpe: float | None

    @property
    def config(self) -> dict[str, Any]:
        """The strategy config pinned into promoted_strategy (what trades)."""
        tickers = self.run.universe_tickers
        return {
            "universe": self.universe,
            "universe_tickers": None if tickers is None else list(tickers),
            "topk": self.params.topk,
            "n_drop": self.params.n_drop,
            "thread_ts": self.run.thread_ts,
            "session_path": self.run.session_path,
        }


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
    lines = [
        ":rocket: *Confirm promotion to paper trading*",
        "The nightly rebalancer will trade this strategy:",
        f"• *Universe:* `{candidate.universe}`",
        f"• *Strategy:* TopkDropoutStrategy — topk={candidate.params.topk},"
        f" n_drop={candidate.params.n_drop}",
        f"• *Workspace:* `{candidate.workspace}`",
        summary.format_summary(candidate.metrics, candidate.sharpe),
    ]
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
    ) -> None:
        self._store = store
        self._recorder = recorder
        self._locate = locate
        self._load_params = load_params

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
        if previous is not None:
            lines.append(
                f":arrows_counterclockwise: Replaced the previously promoted strategy"
                f" (workspace `{previous.workspace_path}`, promoted {previous.promoted_at})."
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
        # Metrics degrade to n/a rather than blocking: the operator already saw
        # (and is acting on) the completion summary built from the same files.
        metrics: dict[str, float] = {}
        try:
            metrics = summary.load_metrics(artifacts.qlib_res_csv)
        except summary.SummaryError as exc:
            logger.warning("promotion confirmation without metrics for %s: %s", thread_ts, exc)
        sharpe: float | None = None
        if artifacts.ret_pkl is not None:
            try:
                sharpe = summary.compute_sharpe(artifacts.ret_pkl)
            except summary.SummaryError:
                sharpe = None
        return PromotionCandidate(
            run=run,
            workspace=Path(artifacts.workspace_path),
            universe=run.universe or FALLBACK_UNIVERSE,
            params=params,
            metrics=metrics,
            sharpe=sharpe,
        )

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

"""Conversational core: refine operator ideas into saved research directives.

Built on ModelRouter.judgment_tool_loop (US-008) — the save_directive tool is
a ToolSpec whose handler persists to StateStore and posts a formatted summary
to the Slack thread; start_research (US-020) launches an RD-Agent run for the
saved directive and records the thread<->session mapping in the runs table.

Durability model: in-memory transcripts are best-effort (bounded, lost on
restart); the durable context is the saved directive, which reloads from
SQLite into the system prompt on every call. A restart loses chit-chat but
never the directive.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from orchestrator import prompts
from orchestrator.llm import LLMError, ModelRouter, RefusalError, ToolSpec
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.rdagent_client import RunHandle
from orchestrator.state import (
    Directive,
    DuplicateRunError,
    PendingInteraction,
    Run,
    StateStore,
)

if TYPE_CHECKING:
    from execution.alpaca_client import Account, Order, PortfolioHistory, Position
    from orchestrator.universe import MaterializedUniverse, UniverseProposal

logger = logging.getLogger(__name__)

# slack_bolt's Say or any equivalent accepting (text=..., thread_ts=...).
SayFn = Callable[..., Any]

# Cap the per-thread transcript sent to the model (user+assistant messages).
MAX_HISTORY_MESSAGES = 40

REFUSAL_REPLY = "I can't help with that request."

# The only universe wired end-to-end today (store + templates + factor source).
# Per-run custom universes are US-023's set_universe tool.
DEFAULT_UNIVERSE = "us_liquid"


class ResearchLauncher(Protocol):
    """What the run-lifecycle tools need from RdAgentClient (stub-friendly)."""

    def start_run(self, directive: str, universe: str) -> RunHandle: ...

    def trace_dir(self, trace_id: str) -> Path: ...

    def trace_id_of(self, session_path: str | Path) -> str: ...

    def stop(self, trace_id: str) -> None: ...

    def resume(
        self,
        trace_id: str,
        session_path: str | Path | None = None,
        *,
        directive: str | None = None,
        universe: str = "",
    ) -> None: ...

class UniverseManager(Protocol):
    """What the set_universe tools need from UniverseService (stub-friendly)."""

    def propose(self, name: str, tickers: Sequence[str]) -> UniverseProposal: ...

    def materialize(self, name: str, tickers: Sequence[str]) -> MaterializedUniverse: ...


class TradingBreaker(Protocol):
    """What the halt/resume tools need from execution.breaker.Breaker."""

    halt_file: Path

    @property
    def halted(self) -> bool: ...

    @property
    def halt_note(self) -> str: ...

    def halt(self, note: str = "") -> None: ...

    def clear_halt(self) -> None: ...


class BrokerReader(Protocol):
    """What the read-only account tools need from AlpacaClient (US-046).

    Strictly the read endpoints — the conversational core must never hold a
    handle that can place, cancel, or liquidate (trading stays with the
    nightly rebalancer; the only trading control here is the breaker halt).
    """

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def list_orders(
        self,
        status: str = "open",
        limit: int | None = None,
        symbols: list[str] | None = None,
        after: str | None = None,
        until: str | None = None,
    ) -> list[Order]: ...

    def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> PortfolioHistory: ...


class HypothesisSteering(Protocol):
    """What the hypothesis-decision tools need from HypothesisPoller.

    The poller's button handlers already own the full submit/resolve/notify
    dance (and post their own outcome to the thread) — the conversational
    tools reuse them verbatim so a spoken "approve" and an Approve click are
    the same code path.
    """

    def approve(self, interaction_id: int, say: SayFn) -> None: ...

    def reject(self, interaction_id: int, say: SayFn) -> None: ...


class PromotionManager(Protocol):
    """What the promotion tools need from PromotionFlow (stub-friendly)."""

    def request_promotion(self, thread_ts: str, say: SayFn) -> None: ...

    def confirm_promotion(self, thread_ts: str, say: SayFn) -> None: ...


START_RESEARCH_SCHEMA: dict[str, Any] = {
    # The run is driven by the thread's saved directive; the only knob is the
    # steering mode (US-045: autonomous is the default).
    "type": "object",
    "properties": {
        "supervised": {
            "type": "boolean",
            "description": (
                "Pass true ONLY when the operator explicitly asks to approve"
                " each hypothesis themselves. Default (false): the run"
                " auto-approves its hypotheses and stops on its own budget."
            ),
        },
    },
}

STOP_RUN_SCHEMA: dict[str, Any] = {
    # No inputs: stops the thread's run.
    "type": "object",
    "properties": {},
}

RESUME_RUN_SCHEMA: dict[str, Any] = {
    # No inputs: resumes the thread's run from its stored session.
    "type": "object",
    "properties": {},
}

SET_UNIVERSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short lowercase snake_case universe name, e.g. ai_semis.",
        },
        "tickers": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Explicit US ticker list — the operator's, or your proposal of"
                " liquid names fitting the idea."
            ),
        },
    },
    "required": ["name", "tickers"],
}

CONFIRM_UNIVERSE_SCHEMA: dict[str, Any] = {
    # No inputs: confirms the thread's stored proposal.
    "type": "object",
    "properties": {},
}

HALT_TRADING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": (
                "Why trading is being halted, in the operator's words — written"
                " into the halt file and the Decision Log."
            ),
        },
    },
}

RESUME_TRADING_SCHEMA: dict[str, Any] = {
    # No inputs: removes the breaker halt file.
    "type": "object",
    "properties": {},
}

CHECK_ACCOUNT_SCHEMA: dict[str, Any] = {
    # No inputs: one fresh snapshot of the paper account and its positions.
    "type": "object",
    "properties": {},
}

CHECK_ORDERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["open", "closed", "all"],
            "description": (
                "Which orders to list: open (still working), closed"
                " (terminal), or all (default — newest first)."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Max orders to return (default 10, max 50).",
        },
    },
}

CHECK_PNL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "period": {
            "type": "string",
            "enum": ["1W", "1M", "3M", "1A", "all"],
            "description": "Lookback window for the P/L history (default 1M).",
        },
    },
}

APPROVE_HYPOTHESIS_SCHEMA: dict[str, Any] = {
    # No inputs: acts on the thread's oldest awaiting hypothesis (FIFO).
    "type": "object",
    "properties": {},
}

REJECT_HYPOTHESIS_SCHEMA: dict[str, Any] = {
    # No inputs: acts on the thread's oldest awaiting hypothesis (FIFO).
    "type": "object",
    "properties": {},
}

PROMOTE_RUN_SCHEMA: dict[str, Any] = {
    # No inputs: promotes the thread's finished run.
    "type": "object",
    "properties": {},
}

CONFIRM_PROMOTION_SCHEMA: dict[str, Any] = {
    # No inputs: confirms the thread's requested promotion.
    "type": "object",
    "properties": {},
}

SAVE_DIRECTIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {
            "type": "string",
            "description": "One concrete, testable sentence describing what to research.",
        },
        "universe_hint": {
            "type": "string",
            "description": "Market/sector/ticker scope the operator gave, if any.",
        },
        "constraints": {
            "type": "string",
            "description": (
                "Risk limits, factor style, holding period — anything the operator"
                " ruled in or out."
            ),
        },
    },
    "required": ["objective"],
}


def format_directive_summary(directive: Directive) -> str:
    """Slack mrkdwn summary posted to the thread when a directive is saved."""
    return (
        f"*Research directive saved* (#{directive.id})\n"
        f"*Objective:* {directive.objective}\n"
        f"*Universe:* {directive.universe_hint or '_none given_'}\n"
        f"*Constraints:* {directive.constraints or '_none given_'}"
    )


def directive_instruction(directive: Directive) -> str:
    """Directive rendered as the run's user_instruction.

    universe_hint is deliberately excluded: the enforced universe is passed to
    start_run separately, and mapping free-text hints onto real universes is
    US-023's set_universe job.
    """
    if directive.constraints:
        return f"{directive.objective}\nConstraints: {directive.constraints}"
    return directive.objective


def format_run_started(run: Run) -> str:
    """Slack mrkdwn confirmation posted to the thread when a run starts."""
    if run.supervised:
        tail = "Hypotheses will be posted here for approval as the loop proposes them."
    else:
        tail = (
            "The loop will try its hypotheses autonomously and narrate each one"
            " here — no approvals needed. It stops on its own after its"
            " hypothesis budget and posts the best result found."
        )
    return (
        "*Research run started*\n"
        f"*Universe:* {run.universe}\n"
        f"*Session:* `{run.session_path}`\n"
        f"{tail}"
    )


def format_universe_proposal(proposal: UniverseProposal) -> str:
    """Slack mrkdwn proposal posted for operator confirmation (before data work)."""
    lines = [
        f"*Custom universe proposed: `{proposal.name}`* ({len(proposal.tickers)} tickers)",
        f"`{', '.join(proposal.tickers)}`",
    ]
    lines.extend(f":warning: {warning}" for warning in proposal.warnings)
    lines.append(
        "Confirm this list and I'll build the universe data; nothing is built until you do."
    )
    return "\n".join(lines)


def format_universe_ready(materialized: MaterializedUniverse) -> str:
    """Slack mrkdwn notice posted once the confirmed universe is materialized."""
    return (
        f"*Universe `{materialized.name}` is ready* ({len(materialized.tickers)} tickers)\n"
        f"*Instruments:* `{materialized.instruments_path}`\n"
        f"*Factor source:* `{materialized.factor_source}`\n"
        f"*Templates:* `{materialized.templates_dir}` (market: {materialized.name})\n"
        "start_research in this thread will now use it."
    )


def format_run_stopped(run: Run, cancelled_interactions: int) -> str:
    """Slack mrkdwn confirmation posted when the operator stops the run."""
    text = (
        ":octagonal_sign: *Research run stopped.*\n"
        f"*Session:* `{run.session_path}`\n"
        "Progress up to the last completed step is checkpointed — say the word"
        " and I'll resume it from there."
    )
    if cancelled_interactions:
        text += (
            f"\n_{cancelled_interactions} open hypothesis prompt(s) above were"
            " cancelled — the run will re-propose after a resume._"
        )
    return text


def format_run_resumed(run: Run) -> str:
    """Slack mrkdwn confirmation posted when a stopped run is resumed."""
    return (
        ":arrow_forward: *Research run resumed*\n"
        f"*Universe:* {run.universe}\n"
        f"*Session:* `{run.session_path}`\n"
        "Picking up from the last checkpoint — new hypotheses will be posted"
        " here for approval."
    )


def format_trading_halted(note: str, halt_file: Path) -> str:
    """Slack mrkdwn confirmation posted when the operator halts trading."""
    return (
        ":octagonal_sign: *Paper trading halted.*\n"
        f"*Reason:* {note}\n"
        f"*Halt file:* `{halt_file}`\n"
        "Every rebalance run will post a halted notice and submit no orders"
        " until you resume trading."
    )


def format_trading_resumed(halt_file: Path) -> str:
    """Slack mrkdwn confirmation posted when the operator resumes trading."""
    return (
        ":arrow_forward: *Paper trading resumed.*\n"
        f"*Halt file:* `{halt_file}` removed — the nightly rebalance will"
        " trade again from its next run."
    )


def _signed_usd(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def _signed_pct(fraction: float) -> str:
    return f"{fraction * 100:+.2f}%"


def format_account_report(
    account: Account, positions: Sequence[Position], trading_state: str
) -> str:
    """Plain-text account snapshot returned to the model by check_account."""
    lines = [
        f"paper account ({account.status.lower() or 'unknown status'})",
        f"equity: ${account.equity:,.2f}",
    ]
    if account.last_equity is not None:
        change = account.equity - account.last_equity
        pct = f" ({_signed_pct(change / account.last_equity)})" if account.last_equity else ""
        lines.append(f"since previous close: {_signed_usd(change)}{pct}")
    lines.append(f"cash: ${account.cash:,.2f}; buying power: ${account.buying_power:,.2f}")
    if not positions:
        lines.append("positions: none (flat)")
    else:
        lines.append(f"positions ({len(positions)}):")
        for p in sorted(positions, key=lambda p: p.symbol):
            value = f"${p.market_value:,.2f}" if p.market_value is not None else "value n/a"
            pl = ""
            if p.unrealized_pl is not None:
                pl = f", unrealized {_signed_usd(p.unrealized_pl)}"
                if p.unrealized_plpc is not None:
                    pl += f" ({_signed_pct(p.unrealized_plpc)})"
            lines.append(f"  {p.symbol}: {p.qty:g} @ avg ${p.avg_entry_price:,.2f}, {value}{pl}")
    lines.append(f"trading: {trading_state}")
    return "\n".join(lines)


def format_orders_report(orders: Sequence[Order], status: str) -> str:
    """Plain-text order list (newest first) returned to the model by check_orders."""
    if not orders:
        return f"no {status} orders found on the paper account"
    lines = [f"{len(orders)} {status} order(s), newest first:"]
    for order in orders:
        stamp = (order.submitted_at or "unknown time").replace("T", " ")[:16]
        qty = order.qty if order.qty is not None else order.filled_qty
        limit = f" @ limit ${order.limit_price:,.2f}" if order.limit_price is not None else ""
        if order.status == "filled" and order.filled_avg_price is not None:
            fill = f"filled {order.filled_qty:g} @ ${order.filled_avg_price:,.2f}"
        elif order.filled_qty:
            fill = f"{order.status}, {order.filled_qty:g} filled"
        else:
            fill = order.status
        lines.append(f"  {stamp} {order.side} {qty:g} {order.symbol}{limit} — {fill}")
    return "\n".join(lines)


def format_pnl_report(history: PortfolioHistory, period: str) -> str:
    """Plain-text equity/P-L history returned to the model by check_pnl."""
    valued = [e for e in history.entries if e.equity]
    if not valued:
        return f"no portfolio history with equity values for period {period}"
    base = history.base_value if history.base_value else valued[0].equity
    last = valued[-1]
    assert last.equity is not None  # `valued` filtered on truthy equity
    lines = [f"portfolio P/L over {period} (daily points):"]
    if base:
        change = last.equity - base
        lines.append(
            f"period total: {_signed_usd(change)} ({_signed_pct(change / base)}) — "
            f"equity ${base:,.2f} -> ${last.equity:,.2f}"
        )
    for entry in valued[-10:]:
        day_pl = "P/L n/a"
        if entry.profit_loss is not None:
            day_pl = _signed_usd(entry.profit_loss)
            if entry.profit_loss_pct is not None:
                day_pl += f" ({_signed_pct(entry.profit_loss_pct)})"
        equity = f"${entry.equity:,.2f}" if entry.equity is not None else "n/a"
        lines.append(f"  {entry.date.isoformat()}: equity {equity}, day {day_pl}")
    if len(valued) > 10:
        lines.append(f"  (showing the last 10 of {len(valued)} days)")
    return "\n".join(lines)


def duplicate_run_message(existing: Run) -> str:
    return (
        f"this thread already has a research run (status: {existing.status}, "
        f"session: {existing.session_path}). One run per thread — follow that "
        "run here, or start a new thread for another run."
    )


def _clean_optional(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _final_text(message: Any) -> str:
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    return "\n".join(parts).strip() or "Done."


class ConversationCore:
    """Per-thread Claude conversations with the desk's Slack-facing tools.

    Share one instance per process (like StateStore/ModelRouter); the Bolt
    message handler calls handle_message for every actionable message.
    ``rdagent`` is injectable for tests; by default it is the real client
    talking to the supervised server_ui instance (US-018).
    """

    def __init__(
        self,
        store: StateStore,
        router: ModelRouter,
        rdagent: ResearchLauncher | None = None,
        universes: UniverseManager | None = None,
        recorder: NotionRecorder | None = None,
        breaker: TradingBreaker | None = None,
        broker: BrokerReader | None = None,
        interactions: HypothesisSteering | None = None,
        promotions: PromotionManager | None = None,
    ) -> None:
        if rdagent is None:
            from orchestrator.rdagent_client import RdAgentClient

            rdagent = RdAgentClient()
        if universes is None:
            from orchestrator.universe import UniverseService

            universes = UniverseService()
        if breaker is None:
            # The real kill switch: ~/rdq-data/breaker/halt, shared with the
            # rebalancer (execution/breaker.py default paths).
            from execution.breaker import Breaker, load_breaker_config

            breaker = Breaker(load_breaker_config())
        if broker is None:
            # Read-only paper-account visibility (US-046): rdq-orchestrator
            # holds the paper-api.alpaca.markets secret, so the same proxy
            # injection that serves the rebalancer serves these reads.
            from execution.alpaca_client import AlpacaClient

            broker = AlpacaClient()
        self._store = store
        self._router = router
        self._rdagent: ResearchLauncher = rdagent
        self._universes: UniverseManager = universes
        self._breaker: TradingBreaker = breaker
        self._broker: BrokerReader = broker
        # Optional Notion audit trail (US-027); None disables recording.
        self._recorder = recorder
        # Optional wiring to the poller's button handlers / promotion flow.
        # None (tests, partial wiring) simply leaves the matching tools out of
        # the loop — the model never sees a tool it cannot execute.
        self._interactions = interactions
        self._promotions = promotions
        self._histories: dict[str, list[dict[str, Any]]] = {}

    def handle_message(self, thread_ts: str, text: str, say: SayFn) -> str:
        """Run one conversational turn; posts the model's reply in-thread.

        Returns the reply text (also posted via say). Refusals and model
        errors are reported in-thread, never raised into the Bolt handler.
        """
        history = self._histories.setdefault(thread_ts, [])
        history.append({"role": "user", "content": text})
        tools = [
            self._save_directive_tool(thread_ts, say),
            self._start_research_tool(thread_ts, say),
            self._stop_run_tool(thread_ts, say),
            self._resume_run_tool(thread_ts, say),
            self._set_universe_tool(thread_ts, say),
            self._confirm_universe_tool(thread_ts, say),
            self._halt_trading_tool(thread_ts, say),
            self._resume_trading_tool(thread_ts, say),
            self._check_account_tool(),
            self._check_orders_tool(),
            self._check_pnl_tool(),
        ]
        if self._interactions is not None:
            tools.append(self._approve_hypothesis_tool(thread_ts, say))
            tools.append(self._reject_hypothesis_tool(thread_ts, say))
        if self._promotions is not None:
            tools.append(self._promote_run_tool(thread_ts, say))
            tools.append(self._confirm_promotion_tool(thread_ts, say))
        try:
            final = self._router.judgment_tool_loop(
                history,
                tools,
                system=self._system_prompt(thread_ts),
            )
        except RefusalError:
            history.pop()  # keep the transcript consistent with what the model accepted
            reply = REFUSAL_REPLY
        except LLMError as exc:
            history.pop()
            logger.exception("model call failed for thread %s", thread_ts)
            reply = f"Model call failed ({exc}). Please try again."
        else:
            reply = _final_text(final)
            history.append({"role": "assistant", "content": reply})
            del history[:-MAX_HISTORY_MESSAGES]
        say(text=reply, thread_ts=thread_ts)
        return reply

    def _system_prompt(self, thread_ts: str) -> str:
        """SYSTEM_PROMPT plus the thread's saved directive reloaded from SQLite."""
        directive = self._store.get_directive(thread_ts)
        if directive is None:
            return prompts.SYSTEM_PROMPT
        return prompts.SYSTEM_PROMPT + "\n\n" + prompts.directive_context(directive)

    def _save_directive_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            objective = str(args.get("objective") or "").strip()
            if not objective:
                raise ValueError("objective must be a non-empty sentence")
            directive = self._store.create_directive(
                thread_ts,
                objective=objective,
                universe_hint=_clean_optional(args.get("universe_hint")),
                constraints=_clean_optional(args.get("constraints")),
            )
            say(text=format_directive_summary(directive), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_idea(
                    thread_ts,
                    raw_idea=self._raw_idea(thread_ts) or objective,
                    directive=directive,
                    universe=self._confirmed_universe_name(thread_ts) or DEFAULT_UNIVERSE,
                )
            logger.info("saved directive #%s for thread %s", directive.id, thread_ts)
            return (
                f"Directive #{directive.id} saved for this thread and the summary"
                " was posted. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="save_directive",
            description=(
                "Persist the refined research directive for this thread and post a"
                " formatted summary. Call it once, when the idea is concrete enough"
                " to research. Saving again replaces the thread's directive."
            ),
            input_schema=SAVE_DIRECTIVE_SCHEMA,
            handler=handler,
        )

    def _start_research_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            supervised = bool(args.get("supervised", False))
            directive = self._store.get_directive(thread_ts)
            if directive is None:
                raise ValueError(
                    "no research directive is saved for this thread yet — refine"
                    " the idea with the operator and call save_directive first"
                )
            existing = self._store.get_run(thread_ts)
            if existing is not None:
                raise ValueError(duplicate_run_message(existing))
            universe = DEFAULT_UNIVERSE
            tickers: list[str] | None = None
            record = self._store.get_thread_universe(thread_ts)
            if record is not None:
                if record.status != "confirmed":
                    raise ValueError(
                        f"universe '{record.name}' is proposed but not confirmed —"
                        " have the operator confirm the ticker list, then call"
                        " confirm_universe before starting the research"
                    )
                universe = record.name
                tickers = list(record.tickers)
            handle = self._rdagent.start_run(directive_instruction(directive), universe)
            session_path = str(self._rdagent.trace_dir(handle.trace_id))
            try:
                run = self._store.create_run(
                    thread_ts,
                    session_path,
                    universe=universe,
                    universe_tickers=tickers,
                    supervised=supervised,
                )
            except DuplicateRunError as exc:
                # Lost a start race — don't leave the just-launched run orphaned.
                self._rdagent.stop(handle.trace_id)
                raise ValueError(duplicate_run_message(exc.existing)) from exc
            say(text=format_run_started(run), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_idea_status(thread_ts, "researching", universe=universe)
            logger.info("started research run %s for thread %s", handle.trace_id, thread_ts)
            if supervised:
                return (
                    f"Research run started SUPERVISED (trace {handle.trace_id}) and"
                    " recorded for this thread; the start notice was posted. Confirm"
                    " briefly to the operator — hypotheses will arrive in this thread"
                    " for their approval."
                )
            return (
                f"Research run started (trace {handle.trace_id}) and recorded for"
                " this thread; the start notice was posted. Confirm briefly to the"
                " operator — the run is autonomous: hypotheses are auto-approved and"
                " narrated in this thread, and the run stops by itself after its"
                " hypothesis budget, posting the best result for promotion."
            )

        return ToolSpec(
            name="start_research",
            description=(
                "Start an RD-Agent research run for this thread's SAVED directive."
                " Requires save_directive to have been called first; only one run"
                " may exist per thread. Call it only when the operator explicitly"
                " asks to start the research. By default the run is autonomous"
                " (hypotheses auto-approved, self-stopping); pass supervised=true"
                " only when the operator explicitly wants to approve each"
                " hypothesis themselves."
            ),
            input_schema=START_RESEARCH_SCHEMA,
            handler=handler,
        )

    def _stop_run_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — stops the thread's run
            run = self._store.get_run(thread_ts)
            if run is None:
                raise ValueError("no research run exists in this thread — nothing to stop")
            if run.status != "running":
                raise ValueError(
                    f"the run in this thread is not running (status: {run.status})"
                    " — nothing to stop"
                )
            self._rdagent.stop(self._rdagent.trace_id_of(run.session_path))
            cancelled = self._cancel_open_interactions(thread_ts)
            run = self._store.update_run_status(thread_ts, "stopped")
            say(text=format_run_stopped(run, cancelled), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_idea_status(thread_ts, "stopped")
            logger.info("stopped research run %s for thread %s", run.session_path, thread_ts)
            return (
                "The research run was stopped and its row marked 'stopped'; the"
                " stop notice was posted. It can be resumed later with resume_run."
                " Confirm briefly to the operator."
            )

        return ToolSpec(
            name="stop_run",
            description=(
                "Stop this thread's in-flight research run (checkpointed — it can"
                " be resumed later with resume_run). Call it only when the operator"
                " explicitly asks to stop, pause, or kill the run."
            ),
            input_schema=STOP_RUN_SCHEMA,
            handler=handler,
        )

    def _resume_run_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — resumes the thread's stored session
            run = self._store.get_run(thread_ts)
            if run is None:
                raise ValueError(
                    "no research run exists in this thread — start one with"
                    " start_research instead"
                )
            if run.status == "running":
                raise ValueError(
                    f"the run in this thread is already running (session:"
                    f" {run.session_path}) — nothing to resume"
                )
            directive = self._store.get_directive(thread_ts)
            if directive is None:
                raise ValueError(
                    "no saved directive for this thread — a resumed run re-asks for"
                    " its instruction, so save_directive must be called first"
                )
            self._rdagent.resume(
                self._rdagent.trace_id_of(run.session_path),
                run.session_path,
                directive=directive_instruction(directive),
                universe=run.universe or DEFAULT_UNIVERSE,
            )
            run = self._store.update_run_status(thread_ts, "running")
            say(text=format_run_resumed(run), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_idea_status(thread_ts, "researching")
            logger.info("resumed research run %s for thread %s", run.session_path, thread_ts)
            return (
                "The run was resumed from its stored session and its row is"
                " 'running' again, so hypothesis polling is re-activated; the"
                " resume notice was posted. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="resume_run",
            description=(
                "Resume this thread's previously stopped research run from its"
                " checkpointed session. Call it only when the operator explicitly"
                " asks to resume/continue the run."
            ),
            input_schema=RESUME_RUN_SCHEMA,
            handler=handler,
        )

    def _cancel_open_interactions(self, thread_ts: str) -> int:
        """Mark the thread's unanswered hypothesis prompts 'cancelled'.

        A stopped run's IPC queues are gone, so pending/editing rows can never
        be submitted; leaving them actionable would wedge the poller's FIFO
        guard after a resume (the resumed run re-proposes under fresh keys).
        """
        cancelled = 0
        for status in ("pending", "editing"):
            for row in self._store.list_pending_interactions(thread_ts, status=status):
                self._store.resolve_pending_interaction(row.id, "cancelled")
                if self._recorder is not None:
                    self._recorder.record_hypothesis_action(row.interaction_key, "cancelled")
                cancelled += 1
        return cancelled

    def _raw_idea(self, thread_ts: str) -> str | None:
        """The operator's first message this process saw for the thread.

        Best effort: the in-memory transcript is lost on restart, so after one
        the earliest retained message (usually the one that triggered the
        save) stands in for the original idea.
        """
        for msg in self._histories.get(thread_ts, []):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None

    def _confirmed_universe_name(self, thread_ts: str) -> str | None:
        record = self._store.get_thread_universe(thread_ts)
        if record is not None and record.status == "confirmed":
            return record.name
        return None

    def _set_universe_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            raw = args.get("tickers")
            if not isinstance(raw, list):
                raise ValueError("tickers must be a list of symbols")
            existing = self._store.get_run(thread_ts)
            if existing is not None:
                raise ValueError(
                    f"this thread already has a research run (status: {existing.status})"
                    " — universes apply to new runs; start a fresh thread for a"
                    " different universe"
                )
            proposal = self._universes.propose(
                str(args.get("name") or ""), [str(t) for t in raw]
            )
            self._store.propose_thread_universe(
                thread_ts, proposal.name, list(proposal.tickers)
            )
            say(text=format_universe_proposal(proposal), thread_ts=thread_ts)
            logger.info(
                "proposed universe '%s' (%d tickers) for thread %s",
                proposal.name,
                len(proposal.tickers),
                thread_ts,
            )
            return (
                f"Universe '{proposal.name}' ({len(proposal.tickers)} tickers) was"
                " posted for confirmation. No data work happened yet — ask the"
                " operator to confirm the ticker list, and call confirm_universe"
                " only after they explicitly confirm."
            )

        return ToolSpec(
            name="set_universe",
            description=(
                "Propose a named custom ticker universe for this thread's future"
                " research run (when the idea targets specific tickers/sectors"
                " rather than the broad market). Posts the list for operator"
                " confirmation; NO data is built until confirm_universe. Proposing"
                " again replaces the thread's proposal. Broad-market ideas should"
                " NOT use this — runs default to the built-in us_liquid universe."
            ),
            input_schema=SET_UNIVERSE_SCHEMA,
            handler=handler,
        )

    def _confirm_universe_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — confirms the thread's stored proposal
            record = self._store.get_thread_universe(thread_ts)
            if record is None:
                raise ValueError(
                    "no universe proposal exists for this thread — call set_universe"
                    " first"
                )
            if record.status == "confirmed":
                raise ValueError(
                    f"universe '{record.name}' is already confirmed for this thread"
                )
            materialized = self._universes.materialize(record.name, list(record.tickers))
            self._store.confirm_thread_universe(thread_ts)
            say(text=format_universe_ready(materialized), thread_ts=thread_ts)
            logger.info(
                "materialized universe '%s' for thread %s", record.name, thread_ts
            )
            return (
                f"Universe '{record.name}' is built (instruments file, factor source,"
                " US templates rendered with market:"
                f" {record.name}) and confirmed for this thread. start_research will"
                " use it. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="confirm_universe",
            description=(
                "Materialize this thread's PROPOSED universe after the operator has"
                " explicitly confirmed the posted ticker list: validates tickers"
                " against the data store, writes the instruments file, regenerates"
                " the factor source, and renders the run's template copy. Never call"
                " it before the operator confirms."
            ),
            input_schema=CONFIRM_UNIVERSE_SCHEMA,
            handler=handler,
        )

    def _halt_trading_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            if self._breaker.halted:
                note = self._breaker.halt_note
                detail = f" ({note})" if note else ""
                raise ValueError(
                    f"trading is already halted{detail} — resume_trading lifts it"
                )
            reason = str(args.get("reason") or "").strip()
            note = reason or f"halted from Slack thread {thread_ts}"
            self._breaker.halt(note)
            say(text=format_trading_halted(note, self._breaker.halt_file), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_decision(
                    title="Trading halted",
                    decision_type="halt",
                    details=f"Reason: {note}. Halt file: {self._breaker.halt_file}.",
                    thread_ts=thread_ts,
                )
            logger.info("trading halted from thread %s: %s", thread_ts, note)
            return (
                "The breaker halt file was written: every rebalance run now exits"
                " with a halted notice and submits no orders until resume_trading."
                " The halt notice was posted. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="halt_trading",
            description=(
                "HALT all paper trading immediately by writing the circuit-breaker"
                " halt file — the nightly rebalancer submits no orders while it"
                " exists. Call it only when the operator explicitly asks to halt or"
                " stop trading; it does not touch research runs (that is stop_run)."
            ),
            input_schema=HALT_TRADING_SCHEMA,
            handler=handler,
        )

    def _resume_trading_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — removes the halt file
            if not self._breaker.halted:
                raise ValueError(
                    "trading is not halted — there is no halt file to remove"
                )
            note = self._breaker.halt_note
            self._breaker.clear_halt()
            say(text=format_trading_resumed(self._breaker.halt_file), thread_ts=thread_ts)
            if self._recorder is not None:
                was = f" (was: {note})" if note else ""
                self._recorder.record_decision(
                    title="Trading resumed",
                    decision_type="resume",
                    details=f"Halt lifted{was}. Halt file {self._breaker.halt_file} removed.",
                    thread_ts=thread_ts,
                )
            logger.info("trading resumed from thread %s (halt note was: %s)", thread_ts, note)
            return (
                "The breaker halt file was removed: the nightly rebalance will"
                " trade again from its next run. The resume notice was posted."
                " Confirm briefly to the operator."
            )

        return ToolSpec(
            name="resume_trading",
            description=(
                "RESUME paper trading by removing the circuit-breaker halt file"
                " written by halt_trading. Call it only when the operator"
                " explicitly asks to resume trading; it does not touch research"
                " runs (that is resume_run)."
            ),
            input_schema=RESUME_TRADING_SCHEMA,
            handler=handler,
        )

    def _trading_state_line(self) -> str:
        """One line of breaker context for the account report (never raises)."""
        try:
            if self._breaker.halted:
                note = self._breaker.halt_note
                return f"HALTED{f' — {note}' if note else ''} (resume_trading lifts it)"
            return "active (nightly rebalancer will trade the promoted strategy)"
        except Exception as exc:  # noqa: BLE001 - breaker state must not sink a read
            return f"breaker state unreadable ({exc})"

    def _check_account_tool(self) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — one fresh snapshot
            account = self._broker.get_account()
            positions = self._broker.get_positions()
            return format_account_report(account, positions, self._trading_state_line())

        return ToolSpec(
            name="check_account",
            description=(
                "READ-ONLY snapshot of the desk's Alpaca paper account: equity,"
                " P/L since the previous close, cash, buying power, every open"
                " position with its unrealized P/L, and whether trading is"
                " halted. Use it whenever the operator asks about the account,"
                " the book, positions, or how we're doing today."
            ),
            input_schema=CHECK_ACCOUNT_SCHEMA,
            handler=handler,
        )

    def _check_orders_tool(self) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            status = str(args.get("status") or "all")
            if status not in ("open", "closed", "all"):
                raise ValueError("status must be one of open/closed/all")
            limit = args.get("limit")
            limit = 10 if limit is None else max(1, min(int(limit), 50))
            orders = self._broker.list_orders(status=status, limit=limit)
            return format_orders_report(orders, status)

        return ToolSpec(
            name="check_orders",
            description=(
                "READ-ONLY list of the paper account's orders (newest first)"
                " with fill status and prices. Use it whenever the operator"
                " asks whether orders were placed or executed — e.g. last"
                " night's rebalance. It cannot place or cancel anything."
            ),
            input_schema=CHECK_ORDERS_SCHEMA,
            handler=handler,
        )

    def _check_pnl_tool(self) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            period = str(args.get("period") or "1M")
            if period not in ("1W", "1M", "3M", "1A", "all"):
                raise ValueError("period must be one of 1W/1M/3M/1A/all")
            history = self._broker.get_portfolio_history(period=period, timeframe="1D")
            return format_pnl_report(history, period)

        return ToolSpec(
            name="check_pnl",
            description=(
                "READ-ONLY daily equity and P/L history of the paper account"
                " over a lookback window (default 1M): period total plus the"
                " last few daily P/L points. Use it when the operator asks"
                " about performance, returns, or P/L over time; for just"
                " today's number, check_account already reports it."
            ),
            input_schema=CHECK_PNL_SCHEMA,
            handler=handler,
        )

    def _awaiting_hypothesis(self, thread_ts: str) -> PendingInteraction:
        """The thread's oldest hypothesis awaiting a decision, or raise.

        FIFO rule (see poller.py): submitted answers go to the run's oldest
        blocked request, so the tools may only ever act on the oldest row.
        """
        rows = self._store.list_pending_interactions(thread_ts, status="pending")
        if rows:
            return rows[0]
        if self._store.list_pending_interactions(thread_ts, status="editing"):
            raise ValueError(
                "the proposed hypothesis is mid-edit — the operator's next plain"
                " reply in this thread is consumed as the revised hypothesis"
                " text, so there is nothing to approve or reject right now"
            )
        raise ValueError(
            "no proposed hypothesis is awaiting a decision in this thread"
        )

    def _approve_hypothesis_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — acts on the thread's oldest awaiting hypothesis
            assert self._interactions is not None  # tool only registered when wired
            row = self._awaiting_hypothesis(thread_ts)
            # The poller handler owns submit/resolve/notify and posts its own
            # outcome (success or a submit-failure notice) to the thread.
            self._interactions.approve(row.id, say)
            resolved = self._store.get_pending_interaction(row.id)
            if resolved is None or resolved.status != "approved":
                return (
                    "Submitting the approval to the run failed (the failure notice"
                    " was posted in-thread) — the hypothesis is still awaiting a"
                    " decision. Suggest the operator try again shortly."
                )
            logger.info("approved hypothesis #%s via chat for thread %s", row.id, thread_ts)
            return (
                "The hypothesis was approved and submitted to the run; the"
                " confirmation was posted. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="approve_hypothesis",
            description=(
                "APPROVE the hypothesis currently awaiting the operator's decision"
                " in this thread (same effect as its Approve button) — the run"
                " implements it next. Call it only when the operator explicitly"
                " approves in words ('approve', 'go ahead with it', 'LGTM')."
                " Never approve on your own judgment or a lukewarm reply."
            ),
            input_schema=APPROVE_HYPOTHESIS_SCHEMA,
            handler=handler,
        )

    def _reject_hypothesis_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — acts on the thread's oldest awaiting hypothesis
            assert self._interactions is not None  # tool only registered when wired
            row = self._awaiting_hypothesis(thread_ts)
            self._interactions.reject(row.id, say)
            resolved = self._store.get_pending_interaction(row.id)
            if resolved is None or resolved.status != "rejected":
                return (
                    "Submitting the rejection to the run failed (the failure notice"
                    " was posted in-thread) — the hypothesis is still awaiting a"
                    " decision. Suggest the operator try again shortly."
                )
            logger.info("rejected hypothesis #%s via chat for thread %s", row.id, thread_ts)
            return (
                "The hypothesis was rejected — the run was told to discard it and"
                " propose a materially different direction; the notice was posted."
                " Confirm briefly to the operator."
            )

        return ToolSpec(
            name="reject_hypothesis",
            description=(
                "REJECT the hypothesis currently awaiting the operator's decision"
                " in this thread (same effect as its Reject button) — the run"
                " discards the idea and proposes a different direction. Call it"
                " only when the operator explicitly rejects in words. For revised"
                " wording they should use the hypothesis message's Edit button."
            ),
            input_schema=REJECT_HYPOTHESIS_SCHEMA,
            handler=handler,
        )

    def _promote_run_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — promotes the thread's finished run
            assert self._promotions is not None  # tool only registered when wired
            # PromotionFlow posts either the confirmation or a refusal itself;
            # capture what it said so the model can relay it faithfully.
            posted: list[str] = []

            def recording_say(**kwargs: Any) -> Any:
                posted.append(str(kwargs.get("text", "")))
                return say(**kwargs)

            self._promotions.request_promotion(thread_ts, recording_say)
            outcome = posted[-1] if posted else "(nothing was posted)"
            logger.info("promotion requested via chat for thread %s", thread_ts)
            return (
                "The promotion request was processed; this was posted in-thread:\n"
                f"{outcome}\n"
                "If that is the confirmation restating the strategy, ask the"
                " operator to explicitly confirm — only then call"
                " confirm_promotion. If it is a refusal, relay the reason."
            )

        return ToolSpec(
            name="promote_run",
            description=(
                "Start promoting this thread's finished research run to paper"
                " trading (same effect as the summary's Promote button): posts a"
                " confirmation restating exactly what the nightly rebalancer"
                " would trade (universe, topk/n_drop, metrics). Nothing is"
                " promoted until confirm_promotion. Call it only when the"
                " operator explicitly asks to promote the run."
            ),
            input_schema=PROMOTE_RUN_SCHEMA,
            handler=handler,
        )

    def _confirm_promotion_tool(self, thread_ts: str, say: SayFn) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args  # no inputs — confirms the thread's requested promotion
            assert self._promotions is not None  # tool only registered when wired
            posted: list[str] = []

            def recording_say(**kwargs: Any) -> Any:
                posted.append(str(kwargs.get("text", "")))
                return say(**kwargs)

            self._promotions.confirm_promotion(thread_ts, recording_say)
            outcome = posted[-1] if posted else "(nothing was posted)"
            logger.info("promotion confirmed via chat for thread %s", thread_ts)
            return (
                "The promotion confirmation was processed; this was posted"
                f" in-thread:\n{outcome}\n"
                "Relay the outcome briefly to the operator."
            )

        return ToolSpec(
            name="confirm_promotion",
            description=(
                "CONFIRM the promotion of this thread's run — pins the strategy"
                " for the nightly paper-trading rebalancer, replacing any"
                " previously promoted strategy. Call it only after promote_run"
                " posted the confirmation AND the operator explicitly confirmed"
                " it (e.g. 'confirm', 'yes, promote it'). Never confirm without"
                " that explicit second yes."
            ),
            input_schema=CONFIRM_PROMOTION_SCHEMA,
            handler=handler,
        )

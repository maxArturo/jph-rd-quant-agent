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
from orchestrator.run_memory import Digest, compose_instruction
from orchestrator.state import (
    Directive,
    DuplicateRunError,
    Run,
    StateStore,
)

if TYPE_CHECKING:
    from execution.alpaca_client import Account, Order, PortfolioHistory, Position
    from ops.run_lock import RunLock
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


class GpuRunner(Protocol):
    """What the GPU-backend run tools need from orchestrator.gpu_backend."""

    status_file: Path

    def launch(
        self,
        thread_ts: str,
        *,
        loop_n: int = 10,
        universe: str | None = None,
        instruction: str | None = None,
    ) -> str: ...

    def stop_unit(self, unit: str) -> None: ...

    def unit_active(self, thread_ts: str) -> bool: ...

    def active_run_lock(self) -> tuple[RunLock | None, RunLock | None]: ...

    def cancel(self) -> str: ...

    def read_status(self) -> dict | None: ...


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


class PromotionManager(Protocol):
    """What the promotion tools need from PromotionFlow (stub-friendly)."""

    def request_promotion(self, thread_ts: str, say: SayFn) -> None: ...

    def confirm_promotion(self, thread_ts: str, say: SayFn) -> None: ...


START_RESEARCH_SCHEMA: dict[str, Any] = {
    # The run is driven by the thread's saved directive. Runs execute on a
    # disposable GPU droplet (2026-08-06 decision) and are always autonomous.
    "type": "object",
    "properties": {
        "supervised": {
            "type": "boolean",
            "description": (
                "DEPRECATED: GPU-backend runs are autonomous only. Passing"
                " true makes the tool refuse with an explanation — never set"
                " it unless the operator insists on per-hypothesis approval."
            ),
        },
        "loop_n": {
            "type": "integer",
            "description": (
                "Hypothesis budget for the run (default 10). Only change it"
                " when the operator names a number."
            ),
        },
        "include_memory": {
            "type": "boolean",
            "description": (
                "Prepend the run-history digest (prior runs + incumbent) to"
                " the run's instruction so it builds on earlier results"
                " (default true). Set false only when the operator explicitly"
                " asks for a clean-slate run without memory."
            ),
        },
    },
}

CHECK_RESEARCH_STATUS_SCHEMA: dict[str, Any] = {
    # No inputs: reports the thread's run status (stage, loops, SOTA count).
    "type": "object",
    "properties": {},
}

STOP_RUN_SCHEMA: dict[str, Any] = {
    # No inputs: stops the thread's run.
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


def format_run_started(run: Run, *, memory_runs: int | None = None) -> str:
    """Slack mrkdwn confirmation posted to the thread when a run starts.

    ``memory_runs`` is how many prior-run digest entries were injected into
    the run's instruction (US-015); None means memory was not included
    (include_memory=false, or no digest builder wired).
    """
    if run.supervised:
        tail = "Hypotheses will be posted here for approval as the loop proposes them."
    else:
        tail = (
            "The loop will try its hypotheses autonomously and narrate each one"
            " here — no approvals needed. It stops on its own after its"
            " hypothesis budget and posts the best result found."
        )
    if memory_runs is None:
        memory = "*Run memory:* not included (clean-slate run)"
    else:
        memory = f"*Run memory:* {memory_runs} prior run(s) included as context"
    return (
        "*Research run started*\n"
        f"*Universe:* {run.universe}\n"
        f"*Session:* `{run.session_path}`\n"
        f"{memory}\n"
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
    Research runs execute exclusively on the GPU backend (``gpu``, US-028
    removed the legacy on-box control plane).
    """

    def __init__(
        self,
        store: StateStore,
        router: ModelRouter,
        universes: UniverseManager | None = None,
        recorder: NotionRecorder | None = None,
        breaker: TradingBreaker | None = None,
        broker: BrokerReader | None = None,
        promotions: PromotionManager | None = None,
        gpu: GpuRunner | None = None,
        digest_builder: Callable[[], Digest] | None = None,
        menu_builder: Callable[[], str] | None = None,
    ) -> None:
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
        # DELIBERATELY no real default: GpuBackend.launch starts billable
        # cloud infrastructure via systemd-run, so callers must opt in
        # explicitly (app.py wires the real one; tests pass stubs). A leaked
        # default provisioned a real droplet from the test suite on
        # 2026-08-06 — None now makes start_research refuse instead.
        self._gpu: GpuRunner | None = gpu
        # Run-history digest for RDQ_USER_INSTRUCTION (US-015). None (tests,
        # partial wiring) skips injection; app.py wires the real
        # run_memory.build_digest_details, which never raises and never
        # stalls a launch (15s Notion budget, local fallback).
        self._digest_builder = digest_builder
        # Data-menu context for the directive-crafting prompt (US-061). None
        # (tests, partial wiring) skips injection; app.py wires
        # prompts.data_menu_context, which never raises (store unreadable
        # degrades to an explicit 'menu unavailable' line).
        self._menu_builder = menu_builder
        self._universes: UniverseManager = universes
        self._breaker: TradingBreaker = breaker
        self._broker: BrokerReader = broker
        # Optional Notion audit trail (US-027); None disables recording.
        self._recorder = recorder
        # Optional wiring to the promotion flow. None (tests, partial wiring)
        # simply leaves the promotion tools out of the loop — the model never
        # sees a tool it cannot execute.
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
            self._set_universe_tool(thread_ts, say),
            self._confirm_universe_tool(thread_ts, say),
            self._halt_trading_tool(thread_ts, say),
            self._resume_trading_tool(thread_ts, say),
            self._check_research_status_tool(thread_ts),
            self._check_account_tool(),
            self._check_orders_tool(),
            self._check_pnl_tool(),
        ]
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
        """SYSTEM_PROMPT + data menu (US-061) + the thread's saved directive.

        The directive reloads from SQLite; the menu rebuilds from the store on
        every call so it tracks daily refreshes. Menu failures degrade to the
        fallback line — this path must never raise into the Bolt handler.
        """
        parts = [prompts.SYSTEM_PROMPT]
        if self._menu_builder is not None:
            try:
                parts.append(self._menu_builder())
            except Exception:
                logger.exception("menu builder raised; degrading to the fallback line")
                parts.append(prompts.MENU_UNAVAILABLE_LINE)
        directive = self._store.get_directive(thread_ts)
        if directive is not None:
            parts.append(prompts.directive_context(directive))
        return "\n\n".join(parts)

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
            if bool(args.get("supervised", False)):
                raise ValueError(
                    "runs now execute on a disposable GPU worker and are"
                    " autonomous only — per-hypothesis approval is not available"
                    " there. Offer to start an autonomous run instead."
                )
            if self._gpu is None:
                raise ValueError(
                    "the GPU research backend is not wired in this deployment —"
                    " research runs cannot be started"
                )
            loop_n = int(args.get("loop_n") or 10)
            if not 1 <= loop_n <= 50:
                raise ValueError("loop_n must be between 1 and 50")
            directive = self._store.get_directive(thread_ts)
            if directive is None:
                raise ValueError(
                    "no research directive is saved for this thread yet — refine"
                    " the idea with the operator and call save_directive first"
                )
            existing = self._store.get_run(thread_ts)
            # US-021: a failed run (crashed pipeline, reaped row) must not
            # brick its thread — only non-failed runs block a fresh start.
            if existing is not None and existing.status != "failed":
                raise ValueError(duplicate_run_message(existing))
            # US-020: one GPU worker at a time, globally — another thread's
            # active run must refuse this launch (its teardown would destroy
            # the shared droplet). A stale lock is broken with a visible note.
            active_lock, broken_lock = self._gpu.active_run_lock()
            if broken_lock is not None:
                say(
                    text=(
                        f":broom: cleared a stale GPU run lock left by"
                        f" {broken_lock.describe()} — the owning unit is no"
                        " longer active."
                    ),
                    thread_ts=thread_ts,
                )
            if active_lock is not None:
                raise ValueError(
                    f"a research run is already active for {active_lock.describe()}"
                    " — only one GPU worker exists at a time, so a second run"
                    " cannot start. Wait for that run to finish, or have the"
                    " operator stop it from its own thread."
                )
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
            # US-015: prepend the run-history digest so the run builds on
            # prior results — the directive always comes first and is never
            # truncated in favor of the digest.
            instruction = directive_instruction(directive)
            memory_runs: int | None = None
            if bool(args.get("include_memory", True)) and self._digest_builder is not None:
                digest = self._digest_builder()
                memory_runs = digest.runs
                instruction = compose_instruction(instruction, digest.text)
            unit = self._gpu.launch(
                thread_ts,
                loop_n=loop_n,
                universe=universe,
                instruction=instruction,
            )
            # session_path holds the pipeline status file until the pipeline
            # rewrites it to the fetched trace dir at completion (promotion
            # locates artifacts through it).
            session_path = str(self._gpu.status_file)
            try:
                run = self._store.create_run(
                    thread_ts,
                    session_path,
                    universe=universe,
                    universe_tickers=tickers,
                    supervised=False,
                    backend="gpu",
                    replace_failed=True,
                )
            except DuplicateRunError as exc:
                # Lost a start race — don't leave the just-launched pipeline up.
                self._gpu.stop_unit(unit)
                raise ValueError(duplicate_run_message(exc.existing)) from exc
            say(text=format_run_started(run, memory_runs=memory_runs), thread_ts=thread_ts)
            if self._recorder is not None:
                self._recorder.record_idea_status(thread_ts, "researching", universe=universe)
            logger.info("started GPU research pipeline %s for thread %s", unit, thread_ts)
            memory_note = (
                f" Run memory: {memory_runs} prior run(s) included as context."
                if memory_runs is not None
                else " Run memory: not included (clean-slate run)."
            )
            return (
                f"GPU research pipeline launched (unit {unit}, budget {loop_n}"
                " hypotheses) and recorded for this thread; the start notice was"
                f" posted.{memory_note}"
                " Confirm briefly to the operator: a GPU droplet is being"
                " provisioned (~$1.57/hr, destroyed automatically when done), the"
                " run is autonomous, per-loop digests will arrive in this thread,"
                " and the final summary names the promotion candidate. Progress"
                " questions -> check_research_status; cancelling -> stop_run."
            )

        return ToolSpec(
            name="start_research",
            description=(
                "Start a research run for this thread's SAVED directive on a"
                " disposable GPU worker droplet. Requires save_directive first;"
                " only one run may exist per thread. Call it only when the"
                " operator explicitly asks to start the research. Runs are"
                " autonomous (hypotheses auto-approved, self-stopping after the"
                " loop_n budget, worker destroyed afterwards)."
            ),
            input_schema=START_RESEARCH_SCHEMA,
            handler=handler,
        )

    def _check_research_status_tool(self, thread_ts: str) -> ToolSpec:
        def handler(args: dict[str, Any]) -> str:
            del args
            run = self._store.get_run(thread_ts)
            if run is None:
                raise ValueError("this thread has no research run — start_research launches one")
            if run.backend != "gpu" or self._gpu is None:
                return f"run status: {run.status} ({run.backend} run, {run.session_path})"
            from orchestrator.gpu_backend import format_gpu_status

            status = self._gpu.read_status()
            if status is not None and status.get("thread_ts") not in (None, thread_ts):
                return (
                    f"run status: {run.status}. The live pipeline status belongs to"
                    " another thread's run (one GPU worker at a time; runs are"
                    " never queued) — this thread's run already finished; its"
                    " summary is earlier in this thread."
                )
            active = self._gpu.unit_active(thread_ts)
            return f"run status: {run.status} (GPU backend)\n" + format_gpu_status(
                status, unit_active=active
            )

        return ToolSpec(
            name="check_research_status",
            description=(
                "Report this thread's research run progress: pipeline stage, loops"
                " finished, SOTA count, latest metrics, promotion candidate. Call"
                " it whenever the operator asks how the run/loops are going. Read-"
                "only; relay the details conversationally."
            ),
            input_schema=CHECK_RESEARCH_STATUS_SCHEMA,
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
            if run.backend != "gpu":
                # Legacy pre-GPU rows (backend 'server_ui') stay readable, but
                # their control plane was removed (US-026/US-028) — there is
                # no process left to stop.
                raise ValueError(
                    f"this run predates the GPU backend (backend: {run.backend})"
                    " and its control plane was decommissioned — there is no"
                    " process to stop. Start fresh research with start_research"
                    " in a new thread."
                )
            if self._gpu is None:
                raise ValueError("the GPU research backend is not wired — cannot cancel")
            # US-020: cancel kills THE worker's tmux session — only the
            # lock-owning thread may do that, or a stop here would kill
            # another thread's run.
            active_lock, broken_lock = self._gpu.active_run_lock()
            if broken_lock is not None:
                say(
                    text=(
                        f":broom: cleared a stale GPU run lock left by"
                        f" {broken_lock.describe()} — the owning unit is no"
                        " longer active."
                    ),
                    thread_ts=thread_ts,
                )
            if active_lock is not None and active_lock.thread_ts != thread_ts:
                raise ValueError(
                    f"the active GPU run belongs to {active_lock.describe()}"
                    " — this thread's run is not the one running. Stop it"
                    " from its own thread."
                )
            message = self._gpu.cancel()
            say(text=f":octagonal_sign: {message}", thread_ts=thread_ts)
            logger.info("cancelled GPU research run for thread %s", thread_ts)
            return (
                "Cancel signal sent to the GPU worker. The pipeline will fetch"
                " the loops finished so far, post the final summary (any SOTA"
                " candidate stays promotable), mark the run stopped, and"
                " destroy the droplet. GPU runs cannot be resumed — a new"
                " start_research begins fresh. Confirm briefly to the operator."
            )

        return ToolSpec(
            name="stop_run",
            description=(
                "Stop this thread's in-flight research run. The run's finished"
                " loops stay promotable, but a stopped GPU run cannot be resumed"
                " — new research means a fresh start_research. Call it only when"
                " the operator explicitly asks to stop, pause, or kill the run."
            ),
            input_schema=STOP_RUN_SCHEMA,
            handler=handler,
        )

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
                " runs."
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

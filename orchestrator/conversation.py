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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from orchestrator import prompts
from orchestrator.llm import LLMError, ModelRouter, RefusalError, ToolSpec
from orchestrator.rdagent_client import RunHandle
from orchestrator.state import Directive, DuplicateRunError, Run, StateStore

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
    """What the start_research tool needs from RdAgentClient (stub-friendly)."""

    def start_run(self, directive: str, universe: str) -> RunHandle: ...

    def trace_dir(self, trace_id: str) -> Path: ...

    def stop(self, trace_id: str) -> None: ...

START_RESEARCH_SCHEMA: dict[str, Any] = {
    # No inputs: the run is driven entirely by the thread's saved directive.
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
    return (
        "*Research run started*\n"
        f"*Universe:* {run.universe}\n"
        f"*Session:* `{run.session_path}`\n"
        "Hypotheses will be posted here for approval as the loop proposes them."
    )


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
    ) -> None:
        if rdagent is None:
            from orchestrator.rdagent_client import RdAgentClient

            rdagent = RdAgentClient()
        self._store = store
        self._router = router
        self._rdagent: ResearchLauncher = rdagent
        self._histories: dict[str, list[dict[str, Any]]] = {}

    def handle_message(self, thread_ts: str, text: str, say: SayFn) -> str:
        """Run one conversational turn; posts the model's reply in-thread.

        Returns the reply text (also posted via say). Refusals and model
        errors are reported in-thread, never raised into the Bolt handler.
        """
        history = self._histories.setdefault(thread_ts, [])
        history.append({"role": "user", "content": text})
        try:
            final = self._router.judgment_tool_loop(
                history,
                [
                    self._save_directive_tool(thread_ts, say),
                    self._start_research_tool(thread_ts, say),
                ],
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
            del args  # no inputs — the saved directive drives the run
            directive = self._store.get_directive(thread_ts)
            if directive is None:
                raise ValueError(
                    "no research directive is saved for this thread yet — refine"
                    " the idea with the operator and call save_directive first"
                )
            existing = self._store.get_run(thread_ts)
            if existing is not None:
                raise ValueError(duplicate_run_message(existing))
            handle = self._rdagent.start_run(directive_instruction(directive), DEFAULT_UNIVERSE)
            session_path = str(self._rdagent.trace_dir(handle.trace_id))
            try:
                run = self._store.create_run(thread_ts, session_path, universe=DEFAULT_UNIVERSE)
            except DuplicateRunError as exc:
                # Lost a start race — don't leave the just-launched run orphaned.
                self._rdagent.stop(handle.trace_id)
                raise ValueError(duplicate_run_message(exc.existing)) from exc
            say(text=format_run_started(run), thread_ts=thread_ts)
            logger.info("started research run %s for thread %s", handle.trace_id, thread_ts)
            return (
                f"Research run started (trace {handle.trace_id}) and recorded for this"
                " thread; the start notice was posted. Confirm briefly to the operator"
                " — hypotheses will arrive in this thread for approval."
            )

        return ToolSpec(
            name="start_research",
            description=(
                "Start an RD-Agent research run for this thread's SAVED directive."
                " Requires save_directive to have been called first; only one run"
                " may exist per thread. Call it only when the operator explicitly"
                " asks to start the research."
            ),
            input_schema=START_RESEARCH_SCHEMA,
            handler=handler,
        )

"""Conversational core: refine operator ideas into saved research directives.

Built on ModelRouter.judgment_tool_loop (US-008) — the save_directive tool is
a ToolSpec whose handler persists to StateStore and posts a formatted summary
to the Slack thread.

Durability model: in-memory transcripts are best-effort (bounded, lost on
restart); the durable context is the saved directive, which reloads from
SQLite into the system prompt on every call. A restart loses chit-chat but
never the directive.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from orchestrator import prompts
from orchestrator.llm import LLMError, ModelRouter, RefusalError, ToolSpec
from orchestrator.state import Directive, StateStore

logger = logging.getLogger(__name__)

# slack_bolt's Say or any equivalent accepting (text=..., thread_ts=...).
SayFn = Callable[..., Any]

# Cap the per-thread transcript sent to the model (user+assistant messages).
MAX_HISTORY_MESSAGES = 40

REFUSAL_REPLY = "I can't help with that request."

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
    """Per-thread Claude conversations with the save_directive tool.

    Share one instance per process (like StateStore/ModelRouter); the Bolt
    message handler calls handle_message for every actionable message.
    """

    def __init__(self, store: StateStore, router: ModelRouter) -> None:
        self._store = store
        self._router = router
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
                [self._save_directive_tool(thread_ts, say)],
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

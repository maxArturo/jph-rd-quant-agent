"""US-009: conversational core — idea -> directive -> echo, with mocked
Anthropic (FakeClient from tests/test_llm.py) and mocked Slack (a recording
say callable). No network anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator import prompts
from orchestrator.conversation import (
    REFUSAL_REPLY,
    ConversationCore,
    format_directive_summary,
)
from orchestrator.llm import ModelRouter
from orchestrator.state import StateStore
from tests.test_llm import (
    FakeClient,
    RefusalMessage,
    message,
    text_block,
    tool_use_block,
)

THREAD = "1751900000.000100"


class RecordingSay:
    """Mocked Slack say(): records (text, thread_ts) per call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, text: str, thread_ts: str) -> None:
        self.calls.append({"text": text, "thread_ts": thread_ts})


def make_core(tmp_path: Path, client: FakeClient) -> tuple[ConversationCore, StateStore]:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    return ConversationCore(store=store, router=ModelRouter(client=client)), store


def save_directive_script(final_reply: str = "Directive saved — ready to research.") -> list[Any]:
    """Model turn 1: call save_directive; turn 2: confirm in text."""
    return [
        message(
            "tool_use",
            [
                tool_use_block(
                    "tu_1",
                    "save_directive",
                    {
                        "objective": "Test whether 12-1 momentum beats SPY",
                        "universe_hint": "US large caps",
                        "constraints": "long-only, monthly rebalance",
                    },
                )
            ],
        ),
        message("end_turn", [text_block(final_reply)]),
    ]


# --- system prompt (acceptance: persona lives in orchestrator/prompts.py) ----


def test_system_prompt_states_persona_and_ground_rules() -> None:
    prompt = prompts.SYSTEM_PROMPT.lower()
    assert "portfolio manager" in prompt and "quant" in prompt
    assert "honest" in prompt  # honest reporting
    assert "never trade" in prompt and "explicit approval" in prompt


# --- idea -> directive -> echo flow ------------------------------------------


def test_idea_to_directive_to_echo_flow(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=save_directive_script())
    core, store = make_core(tmp_path, client)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "momentum on big US names?", say)

    # persisted: {objective, universe_hint, constraints} in the directives table
    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Test whether 12-1 momentum beats SPY"
    assert directive.universe_hint == "US large caps"
    assert directive.constraints == "long-only, monthly rebalance"

    # echoed: formatted summary posted to the thread, then the final reply
    assert [c["thread_ts"] for c in say.calls] == [THREAD, THREAD]
    summary, final = say.calls[0]["text"], say.calls[1]["text"]
    assert summary == format_directive_summary(directive)
    assert "Test whether 12-1 momentum beats SPY" in summary
    assert final == "Directive saved — ready to research." == reply

    # the tool result told the model the save happened
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert f"#{directive.id} saved" in tool_result["content"]


def test_optional_directive_fields_default_to_none(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message(
                "tool_use",
                [tool_use_block("tu_1", "save_directive", {"objective": "Objective only"})],
            ),
            message("end_turn", [text_block("Saved.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    core.handle_message(THREAD, "idea", RecordingSay())
    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Objective only"
    assert directive.universe_hint is None
    assert directive.constraints is None


def test_empty_objective_is_rejected_and_nothing_persisted(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message(
                "tool_use",
                [tool_use_block("tu_1", "save_directive", {"objective": "   "})],
            ),
            message("end_turn", [text_block("That objective was empty.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message(THREAD, "idea", say)

    assert store.get_directive(THREAD) is None
    # the failure went back to the model as an is_error tool_result
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "objective" in tool_result["content"]
    # no summary was posted — only the final reply
    assert [c["text"] for c in say.calls] == ["That objective was empty."]


# --- conversation context ------------------------------------------------------


def test_history_accumulates_across_turns(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message("end_turn", [text_block("What horizon?")]),
            message("end_turn", [text_block("Got it.")]),
        ]
    )
    core, _ = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message(THREAD, "momentum idea", say)
    core.handle_message(THREAD, "12 months", say)

    second_turn = client.stream_calls[1]["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "user"]
    assert second_turn[0]["content"] == "momentum idea"
    assert second_turn[1]["content"] == "What horizon?"
    assert second_turn[2]["content"] == "12 months"


def test_threads_have_independent_histories(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message("end_turn", [text_block("a")]),
            message("end_turn", [text_block("b")]),
        ]
    )
    core, _ = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message("111.000", "first thread", say)
    core.handle_message("222.000", "second thread", say)
    assert client.stream_calls[1]["messages"] == [
        {"role": "user", "content": "second thread"}
    ]


def test_directive_context_reloads_from_sqlite_after_restart(tmp_path: Path) -> None:
    """Acceptance: create directive, recreate the app objects, directive is
    retrievable by thread AND flows back into the model's context."""
    core1, _ = make_core(tmp_path, FakeClient(judgment_messages=save_directive_script()))
    core1.handle_message(THREAD, "momentum on big US names?", RecordingSay())

    # simulated restart: brand-new store + core over the same sqlite file
    client2 = FakeClient(judgment_messages=[message("end_turn", [text_block("Recap...")])])
    core2, store2 = make_core(tmp_path, client2)

    directive = store2.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Test whether 12-1 momentum beats SPY"

    core2.handle_message(THREAD, "where were we?", RecordingSay())
    system = client2.stream_calls[0]["system"]
    assert system.startswith(prompts.SYSTEM_PROMPT)
    assert "Test whether 12-1 momentum beats SPY" in system
    assert "US large caps" in system


def test_system_prompt_has_no_directive_context_before_save(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Tell me more.")])])
    core, _ = make_core(tmp_path, client)
    core.handle_message(THREAD, "vague idea", RecordingSay())
    assert client.stream_calls[0]["system"] == prompts.SYSTEM_PROMPT


# --- failure handling ------------------------------------------------------------


def test_refusal_posts_notice_and_keeps_history_clean(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            RefusalMessage(),
            message("end_turn", [text_block("Happy to help with research.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "do something sketchy", say)
    assert reply == REFUSAL_REPLY
    assert say.calls == [{"text": REFUSAL_REPLY, "thread_ts": THREAD}]
    assert store.get_directive(THREAD) is None

    # the refused turn was rolled back — next turn starts a clean transcript
    core.handle_message(THREAD, "ok, a real idea", say)
    assert client.stream_calls[1]["messages"] == [
        {"role": "user", "content": "ok, a real idea"}
    ]

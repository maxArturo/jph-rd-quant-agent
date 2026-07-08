"""Unit tests for orchestrator/llm.py (US-008) — mocked Anthropic client only."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orchestrator.llm import (
    JUDGMENT_FALLBACK_MODEL,
    JUDGMENT_MODEL,
    SERVER_SIDE_FALLBACK_BETA,
    UTILITY_MODEL,
    ModelRouter,
    RefusalError,
    ToolLoopLimitError,
    ToolSpec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(block_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def message(stop_reason: str, content: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content if content is not None else [text_block("ok")],
        stop_details=None,
    )


class FakeStream:
    """Stands in for the context manager returned by beta.messages.stream()."""

    def __init__(self, final_message: Any) -> None:
        self._final_message = final_message

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._final_message


class FakeClient:
    """Records every call; serves scripted messages in order."""

    def __init__(self, judgment_messages: list[Any] | None = None,
                 utility_message: Any | None = None) -> None:
        self.stream_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self._judgment_messages = list(judgment_messages or [])
        self._utility_message = utility_message

        outer = self

        def stream(**kwargs: Any) -> FakeStream:
            outer.stream_calls.append(kwargs)
            return FakeStream(outer._judgment_messages.pop(0))

        def create(**kwargs: Any) -> Any:
            outer.create_calls.append(kwargs)
            return outer._utility_message

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        self.messages = SimpleNamespace(create=create)


# --- model selection per path -------------------------------------------------


def test_judgment_uses_fable_with_fallback_beta_and_streaming() -> None:
    client = FakeClient(judgment_messages=[message("end_turn")])
    router = ModelRouter(client=client)

    result = router.judgment([{"role": "user", "content": "hi"}])

    assert result.content[0].text == "ok"
    assert len(client.stream_calls) == 1  # streaming path, not create()
    assert client.create_calls == []
    call = client.stream_calls[0]
    assert call["model"] == JUDGMENT_MODEL
    assert call["betas"] == [SERVER_SIDE_FALLBACK_BETA]
    assert call["fallbacks"] == [{"model": JUDGMENT_FALLBACK_MODEL}]


def test_judgment_omits_thinking_parameter() -> None:
    client = FakeClient(judgment_messages=[message("end_turn")])
    ModelRouter(client=client).judgment([{"role": "user", "content": "hi"}])
    assert "thinking" not in client.stream_calls[0]


def test_judgment_passes_system_and_tools() -> None:
    client = FakeClient(judgment_messages=[message("end_turn")])
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    ModelRouter(client=client).judgment(
        [{"role": "user", "content": "hi"}], system="be a PM", tools=tools
    )
    call = client.stream_calls[0]
    assert call["system"] == "be a PM"
    assert call["tools"] == tools


def test_utility_uses_haiku_via_create() -> None:
    client = FakeClient(utility_message=message("end_turn", [text_block("done")]))
    router = ModelRouter(client=client)

    result = router.utility([{"role": "user", "content": "classify"}], system="s")

    assert result.content[0].text == "done"
    assert client.stream_calls == []
    call = client.create_calls[0]
    assert call["model"] == UTILITY_MODEL
    assert call["system"] == "s"


def test_model_ids_only_defined_in_llm_module() -> None:
    """The tier policy lives in orchestrator/llm.py alone."""
    for model_id in (JUDGMENT_MODEL, UTILITY_MODEL, JUDGMENT_FALLBACK_MODEL):
        proc = subprocess.run(
            ["grep", "-rl", "--include=*.py", model_id,
             "orchestrator", "execution", "data", "research", "ops"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        hits = [line for line in proc.stdout.splitlines() if line]
        assert hits == ["orchestrator/llm.py"], (
            f"model id {model_id!r} found outside orchestrator/llm.py: {hits}"
        )


# --- refusal handling ----------------------------------------------------------


class RefusalMessage:
    """content raises if touched — proves stop_reason is checked first."""

    stop_reason = "refusal"
    stop_details = SimpleNamespace(category="cyber", explanation="declined")

    @property
    def content(self) -> Any:
        raise AssertionError("content was read before checking stop_reason")


def test_judgment_raises_refusal_error_without_reading_content() -> None:
    client = FakeClient(judgment_messages=[RefusalMessage()])
    with pytest.raises(RefusalError) as excinfo:
        ModelRouter(client=client).judgment([{"role": "user", "content": "hi"}])
    assert excinfo.value.stop_details.category == "cyber"


def test_utility_raises_refusal_error() -> None:
    client = FakeClient(utility_message=RefusalMessage())
    with pytest.raises(RefusalError):
        ModelRouter(client=client).utility([{"role": "user", "content": "hi"}])


# --- tool-use loop --------------------------------------------------------------


def make_tool(name: str = "get_price", handler: Any = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}},
        handler=handler or (lambda args: f"price of {args['ticker']} is 42"),
    )


def test_tool_loop_executes_tool_and_terminates_on_end_turn() -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> str:
        calls.append(args)
        return "42.5"

    client = FakeClient(
        judgment_messages=[
            message("tool_use", [tool_use_block("tu_1", "get_price", {"ticker": "AAPL"})]),
            message("end_turn", [text_block("AAPL trades at 42.5")]),
        ]
    )
    router = ModelRouter(client=client)

    result = router.judgment_tool_loop(
        [{"role": "user", "content": "price of AAPL?"}], [make_tool(handler=handler)]
    )

    assert result.stop_reason == "end_turn"
    assert result.content[0].text == "AAPL trades at 42.5"
    assert calls == [{"ticker": "AAPL"}]
    # second request must carry the assistant turn + the tool_result back
    second = client.stream_calls[1]["messages"]
    assert second[1]["role"] == "assistant"
    tool_results = second[2]["content"]
    assert second[2]["role"] == "user"
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "42.5"}
    ]
    # tool schemas were passed on every iteration
    assert client.stream_calls[0]["tools"][0]["name"] == "get_price"


def test_tool_loop_returns_error_result_for_unknown_tool() -> None:
    client = FakeClient(
        judgment_messages=[
            message("tool_use", [tool_use_block("tu_1", "nope", {})]),
            message("end_turn"),
        ]
    )
    ModelRouter(client=client).judgment_tool_loop(
        [{"role": "user", "content": "x"}], [make_tool()]
    )
    result = client.stream_calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "unknown tool 'nope'" in result["content"]


def test_tool_loop_converts_handler_exception_to_error_result() -> None:
    def boom(args: dict[str, Any]) -> str:
        raise ValueError("bad ticker")

    client = FakeClient(
        judgment_messages=[
            message("tool_use", [tool_use_block("tu_1", "get_price", {"ticker": "??"})]),
            message("end_turn"),
        ]
    )
    ModelRouter(client=client).judgment_tool_loop(
        [{"role": "user", "content": "x"}], [make_tool(handler=boom)]
    )
    result = client.stream_calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "bad ticker" in result["content"]


def test_tool_loop_handles_parallel_tool_calls_in_one_user_message() -> None:
    client = FakeClient(
        judgment_messages=[
            message(
                "tool_use",
                [
                    tool_use_block("tu_1", "get_price", {"ticker": "AAPL"}),
                    tool_use_block("tu_2", "get_price", {"ticker": "MSFT"}),
                ],
            ),
            message("end_turn"),
        ]
    )
    ModelRouter(client=client).judgment_tool_loop(
        [{"role": "user", "content": "x"}], [make_tool()]
    )
    results = client.stream_calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]


def test_tool_loop_raises_after_max_iterations() -> None:
    endless = [
        message("tool_use", [tool_use_block(f"tu_{i}", "get_price", {"ticker": "AAPL"})])
        for i in range(3)
    ]
    client = FakeClient(judgment_messages=endless)
    with pytest.raises(ToolLoopLimitError):
        ModelRouter(client=client).judgment_tool_loop(
            [{"role": "user", "content": "x"}], [make_tool()], max_iterations=3
        )


def test_tool_loop_propagates_refusal() -> None:
    client = FakeClient(judgment_messages=[RefusalMessage()])
    with pytest.raises(RefusalError):
        ModelRouter(client=client).judgment_tool_loop(
            [{"role": "user", "content": "x"}], [make_tool()]
        )

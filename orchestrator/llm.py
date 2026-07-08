"""Central model router and Claude client wrapper.

Tier policy (enforced here and nowhere else):
  - judgment(): claude-fable-5 — decisions, conversation, anything requiring
    judgment. Streams, opts into server-side refusal fallbacks (claude-opus-4-8),
    and checks stop_reason == "refusal" before touching content.
  - utility(): claude-haiku-4-5 — cheap mechanical work (classification,
    formatting, extraction).

Model IDs must appear only in this module (tests/test_llm.py enforces this).

The judgment path deliberately omits the `thinking` parameter: claude-fable-5
has always-on adaptive thinking and returns 400 for any explicit thinking
config (including {"type": "disabled"}).

Auth: like every Anthropic call in this repo, real credentials are injected by
the OneCLI proxy when running under `onecli run` — a placeholder API key is
fine and expected (the proxy overrides client-sent auth headers).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

JUDGMENT_MODEL = "claude-fable-5"
JUDGMENT_FALLBACK_MODEL = "claude-opus-4-8"
UTILITY_MODEL = "claude-haiku-4-5"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-06-01"

DEFAULT_JUDGMENT_MAX_TOKENS = 8192
DEFAULT_UTILITY_MAX_TOKENS = 1024
DEFAULT_MAX_TOOL_ITERATIONS = 20

_PLACEHOLDER_API_KEY = "placeholder-injected-by-onecli-proxy"


class LLMError(Exception):
    """Base class for model-router errors."""


class RefusalError(LLMError):
    """The model (and any fallback) declined the request (stop_reason == 'refusal')."""

    def __init__(self, stop_details: Any = None) -> None:
        category = getattr(stop_details, "category", None)
        super().__init__(f"model refused the request (category={category})")
        self.stop_details = stop_details


class ToolLoopLimitError(LLMError):
    """The tool-use loop did not reach end_turn within the iteration limit."""


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: API schema plus the local handler that executes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ModelRouter:
    """Routes calls to the right Claude tier. Share one instance per process."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic  # lazy: keeps offline tests and pyright startup fast

            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", _PLACEHOLDER_API_KEY)
            )
        self._client = client

    def judgment(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int = DEFAULT_JUDGMENT_MAX_TOKENS,
    ) -> Any:
        """One streamed claude-fable-5 call with server-side refusal fallback.

        Returns the final Message. Raises RefusalError when the whole fallback
        chain refused — checked before content is read.
        """
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = list(tools)
        # No `thinking` key here on purpose — see module docstring.
        with self._client.beta.messages.stream(
            model=JUDGMENT_MODEL,
            max_tokens=max_tokens,
            messages=list(messages),
            betas=[SERVER_SIDE_FALLBACK_BETA],
            fallbacks=[{"model": JUDGMENT_FALLBACK_MODEL}],
            **kwargs,
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RefusalError(getattr(message, "stop_details", None))
        return message

    def utility(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_UTILITY_MAX_TOKENS,
    ) -> Any:
        """One claude-haiku-4-5 call for cheap mechanical work."""
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        message = self._client.messages.create(
            model=UTILITY_MODEL,
            max_tokens=max_tokens,
            messages=list(messages),
            **kwargs,
        )
        if message.stop_reason == "refusal":
            raise RefusalError(getattr(message, "stop_details", None))
        return message

    def judgment_tool_loop(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_JUDGMENT_MAX_TOKENS,
        max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> Any:
        """Run judgment() in a tool-use loop until the model stops calling tools.

        Executes registered ToolSpec handlers for each tool_use block, feeds all
        tool_result blocks back in a single user message (parallel tool use),
        and returns the final Message on end_turn. Tool handler exceptions are
        returned to the model as is_error results, not raised.
        """
        registry = {spec.name: spec for spec in tools}
        api_tools = [spec.to_api() for spec in tools]
        convo: list[dict[str, Any]] = [dict(m) for m in messages]
        for _ in range(max_iterations):
            message = self.judgment(
                convo, system=system, tools=api_tools, max_tokens=max_tokens
            )
            if message.stop_reason != "tool_use":
                return message
            convo.append({"role": "assistant", "content": message.content})
            convo.append({"role": "user", "content": self._run_tools(message.content, registry)})
        raise ToolLoopLimitError(
            f"tool loop did not reach end_turn within {max_iterations} iterations"
        )

    @staticmethod
    def _run_tools(
        content: Sequence[Any], registry: Mapping[str, ToolSpec]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            spec = registry.get(block.name)
            if spec is None:
                results.append(_tool_result(block.id, f"Error: unknown tool '{block.name}'", True))
                continue
            try:
                output = spec.handler(block.input)
            except Exception as exc:  # noqa: BLE001 — errors go back to the model
                results.append(_tool_result(block.id, f"Error: {exc}", True))
            else:
                results.append(_tool_result(block.id, output, False))
        return results


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        result["is_error"] = True
    return result

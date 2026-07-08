"""LLM backend probe for RD-Agent (US-004).

Verifies the two provider paths RD-Agent depends on, end-to-end through the
OneCLI proxy:

1. Chat: a JSON-mode, hypothesis-shaped prompt to ``CHAT_MODEL``
   (default ``anthropic/claude-sonnet-5``) via LiteLLM, response validated
   against ``HYPOTHESIS_SCHEMA``.
2. Embeddings: one ``EMBEDDING_MODEL`` (default ``voyage/voyage-3.5-lite``)
   call via LiteLLM, asserting a non-empty float vector.

Run it through the proxy — real keys are injected there; the placeholder env
values only satisfy LiteLLM's client-side key-presence checks:

    onecli run --agent rdq-research -- .venv/bin/python research/probe_llm.py

Exits 0 when both probes pass, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import jsonschema

PLACEHOLDER_KEY = "placeholder-injected-by-onecli-proxy"
DEFAULT_CHAT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_EMBEDDING_MODEL = "voyage/voyage-3.5-lite"

# The shape RD-Agent's research loop expects a hypothesis proposal to take.
HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["hypothesis", "rationale", "confidence"],
    "additionalProperties": False,
}

HYPOTHESIS_PROMPT = (
    "You are a quant research assistant. Propose one testable factor "
    "hypothesis about US equity momentum. Respond with ONLY a JSON object "
    "(no markdown fences, no prose) with exactly these keys: "
    '"hypothesis" (string), "rationale" (string), '
    '"confidence" (number between 0 and 1).'
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating markdown fences.

    LiteLLM's Anthropic JSON mode can still wrap the object in ```json fences.
    """
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model reply is not valid JSON: {exc}: {text[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"model reply is JSON but not an object: {text[:200]!r}")
    return payload


def validate_hypothesis(payload: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if payload is not hypothesis-shaped."""
    jsonschema.validate(payload, HYPOTHESIS_SCHEMA)


def set_placeholder_keys() -> None:
    """Client-side key-presence placeholders; the proxy injects real keys."""
    os.environ.setdefault("ANTHROPIC_API_KEY", PLACEHOLDER_KEY)
    os.environ.setdefault("VOYAGE_API_KEY", PLACEHOLDER_KEY)


def probe_chat(model: str) -> dict[str, Any]:
    """JSON-mode hypothesis prompt through LiteLLM; returns the validated dict."""
    import litellm  # heavy import — keep lazy so offline tests stay fast

    resp: Any = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": HYPOTHESIS_PROMPT}],
        response_format={"type": "json_object"},
        max_tokens=1024,
        # no temperature: litellm rejects temperature!=1 for claude-sonnet-5
    )
    content = resp.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"empty chat completion content: {content!r}")
    payload = extract_json_object(content)
    validate_hypothesis(payload)
    return payload


def probe_embedding(model: str) -> list[float]:
    """One embedding call through LiteLLM; returns the (non-empty) vector."""
    import litellm  # heavy import — keep lazy so offline tests stay fast

    resp: Any = litellm.embedding(model=model, input=["US equity momentum factor probe"])
    vector = resp.data[0]["embedding"]
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError(f"embedding response has no vector: {vector!r}")
    if not all(isinstance(x, float) for x in vector):
        raise ValueError("embedding vector contains non-float entries")
    return vector


def main() -> int:
    set_placeholder_keys()
    chat_model = os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    failures = 0

    try:
        payload = probe_chat(chat_model)
        print(f"PASS chat [{chat_model}]: schema-valid hypothesis "
              f"(confidence={payload['confidence']})")
    except Exception as exc:  # noqa: BLE001 — probe reports any failure mode
        failures += 1
        print(f"FAIL chat [{chat_model}]: {exc}")

    try:
        vector = probe_embedding(embedding_model)
        print(f"PASS embedding [{embedding_model}]: {len(vector)}-dim vector")
    except Exception as exc:  # noqa: BLE001 — probe reports any failure mode
        failures += 1
        print(f"FAIL embedding [{embedding_model}]: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

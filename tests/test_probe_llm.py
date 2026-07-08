"""Offline tests for the US-004 LLM probe helpers and .env.example contract."""

from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

from research.probe_llm import (
    HYPOTHESIS_SCHEMA,
    extract_json_object,
    validate_hypothesis,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / "research" / ".env.example"

GOOD_HYPOTHESIS = {
    "hypothesis": "12-1 month momentum predicts next-month returns",
    "rationale": "Underreaction and herding sustain trends",
    "confidence": 0.6,
}


class TestExtractJsonObject:
    def test_bare_json(self) -> None:
        assert extract_json_object('{"ok": true}') == {"ok": True}

    def test_markdown_fenced_json(self) -> None:
        text = '```json\n{"hypothesis": "x"}\n```'
        assert extract_json_object(text) == {"hypothesis": "x"}

    def test_fence_without_language_tag(self) -> None:
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            extract_json_object("not json at all")

    def test_json_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an object"):
            extract_json_object("[1, 2, 3]")


class TestHypothesisSchema:
    def test_valid_payload_passes(self) -> None:
        validate_hypothesis(GOOD_HYPOTHESIS)

    def test_missing_key_rejected(self) -> None:
        payload = {k: v for k, v in GOOD_HYPOTHESIS.items() if k != "confidence"}
        with pytest.raises(jsonschema.ValidationError):
            validate_hypothesis(payload)

    def test_out_of_range_confidence_rejected(self) -> None:
        with pytest.raises(jsonschema.ValidationError):
            validate_hypothesis({**GOOD_HYPOTHESIS, "confidence": 1.5})

    def test_extra_key_rejected(self) -> None:
        with pytest.raises(jsonschema.ValidationError):
            validate_hypothesis({**GOOD_HYPOTHESIS, "extra": "nope"})

    def test_schema_requires_all_three_keys(self) -> None:
        assert set(HYPOTHESIS_SCHEMA["required"]) == {"hypothesis", "rationale", "confidence"}


class TestEnvExample:
    """Pin the US-004 acceptance criteria for research/.env.example."""

    def test_models_and_placeholder_keys(self) -> None:
        text = ENV_EXAMPLE.read_text()
        assert "CHAT_MODEL=anthropic/claude-sonnet-5" in text
        assert "EMBEDDING_MODEL=voyage/voyage-3.5-lite" in text
        assert "ANTHROPIC_API_KEY=placeholder" in text
        assert "VOYAGE_API_KEY=placeholder" in text

    def test_no_openai_variables(self) -> None:
        assignments = [
            line
            for line in ENV_EXAMPLE.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        assert assignments, "expected variable assignments in .env.example"
        for line in assignments:
            assert "OPENAI" not in line.upper(), f"OpenAI variable found: {line}"

    def test_no_real_looking_secrets(self) -> None:
        for line in ENV_EXAMPLE.read_text().splitlines():
            if "_API_KEY=" in line:
                value = line.split("=", 1)[1]
                assert value.startswith("placeholder"), f"non-placeholder key: {line}"

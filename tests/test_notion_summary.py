"""Offline tests for the plain-language Notion write-up (ops/notion_summary.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.notion_summary import (
    SUMMARY_PROMPT,
    build_facts,
    create_summary_page,
    load_notes_database_id,
    note_properties,
    text_to_blocks,
)

CONTEXT = {
    "run_date": "2026-08-06",
    "universe": "us_liquid",
    "directive": "Focus on volume/price divergence",
    "loops_total": 10,
    "sota_count": 2,
    "candidate": {
        "loop": 5,
        "hypothesis": "Bounded intraday factors add orthogonal alpha",
        "feedback_reason": "beat the previous best",
        "metrics": {"IC": 0.0186, "ARR": 0.7128, "MDD": -0.14},
    },
}


class TestFactsAndPrompt:
    def test_facts_cover_the_story(self) -> None:
        facts = build_facts(CONTEXT)
        assert "Focus on volume/price divergence" in facts
        assert "Bounded intraday factors" in facts
        assert "ARR: 0.7128" in facts
        assert "Hypotheses tested: 10" in facts

    def test_prompt_demands_nontechnical_prose_and_caveats(self) -> None:
        assert "NO trading or machine-learning" in SUMMARY_PROMPT
        assert "caveats" in SUMMARY_PROMPT.lower()
        assert "paper" in SUMMARY_PROMPT


class TestBlocks:
    def test_paragraphs_become_blocks(self) -> None:
        blocks = text_to_blocks("First paragraph.\n\nSecond paragraph.")
        assert len(blocks) == 2
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "First paragraph."

    def test_long_paragraph_is_chunked_under_notion_cap(self) -> None:
        blocks = text_to_blocks("x" * 4000)
        assert len(blocks) == 3
        assert all(
            len(b["paragraph"]["rich_text"][0]["text"]["content"]) <= 1900 for b in blocks
        )


class TestNoteProperties:
    def test_row_properties_from_context(self) -> None:
        props = note_properties("Strategy note", CONTEXT)
        assert props["Note"]["title"][0]["text"]["content"] == "Strategy note"
        assert props["Run Date"]["date"]["start"] == "2026-08-06"
        assert props["Universe"]["rich_text"][0]["text"]["content"] == "us_liquid"
        assert (
            props["Directive"]["rich_text"][0]["text"]["content"]
            == "Focus on volume/price divergence"
        )
        assert (
            props["Hypothesis"]["rich_text"][0]["text"]["content"]
            == "Bounded intraday factors add orthogonal alpha"
        )
        assert props["IC"]["number"] == pytest.approx(0.0186)
        assert props["ARR"]["number"] == pytest.approx(0.7128)
        assert props["MDD"]["number"] == pytest.approx(-0.14)
        assert "Sharpe" not in props  # not in the context metrics

    def test_sparse_context_omits_optional_properties(self) -> None:
        props = note_properties("t", {})
        assert set(props) == {"Note", "Universe"}
        assert props["Universe"]["rich_text"][0]["text"]["content"] == "us_liquid"

    def test_non_finite_metrics_are_dropped(self) -> None:
        context = {"candidate": {"metrics": {"IC": float("nan"), "Sharpe": 1.2}}}
        props = note_properties("t", context)
        assert "IC" not in props
        assert props["Sharpe"]["number"] == pytest.approx(1.2)


class StubNotion:
    def __init__(self, url: str | None = "https://www.notion.so/abc") -> None:
        self.pages: list[tuple[dict, dict, list | None]] = []
        self._url = url

    def create_page(self, parent, properties, children=None):  # noqa: ANN001
        self.pages.append((parent, properties, children))
        page = {"id": "39e9b1a4-36cf-0000-0000-000000000000"}
        if self._url:
            page["url"] = self._url
        return page


class TestCreatePage:
    def test_creates_database_row_with_body(self) -> None:
        client = StubNotion()
        url = create_summary_page(client, "db-notes", "Strategy note", "One.\n\nTwo.", CONTEXT)
        assert url == "https://www.notion.so/abc"
        parent, properties, children = client.pages[0]
        assert parent == {"type": "database_id", "database_id": "db-notes"}
        assert properties["Note"]["title"][0]["text"]["content"] == "Strategy note"
        assert children is not None and len(children) == 2

    def test_url_fallback_from_page_id(self) -> None:
        url = create_summary_page(StubNotion(url=None), "db", "t", "body", CONTEXT)
        assert url == "https://www.notion.so/" + "39e9b1a436cf" + "0" * 20

    def test_title_clipped(self) -> None:
        client = StubNotion()
        create_summary_page(client, "db", "t" * 300, "body", CONTEXT)
        assert len(client.pages[0][1]["Note"]["title"][0]["text"]["content"]) == 120


class TestConfig:
    def test_reads_notes_database_id(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("notion:\n  databases:\n    strategy_notes: 1234-abcd\n")
        assert load_notes_database_id(config) == "1234-abcd"

    def test_missing_database_is_actionable(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("notion:\n  databases: {}\n")
        with pytest.raises(RuntimeError, match="bootstrap_notion"):
            load_notes_database_id(config)

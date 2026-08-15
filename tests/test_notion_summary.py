"""Offline tests for the plain-language Notion write-up (ops/notion_summary.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.notion_summary import (
    RUN_SUMMARY_SCHEMA_VERSION,
    SUMMARY_PROMPT,
    build_facts,
    build_run_summary,
    create_summary_page,
    load_notes_database_id,
    loop_outcome,
    note_properties,
    parse_run_summary,
    run_summary_blocks,
    text_to_blocks,
)

CONTEXT = {
    "run_date": "2026-08-06",
    "universe": "us_liquid",
    "directive": "Focus on volume/price divergence",
    "loops_total": 10,
    "sota_count": 2,
    "run_status": "completed",
    "instrument_hash": "6fbafedc13ed9a52",
    "test_end": "2026-06-12",
    "confirmation_window": ["2026-06-15", "2026-08-13"],
    "loops": [
        {"loop": 0, "action": "factor", "hypothesis": "Momentum works", "decision": False},
        {
            "loop": 5,
            "action": "factor",
            "hypothesis": "Bounded intraday factors add orthogonal alpha",
            "decision": True,
            "metrics": {"IC": 0.0186, "ARR": 0.7128, "MDD": -0.14},
        },
        {"loop": 6, "action": "model", "hypothesis": "Deeper GBDT", "decision": None},
    ],
    "candidate": {
        "loop": 5,
        "hypothesis": "Bounded intraday factors add orthogonal alpha",
        "feedback_reason": "beat the previous best",
        "metrics": {"IC": 0.0186, "ARR": 0.7128, "MDD": -0.14, "IR": 1.72},
        "factors": ["intraday_range_z", "downside_share_60"],
        "window": ["2025-01-02", "2026-07-10"],
    },
    "incumbent": {
        "workspace": "/y/e05ad9b46f4d",
        "metrics": {"IC": 0.0217, "ARR": 0.5936, "MDD": -0.2665},
        "window": ["2025-01-02", "2026-07-10"],
    },
}


class TestFactsAndPrompt:
    def test_facts_cover_the_story(self) -> None:
        facts = build_facts(CONTEXT)
        assert "Focus on volume/price divergence" in facts
        assert "Bounded intraday factors" in facts
        assert "ARR: 0.7128" in facts
        assert "Hypotheses tested: 10" in facts

    def test_facts_include_factors_window_and_incumbent(self) -> None:
        facts = build_facts(CONTEXT)
        assert "intraday_range_z, downside_share_60" in facts
        assert "Historical test window: 2025-01-02 to 2026-07-10" in facts
        assert "incumbent" in facts
        assert "ARR 0.5936" in facts
        assert "SAME test window" in facts

    def test_facts_flag_incumbent_window_mismatch(self) -> None:
        context = dict(CONTEXT)
        context["incumbent"] = {
            "metrics": {"ARR": 0.5936},
            "window": ["2025-01-02", "2026-08-11"],
        }
        facts = build_facts(context)
        assert "DIFFERENT test window" in facts
        assert "not directly comparable" in facts

    def test_facts_without_incumbent_say_none_promoted(self) -> None:
        context = {k: v for k, v in CONTEXT.items() if k != "incumbent"}
        context["incumbent"] = None
        facts = build_facts(context)
        assert "none — nothing is promoted yet" in facts

    def test_prompt_demands_nontechnical_prose_and_caveats(self) -> None:
        assert "NO trading or machine-learning" in SUMMARY_PROMPT
        assert "caveats" in SUMMARY_PROMPT.lower()
        assert "paper" in SUMMARY_PROMPT
        assert "incumbent" in SUMMARY_PROMPT


class TestBlocks:
    def test_paragraphs_become_blocks(self) -> None:
        blocks = text_to_blocks("First paragraph.\n\nSecond paragraph.")
        assert len(blocks) == 2
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "First paragraph."

    def test_long_paragraph_is_chunked_under_notion_cap(self) -> None:
        blocks = text_to_blocks("x" * 4000)
        assert len(blocks) == 3
        assert all(len(b["paragraph"]["rich_text"][0]["text"]["content"]) <= 1900 for b in blocks)


class TestRunSummary:
    def test_schema_version_and_core_fields(self) -> None:
        summary = build_run_summary(CONTEXT)
        assert summary["schema_version"] == RUN_SUMMARY_SCHEMA_VERSION
        assert summary["directive"] == "Focus on volume/price divergence"
        assert summary["status"] == "completed"
        assert summary["universe"] == {
            "name": "us_liquid",
            "instrument_hash": "6fbafedc13ed9a52",
        }
        assert summary["windows"] == {
            "test": ["2025-01-02", "2026-07-10"],
            "test_end": "2026-06-12",
            "confirmation": ["2026-06-15", "2026-08-13"],
        }
        assert summary["winner"]["loop"] == 5
        assert summary["winner"]["metrics"]["IR"] == pytest.approx(1.72)
        assert summary["winner"]["factors"] == ["intraday_range_z", "downside_share_60"]

    def test_per_hypothesis_outcomes(self) -> None:
        summary = build_run_summary(CONTEXT)
        outcomes = {h["loop"]: h["outcome"] for h in summary["hypotheses"]}
        assert outcomes == {0: "rejected", 5: "SOTA", 6: "failed"}
        assert summary["hypotheses"][1]["hypothesis"].startswith("Bounded intraday")
        assert summary["hypotheses"][1]["metrics"]["IC"] == pytest.approx(0.0186)

    def test_outcome_mapping(self) -> None:
        assert loop_outcome({"decision": True}) == "SOTA"
        assert loop_outcome({"decision": False}) == "rejected"
        assert loop_outcome({"decision": None}) == "failed"
        assert loop_outcome({}) == "failed"

    def test_sparse_context_degrades_to_nones(self) -> None:
        summary = build_run_summary({})
        assert summary["schema_version"] == RUN_SUMMARY_SCHEMA_VERSION
        assert summary["universe"] == {"name": "us_liquid", "instrument_hash": None}
        assert summary["hypotheses"] == []
        assert summary["winner"] is None

    def test_blocks_are_json_code_blocks(self) -> None:
        blocks = run_summary_blocks(build_run_summary(CONTEXT))
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "json"
        for rich in blocks[0]["code"]["rich_text"]:
            assert len(rich["text"]["content"]) <= 1900

    def test_round_trip_through_blocks(self) -> None:
        summary = build_run_summary(CONTEXT)
        assert parse_run_summary(run_summary_blocks(summary)) == summary

    def test_chunk_boundary_over_2000_chars_round_trips(self) -> None:
        context = dict(CONTEXT)
        context["loops"] = [
            {"loop": n, "hypothesis": f"h{n} " + "x" * 500, "decision": False}
            for n in range(10)
        ]
        summary = build_run_summary(context)
        blocks = run_summary_blocks(summary)
        elements = [rich for block in blocks for rich in block["code"]["rich_text"]]
        assert len(elements) > 1  # actually crossed the chunk boundary
        assert all(len(rich["text"]["content"]) <= 1900 for rich in elements)
        assert parse_run_summary(blocks) == summary

    def test_reader_handles_api_read_shape(self) -> None:
        summary = build_run_summary(CONTEXT)
        blocks = run_summary_blocks(summary)
        for block in blocks:  # Notion GET returns plain_text, not text.content
            block["code"]["rich_text"] = [
                {"type": "text", "plain_text": rich["text"]["content"]}
                for rich in block["code"]["rich_text"]
            ]
        assert parse_run_summary(blocks) == summary

    def test_reader_ignores_prose_and_returns_none_without_summary(self) -> None:
        assert parse_run_summary(text_to_blocks("Just prose.")) is None
        assert parse_run_summary([]) is None
        broken = [
            {
                "type": "code",
                "code": {
                    "language": "json",
                    "rich_text": [{"type": "text", "text": {"content": "{not json"}}],
                },
            }
        ]
        assert parse_run_summary(broken) is None


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
        assert children is not None
        assert [b["type"] for b in children[:2]] == ["paragraph", "paragraph"]
        # US-013: the run_summary rides the same page, after the prose.
        assert parse_run_summary(children) == build_run_summary(CONTEXT)

    def test_url_fallback_from_page_id(self) -> None:
        url = create_summary_page(StubNotion(url=None), "db", "t", "body", CONTEXT)
        assert url == "https://www.notion.so/" + "39e9b1a436cf" + "0" * 20

    def test_title_clipped(self) -> None:
        client = StubNotion()
        create_summary_page(client, "db", "t" * 300, "body", CONTEXT)
        assert len(client.pages[0][1]["Note"]["title"][0]["text"]["content"]) == 120

    def test_page_body_carries_survivorship_caveat(self) -> None:
        """US-025: the standing delisted-names caveat rides every write-up,
        between the prose and the run_summary JSON — which must still parse."""
        from ops.promotion_gate import SURVIVORSHIP_CAVEAT

        client = StubNotion()
        create_summary_page(client, "db", "t", "One.\n\nTwo.", CONTEXT)
        children = client.pages[0][2]
        assert children is not None
        paragraphs = [
            block["paragraph"]["rich_text"][0]["text"]["content"]
            for block in children
            if block["type"] == "paragraph"
        ]
        assert paragraphs[-1] == SURVIVORSHIP_CAVEAT
        assert "docs/decisions.md" in SURVIVORSHIP_CAVEAT and "US-025" in SURVIVORSHIP_CAVEAT
        assert parse_run_summary(children) == build_run_summary(CONTEXT)


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

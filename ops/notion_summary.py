"""Plain-language Notion write-up of a successful research run.

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.notion_summary \
        --context /path/to/context.json

Reads a small context JSON (built by ops/gpu_pipeline.py at completion:
directive, universe, loop counts, and the winning candidate's hypothesis +
metrics), asks the judgment model for a NONTECHNICAL summary of the result
and the investing approach, and creates a row in the Strategy Notes database
(the prose lands in the row's page body). Prints the page URL on stdout —
the caller posts it to Slack.

The page body also carries a machine-readable ``run_summary`` JSON (US-013):
a fenced ``json`` code block after the prose, chunked to Notion's ~2000-char
rich-text element limit, with a ``schema_version`` field. It is the read-back
contract for run memory (US-014) — ``parse_run_summary`` beside the writer
reassembles and parses it from a page's block children.

Why a database row: the notes accumulate one per run, and a database keeps
them sortable/filterable (run date, universe, headline metrics) as they grow —
loose child pages under the parent page do not scale. The Decision Log is
still not an option: its rich_text rows clip at 2000 chars, and per
docs/reference/notion-schema.md each database has exactly one writer — this
module is the sole writer of Strategy Notes.

Must run under `onecli run --agent rdq-orchestrator` — the proxy injects both
the Anthropic key (for the summary) and the Notion bearer (app connection).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "orchestrator" / "config.yaml"

# Notion rich_text objects clip server-side at 2000 chars; stay under it.
_BLOCK_CHAR_LIMIT = 1900

# Notion caps a block's rich_text array at 100 elements.
_CODE_RICH_TEXT_LIMIT = 100

# Bump when the run_summary shape changes; readers key on it (US-014).
RUN_SUMMARY_SCHEMA_VERSION = 1

# Candidate metrics mirrored into sortable number properties (schema keys).
_METRIC_PROPERTIES = ("IC", "ARR", "MDD", "Sharpe")

SUMMARY_PROMPT = """\
You are writing for a smart, curious reader with NO trading or machine-learning
background — the tone of a good newspaper's finance explainer. Using only the
facts below, write a summary (4-6 short paragraphs, no headings, no bullet
lists, no jargon) covering:

1. What question the research asked (the operator's directive, in plain words).
2. What the winning approach actually does when picking stocks — translate the
   hypothesis into everyday language (e.g. "it favors stocks that have been
   drifting up steadily rather than jumping around").
3. How it did in the historical test — translate the metrics: ARR is the
   yearly return the strategy would have earned ABOVE the market after trading
   costs; MDD is the worst peak-to-trough loss along the way; IC measures how
   often its daily stock rankings pointed the right way (small positive numbers
   are normal and useful). When the facts include the currently live
   (incumbent) strategy's results, say plainly whether the new approach beat
   it and by how much — and if the facts say the test windows differ, say the
   comparison is not apples-to-apples.
4. Honest caveats: this is a simulation on past data; past results do not
   guarantee future ones; the strategy trades paper money until explicitly
   promoted, and even then it trades a paper account.

Facts:
{facts}

Return ONLY the summary paragraphs, separated by blank lines."""


def build_facts(context: dict) -> str:
    candidate = context.get("candidate") or {}
    metrics = candidate.get("metrics") or {}
    lines = [
        f"Run date: {context.get('run_date', 'unknown')}",
        f"Universe traded: {context.get('universe') or 'us_liquid'} "
        f"({context.get('universe_size', 'unknown')} stocks)",
        f"Operator directive: {context.get('directive') or '(none — open-ended exploration)'}",
        f"Hypotheses tested: {context.get('loops_total', '?')} "
        f"(new-best results: {context.get('sota_count', '?')})",
        f"Winning hypothesis (loop {candidate.get('loop', '?')}):"
        f" {candidate.get('hypothesis', 'n/a')}",
        f"Why the system kept it: {candidate.get('feedback_reason') or 'n/a'}",
    ]
    if candidate.get("factors"):
        lines.append(f"Signals the winning strategy uses: {', '.join(candidate['factors'])}")
    window = candidate.get("window")
    if window:
        lines.append(f"Historical test window: {window[0]} to {window[1]}")
    for label, key in (("IC", "IC"), ("ARR", "ARR"), ("MDD", "MDD"), ("Sharpe", "Sharpe")):
        if key in metrics:
            lines.append(f"{label}: {metrics[key]:.4f}")
    lines.extend(_incumbent_facts(context.get("incumbent"), window))
    return "\n".join(lines)


def _incumbent_facts(incumbent: dict | None, candidate_window: list | None) -> list[str]:
    """Baseline facts — what the new result must beat to matter (2026-08-12 gap)."""
    if incumbent is None:
        return ["Currently live (incumbent) strategy: none — nothing is promoted yet"]
    metrics = incumbent.get("metrics") or {}
    parts = ", ".join(f"{k} {metrics[k]:.4f}" for k in ("IC", "ARR", "MDD") if k in metrics)
    if not parts:
        return ["Currently live (incumbent) strategy: metrics unavailable"]
    window = incumbent.get("window")
    if window and candidate_window and window == candidate_window:
        note = " on the SAME test window as the new result"
    elif window and candidate_window:
        note = (
            f" on a DIFFERENT test window ({window[0]} to {window[1]}) — "
            "not directly comparable with the new result"
        )
    else:
        note = ""
    return [f"Currently live (incumbent) strategy's results{note}: {parts}"]


def generate_summary(context: dict) -> str:
    from orchestrator.llm import ModelRouter

    router = ModelRouter()
    message = router.judgment(
        [{"role": "user", "content": SUMMARY_PROMPT.format(facts=build_facts(context))}],
        max_tokens=2000,
    )
    text = "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise RuntimeError("the model returned no summary text")
    return text


def text_to_blocks(text: str) -> list[dict]:
    """Paragraphs -> Notion paragraph blocks, chunked under the 2000-char cap."""
    blocks: list[dict] = []
    for paragraph in filter(None, (p.strip() for p in text.split("\n\n"))):
        for start in range(0, len(paragraph), _BLOCK_CHAR_LIMIT):
            chunk = paragraph[start : start + _BLOCK_CHAR_LIMIT]
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]},
                }
            )
    return blocks


def loop_outcome(loop: dict) -> str:
    """Per-hypothesis outcome: SOTA (adopted), rejected (verdict against), or
    failed (no verdict recorded — the loop crashed or never finished)."""
    decision = loop.get("decision")
    if decision is None:
        return "failed"
    return "SOTA" if decision else "rejected"


def build_run_summary(context: dict) -> dict:
    """The machine-readable record of a run (US-013) — everything a later run
    needs to not re-propose rejected ideas, straight from the pipeline context."""
    candidate = context.get("candidate") or {}
    winner = None
    if candidate.get("loop") is not None or candidate.get("hypothesis"):
        winner = {
            "loop": candidate.get("loop"),
            "hypothesis": candidate.get("hypothesis"),
            "metrics": candidate.get("metrics") or {},
            "factors": candidate.get("factors"),
        }
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_date": context.get("run_date"),
        "status": context.get("run_status"),
        "directive": context.get("directive"),
        "universe": {
            "name": context.get("universe") or "us_liquid",
            "instrument_hash": context.get("instrument_hash"),
        },
        "windows": {
            "test": candidate.get("window"),
            "test_end": context.get("test_end"),
            "confirmation": context.get("confirmation_window"),
        },
        "hypotheses": [
            {
                "loop": loop.get("loop"),
                "action": loop.get("action"),
                "hypothesis": loop.get("hypothesis"),
                "outcome": loop_outcome(loop),
                "metrics": loop.get("metrics"),
            }
            for loop in context.get("loops") or []
        ],
        "winner": winner,
    }


def run_summary_blocks(summary: dict) -> list[dict]:
    """run_summary -> fenced ``json`` code block(s), chunked under Notion's
    2000-char rich-text element cap (and the 100-elements-per-block cap)."""
    text = json.dumps(summary, indent=2)
    chunks = [
        text[start : start + _BLOCK_CHAR_LIMIT]
        for start in range(0, len(text), _BLOCK_CHAR_LIMIT)
    ]
    return [
        {
            "object": "block",
            "type": "code",
            "code": {
                "language": "json",
                "rich_text": [
                    {"type": "text", "text": {"content": chunk}}
                    for chunk in chunks[start : start + _CODE_RICH_TEXT_LIMIT]
                ],
            },
        }
        for start in range(0, len(chunks), _CODE_RICH_TEXT_LIMIT)
    ]


def parse_run_summary(blocks: list[dict]) -> dict | None:
    """Reassemble and parse the run_summary from a page's block children.

    Accepts both the write shape (text.content) and the API read shape
    (plain_text). Returns None when the page has no parseable run_summary —
    the caller (US-014 digest builder) degrades that run, never raises.
    """
    parts: list[str] = []
    for block in blocks:
        if block.get("type") != "code":
            continue
        code = block.get("code") or {}
        if code.get("language") != "json":
            continue
        for rich in code.get("rich_text") or []:
            content = (rich.get("text") or {}).get("content")
            if content is None:
                content = rich.get("plain_text")
            parts.append(content or "")
    if not parts:
        return None
    try:
        parsed = json.loads("".join(parts))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "schema_version" in parsed:
        return parsed
    return None


def note_properties(title: str, context: dict) -> dict[str, Any]:
    """Strategy Notes row properties (docs/reference/notion-schema.md)."""
    from orchestrator.notion_recorder import (
        date_property,
        number_property,
        rich_text_property,
        title_property,
    )

    properties: dict[str, Any] = {
        "Note": title_property(title),
        "Universe": rich_text_property(str(context.get("universe") or "us_liquid")),
    }
    if context.get("run_date"):
        properties["Run Date"] = date_property(str(context["run_date"]))
    if context.get("directive"):
        properties["Directive"] = rich_text_property(str(context["directive"]))
    candidate = context.get("candidate") or {}
    if candidate.get("hypothesis"):
        properties["Hypothesis"] = rich_text_property(str(candidate["hypothesis"]))
    metrics = candidate.get("metrics") or {}
    for key in _METRIC_PROPERTIES:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            properties[key] = number_property(float(value))
    return properties


def load_notes_database_id(config_path: Path = CONFIG_PATH) -> str:
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    databases = (data.get("notion") or {}).get("databases") or {}
    db_id = databases.get("strategy_notes")
    if not db_id:
        raise RuntimeError(
            f"no notion.databases.strategy_notes in {config_path} — run"
            " ops/bootstrap_notion.py to create the Strategy Notes database"
        )
    return str(db_id)


def create_summary_page(client, database_id: str, title: str, summary: str, context: dict) -> str:
    """Create the Strategy Notes row (prose + run_summary JSON in the page
    body); returns its URL."""
    page = client.create_page(
        {"type": "database_id", "database_id": database_id},
        note_properties(title, context),
        children=text_to_blocks(summary) + run_summary_blocks(build_run_summary(context)),
    )
    url = page.get("url")
    if not url:
        # Fallback: Notion page URLs are the id with dashes stripped.
        url = "https://www.notion.so/" + str(page.get("id", "")).replace("-", "")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=Path, required=True, help="context JSON from the pipeline"
    )
    parser.add_argument("--title", default=None, help="page title (default derived from context)")
    args = parser.parse_args(argv)

    context = json.loads(args.context.read_text())
    default_title = (
        f"Strategy note — {context.get('universe') or 'us_liquid'}"
        f" run {context.get('run_date', '')}"
    )
    title = args.title or default_title.strip()

    summary = generate_summary(context)

    from orchestrator.notion_client import NotionClient

    url = create_summary_page(NotionClient(), load_notes_database_id(), title, summary, context)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

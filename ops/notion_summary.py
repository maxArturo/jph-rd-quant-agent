"""Plain-language Notion write-up of a successful research run.

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.notion_summary \
        --context /path/to/context.json

Reads a small context JSON (built by ops/gpu_pipeline.py at completion:
directive, universe, loop counts, and the winning candidate's hypothesis +
metrics), asks the judgment model for a NONTECHNICAL summary of the result
and the investing approach, and creates a row in the Strategy Notes database
(the prose lands in the row's page body). Prints the page URL on stdout —
the caller posts it to Slack.

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
   are normal and useful).
4. Honest caveats: this is a simulation on past data; past results do not
   guarantee future ones; {account_context}.

Facts:
{facts}

Return ONLY the summary paragraphs, separated by blank lines."""

# Stated when the caller's context carries no account_context (US-017): the
# pipeline passes the real slot semantics; a bare CLI run gets this fallback.
DEFAULT_ACCOUNT_CONTEXT = (
    "the strategy trades no account until an operator explicitly promotes it"
)


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
    for label, key in (("IC", "IC"), ("ARR", "ARR"), ("MDD", "MDD"), ("Sharpe", "Sharpe")):
        if key in metrics:
            lines.append(f"{label}: {metrics[key]:.4f}")
    return "\n".join(lines)


def build_prompt(context: dict) -> str:
    """The full summary prompt: facts plus the caller-stated account context."""
    account_context = str(context.get("account_context") or DEFAULT_ACCOUNT_CONTEXT)
    return SUMMARY_PROMPT.format(facts=build_facts(context), account_context=account_context)


def generate_summary(context: dict) -> str:
    from orchestrator.llm import ModelRouter

    router = ModelRouter()
    message = router.judgment(
        [{"role": "user", "content": build_prompt(context)}],
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
    """Create the Strategy Notes row (prose in the page body); returns its URL."""
    page = client.create_page(
        {"type": "database_id", "database_id": database_id},
        note_properties(title, context),
        children=text_to_blocks(summary),
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

"""Plain-language Notion write-up of a successful research run.

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.notion_summary \
        --context /path/to/context.json

Reads a small context JSON (built by ops/gpu_pipeline.py at completion:
directive, universe, loop counts, and the winning candidate's hypothesis +
metrics), asks the judgment model for a NONTECHNICAL summary of the result
and the investing approach, and creates a Notion PAGE (with body paragraphs)
under the configured parent page. Prints the page URL on stdout — the caller
posts it to Slack.

Why a standalone page: the Decision Log schema has no url property and its
rich_text rows clip at 2000 chars; per docs/reference/notion-schema.md each
database has exactly one writer. A child page of the parent workspace page
collides with nobody and holds unlimited prose.

Must run under `onecli run --agent rdq-orchestrator` — the proxy injects both
the Anthropic key (for the summary) and the Notion bearer (app connection).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "orchestrator" / "config.yaml"

# Notion rich_text objects clip server-side at 2000 chars; stay under it.
_BLOCK_CHAR_LIMIT = 1900
_TITLE_LIMIT = 120

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
    for label, key in (("IC", "IC"), ("ARR", "ARR"), ("MDD", "MDD"), ("Sharpe", "Sharpe")):
        if key in metrics:
            lines.append(f"{label}: {metrics[key]:.4f}")
    return "\n".join(lines)


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


def load_parent_page_id(config_path: Path = CONFIG_PATH) -> str:
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    parent = (data.get("notion") or {}).get("parent_page_id")
    if not parent:
        raise RuntimeError(
            f"no notion.parent_page_id in {config_path} — run ops/bootstrap_notion.py"
        )
    return str(parent)


def create_summary_page(client, parent_page_id: str, title: str, summary: str) -> str:
    """Create the page; returns its URL."""
    page = client.create_page(
        {"type": "page_id", "page_id": parent_page_id},
        {"title": {"title": [{"type": "text", "text": {"content": title[:_TITLE_LIMIT]}}]}},
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

    url = create_summary_page(NotionClient(), load_parent_page_id(), title, summary)
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

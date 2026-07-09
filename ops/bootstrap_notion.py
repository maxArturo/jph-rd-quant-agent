"""Bootstrap the five Notion databases under the parent page.

Creates Research Ideas, Hypothesis Log, Backtest Results, Decision Log and
Trade Ledger with the property schemas defined in
docs/reference/notion-schema.md (keep database_properties() below in sync
with that document), then writes the database ids into
orchestrator/config.yaml.

Idempotent: existing child databases under the parent page are matched by
title and reused — rerunning never duplicates a database.

Run through the OneCLI proxy (auth is connector-injected, never in code):

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.bootstrap_notion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from orchestrator.notion_client import NotionClient

DEFAULT_PARENT_PAGE_ID = "3979b1a4-36cf-8046-baa5-cc14c1ca7665"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "orchestrator" / "config.yaml"

# Title of the database every relation property points at. It must be created
# first so the others can reference its id.
IDEAS_TITLE = "Research Ideas"

_SELECT = {
    "idea_status": ["proposed", "researching", "stopped", "completed", "failed", "promoted"],
    "hypothesis_action": [
        "pending",
        "approved",
        "edited",
        "rejected",
        "auto_approved",
        "cancelled",
    ],
    "decision_type": ["promotion", "halt", "resume", "universe", "other"],
    "order_side": ["buy", "sell"],
    "order_status": [
        "submitted",
        "filled",
        "partially_filled",
        "rejected",
        "cancelled",
        "expired",
    ],
}


def _select(options_key: str) -> dict[str, Any]:
    return {"select": {"options": [{"name": name} for name in _SELECT[options_key]]}}


def _number() -> dict[str, Any]:
    return {"number": {"format": "number"}}


def _relation(ideas_db_id: str) -> dict[str, Any]:
    # single_property: no synced back-reference property on Research Ideas.
    return {"relation": {"database_id": ideas_db_id, "single_property": {}}}


def database_properties(ideas_db_id: str) -> dict[str, dict[str, Any]]:
    """Property schema per database title (docs/reference/notion-schema.md).

    ``ideas_db_id`` is the Research Ideas database id that relation properties
    point at; pass a placeholder when building the Research Ideas schema
    itself (it contains no relations).
    """
    return {
        IDEAS_TITLE: {
            "Idea": {"title": {}},
            "Raw Idea": {"rich_text": {}},
            "Directive": {"rich_text": {}},
            "Universe": {"rich_text": {}},
            "Status": _select("idea_status"),
            "Thread": {"url": {}},
            "Thread TS": {"rich_text": {}},
        },
        "Hypothesis Log": {
            "Hypothesis": {"title": {}},
            "Idea": _relation(ideas_db_id),
            "Details": {"rich_text": {}},
            "Action": _select("hypothesis_action"),
            "Operator Input": {"rich_text": {}},
            "Interaction Key": {"rich_text": {}},
        },
        "Backtest Results": {
            "Experiment": {"title": {}},
            "Idea": _relation(ideas_db_id),
            "IC": _number(),
            "ICIR": _number(),
            "Rank IC": _number(),
            "ARR": _number(),
            "IR": _number(),
            "MDD": _number(),
            "Sharpe": _number(),
            "SOTA": {"checkbox": {}},
            "Workspace": {"rich_text": {}},
            "Universe": {"rich_text": {}},
        },
        "Decision Log": {
            "Decision": {"title": {}},
            "Type": _select("decision_type"),
            "Details": {"rich_text": {}},
            "Idea": _relation(ideas_db_id),
            "Decided At": {"date": {}},
        },
        "Trade Ledger": {
            "Order": {"title": {}},
            "Order ID": {"rich_text": {}},
            "Symbol": {"rich_text": {}},
            "Side": _select("order_side"),
            "Qty": _number(),
            "Limit Price": _number(),
            "Status": _select("order_status"),
            "Filled Qty": _number(),
            "Filled Avg Price": _number(),
            "Submitted At": {"date": {}},
            "Notes": {"rich_text": {}},
        },
    }


# config.yaml key per database title.
CONFIG_KEYS = {
    IDEAS_TITLE: "research_ideas",
    "Hypothesis Log": "hypothesis_log",
    "Backtest Results": "backtest_results",
    "Decision Log": "decision_log",
    "Trade Ledger": "trade_ledger",
}


def bootstrap(client: NotionClient, parent_page_id: str) -> dict[str, dict[str, str]]:
    """Ensure all five databases exist; return title -> {id, action}.

    ``action`` is "created" or "exists" so callers can report what happened.
    """
    existing = client.list_child_databases(parent_page_id)
    outcome: dict[str, dict[str, str]] = {}

    def ensure(title: str, properties: dict[str, Any]) -> str:
        if title in existing:
            outcome[title] = {"id": existing[title], "action": "exists"}
            return existing[title]
        created = client.create_database(parent_page_id, title, properties)
        outcome[title] = {"id": created["id"], "action": "created"}
        return created["id"]

    # Research Ideas first: every relation property points at it.
    ideas_id = ensure(IDEAS_TITLE, database_properties("placeholder")[IDEAS_TITLE])
    schemas = database_properties(ideas_id)
    for title in CONFIG_KEYS:
        if title != IDEAS_TITLE:
            ensure(title, schemas[title])
    return outcome


def write_config(
    config_path: Path, parent_page_id: str, outcome: dict[str, dict[str, str]]
) -> None:
    """Merge the database ids into config_path under the ``notion:`` key.

    Other top-level keys in an existing file are preserved (comments are not —
    the file is machine-managed by this script).
    """
    config: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text())
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"{config_path} must hold a YAML mapping, got: {type(loaded)}")
            config = loaded
    config["notion"] = {
        "parent_page_id": parent_page_id,
        "databases": {CONFIG_KEYS[title]: info["id"] for title, info in outcome.items()},
    }
    header = (
        "# Orchestrator configuration. The notion: section is machine-managed by\n"
        "# ops/bootstrap_notion.py — rerun it rather than editing ids by hand.\n"
        "# Database ids are not secrets (auth is injected by the OneCLI proxy).\n"
    )
    config_path.write_text(header + yaml.safe_dump(config, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the five Notion databases under the parent page."
    )
    parser.add_argument(
        "--parent-page-id",
        default=DEFAULT_PARENT_PAGE_ID,
        help="Notion page the databases live under (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="config.yaml to write database ids into (default: %(default)s)",
    )
    opts = parser.parse_args(argv)

    client = NotionClient()
    outcome = bootstrap(client, opts.parent_page_id)
    write_config(opts.config, opts.parent_page_id, outcome)

    for title, info in outcome.items():
        print(f"{info['action']:>7}  {title}: {info['id']}")
    print(f"ids written to {opts.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

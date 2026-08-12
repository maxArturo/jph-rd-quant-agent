"""Bootstrap the Notion databases (paper page, and the LIVE page with --live).

Creates Research Ideas, Hypothesis Log, Backtest Results, Decision Log,
Trade Ledger, Account Snapshots and Strategy Notes under the paper parent
page with the property schemas defined in docs/reference/notion-schema.md
(keep database_properties() below in sync with that document), then writes
the database ids into orchestrator/config.yaml.

With ``--live`` it additionally ensures the real-money reporting surface
(US-011): a sibling page titled "Automated AI Quant Investment — LIVE 🔴"
with an intro callout stating it reflects a real-money account, holding
Trade Ledger (Live) and Account Snapshots (Live) with schemas identical to
their paper counterparts. An operator-created page with that exact title is
adopted (found via Notion search — task D of the go-live checklist lets the
operator create and share the page when the integration cannot create
siblings itself).

Idempotent: existing pages/databases are matched by exact title and reused —
rerunning never duplicates anything, and the paper page and its databases
are never touched by the live path.

Run through the OneCLI proxy (auth is connector-injected, never in code):

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.bootstrap_notion [--live]
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

# --- live reporting surface (US-011) ------------------------------------------
# The live page is a SIBLING of the paper parent page, unmistakably labeled;
# paper and live records can never be confused. Match titles EXACTLY.
LIVE_PAGE_TITLE = "Automated AI Quant Investment — LIVE 🔴"
LIVE_PAGE_INTRO = (
    "This page reflects a REAL-MONEY Alpaca account. Every row under it is a "
    "real order or real account state — not paper trading. Paper records live "
    'under "Automated AI Quant Investment".'
)
# live database title -> the paper database whose property schema it copies.
LIVE_SOURCE_TITLES = {
    "Trade Ledger (Live)": "Trade Ledger",
    "Account Snapshots (Live)": "Account Snapshots",
}


class LiveBootstrapError(RuntimeError):
    """The live page could not be created or adopted (see operator task D)."""

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
    "snapshot_outcome": [
        "traded",
        "no_trade",
        "gate_rejected",
        "breaker_tripped",
        "halted",
    ],
}


def _select(options_key: str) -> dict[str, Any]:
    return {"select": {"options": [{"name": name} for name in _SELECT[options_key]]}}


def _number() -> dict[str, Any]:
    return {"number": {"format": "number"}}


def _percent() -> dict[str, Any]:
    # Values are stored as fractions (0.0125) and rendered by Notion as 1.25%.
    return {"number": {"format": "percent"}}


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
        "Account Snapshots": {
            "Snapshot": {"title": {}},
            "Date": {"date": {}},
            "Equity": _number(),
            "Cash": _number(),
            "Long Value": _number(),
            "Short Value": _number(),
            "Positions": _number(),
            "Day P/L": _number(),
            "Day P/L %": _percent(),
            "P/L Day": {"date": {}},
            "Orders Placed": _number(),
            "Orders Filled": _number(),
            "Outcome": _select("snapshot_outcome"),
            "Breaker": {"rich_text": {}},
            "Notes": {"rich_text": {}},
        },
        "Strategy Notes": {
            "Note": {"title": {}},
            "Run Date": {"date": {}},
            "Universe": {"rich_text": {}},
            "Directive": {"rich_text": {}},
            "Hypothesis": {"rich_text": {}},
            "IC": _number(),
            "ARR": _number(),
            "MDD": _number(),
            "Sharpe": _number(),
        },
    }


def live_database_properties() -> dict[str, dict[str, Any]]:
    """Property schema per LIVE database title — identical to the paper twins.

    Neither live database has a relation property, so the ideas-db id
    placeholder is never embedded in the schemas.
    """
    paper = database_properties("unused-live-schemas-have-no-relations")
    return {live: paper[source] for live, source in LIVE_SOURCE_TITLES.items()}


# config.yaml key per database title.
CONFIG_KEYS = {
    IDEAS_TITLE: "research_ideas",
    "Hypothesis Log": "hypothesis_log",
    "Backtest Results": "backtest_results",
    "Decision Log": "decision_log",
    "Trade Ledger": "trade_ledger",
    "Account Snapshots": "account_snapshots",
    "Strategy Notes": "strategy_notes",
}

# config.yaml key per LIVE database title (kept out of CONFIG_KEYS: the paper
# bootstrap loop iterates CONFIG_KEYS and must never create these under the
# paper page).
LIVE_CONFIG_KEYS = {
    "Trade Ledger (Live)": "trade_ledger_live",
    "Account Snapshots (Live)": "account_snapshots_live",
}

ALL_CONFIG_KEYS = {**CONFIG_KEYS, **LIVE_CONFIG_KEYS}


def bootstrap(client: NotionClient, parent_page_id: str) -> dict[str, dict[str, str]]:
    """Ensure all seven databases exist; return title -> {id, action}.

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


def _page_title(page: dict[str, Any]) -> str:
    """Plain-text title of a page object (the sole ``title``-type property)."""
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(
                part.get("plain_text") or part.get("text", {}).get("content", "")
                for part in prop.get("title", [])
            )
    return ""


def _intro_callout_block() -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": LIVE_PAGE_INTRO}}],
            "icon": {"type": "emoji", "emoji": "🔴"},
            "color": "red_background",
        },
    }


def ensure_live_page(client: NotionClient, paper_page_id: str) -> dict[str, str]:
    """Adopt or create the LIVE sibling page; return {id, action}.

    Adoption matches the EXACT title via Notion search (covers an
    operator-created page anywhere the integration can see, including
    workspace level); an adopted page missing the real-money intro callout
    gets it appended. Creation places the page under the paper page's own
    parent — a true sibling. When the paper page has no page parent (it is
    workspace-level, which the API cannot create siblings of), raise
    LiveBootstrapError telling the operator to create + share the page
    (go-live checklist task D) so a rerun adopts it.
    """
    for page in client.search_pages(LIVE_PAGE_TITLE):
        if _page_title(page) == LIVE_PAGE_TITLE:
            if not any(
                block.get("type") == "callout" for block in client.list_children(page["id"])
            ):
                client.append_children(page["id"], [_intro_callout_block()])
            return {"id": page["id"], "action": "exists"}

    parent = client.get_page(paper_page_id).get("parent", {})
    if parent.get("type") != "page_id":
        raise LiveBootstrapError(
            f"no page titled {LIVE_PAGE_TITLE!r} is shared with the integration, and "
            f"the paper page's parent is {parent.get('type', 'unknown')!r} (not a page), "
            "so a sibling cannot be created via the API. Create an empty page with "
            "exactly that title, share it with the integration, and rerun --live to "
            "adopt it (ops/runbook.md go-live checklist, task D)."
        )
    created = client.create_page(
        parent={"type": "page_id", "page_id": parent["page_id"]},
        properties={"title": {"title": [{"type": "text", "text": {"content": LIVE_PAGE_TITLE}}]}},
        children=[_intro_callout_block()],
    )
    return {"id": created["id"], "action": "created"}


def bootstrap_live(
    client: NotionClient, paper_page_id: str
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Ensure the LIVE page and its two databases; return (page, title -> {id, action}).

    The paper page and its databases are never read or written beyond
    resolving the paper page's parent for sibling creation.
    """
    live_page = ensure_live_page(client, paper_page_id)
    existing = client.list_child_databases(live_page["id"])
    outcome: dict[str, dict[str, str]] = {}
    for title, properties in live_database_properties().items():
        if title in existing:
            outcome[title] = {"id": existing[title], "action": "exists"}
        else:
            created = client.create_database(live_page["id"], title, properties)
            outcome[title] = {"id": created["id"], "action": "created"}
    return live_page, outcome


def write_config(
    config_path: Path,
    parent_page_id: str,
    outcome: dict[str, dict[str, str]],
    live_parent_page_id: str | None = None,
) -> None:
    """Merge the database ids into config_path under the ``notion:`` key.

    Other top-level keys in an existing file are preserved, and so are
    notion.databases entries this run did not produce — a paper-only rerun
    never drops previously bootstrapped live ids, and vice versa (comments
    are not preserved — the file is machine-managed by this script).
    """
    config: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text())
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"{config_path} must hold a YAML mapping, got: {type(loaded)}")
            config = loaded
    notion = config.get("notion")
    notion = dict(notion) if isinstance(notion, dict) else {}
    databases = notion.get("databases")
    databases = dict(databases) if isinstance(databases, dict) else {}
    databases.update({ALL_CONFIG_KEYS[title]: info["id"] for title, info in outcome.items()})
    notion["parent_page_id"] = parent_page_id
    if live_parent_page_id is not None:
        notion["live_parent_page_id"] = live_parent_page_id
    notion["databases"] = databases
    config["notion"] = notion
    header = (
        "# Orchestrator configuration. The notion: section is machine-managed by\n"
        "# ops/bootstrap_notion.py — rerun it rather than editing ids by hand.\n"
        "# Database ids are not secrets (auth is injected by the OneCLI proxy).\n"
    )
    config_path.write_text(header + yaml.safe_dump(config, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the Notion databases under the parent page."
    )
    parser.add_argument(
        "--parent-page-id",
        default=DEFAULT_PARENT_PAGE_ID,
        help="Notion page the paper databases live under (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="config.yaml to write database ids into (default: %(default)s)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            f"also ensure the {LIVE_PAGE_TITLE!r} sibling page and its "
            "Trade Ledger (Live) / Account Snapshots (Live) databases"
        ),
    )
    opts = parser.parse_args(argv)

    client = NotionClient()
    outcome = bootstrap(client, opts.parent_page_id)
    live_page_id: str | None = None
    if opts.live:
        try:
            live_page, live_outcome = bootstrap_live(client, opts.parent_page_id)
        except LiveBootstrapError as exc:
            print(f"live bootstrap failed: {exc}", file=sys.stderr)
            return 1
        live_page_id = live_page["id"]
        print(f"{live_page['action']:>7}  {LIVE_PAGE_TITLE}: {live_page_id}")
        outcome.update(live_outcome)
    write_config(opts.config, opts.parent_page_id, outcome, live_parent_page_id=live_page_id)

    for title, info in outcome.items():
        print(f"{info['action']:>7}  {title}: {info['id']}")
    print(f"ids written to {opts.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

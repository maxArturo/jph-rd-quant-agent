"""Read-only live-account status from the Live Notion databases (US-012).

The read-path decision for the live channel's check tools (PRD US-053,
recorded in docs/decisions.md) is option (b): the orchestrator answers from
the Trade Ledger (Live) / Account Snapshots (Live) Notion databases — whose
sole writer is the live rebalancer — plus orchestrator state. The
orchestrator identity gains NO live broker access; nothing in orchestrator/
may construct an AlpacaClient on the live host (a source-grep test
enforces it).

Database ids come from ``notion.databases.{account_snapshots_live,
trade_ledger_live}`` in orchestrator/config.yaml (written by
``bootstrap_notion --live``). Ids that are still unset — or databases with
no rows yet — mean "no live data yet": the reader returns None/empty and
the tools answer gracefully instead of erroring, because before the first
live rebalance there is genuinely nothing to report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from orchestrator.notion_client import NotionClient

logger = logging.getLogger(__name__)


def _plain_text(prop: dict[str, Any] | None) -> str:
    """Concatenated text of a Notion title/rich_text property value."""
    if not isinstance(prop, dict):
        return ""
    items = prop.get("title") or prop.get("rich_text") or []
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("plain_text")
        if not isinstance(text, str):
            text = (item.get("text") or {}).get("content", "")
        if text:
            parts.append(text)
    return "".join(parts)


def _number(props: dict[str, Any], name: str) -> float | None:
    value = (props.get(name) or {}).get("number")
    return float(value) if isinstance(value, (int, float)) else None


def _select(props: dict[str, Any], name: str) -> str | None:
    select = (props.get(name) or {}).get("select")
    name_value = select.get("name") if isinstance(select, dict) else None
    return name_value if isinstance(name_value, str) else None


def _date(props: dict[str, Any], name: str) -> str | None:
    date = (props.get(name) or {}).get("date")
    start = date.get("start") if isinstance(date, dict) else None
    return start if isinstance(start, str) else None


def _text(props: dict[str, Any], name: str) -> str:
    return _plain_text(props.get(name))


@dataclass(frozen=True)
class LiveSnapshot:
    """One Account Snapshots (Live) row, as written by the live rebalancer."""

    date: str | None
    equity: float | None
    cash: float | None
    positions: float | None
    orders_placed: float | None
    orders_filled: float | None
    outcome: str | None
    day_pl: float | None
    day_pl_pct: float | None  # fraction, like the Notion percent property
    pl_day: str | None
    breaker: str
    notes: str


@dataclass(frozen=True)
class LiveOrder:
    """One Trade Ledger (Live) row, as written by the live rebalancer."""

    title: str
    symbol: str
    side: str | None
    status: str | None
    qty: float | None
    filled_qty: float | None
    limit_price: float | None
    filled_avg_price: float | None
    submitted_at: str | None
    notes: str


def snapshot_from_page(page: dict[str, Any]) -> LiveSnapshot:
    props = page.get("properties") or {}
    return LiveSnapshot(
        date=_date(props, "Date"),
        equity=_number(props, "Equity"),
        cash=_number(props, "Cash"),
        positions=_number(props, "Positions"),
        orders_placed=_number(props, "Orders Placed"),
        orders_filled=_number(props, "Orders Filled"),
        outcome=_select(props, "Outcome"),
        day_pl=_number(props, "Day P/L"),
        day_pl_pct=_number(props, "Day P/L %"),
        pl_day=_date(props, "P/L Day"),
        breaker=_text(props, "Breaker"),
        notes=_text(props, "Notes"),
    )


def order_from_page(page: dict[str, Any]) -> LiveOrder:
    props = page.get("properties") or {}
    return LiveOrder(
        title=_text(props, "Order"),
        symbol=_text(props, "Symbol"),
        side=_select(props, "Side"),
        status=_select(props, "Status"),
        qty=_number(props, "Qty"),
        filled_qty=_number(props, "Filled Qty"),
        limit_price=_number(props, "Limit Price"),
        filled_avg_price=_number(props, "Filled Avg Price"),
        submitted_at=_date(props, "Submitted At"),
        notes=_text(props, "Notes"),
    )


class LiveStatusReader:
    """Reads the live rebalancer's Notion records for the check_live_* tools.

    Either database id may be None (``bootstrap_notion --live`` not run yet):
    the matching reads return nothing instead of raising, which the tools
    report as "no live data yet". Notion outages DO raise — the tool loop
    surfaces them as error tool results, exactly like paper broker errors.

    The row schemas are shared with the paper databases, so the reader is
    schema-generic: the live rebalancer reuses it pointed at the PAPER
    Account Snapshots database for its same-day paper-vs-live summary line
    (US-014, read-only — the one-writer-per-DB rule is about writes).
    """

    def __init__(
        self,
        notion: NotionClient,
        snapshots_db: str | None = None,
        ledger_db: str | None = None,
    ) -> None:
        self._notion = notion
        self._snapshots_db = snapshots_db
        self._ledger_db = ledger_db

    def latest_snapshot(self) -> LiveSnapshot | None:
        """The most recent Account Snapshots (Live) row, or None."""
        rows = self.snapshot_history(limit=1)
        return rows[0] if rows else None

    def snapshot_history(self, limit: int = 30) -> list[LiveSnapshot]:
        """Up to ``limit`` snapshot rows, newest first (by the Date property)."""
        if self._snapshots_db is None:
            return []
        pages = self._notion.query_db(
            self._snapshots_db,
            sorts=[{"property": "Date", "direction": "descending"}],
            page_size=limit,
            max_results=limit,
        )
        return [snapshot_from_page(page) for page in pages]

    def recent_orders(self, limit: int = 10) -> list[LiveOrder]:
        """Up to ``limit`` ledger rows, newest first (by row creation time).

        created_time (never absent) rather than Submitted At: a row created
        by ``record_final`` after a submit-time outage can lack the stamp.
        """
        if self._ledger_db is None:
            return []
        pages = self._notion.query_db(
            self._ledger_db,
            sorts=[{"timestamp": "created_time", "direction": "descending"}],
            page_size=limit,
            max_results=limit,
        )
        return [order_from_page(page) for page in pages]

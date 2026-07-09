"""Notion Trade Ledger writer (US-035).

One row per order the rebalancer submitted, updated with its terminal fill
or rejection (docs/reference/notion-schema.md). This module is the Trade
Ledger's SOLE writer (one-writer-per-DB convention) — the orchestrator's
NotionRecorder never touches it, and reconciliation (US-037) can therefore
treat every row as rebalancer output.

Writes are best-effort BY DESIGN, like NotionRecorder: by the time a ledger
row is due the order is already live at Alpaca, so a Notion outage must not
turn a completed trade into a failed rebalance. Unlike NotionRecorder,
failures are not just logged — they accumulate in ``TradeLedger.failures``
so the daily Slack summary can surface them (an invisible audit gap would
defeat reconciliation).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from execution.alpaca_client import Order
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import (
    date_property,
    number_property,
    rich_text_property,
    select_property,
    title_property,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Alpaca order status -> Trade Ledger Status select vocabulary
# (notion-schema.md: submitted / filled / partially_filled / rejected /
# cancelled / expired).
_STATUS_MAP = {
    "filled": "filled",
    "partially_filled": "partially_filled",
    "canceled": "cancelled",
    "expired": "expired",
    "rejected": "rejected",
}


def ledger_status(status: str, filled_qty: float) -> str:
    """Map an Alpaca order status onto the ledger's Status vocabulary.

    Statuses outside the map (accepted, new, done_for_day, ...) are orders
    that may still fill: partially_filled when shares already crossed,
    otherwise still just submitted.
    """
    mapped = _STATUS_MAP.get(status)
    if mapped is not None:
        return mapped
    return "partially_filled" if filled_qty > 0 else "submitted"


class TradeLedger:
    """Best-effort writer for the Trade Ledger Notion database.

    One instance per rebalance run: ``record_submitted`` right after each
    order goes in, ``record_final`` with the post-poll snapshot. Page ids
    are kept in-memory keyed by Alpaca order id; if the creation write
    failed (or never happened), ``record_final`` creates the full row
    instead of updating, so a transient outage at submit time still leaves
    one complete row per order.
    """

    def __init__(self, notion: NotionClient, database_id: str) -> None:
        self._notion = notion
        self._database_id = database_id
        self._pages: dict[str, str] = {}
        self.failures: list[str] = []

    def record_submitted(self, order: Order, as_of: dt.date, note: str = "") -> str | None:
        """Create the order's row (Status 'submitted'); returns the page id."""
        return self._guarded(
            f"record_submitted {order.side} {order.symbol} ({order.id})",
            lambda: self._create_row(order, as_of, "submitted", note),
        )

    def record_final(self, order: Order, as_of: dt.date, note: str = "") -> None:
        """Update the order's row with its final status, fill qty, and price."""
        self._guarded(
            f"record_final {order.status} {order.symbol} ({order.id})",
            lambda: self._record_final(order, as_of, note),
        )

    def _create_row(self, order: Order, as_of: dt.date, status: str, note: str) -> str:
        properties = self._order_properties(order, as_of, status, note)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._database_id}, properties
        )
        self._pages[order.id] = page["id"]
        return page["id"]

    def _record_final(self, order: Order, as_of: dt.date, note: str) -> None:
        status = ledger_status(order.status, order.filled_qty)
        page_id = self._pages.get(order.id)
        if page_id is None:
            self._create_row(order, as_of, status, note)
            return
        properties: dict[str, Any] = {
            "Status": select_property(status),
            "Filled Qty": number_property(order.filled_qty),
        }
        if order.filled_avg_price is not None:
            properties["Filled Avg Price"] = number_property(order.filled_avg_price)
        if note:
            properties["Notes"] = rich_text_property(note)
        self._notion.update_page(page_id, properties=properties)

    def _order_properties(
        self, order: Order, as_of: dt.date, status: str, note: str
    ) -> dict[str, Any]:
        qty = order.qty if order.qty is not None else order.filled_qty
        properties: dict[str, Any] = {
            "Order": title_property(
                f"{as_of.isoformat()} {order.side.upper()} {qty:g} {order.symbol}"
            ),
            "Order ID": rich_text_property(order.id),
            "Symbol": rich_text_property(order.symbol),
            "Side": select_property(order.side),
            "Status": select_property(status),
            "Filled Qty": number_property(order.filled_qty),
        }
        if order.qty is not None:
            properties["Qty"] = number_property(order.qty)
        if order.limit_price is not None:
            properties["Limit Price"] = number_property(order.limit_price)
        if order.filled_avg_price is not None:
            properties["Filled Avg Price"] = number_property(order.filled_avg_price)
        if order.submitted_at:
            properties["Submitted At"] = date_property(order.submitted_at)
        if note:
            properties["Notes"] = rich_text_property(note)
        return properties

    def _guarded(self, action: str, fn: Callable[[], _T]) -> _T | None:
        """Run one write; log-and-collect so trading flow never breaks."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the trade already happened; never re-raise
            logger.exception("Trade Ledger write failed: %s", action)
            self.failures.append(f"{action}: {exc}")
            return None

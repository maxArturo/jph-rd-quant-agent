"""Notion Account Snapshots writer (US-047).

One row per rebalance day with the paper account's status: equity, cash,
exposure, position count, the previous completed trading day's P/L (from
Alpaca portfolio history — at pre-open run time equity-vs-last_equity is
always ~0, so the last COMPLETED day is the honest daily number), order
counts, the day's outcome, and breaker state. This module is the Account
Snapshots database's SOLE writer (one-writer-per-DB convention).

Best-effort like the Trade Ledger: a Notion outage must never turn a
completed rebalance into a failure, so every write logs-and-collects into
``AccountSnapshotLog.failures`` for the daily Slack summary to surface.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

from execution.alpaca_client import Account, PortfolioEntry, PortfolioHistory, Position
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import (
    date_property,
    number_property,
    rich_text_property,
    select_property,
    title_property,
)

logger = logging.getLogger(__name__)

# Outcome select vocabulary (notion-schema.md): every day the pipeline got a
# broker snapshot lands in exactly one of these.
OUTCOMES = frozenset({"traded", "no_trade", "gate_rejected", "breaker_tripped", "halted"})


def previous_day_pnl(
    history: PortfolioHistory | None, as_of: dt.date
) -> PortfolioEntry | None:
    """The latest completed-day P/L point strictly before ``as_of``.

    Pre-open snapshots must not report the (empty) current day's point as
    "daily P/L" — the previous trading day is the last completed one.
    """
    if history is None:
        return None
    candidates = [
        entry
        for entry in history.entries
        if entry.date < as_of and entry.profit_loss is not None
    ]
    return candidates[-1] if candidates else None


class AccountSnapshotLog:
    """Best-effort writer for the Account Snapshots Notion database."""

    def __init__(self, notion: NotionClient, database_id: str) -> None:
        self._notion = notion
        self._database_id = database_id
        self.failures: list[str] = []

    def record_daily(
        self,
        as_of: dt.date,
        account: Account,
        positions: Sequence[Position],
        outcome: str,
        orders_placed: int = 0,
        orders_filled: int = 0,
        breaker_state: str = "",
        history: PortfolioHistory | None = None,
        note: str = "",
    ) -> str | None:
        """Create the day's snapshot row; returns the page id (None on failure)."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}, got {outcome!r}")
        try:
            return self._create_row(
                as_of,
                account,
                positions,
                outcome,
                orders_placed,
                orders_filled,
                breaker_state,
                history,
                note,
            )
        except Exception as exc:  # noqa: BLE001 - the rebalance already happened; never re-raise
            logger.exception("Account Snapshot write failed for %s", as_of)
            self.failures.append(f"record_daily {as_of}: {exc}")
            return None

    def _create_row(
        self,
        as_of: dt.date,
        account: Account,
        positions: Sequence[Position],
        outcome: str,
        orders_placed: int,
        orders_filled: int,
        breaker_state: str,
        history: PortfolioHistory | None,
        note: str,
    ) -> str:
        properties: dict[str, Any] = {
            "Snapshot": title_property(f"{as_of.isoformat()} — equity ${account.equity:,.2f}"),
            "Date": date_property(as_of.isoformat()),
            "Equity": number_property(account.equity),
            "Cash": number_property(account.cash),
            "Positions": number_property(len(positions)),
            "Orders Placed": number_property(orders_placed),
            "Orders Filled": number_property(orders_filled),
            "Outcome": select_property(outcome),
        }
        if account.long_market_value is not None:
            properties["Long Value"] = number_property(account.long_market_value)
        if account.short_market_value is not None:
            properties["Short Value"] = number_property(account.short_market_value)
        day = previous_day_pnl(history, as_of)
        if day is not None:
            assert day.profit_loss is not None  # previous_day_pnl filters on it
            properties["Day P/L"] = number_property(day.profit_loss)
            if day.profit_loss_pct is not None:
                # Property uses Notion's percent format: fractions render as %.
                properties["Day P/L %"] = number_property(day.profit_loss_pct)
            properties["P/L Day"] = date_property(day.date.isoformat())
        if breaker_state:
            properties["Breaker"] = rich_text_property(breaker_state)
        if note:
            properties["Notes"] = rich_text_property(note)
        page = self._notion.create_page(
            {"type": "database_id", "database_id": self._database_id}, properties
        )
        return page["id"]

"""Reconcile the Notion Trade Ledger against Alpaca paper order history (US-037).

For a date range (America/New_York trading dates, judged by each order's
submitted_at — the same "today" convention as execution/rebalance.py), this
script:

1. pulls every Alpaca order submitted in the range (GET /v2/orders, paged
   backwards via the ``until`` bound),
2. pulls every Trade Ledger row whose Submitted At falls in the range,
3. joins them on the Alpaca order id (Trade Ledger "Order ID" — the
   reconciliation key per docs/reference/notion-schema.md) and compares
   Symbol / Side / Qty / Limit Price / Status / Filled Qty / Filled Avg Price
   (Alpaca statuses mapped through execution.ledger.ledger_status, the same
   mapping the writer used).

Exit codes: 0 = every order matches its ledger row exactly; 1 = mismatches
(each printed with the order id and the differing fields); 2 = the
comparison itself could not run (config/auth/HTTP failure).

Because the Trade Ledger has exactly one writer (the rebalancer,
execution/ledger.py), every discrepancy is meaningful: a missing ledger row
means either a ledger write failed or something other than the rebalancer
traded the account; an orphan ledger row means the broker no longer reports
an order we recorded. Note a ledger row whose Status is still 'submitted'
against a now-filled Alpaca order is a real finding, not noise — it means
the run's fill poll timed out and record_final never saw the fill.

Run through the OneCLI proxy (Alpaca + Notion auth are both injected for
rdq-exec-paper; never in code):

    onecli run --agent rdq-exec-paper -- .venv/bin/python -m ops.reconcile \\
        --start 2026-07-01 --end 2026-07-09
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution.alpaca_client import AlpacaClient, AlpacaError, Order
from execution.ledger import ledger_status
from execution.rebalance import MARKET_TZ, submitted_market_date
from orchestrator.notion_client import NotionClient, NotionError
from orchestrator.notion_recorder import (
    DEFAULT_CONFIG_PATH,
    RecorderConfigError,
    load_notion_databases,
)

# Alpaca's GET /v2/orders page cap.
ORDERS_PAGE_LIMIT = 500


class ReconcileError(Exception):
    """The reconciliation could not be carried out (distinct from a mismatch)."""


@dataclass(frozen=True)
class LedgerRow:
    """One Trade Ledger page, parsed down to the reconcilable fields."""

    page_id: str
    order_id: str
    symbol: str | None
    side: str | None
    qty: float | None
    limit_price: float | None
    status: str | None
    filled_qty: float | None
    filled_avg_price: float | None
    submitted_at: str | None


@dataclass(frozen=True)
class Mismatch:
    order_id: str
    kind: str
    detail: str

    def describe(self) -> str:
        return f"{self.kind} [{self.order_id}]: {self.detail}"


# ---------------------------------------------------------------------------
# Notion page parsing
# ---------------------------------------------------------------------------


def _rich_text(prop: dict[str, Any] | None) -> str:
    if not prop:
        return ""
    parts = prop.get("rich_text") or []
    return "".join(
        str(part.get("plain_text") or part.get("text", {}).get("content", "")) for part in parts
    )


def _number(prop: dict[str, Any] | None) -> float | None:
    if not prop:
        return None
    value = prop.get("number")
    return None if value is None else float(value)


def _select(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None
    selected = prop.get("select")
    return None if selected is None else str(selected.get("name"))


def _date_start(prop: dict[str, Any] | None) -> str | None:
    if not prop:
        return None
    date = prop.get("date")
    return None if date is None else str(date.get("start"))


def parse_ledger_page(page: dict[str, Any]) -> LedgerRow:
    """Parse a Trade Ledger query result page; absent properties become None."""
    props = page.get("properties", {})
    return LedgerRow(
        page_id=str(page.get("id", "")),
        order_id=_rich_text(props.get("Order ID")),
        symbol=_rich_text(props.get("Symbol")) or None,
        side=_select(props.get("Side")),
        qty=_number(props.get("Qty")),
        limit_price=_number(props.get("Limit Price")),
        status=_select(props.get("Status")),
        filled_qty=_number(props.get("Filled Qty")),
        filled_avg_price=_number(props.get("Filled Avg Price")),
        submitted_at=_date_start(props.get("Submitted At")),
    )


def _market_date(stamp: str | None) -> dt.date | None:
    """The America/New_York date of an ISO timestamp (Z-suffix tolerated)."""
    if not stamp:
        return None
    try:
        parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(MARKET_TZ).date()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_ledger_rows(
    notion: NotionClient, database_id: str, start: dt.date, end: dt.date
) -> list[LedgerRow]:
    """Trade Ledger rows whose Submitted At falls in [start, end] Eastern.

    The Notion date filter is widened by a day on each side (Notion compares
    date-only bounds without our market-timezone convention); the precise
    range cut happens client-side on the parsed timestamp. Rows with no
    Submitted At cannot be placed in any range and are excluded — their
    broker order (if any) will then surface as a missing ledger row.
    """
    date_filter = {
        "and": [
            {
                "property": "Submitted At",
                "date": {"on_or_after": (start - dt.timedelta(days=1)).isoformat()},
            },
            {
                "property": "Submitted At",
                "date": {"on_or_before": (end + dt.timedelta(days=1)).isoformat()},
            },
        ]
    }
    pages = notion.query_db(database_id, filter=date_filter)
    rows = [parse_ledger_page(page) for page in pages]
    return [row for row in rows if _in_range(_market_date(row.submitted_at), start, end)]


def fetch_broker_orders(
    client: AlpacaClient, start: dt.date, end: dt.date, page_limit: int = ORDERS_PAGE_LIMIT
) -> list[Order]:
    """Every Alpaca order submitted in [start, end] Eastern.

    Pages backwards: Alpaca returns newest-first, so when a page comes back
    full the oldest submitted_at in it becomes the next ``until`` bound.
    Overlapping boundary rows are deduped by order id.
    """
    after = _utc_bound(start)
    until = _utc_bound(end + dt.timedelta(days=1))
    seen: dict[str, Order] = {}
    while True:
        batch = client.list_orders(status="all", limit=page_limit, after=after, until=until)
        for order in batch:
            seen[order.id] = order
        if len(batch) < page_limit:
            break
        stamps = sorted(order.submitted_at for order in batch if order.submitted_at)
        if not stamps or stamps[0] == until:
            raise ReconcileError(
                f"cannot page past {page_limit} orders sharing submitted_at {until!r} — "
                "narrow the date range"
            )
        until = stamps[0]
    return [
        order for order in seen.values() if _in_range(submitted_market_date(order), start, end)
    ]


def _utc_bound(day: dt.date) -> str:
    """Eastern midnight of ``day`` as an RFC3339 UTC timestamp."""
    eastern_midnight = dt.datetime.combine(day, dt.time.min, tzinfo=MARKET_TZ)
    return eastern_midnight.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _in_range(day: dt.date | None, start: dt.date, end: dt.date) -> bool:
    return day is not None and start <= day <= end


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def expected_ledger_fields(order: Order) -> dict[str, Any]:
    """What the order's Trade Ledger row must say, field by field."""
    return {
        "Symbol": order.symbol,
        "Side": order.side,
        "Qty": order.qty,
        "Limit Price": order.limit_price,
        "Status": ledger_status(order.status, order.filled_qty),
        "Filled Qty": order.filled_qty,
        "Filled Avg Price": order.filled_avg_price,
    }


def _row_fields(row: LedgerRow) -> dict[str, Any]:
    return {
        "Symbol": row.symbol,
        "Side": row.side,
        "Qty": row.qty,
        "Limit Price": row.limit_price,
        "Status": row.status,
        "Filled Qty": row.filled_qty,
        "Filled Avg Price": row.filled_avg_price,
    }


def _values_match(ledger_value: Any, broker_value: Any) -> bool:
    if ledger_value is None or broker_value is None:
        return ledger_value is None and broker_value is None
    if isinstance(ledger_value, int | float) and isinstance(broker_value, int | float):
        return math.isclose(ledger_value, broker_value, rel_tol=1e-9, abs_tol=1e-6)
    return ledger_value == broker_value


def _fmt(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def compare_fields(row: LedgerRow, order: Order) -> list[str]:
    """Human lines for every field where the ledger disagrees with Alpaca."""
    expected = expected_ledger_fields(order)
    actual = _row_fields(row)
    return [
        f"{field}: ledger={_fmt(actual[field])} alpaca={_fmt(expected[field])}"
        for field in expected
        if not _values_match(actual[field], expected[field])
    ]


def reconcile(ledger_rows: Iterable[LedgerRow], orders: Iterable[Order]) -> list[Mismatch]:
    """All discrepancies between the ledger rows and the broker orders."""
    mismatches: list[Mismatch] = []
    by_order_id: dict[str, list[LedgerRow]] = {}
    for row in ledger_rows:
        if not row.order_id:
            mismatches.append(
                Mismatch("?", "ledger_row_without_order_id", f"ledger page {row.page_id}")
            )
            continue
        by_order_id.setdefault(row.order_id, []).append(row)

    broker_by_id = {order.id: order for order in orders}

    for order_id in sorted(by_order_id):
        rows = by_order_id[order_id]
        if len(rows) > 1:
            pages = ", ".join(row.page_id for row in rows)
            mismatches.append(
                Mismatch(order_id, "duplicate_ledger_rows", f"{len(rows)} rows: pages {pages}")
            )
        if order_id not in broker_by_id:
            mismatches.append(
                Mismatch(
                    order_id,
                    "orphan_ledger_row",
                    f"ledger page {rows[0].page_id} matches no Alpaca order in the range",
                )
            )

    for order_id in sorted(broker_by_id):
        order = broker_by_id[order_id]
        rows = by_order_id.get(order_id)
        if not rows:
            qty = _fmt(order.qty if order.qty is not None else order.filled_qty)
            mismatches.append(
                Mismatch(
                    order_id,
                    "missing_ledger_row",
                    f"no Trade Ledger row for {order.side} {qty} {order.symbol} "
                    f"(submitted {order.submitted_at or 'unknown'})",
                )
            )
            continue
        diffs = compare_fields(rows[0], order)
        if diffs:
            mismatches.append(Mismatch(order_id, "field_mismatch", "; ".join(diffs)))

    return mismatches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_reconcile(
    notion: NotionClient,
    alpaca: AlpacaClient,
    trade_ledger_db_id: str,
    start: dt.date,
    end: dt.date,
    out: Callable[[str], None] = print,
) -> int:
    """Fetch both sides, compare, report. Returns the process exit code."""
    ledger_rows = fetch_ledger_rows(notion, trade_ledger_db_id, start, end)
    orders = fetch_broker_orders(alpaca, start, end)
    mismatches = reconcile(ledger_rows, orders)
    scope = f"{start.isoformat()}..{end.isoformat()}"
    if not mismatches:
        out(
            f"OK {scope}: {len(orders)} Alpaca order(s) match "
            f"{len(ledger_rows)} Trade Ledger row(s)"
        )
        return 0
    for mismatch in mismatches:
        out(f"MISMATCH {mismatch.describe()}")
    out(
        f"FAIL {scope}: {len(mismatches)} mismatch(es) across {len(orders)} Alpaca order(s) "
        f"and {len(ledger_rows)} Trade Ledger row(s)"
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile the Notion Trade Ledger against Alpaca paper order history."
    )
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=None,
        help="first America/New_York trading date, YYYY-MM-DD (default: --end)",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=None,
        help="last America/New_York trading date, YYYY-MM-DD (default: today Eastern)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="orchestrator/config.yaml holding the Trade Ledger database id",
    )
    args = parser.parse_args(argv)

    end = args.end if args.end is not None else dt.datetime.now(tz=MARKET_TZ).date()
    start = args.start if args.start is not None else end
    if start > end:
        parser.error(f"--start {start} is after --end {end}")

    try:
        databases = load_notion_databases(args.config_path)
        return run_reconcile(NotionClient(), AlpacaClient(), databases.trade_ledger, start, end)
    except (RecorderConfigError, NotionError, AlpacaError, ReconcileError) as exc:
        print(f"reconcile failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

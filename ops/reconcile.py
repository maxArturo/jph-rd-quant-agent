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

Exit codes: 0 = every order matches its ledger row exactly (or, with
--update, every mismatch was repaired); 1 = unresolved mismatches (each
printed with the order id and the differing fields); 2 = the comparison
itself could not run (config/auth/HTTP failure).

``--update`` (US-019) additionally repairs the fill-poll-timeout case: a
ledger row whose ONLY differences from the broker are the fill-state fields
(Status / Filled Qty / Filled Avg Price) is patched to the values the
rebalancer's record_final would have written (same ledger_status mapping and
property shapes). Rows that disagree on identity fields (Symbol / Side /
Qty / Limit Price), orphans, duplicates, and missing rows are never touched
— those mean corruption or an out-of-band trade, and stay unresolved for a
human. This is the one sanctioned writer besides execution/ledger.py, and it
only ever writes broker truth. ``--notify`` posts a one-line Slack summary
when (and only when) mismatches were found — the daily timer
(ops/rdq-reconcile.timer, weekday 16:15 America/New_York) runs
``--update --notify --lookback 4`` so quiet days stay silent.

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
    select_property,
)

# Alpaca's GET /v2/orders page cap.
ORDERS_PAGE_LIMIT = 500

# The fill-state fields --update may repair (broker truth only). Everything
# else on a ledger row is identity the rebalancer wrote at submit time — a
# disagreement there is corruption, not a late fill, and is never patched.
FILL_FIELDS = ("Status", "Filled Qty", "Filled Avg Price")


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


def diff_fields(row: LedgerRow, order: Order) -> dict[str, tuple[Any, Any]]:
    """Every disagreeing field, mapped to its (ledger value, broker value)."""
    expected = expected_ledger_fields(order)
    actual = _row_fields(row)
    return {
        field: (actual[field], expected[field])
        for field in expected
        if not _values_match(actual[field], expected[field])
    }


def compare_fields(row: LedgerRow, order: Order) -> list[str]:
    """Human lines for every field where the ledger disagrees with Alpaca."""
    return [
        f"{field}: ledger={_fmt(ledger_value)} alpaca={_fmt(broker_value)}"
        for field, (ledger_value, broker_value) in diff_fields(row, order).items()
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
# Update mode (US-019)
# ---------------------------------------------------------------------------


def _fill_update_properties(order: Order, fields: Iterable[str]) -> dict[str, Any]:
    """Notion property payload setting ``fields`` (all in FILL_FIELDS) to
    broker truth — the same values record_final would have written."""
    expected = expected_ledger_fields(order)
    properties: dict[str, Any] = {}
    for field in fields:
        if field == "Status":
            properties[field] = select_property(str(expected[field]))
        else:
            # None clears a stale number (Notion treats null as "unset").
            properties[field] = {"number": expected[field]}
    return properties


def apply_fill_updates(
    notion: NotionClient,
    ledger_rows: Iterable[LedgerRow],
    orders: Iterable[Order],
    mismatches: Sequence[Mismatch],
    out: Callable[[str], None] = print,
) -> tuple[list[str], list[Mismatch]]:
    """Repair pure fill-state mismatches; return (updated order ids, unresolved).

    Only a ``field_mismatch`` whose every differing field is in FILL_FIELDS is
    patched (the fill-poll-timeout case: submitted-then-filled). Identity
    disagreements, orphans, duplicates, missing rows, and failed patches stay
    in the unresolved list.
    """
    rows_by_id: dict[str, list[LedgerRow]] = {}
    for row in ledger_rows:
        if row.order_id:
            rows_by_id.setdefault(row.order_id, []).append(row)
    orders_by_id = {order.id: order for order in orders}

    updated: list[str] = []
    unresolved: list[Mismatch] = []
    for mismatch in mismatches:
        rows = rows_by_id.get(mismatch.order_id)
        order = orders_by_id.get(mismatch.order_id)
        if mismatch.kind != "field_mismatch" or not rows or order is None:
            unresolved.append(mismatch)
            continue
        diffs = diff_fields(rows[0], order)
        if any(field not in FILL_FIELDS for field in diffs):
            unresolved.append(mismatch)
            continue
        try:
            notion.update_page(
                rows[0].page_id, properties=_fill_update_properties(order, diffs)
            )
        except NotionError as exc:
            out(f"UPDATE FAILED [{mismatch.order_id}]: {exc}")
            unresolved.append(mismatch)
            continue
        out(f"UPDATED [{mismatch.order_id}]: {', '.join(diffs)} set to final fill state")
        updated.append(mismatch.order_id)
    return updated, unresolved


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
    update: bool = False,
    notify: Callable[[str], None] | None = None,
) -> int:
    """Fetch both sides, compare, report (and optionally repair).

    Returns the process exit code: 0 when the ledger matches (or every
    mismatch was repaired by ``update``), 1 when unresolved mismatches
    remain. ``notify`` is called with a one-line summary only when
    mismatches were found.
    """
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
    unresolved = list(mismatches)
    summary = (
        f"{scope}: {len(mismatches)} mismatch(es) across {len(orders)} Alpaca order(s) "
        f"and {len(ledger_rows)} Trade Ledger row(s)"
    )
    if update:
        updated, unresolved = apply_fill_updates(notion, ledger_rows, orders, mismatches, out)
        summary += (
            f"; {len(updated)} ledger row(s) updated to final fill state, "
            f"{len(unresolved)} unresolved"
        )
    out(("FAIL " if unresolved else "FIXED ") + summary)
    if notify is not None:
        notify(f":mag: Trade Ledger reconcile {summary}")
    return 1 if unresolved else 0


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
    parser.add_argument(
        "--lookback",
        type=int,
        default=0,
        help="reconcile [end - N calendar days, end] instead of just end "
        "(timer mode; conflicts with --start)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="repair pure fill-state mismatches (Status / Filled Qty / Filled Avg "
        "Price) to broker truth; identity mismatches are only reported",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="post a one-line Slack summary when mismatches were found "
        "(quiet days post nothing)",
    )
    args = parser.parse_args(argv)

    if args.lookback < 0:
        parser.error(f"--lookback must be >= 0, got {args.lookback}")
    if args.lookback and args.start is not None:
        parser.error("--lookback conflicts with --start (pick one way to set the range)")
    end = args.end if args.end is not None else dt.datetime.now(tz=MARKET_TZ).date()
    start = args.start if args.start is not None else end - dt.timedelta(days=args.lookback)
    if start > end:
        parser.error(f"--start {start} is after --end {end}")

    notify: Callable[[str], None] | None = None
    if args.notify:
        from execution.rebalance import slack_notifier
        from orchestrator.config import ConfigError

        try:
            notify = slack_notifier()
        except ConfigError as exc:
            print(f"reconcile failed: --notify needs a Slack config: {exc}", file=sys.stderr)
            return 2

    try:
        databases = load_notion_databases(args.config_path)
        return run_reconcile(
            NotionClient(),
            AlpacaClient(),
            databases.trade_ledger,
            start,
            end,
            update=args.update,
            notify=notify,
        )
    except (RecorderConfigError, NotionError, AlpacaError, ReconcileError) as exc:
        print(f"reconcile failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

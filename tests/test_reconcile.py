"""US-037: ledger reconciliation — fixture ledger/order data through the real clients."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from execution.alpaca_client import AlpacaClient
from ops.reconcile import (
    LedgerRow,
    Mismatch,
    ReconcileError,
    compare_fields,
    fetch_broker_orders,
    fetch_ledger_rows,
    main,
    parse_ledger_page,
    reconcile,
    run_reconcile,
)
from orchestrator.notion_client import NotionClient
from tests.test_alpaca_client import FakeResponse as BrokerResponse
from tests.test_alpaca_client import FakeSession as BrokerSession
from tests.test_notion_client import FakeResponse as NotionResponse
from tests.test_notion_client import FakeSession as NotionSession
from tests.test_rebalance import make_order

START = dt.date(2026, 7, 9)
END = dt.date(2026, 7, 9)
DB_ID = "db-trade-ledger"


# ---------------------------------------------------------------- fixtures


def ledger_page(
    page_id: str = "page-1",
    order_id: str = "ord-1",
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float | None = 10.0,
    limit_price: float | None = 100.0,
    status: str = "filled",
    filled_qty: float | None = 10.0,
    filled_avg_price: float | None = 100.0,
    submitted_at: str | None = "2026-07-09T13:30:00Z",
) -> dict[str, Any]:
    """A Notion query-result page shaped like a Trade Ledger row."""

    def text(content: str) -> dict[str, Any]:
        return {"rich_text": [{"plain_text": content}]}

    def number(value: float | None) -> dict[str, Any]:
        return {"number": value}

    def select(name: str | None) -> dict[str, Any]:
        return {"select": None if name is None else {"name": name}}

    properties: dict[str, Any] = {
        "Order ID": text(order_id),
        "Symbol": text(symbol),
        "Side": select(side),
        "Qty": number(qty),
        "Limit Price": number(limit_price),
        "Status": select(status),
        "Filled Qty": number(filled_qty),
        "Filled Avg Price": number(filled_avg_price),
    }
    if submitted_at is not None:
        properties["Submitted At"] = {"date": {"start": submitted_at}}
    return {"object": "page", "id": page_id, "properties": properties}


def ledger_row(**overrides: Any) -> LedgerRow:
    return parse_ledger_page(ledger_page(**overrides))


def query_response(pages: list[dict[str, Any]]) -> NotionResponse:
    return NotionResponse(200, {"object": "list", "results": pages, "has_more": False})


def order_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ord-1",
        "client_order_id": "rdq-2026-07-09-buy-AAPL",
        "symbol": "AAPL",
        "qty": "10",
        "notional": None,
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "100",
        "status": "filled",
        "filled_qty": "10",
        "filled_avg_price": "100",
        "submitted_at": "2026-07-09T13:30:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- parsing


def test_parse_ledger_page_round_trip() -> None:
    row = parse_ledger_page(ledger_page(page_id="p-9", order_id="ord-9", status="cancelled"))
    assert row == LedgerRow(
        page_id="p-9",
        order_id="ord-9",
        symbol="AAPL",
        side="buy",
        qty=10.0,
        limit_price=100.0,
        status="cancelled",
        filled_qty=10.0,
        filled_avg_price=100.0,
        submitted_at="2026-07-09T13:30:00Z",
    )


def test_parse_ledger_page_tolerates_missing_properties() -> None:
    row = parse_ledger_page({"object": "page", "id": "p-1", "properties": {}})
    assert row.order_id == ""
    assert row.qty is None
    assert row.status is None
    assert row.submitted_at is None


def test_parse_ledger_page_reads_text_content_fallback() -> None:
    page = ledger_page()
    page["properties"]["Order ID"] = {"rich_text": [{"text": {"content": "ord-77"}}]}
    assert parse_ledger_page(page).order_id == "ord-77"


# ---------------------------------------------------------------- reconcile: AC cases


def test_clean_match_reports_nothing() -> None:
    orders = [
        make_order(),
        make_order(
            id="ord-2",
            symbol="MSFT",
            side="sell",
            qty=5.0,
            limit_price=400.0,
            status="canceled",
            filled_qty=0.0,
            filled_avg_price=None,
        ),
    ]
    rows = [
        ledger_row(),
        ledger_row(
            page_id="page-2",
            order_id="ord-2",
            symbol="MSFT",
            side="sell",
            qty=5.0,
            limit_price=400.0,
            status="cancelled",  # ledger vocabulary for Alpaca "canceled"
            filled_qty=0.0,
            filled_avg_price=None,
        ),
    ]
    assert reconcile(rows, orders) == []


def test_missing_ledger_row_is_a_mismatch() -> None:
    (mismatch,) = reconcile([], [make_order()])
    assert mismatch.kind == "missing_ledger_row"
    assert mismatch.order_id == "ord-1"
    assert "buy 10 AAPL" in mismatch.detail


def test_quantity_mismatch_names_the_field_and_both_values() -> None:
    (mismatch,) = reconcile([ledger_row(qty=12.0)], [make_order()])
    assert mismatch.kind == "field_mismatch"
    assert mismatch.order_id == "ord-1"
    assert "Qty: ledger=12 alpaca=10" in mismatch.detail


# ---------------------------------------------------------------- reconcile: edges


def test_orphan_ledger_row_is_a_mismatch() -> None:
    (mismatch,) = reconcile([ledger_row(order_id="ord-gone")], [])
    assert mismatch.kind == "orphan_ledger_row"
    assert mismatch.order_id == "ord-gone"
    assert "page-1" in mismatch.detail


def test_duplicate_ledger_rows_are_a_mismatch() -> None:
    rows = [ledger_row(), ledger_row(page_id="page-2")]
    mismatches = reconcile(rows, [make_order()])
    assert [m.kind for m in mismatches] == ["duplicate_ledger_rows"]
    assert "page-1" in mismatches[0].detail and "page-2" in mismatches[0].detail


def test_ledger_row_without_order_id_is_a_mismatch() -> None:
    (mismatch,) = reconcile([ledger_row(order_id="")], [])
    assert mismatch.kind == "ledger_row_without_order_id"
    assert "page-1" in mismatch.detail


def test_stale_submitted_status_against_filled_order_mismatches() -> None:
    # Fill poll timed out at run time: ledger still says submitted/0 shares.
    row = ledger_row(status="submitted", filled_qty=0.0, filled_avg_price=None)
    (mismatch,) = reconcile([row], [make_order()])
    diffs = mismatch.detail
    assert "Status: ledger=submitted alpaca=filled" in diffs
    assert "Filled Qty: ledger=0 alpaca=10" in diffs
    assert "Filled Avg Price: ledger=(none) alpaca=100" in diffs


def test_none_vs_value_counts_as_a_difference() -> None:
    order = make_order(status="accepted", filled_qty=0.0, filled_avg_price=None)
    row = ledger_row(status="submitted", filled_qty=0.0, filled_avg_price=None)
    assert compare_fields(row, order) == []
    assert compare_fields(ledger_row(filled_avg_price=None), make_order()) == [
        "Filled Avg Price: ledger=(none) alpaca=100"
    ]


def test_float_noise_within_tolerance_matches() -> None:
    assert compare_fields(ledger_row(limit_price=100.0000000001), make_order()) == []


# ---------------------------------------------------------------- fetching


def test_fetch_ledger_rows_filters_by_market_date() -> None:
    in_range = ledger_page(page_id="p-in", order_id="ord-in")
    # 2026-07-09 01:00 UTC is 2026-07-08 21:00 Eastern — the widened Notion
    # filter returns it, the client-side range cut must drop it.
    prior_day = ledger_page(
        page_id="p-out", order_id="ord-out", submitted_at="2026-07-09T01:00:00Z"
    )
    undated = ledger_page(page_id="p-undated", order_id="ord-undated", submitted_at=None)
    session = NotionSession([query_response([in_range, prior_day, undated])])
    rows = fetch_ledger_rows(NotionClient(session=session), DB_ID, START, END)

    assert [row.page_id for row in rows] == ["p-in"]
    (call,) = session.calls
    assert call["url"].endswith(f"/v1/databases/{DB_ID}/query")
    assert call["json"]["filter"] == {
        "and": [
            {"property": "Submitted At", "date": {"on_or_after": "2026-07-08"}},
            {"property": "Submitted At", "date": {"on_or_before": "2026-07-10"}},
        ]
    }


def test_fetch_broker_orders_pages_backwards_and_dedupes() -> None:
    newest = order_row(id="ord-3", submitted_at="2026-07-09T15:00:00Z")
    middle = order_row(id="ord-2", submitted_at="2026-07-09T14:00:00Z")
    oldest = order_row(id="ord-1", submitted_at="2026-07-09T13:30:00Z")
    session = BrokerSession(
        [
            BrokerResponse(200, [newest, middle]),  # full page of 2 -> keep paging
            BrokerResponse(200, [middle, oldest]),  # boundary row repeats -> dedupe
            BrokerResponse(200, [oldest]),  # short page -> stop
        ]
    )
    orders = fetch_broker_orders(AlpacaClient(session=session), START, END, page_limit=2)

    assert sorted(order.id for order in orders) == ["ord-1", "ord-2", "ord-3"]
    first, second, third = session.calls
    assert first["params"]["status"] == "all"
    assert first["params"]["limit"] == "2"
    assert first["params"]["after"] == "2026-07-09T04:00:00Z"  # Eastern midnight, EDT
    assert first["params"]["until"] == "2026-07-10T04:00:00Z"
    assert second["params"]["until"] == "2026-07-09T14:00:00Z"  # oldest stamp of page 1
    assert third["params"]["until"] == "2026-07-09T13:30:00Z"  # oldest stamp of page 2


def test_fetch_broker_orders_drops_out_of_range_orders() -> None:
    session = BrokerSession(
        [BrokerResponse(200, [order_row(), order_row(id="ord-8", submitted_at=None)])]
    )
    orders = fetch_broker_orders(AlpacaClient(session=session), START, END)
    assert [order.id for order in orders] == ["ord-1"]


def test_fetch_broker_orders_refuses_a_stuck_page() -> None:
    same_stamp = "2026-07-09T13:30:00Z"
    page = [order_row(id=f"ord-{i}", submitted_at=same_stamp) for i in range(2)]
    session = BrokerSession([BrokerResponse(200, page), BrokerResponse(200, page)])
    with pytest.raises(ReconcileError, match="narrow the date range"):
        fetch_broker_orders(AlpacaClient(session=session), START, END, page_limit=2)


# ---------------------------------------------------------------- run_reconcile / CLI


def make_clients(
    pages: list[dict[str, Any]], order_rows: list[dict[str, Any]]
) -> tuple[NotionClient, AlpacaClient]:
    notion = NotionClient(session=NotionSession([query_response(pages)]))
    alpaca = AlpacaClient(session=BrokerSession([BrokerResponse(200, order_rows)]))
    return notion, alpaca


def test_run_reconcile_clean_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    notion, alpaca = make_clients([ledger_page()], [order_row()])
    assert run_reconcile(notion, alpaca, DB_ID, START, END) == 0
    out = capsys.readouterr().out
    assert "OK 2026-07-09..2026-07-09: 1 Alpaca order(s) match 1 Trade Ledger row(s)" in out


def test_run_reconcile_mismatch_exits_one_and_prints_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    notion, alpaca = make_clients(
        [ledger_page(qty=12.0)],
        [order_row(), order_row(id="ord-2", symbol="MSFT", submitted_at="2026-07-09T14:00:00Z")],
    )
    assert run_reconcile(notion, alpaca, DB_ID, START, END) == 1
    out = capsys.readouterr().out
    assert "MISMATCH field_mismatch [ord-1]: Qty: ledger=12 alpaca=10" in out
    assert "MISMATCH missing_ledger_row [ord-2]" in out
    assert "FAIL 2026-07-09..2026-07-09: 2 mismatch(es)" in out


def test_main_exits_two_on_missing_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["--start", "2026-07-09", "--end", "2026-07-09", "--config-path", str(tmp_path / "no.yaml")]
    )
    assert exit_code == 2
    assert "reconcile failed:" in capsys.readouterr().err


def test_main_rejects_inverted_range() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--start", "2026-07-10", "--end", "2026-07-09"])
    assert excinfo.value.code == 2


def test_mismatch_describe_format() -> None:
    mismatch = Mismatch("ord-1", "field_mismatch", "Qty: ledger=12 alpaca=10")
    assert mismatch.describe() == "field_mismatch [ord-1]: Qty: ledger=12 alpaca=10"

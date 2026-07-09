"""US-035: Trade Ledger writes — payload asserts against a mocked Notion session."""

from __future__ import annotations

import datetime as dt
from typing import Any

from execution.ledger import TradeLedger, ledger_status
from orchestrator.notion_client import NotionClient
from tests.test_notion_client import FakeResponse, FakeSession
from tests.test_rebalance import make_order

AS_OF = dt.date(2026, 7, 9)
DB_ID = "db-trade-ledger"


def page_response(page_id: str = "page-1") -> FakeResponse:
    return FakeResponse(200, {"object": "page", "id": page_id})


def make_ledger(responses: list[FakeResponse]) -> tuple[TradeLedger, FakeSession]:
    session = FakeSession(responses)
    return TradeLedger(NotionClient(session=session), DB_ID), session


def prop(call: dict[str, Any], name: str) -> Any:
    return call["json"]["properties"][name]


# ---------------------------------------------------------------- submitted rows


def test_record_submitted_creates_full_row() -> None:
    ledger, session = make_ledger([page_response()])
    order = make_order(status="accepted", filled_qty=0.0, filled_avg_price=None)

    page_id = ledger.record_submitted(order, AS_OF)

    assert page_id == "page-1"
    assert ledger.failures == []
    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/pages")
    assert call["json"]["parent"] == {"type": "database_id", "database_id": DB_ID}
    assert prop(call, "Order")["title"][0]["text"]["content"] == "2026-07-09 BUY 10 AAPL"
    assert prop(call, "Order ID")["rich_text"][0]["text"]["content"] == "ord-1"
    assert prop(call, "Symbol")["rich_text"][0]["text"]["content"] == "AAPL"
    assert prop(call, "Side") == {"select": {"name": "buy"}}
    assert prop(call, "Qty") == {"number": 10.0}
    assert prop(call, "Limit Price") == {"number": 100.0}
    assert prop(call, "Status") == {"select": {"name": "submitted"}}
    assert prop(call, "Filled Qty") == {"number": 0.0}
    assert prop(call, "Submitted At") == {"date": {"start": "2026-07-09T13:30:00Z"}}
    assert "Filled Avg Price" not in call["json"]["properties"]


def test_record_submitted_note_lands_in_notes() -> None:
    ledger, session = make_ledger([page_response()])
    ledger.record_submitted(
        make_order(status="accepted", filled_qty=0.0, filled_avg_price=None),
        AS_OF,
        note="pre-open batch",
    )
    (call,) = session.calls
    assert prop(call, "Notes")["rich_text"][0]["text"]["content"] == "pre-open batch"


# ---------------------------------------------------------------- final rows


def test_record_final_updates_the_submitted_page() -> None:
    ledger, session = make_ledger([page_response("page-7"), page_response("page-7")])
    submitted = make_order(status="accepted", filled_qty=0.0, filled_avg_price=None)
    ledger.record_submitted(submitted, AS_OF)

    ledger.record_final(make_order(status="filled", filled_qty=10.0), AS_OF)

    assert ledger.failures == []
    update = session.calls[1]
    assert update["method"] == "PATCH"
    assert update["url"].endswith("/v1/pages/page-7")
    assert prop(update, "Status") == {"select": {"name": "filled"}}
    assert prop(update, "Filled Qty") == {"number": 10.0}
    assert prop(update, "Filled Avg Price") == {"number": 100.0}


def test_record_final_without_prior_row_creates_one() -> None:
    # If the submit-time write failed (or never ran), the final record must
    # still leave one complete row for reconciliation.
    ledger, session = make_ledger([page_response()])
    ledger.record_final(make_order(status="rejected", filled_qty=0.0), AS_OF)

    (call,) = session.calls
    assert call["method"] == "POST"
    assert prop(call, "Status") == {"select": {"name": "rejected"}}
    assert prop(call, "Order ID")["rich_text"][0]["text"]["content"] == "ord-1"


def test_record_final_rejection_note() -> None:
    ledger, session = make_ledger([page_response(), page_response()])
    order = make_order(status="accepted", filled_qty=0.0, filled_avg_price=None)
    ledger.record_submitted(order, AS_OF)
    ledger.record_final(
        make_order(status="canceled", filled_qty=0.0, filled_avg_price=None),
        AS_OF,
        note="cancelled by operator",
    )
    update = session.calls[1]
    assert prop(update, "Status") == {"select": {"name": "cancelled"}}
    assert prop(update, "Notes")["rich_text"][0]["text"]["content"] == "cancelled by operator"


# ---------------------------------------------------------------- status mapping


def test_ledger_status_maps_alpaca_vocabulary() -> None:
    assert ledger_status("filled", 10.0) == "filled"
    assert ledger_status("partially_filled", 5.0) == "partially_filled"
    assert ledger_status("canceled", 0.0) == "cancelled"
    assert ledger_status("expired", 0.0) == "expired"
    assert ledger_status("rejected", 0.0) == "rejected"
    # Non-terminal statuses: partially_filled once shares crossed, else submitted.
    assert ledger_status("accepted", 0.0) == "submitted"
    assert ledger_status("new", 0.0) == "submitted"
    assert ledger_status("accepted", 3.0) == "partially_filled"


# ---------------------------------------------------------------- best-effort


def test_write_failures_collect_and_never_raise() -> None:
    # 400 = NotionError with no retry; both writes must swallow it.
    ledger, session = make_ledger(
        [FakeResponse(400, {"message": "bad payload"}), FakeResponse(400, {"message": "nope"})]
    )
    order = make_order(status="accepted", filled_qty=0.0, filled_avg_price=None)

    assert ledger.record_submitted(order, AS_OF) is None
    ledger.record_final(make_order(status="filled"), AS_OF)

    assert len(ledger.failures) == 2
    assert "record_submitted buy AAPL (ord-1)" in ledger.failures[0]
    assert "record_final filled AAPL (ord-1)" in ledger.failures[1]
    assert len(session.calls) == 2

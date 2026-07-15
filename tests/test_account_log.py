"""US-047: Account Snapshots — writer payloads + the rebalance days that write one.

Writer tests assert payloads against a mocked Notion session (test_ledger.py
pattern); pipeline tests drive the REAL run_rebalance over FakeBroker with a
portfolio-history route added, proving which days produce a row and that
Notion/history outages degrade to summary warnings, never aborts.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from execution.account_log import AccountSnapshotLog, previous_day_pnl
from execution.alpaca_client import Account, PortfolioEntry, PortfolioHistory, Position
from execution.order_gate import Limits
from orchestrator.notion_client import NotionClient
from orchestrator.state import StateStore
from tests.test_alpaca_client import FakeResponse
from tests.test_notion_client import FakeResponse as NotionResponse
from tests.test_notion_client import FakeSession as NotionSession
from tests.test_rebalance import AS_OF, STORE_DAYS, FakeBroker, run, write_bins
from tests.test_signal import write_calendar, write_conf, write_pred
from tests.test_trading_halt import make_breaker

DB_ID = "db-account-snapshots"

ACCOUNT = Account(
    id="acct-1",
    status="ACTIVE",
    currency="USD",
    equity=101_250.5,
    cash=41_000.25,
    buying_power=82_000.0,
    last_equity=100_000.0,
    long_market_value=60_250.25,
    short_market_value=0.0,
)

POSITIONS = [
    Position("AAPL", 250.0, "long", 201.0, 202.0, 50_500.0),
    Position("MSFT", 125.0, "long", 402.0, 401.0, 50_125.0),
]

# 2026-07-07 / 2026-07-08 pre-open (09:30 Eastern) daily points — both strictly
# before AS_OF (2026-07-09), so the 07-08 point is the "previous completed day".
HISTORY = PortfolioHistory(
    timeframe="1D",
    base_value=100_000.0,
    entries=[
        PortfolioEntry(dt.date(2026, 7, 7), 100_000.0, 0.0, 0.0),
        PortfolioEntry(dt.date(2026, 7, 8), 101_250.5, 1_250.5, 0.012505),
    ],
)

HISTORY_ROW = {
    "timestamp": [1783431000, 1783517400],
    "equity": [100000.0, 101250.5],
    "profit_loss": [0.0, 1250.5],
    "profit_loss_pct": [0.0, 0.012505],
    "base_value": 100000.0,
    "timeframe": "1D",
}


def page_response(page_id: str = "page-snap") -> NotionResponse:
    return NotionResponse(200, {"object": "page", "id": page_id})


def make_log(responses: list[NotionResponse]) -> tuple[AccountSnapshotLog, NotionSession]:
    session = NotionSession(responses)
    client = NotionClient(session=session, sleep=lambda _s: None, max_retries=0)
    return AccountSnapshotLog(client, DB_ID), session


def props(call: dict[str, Any]) -> dict[str, Any]:
    return call["json"]["properties"]


# ------------------------------------------------------------ previous_day_pnl


def test_previous_day_pnl_picks_latest_completed_day() -> None:
    entry = previous_day_pnl(HISTORY, AS_OF)
    assert entry is not None
    assert entry.date == dt.date(2026, 7, 8)
    assert entry.profit_loss == 1_250.5


def test_previous_day_pnl_ignores_as_of_and_later_points() -> None:
    padded = PortfolioHistory(
        timeframe="1D",
        base_value=None,
        entries=[*HISTORY.entries, PortfolioEntry(AS_OF, 101_000.0, -250.5, -0.0025)],
    )
    entry = previous_day_pnl(padded, AS_OF)
    assert entry is not None
    assert entry.date == dt.date(2026, 7, 8)


def test_previous_day_pnl_none_without_usable_points() -> None:
    assert previous_day_pnl(None, AS_OF) is None
    empty = PortfolioHistory(timeframe="1D", base_value=None, entries=[])
    assert previous_day_pnl(empty, AS_OF) is None
    unpriced = PortfolioHistory(
        timeframe="1D",
        base_value=None,
        entries=[PortfolioEntry(dt.date(2026, 7, 8), None, None, None)],
    )
    assert previous_day_pnl(unpriced, AS_OF) is None


# ------------------------------------------------------------------ writer


def test_record_daily_creates_full_row() -> None:
    log, session = make_log([page_response()])

    page_id = log.record_daily(
        AS_OF,
        ACCOUNT,
        POSITIONS,
        "traded",
        orders_placed=2,
        orders_filled=2,
        breaker_state="breaker: normal (high-water mark $101,250.50)",
        history=HISTORY,
    )

    assert page_id == "page-snap"
    assert log.failures == []
    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/pages")
    assert call["json"]["parent"] == {"type": "database_id", "database_id": DB_ID}
    row = props(call)
    assert row["Snapshot"]["title"][0]["text"]["content"] == (
        "2026-07-09 — equity $101,250.50"
    )
    assert row["Date"] == {"date": {"start": "2026-07-09"}}
    assert row["Equity"] == {"number": 101_250.5}
    assert row["Cash"] == {"number": 41_000.25}
    assert row["Long Value"] == {"number": 60_250.25}
    assert row["Short Value"] == {"number": 0.0}
    assert row["Positions"] == {"number": 2}
    assert row["Day P/L"] == {"number": 1_250.5}
    assert row["Day P/L %"] == {"number": 0.012505}
    assert row["P/L Day"] == {"date": {"start": "2026-07-08"}}
    assert row["Orders Placed"] == {"number": 2}
    assert row["Orders Filled"] == {"number": 2}
    assert row["Outcome"] == {"select": {"name": "traded"}}
    assert "high-water mark" in row["Breaker"]["rich_text"][0]["text"]["content"]
    assert "Notes" not in row


def test_record_daily_omits_optional_fields() -> None:
    bare = Account(
        id="a", status="ACTIVE", currency="USD",
        equity=100_000.0, cash=100_000.0, buying_power=200_000.0,
    )
    log, session = make_log([page_response()])

    log.record_daily(AS_OF, bare, [], "no_trade", note="book on target")

    row = props(session.calls[0])
    for absent in ("Long Value", "Short Value", "Day P/L", "Day P/L %", "P/L Day", "Breaker"):
        assert absent not in row
    assert row["Positions"] == {"number": 0}
    assert row["Outcome"] == {"select": {"name": "no_trade"}}
    assert row["Notes"]["rich_text"][0]["text"]["content"] == "book on target"


def test_record_daily_collects_failures_instead_of_raising() -> None:
    log, _ = make_log([NotionResponse(500, {"message": "boom"})])

    page_id = log.record_daily(AS_OF, ACCOUNT, POSITIONS, "traded")

    assert page_id is None
    assert len(log.failures) == 1
    assert "record_daily 2026-07-09" in log.failures[0]


def test_record_daily_refuses_unknown_outcome() -> None:
    log, _ = make_log([])
    with pytest.raises(ValueError, match="outcome"):
        log.record_daily(AS_OF, ACCOUNT, POSITIONS, "sideways")


# ------------------------------------------------- rebalance pipeline wiring


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    """Mirror of tests/test_rebalance.py's env fixture (same shared helpers):
    topk=2/n_drop=1 selects AAPL+MSFT at 0.5 each on the $100k account."""
    store = tmp_path / "us_data"
    write_calendar(store / "calendars" / "day.txt", STORE_DAYS)
    write_bins(store, "AAPL", [199.0, 200.0], [1.0, 1.0])
    write_bins(store, "MSFT", [398.0, 400.0], [1.0, 1.0])
    write_bins(store, "NVDA", [99.0, 100.0], [1.0, 1.0])

    workspace = tmp_path / "workspace"
    write_conf(workspace, "conf.yaml", topk=2, n_drop=1)
    write_pred(workspace, {"2026-07-08": {"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.1}})

    db_path = tmp_path / "state.sqlite"
    StateStore(db_path).set_promoted_strategy(
        str(workspace), {"universe": "us_liquid", "topk": 2, "n_drop": 1}
    )
    limits = Limits(
        max_order_notional_usd=60_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    return SimpleNamespace(
        store=store,
        workspace=workspace,
        db_path=db_path,
        breaker=make_breaker(tmp_path),
        limits=limits,
    )


def route_history(broker: FakeBroker, row: dict[str, Any] | None = None) -> None:
    payload = HISTORY_ROW if row is None else row
    broker.session.route(
        "GET",
        "/v2/account/portfolio/history",
        lambda _p, _j: FakeResponse(200, payload),
    )


def test_traded_day_writes_traded_snapshot(env: SimpleNamespace) -> None:
    broker = FakeBroker()
    route_history(broker)
    log, session = make_log([page_response()])
    notes: list[str] = []

    assert run(env, broker, notes, snapshots=log) == 0

    row = props(session.calls[0])
    assert row["Outcome"] == {"select": {"name": "traded"}}
    assert row["Orders Placed"] == {"number": 2}
    assert row["Orders Filled"] == {"number": 2}
    assert row["Day P/L"] == {"number": 1250.5}
    assert row["P/L Day"] == {"date": {"start": "2026-07-08"}}
    assert "WARNING" not in notes[-1]


def test_dry_run_writes_no_snapshot(env: SimpleNamespace) -> None:
    broker = FakeBroker()
    log, session = make_log([])
    notes: list[str] = []

    assert run(env, broker, notes, dry_run=True, snapshots=log) == 0

    assert session.calls == []


def test_halted_day_writes_halted_snapshot(env: SimpleNamespace) -> None:
    env.breaker.halt("operator kill switch")
    broker = FakeBroker()
    route_history(broker)
    log, session = make_log([page_response()])
    notes: list[str] = []

    assert run(env, broker, notes, snapshots=log) == 0

    row = props(session.calls[0])
    assert row["Outcome"] == {"select": {"name": "halted"}}
    assert "operator kill switch" in row["Notes"]["rich_text"][0]["text"]["content"]
    assert broker.session.posts() == []  # halted: nothing traded, row still written


def test_history_outage_degrades_to_warning_row_without_day_pnl(
    env: SimpleNamespace,
) -> None:
    broker = FakeBroker()
    broker.session.route(
        "GET",
        "/v2/account/portfolio/history",
        lambda _p, _j: FakeResponse(500, {"message": "boom"}, text="boom"),
    )
    log, session = make_log([page_response()])
    notes: list[str] = []

    assert run(env, broker, notes, snapshots=log) == 0

    row = props(session.calls[0])  # the row is still written, minus Day P/L
    assert row["Outcome"] == {"select": {"name": "traded"}}
    assert "Day P/L" not in row
    assert "WARNING: Account Snapshot: portfolio history fetch" in notes[-1]


def test_notion_outage_becomes_summary_warning_not_abort(env: SimpleNamespace) -> None:
    broker = FakeBroker()
    route_history(broker)
    log, _ = make_log([NotionResponse(500, {"message": "boom"})])
    notes: list[str] = []

    assert run(env, broker, notes, snapshots=log) == 0

    assert "WARNING: Account Snapshot: record_daily 2026-07-09" in notes[-1]

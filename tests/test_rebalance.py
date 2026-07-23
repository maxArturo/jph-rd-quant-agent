"""US-034: rebalance pipeline — integration over a mocked Alpaca + every abort path."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest

from execution.alpaca_client import AlpacaClient, Order
from execution.breaker import Breaker, BreakerConfig
from execution.ledger import TradeLedger
from execution.order_gate import Limits, ProposedOrder
from execution.rebalance import (
    RebalanceError,
    _strategy_params,
    build_reference_prices,
    cap_buys_to_buying_power,
    day_traded_notional,
    fill_summary,
    format_daily_summary,
    latest_store_price,
    orders_submitted_on,
    poll_fills,
    run_rebalance,
    submitted_market_date,
)
from orchestrator.notion_client import NotionClient
from orchestrator.state import StateStore
from tests.test_alpaca_client import FakeResponse
from tests.test_notion_client import FakeResponse as NotionResponse
from tests.test_notion_client import FakeSession as NotionSession
from tests.test_signal import write_calendar, write_conf, write_pred

AS_OF = dt.date(2026, 7, 9)  # Thursday; store calendar ends the day before
STORE_DAYS = ["2026-07-06", "2026-07-07", "2026-07-08"]


# ---------------------------------------------------------------- fakes


class RoutedSession:
    """Routes requests by (method, path) and records every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.handlers: dict[tuple[str, str], Any] = {}

    def route(self, method: str, path: str, handler: Any) -> None:
        self.handlers[(method, path)] = handler

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        path = urlparse(url).path
        self.calls.append({"method": method, "path": path, "params": params, "json": json})
        handler = self.handlers.get((method, path))
        if handler is None:
            raise AssertionError(f"unexpected request: {method} {path}")
        return handler(params, json)

    def posts(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == "POST"]


class FakeBroker:
    """A routed mock of the Alpaca paper API, driven through the REAL client.

    ``fill_countdown`` = how many post-submit order polls stay unfilled before
    everything fills; None = never fills (timeout path).
    """

    def __init__(
        self,
        equity: float = 100_000.0,
        positions: list[dict[str, Any]] | None = None,
        trading_days: set[str] | None = None,
        existing_orders: list[dict[str, Any]] | None = None,
        fill_countdown: int | None = 0,
        post_fail_after: int | None = None,
        buying_power: float | None = None,
    ) -> None:
        self.session = RoutedSession()
        self.positions = positions or []
        self.trading_days = trading_days if trading_days is not None else {AS_OF.isoformat()}
        self.orders: list[dict[str, Any]] = list(existing_orders or [])
        self.fill_countdown = fill_countdown
        self.post_fail_after = post_fail_after
        self.submitted: list[dict[str, Any]] = []

        account_row = {
            "id": "acct-1",
            "status": "ACTIVE",
            "currency": "USD",
            "equity": str(equity),
            "cash": str(equity),
            "buying_power": str(equity * 2 if buying_power is None else buying_power),
        }
        self.session.route("GET", "/v2/account", lambda p, j: FakeResponse(200, account_row))
        self.session.route(
            "GET", "/v2/positions", lambda p, j: FakeResponse(200, list(self.positions))
        )
        self.session.route("GET", "/v2/calendar", self._calendar)
        self.session.route("GET", "/v2/orders", self._list_orders)
        self.session.route("POST", "/v2/orders", self._place_order)

    def client(self) -> AlpacaClient:
        return AlpacaClient(session=self.session)

    def _calendar(self, params: dict[str, str], _json: Any) -> FakeResponse:
        rows = [
            {"date": day, "open": "09:30", "close": "16:00"}
            for day in sorted(self.trading_days)
            if params["start"] <= day <= params["end"]
        ]
        return FakeResponse(200, rows)

    def _list_orders(self, _params: Any, _json: Any) -> FakeResponse:
        if any(o["status"] == "accepted" for o in self.orders):
            if self.fill_countdown is not None:
                if self.fill_countdown <= 0:
                    for order in self.orders:
                        if order["status"] == "accepted":
                            order["status"] = "filled"
                            order["filled_qty"] = order["qty"]
                            order["filled_avg_price"] = order["limit_price"]
                else:
                    self.fill_countdown -= 1
        return FakeResponse(200, [dict(o) for o in self.orders])

    def _place_order(self, _params: Any, payload: dict[str, Any]) -> FakeResponse:
        if self.post_fail_after is not None and len(self.submitted) >= self.post_fail_after:
            return FakeResponse(403, {"code": 40110000, "message": "access denied"})
        row = {
            "id": f"ord-{len(self.orders) + 1}",
            "client_order_id": payload["client_order_id"],
            "symbol": payload["symbol"],
            "qty": payload["qty"],
            "notional": None,
            "side": payload["side"],
            "type": payload["type"],
            "time_in_force": payload["time_in_force"],
            "limit_price": payload["limit_price"],
            "status": "accepted",
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": f"{AS_OF.isoformat()}T13:31:00Z",
        }
        self.orders.append(row)
        self.submitted.append(row)
        return FakeResponse(200, dict(row))


def position_row(
    symbol: str, qty: float, current_price: float | None
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "qty": str(qty),
        "side": "long" if qty >= 0 else "short",
        "avg_entry_price": "100",
        "current_price": None if current_price is None else str(current_price),
        "market_value": None if current_price is None else str(qty * current_price),
    }


def make_order(**overrides: Any) -> Order:
    base: dict[str, Any] = {
        "id": "ord-1",
        "client_order_id": "rdq-x",
        "symbol": "AAPL",
        "qty": 10.0,
        "notional": None,
        "side": "buy",
        "order_type": "limit",
        "time_in_force": "day",
        "limit_price": 100.0,
        "status": "filled",
        "filled_qty": 10.0,
        "filled_avg_price": 100.0,
        "submitted_at": "2026-07-09T13:30:00Z",
    }
    base.update(overrides)
    return Order(**base)


# ---------------------------------------------------------------- fixtures


def write_bins(store: Path, symbol: str, closes: list[float], factors: list[float]) -> None:
    feature_dir = store / "features" / symbol.lower()
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.array([0.0, *closes], dtype="<f").tofile(feature_dir / "close.day.bin")
    np.array([0.0, *factors], dtype="<f").tofile(feature_dir / "factor.day.bin")


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    """Store + promoted workspace + state DB + breaker/limits for a happy run.

    topk=2/n_drop=1 over {AAPL: 0.9, MSFT: 0.8, NVDA: 0.1} selects AAPL+MSFT
    at 0.5 weight each; on $100k with AAPL@200/MSFT@400 that is exactly
    buy 250 AAPL @ 201.00 and buy 125 MSFT @ 402.00.
    """
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

    breaker = Breaker(
        BreakerConfig(max_daily_notional_usd=200_000.0, max_drawdown_pct=20.0),
        halt_file=tmp_path / "halt",
        high_water_mark_file=tmp_path / "hwm.json",
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
        breaker=breaker,
        limits=limits,
        hwm_file=tmp_path / "hwm.json",
    )


def run(env: SimpleNamespace, broker: FakeBroker, notes: list[str], **overrides: Any) -> int:
    kwargs: dict[str, Any] = dict(
        dry_run=False,
        as_of=AS_OF,
        db_path=env.db_path,
        store_path=env.store,
        limits=env.limits,
        breaker=env.breaker,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.1,
        sleep=lambda _s: None,
    )
    kwargs.update(overrides)
    return run_rebalance(broker.client(), notes.append, **kwargs)


# ---------------------------------------------------------------- dry run (AC 2)


def test_dry_run_prints_exact_orders_and_submits_nothing(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "dry run — nothing submitted" in out
    assert "buy 250 AAPL @ 201.00" in out
    assert "buy 125 MSFT @ 402.00" in out
    assert "2 orders" in out
    assert broker.session.posts() == []


def test_dry_run_runs_gate_and_breaker_before_printing(env: SimpleNamespace) -> None:
    tight = Limits(
        max_order_notional_usd=1_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, dry_run=True, limits=tight) == 1
    assert "max_order_notional_usd" in notes[0]
    assert broker.session.posts() == []


# ---------------------------------------------------------------- live run


def test_live_run_submits_exact_orders_and_polls_fills(env: SimpleNamespace) -> None:
    broker = FakeBroker(fill_countdown=1)  # unfilled on the first poll, filled on the second
    notes: list[str] = []
    sleeps: list[float] = []
    assert run(env, broker, notes, sleep=sleeps.append) == 0

    posts = [c["json"] for c in broker.session.posts()]
    assert [(p["symbol"], p["side"], p["qty"], p["limit_price"]) for p in posts] == [
        ("AAPL", "buy", "250", "201"),
        ("MSFT", "buy", "125", "402"),
    ]
    assert posts[0]["client_order_id"] == "rdq-2026-07-09-buy-AAPL"
    assert posts[0]["time_in_force"] == "day"
    assert sleeps == [0.1]
    assert len(notes) == 1
    assert "2/2 orders filled" in notes[0]
    assert "buy 250 AAPL: filled @ $201.00" in notes[0]


# ------------------------------------------------- buying-power cap (2026-07-23)


def test_cap_defers_buys_beyond_buying_power_and_keeps_sells() -> None:
    orders = [
        ProposedOrder(symbol="COST", side="sell", qty=3, limit_price=900.0),
        ProposedOrder(symbol="AAPL", side="buy", qty=10, limit_price=100.0),  # 1,000
        ProposedOrder(symbol="MSFT", side="buy", qty=10, limit_price=400.0),  # 4,000
        ProposedOrder(symbol="NVDA", side="buy", qty=1, limit_price=200.0),  # 200
    ]
    kept, deferred = cap_buys_to_buying_power(orders, 1_500.0)
    # The sell never consumes buying power; MSFT is over the remaining $500,
    # but NVDA after it still fits — order stays deterministic, gaps close up.
    assert [o.symbol for o in kept] == ["COST", "AAPL", "NVDA"]
    assert [s.symbol for s in deferred] == ["MSFT"]
    assert deferred[0].reason == "insufficient_buying_power"
    assert "buy 10 MSFT @ 400.00" in deferred[0].message


def test_cap_exactly_at_buying_power_passes() -> None:
    orders = [ProposedOrder(symbol="AAPL", side="buy", qty=10, limit_price=100.0)]
    kept, deferred = cap_buys_to_buying_power(orders, 1_000.0)
    assert [o.symbol for o in kept] == ["AAPL"]
    assert deferred == []


def test_cap_sells_pass_with_zero_buying_power() -> None:
    orders = [
        ProposedOrder(symbol="COST", side="sell", qty=3, limit_price=900.0),
        ProposedOrder(symbol="AAPL", side="buy", qty=1, limit_price=100.0),
    ]
    kept, deferred = cap_buys_to_buying_power(orders, 0.0)
    assert [o.symbol for o in kept] == ["COST"]
    assert [s.symbol for s in deferred] == ["AAPL"]


def test_live_run_defers_buys_over_buying_power_and_warns(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    # The env plan is buy 250 AAPL @ 201 ($50,250) then buy 125 MSFT @ 402
    # ($50,250): with $60k buying power AAPL fits, MSFT must be deferred —
    # NOT submitted into a mid-batch 403 (the 2026-07-23 incident).
    broker = FakeBroker(buying_power=60_000.0)
    notes: list[str] = []
    assert run(env, broker, notes) == 0

    posts = [c["json"] for c in broker.session.posts()]
    assert [p["symbol"] for p in posts] == ["AAPL"]
    assert "skipped: MSFT" in capsys.readouterr().out
    assert len(notes) == 1
    assert "1/1 orders filled" in notes[0]
    assert "WARNING: MSFT: buy 125 MSFT @ 402.00 deferred" in notes[0]
    assert "buying power" in notes[0]


def test_live_run_all_buys_deferred_is_a_no_trade_day(env: SimpleNamespace) -> None:
    broker = FakeBroker(buying_power=1_000.0)
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    assert broker.session.posts() == []
    assert "no orders — every buy deferred for buying power" in notes[0]
    assert "WARNING: AAPL" in notes[0]
    assert "WARNING: MSFT" in notes[0]


def test_full_exit_of_dropped_position_uses_snapshot_price_fallback(
    env: SimpleNamespace,
) -> None:
    # TSLA is held but absent from the store AND the predictions: it must be
    # fully exited, priced from the position snapshot's current_price.
    broker = FakeBroker(positions=[position_row("TSLA", 10, 300.0)])
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    posts = [c["json"] for c in broker.session.posts()]
    assert posts[0]["symbol"] == "TSLA"  # sells come first
    assert posts[0]["side"] == "sell"
    assert posts[0]["qty"] == "10"
    assert posts[0]["limit_price"] == "298.5"  # 300 * 0.995, floored to the cent


def test_nothing_to_trade_exits_zero_without_submitting(env: SimpleNamespace) -> None:
    broker = FakeBroker(
        positions=[position_row("AAPL", 250, 200.0), position_row("MSFT", 125, 400.0)]
    )
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    assert broker.session.posts() == []
    assert "no orders — book already on target" in notes[0]


def test_poll_timeout_reports_unfilled_and_exits_zero(env: SimpleNamespace) -> None:
    broker = FakeBroker(fill_countdown=None)  # never fills (pre-open submission)
    notes: list[str] = []
    assert run(env, broker, notes, poll_timeout_seconds=0.3, poll_interval_seconds=0.1) == 0
    assert "0/2 orders filled" in notes[0]
    assert "may still fill after market open" in notes[0]


# ---------------------------------------------------------------- abort paths (AC 3/4)


def test_market_closed_aborts_nonzero_and_notifies(env: SimpleNamespace) -> None:
    broker = FakeBroker(trading_days=set())
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "market closed" in notes[0]
    assert broker.session.posts() == []


def test_no_promoted_strategy_aborts_nonzero(env: SimpleNamespace, tmp_path: Path) -> None:
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, db_path=tmp_path / "absent.sqlite") == 1
    assert "promote" in notes[0]
    assert broker.session.posts() == []


def test_stale_predictions_abort_nonzero(env: SimpleNamespace) -> None:
    # Rewrite the pred with a cross-section older than the last trading day.
    pred = next(env.workspace.glob("mlruns/**/pred.pkl"))
    pred.unlink()
    write_pred(env.workspace, {"2026-07-07": {"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.1}})
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "stale" in notes[0]
    assert broker.session.posts() == []


def test_halt_file_exits_zero_with_halted_notice(env: SimpleNamespace) -> None:
    env.breaker.halt("weekend maintenance")
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    assert "halted" in notes[0]
    assert "weekend maintenance" in notes[0]
    assert broker.session.posts() == []


def test_gate_rejection_aborts_nonzero_naming_the_limit(env: SimpleNamespace) -> None:
    tight = Limits(
        max_order_notional_usd=1_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, limits=tight) == 1
    assert "max_order_notional_usd" in notes[0]
    assert broker.session.posts() == []


def test_breaker_drawdown_trip_aborts_nonzero(env: SimpleNamespace) -> None:
    env.hwm_file.write_text(json.dumps({"high_water_mark": 200_000.0}) + "\n")
    broker = FakeBroker()  # equity 100k = 50% below the mark
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "max_drawdown_pct" in notes[0]
    assert broker.session.posts() == []


def test_breaker_daily_notional_trip_aborts_nonzero(env: SimpleNamespace) -> None:
    already_traded = {
        "id": "ord-old",
        "client_order_id": "rdq-earlier",
        "symbol": "SPY",
        "qty": "1000",
        "notional": None,
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "250",
        "status": "filled",
        "filled_qty": "1000",
        "filled_avg_price": "250",  # $250k already traded today, over the $200k cap
        "submitted_at": f"{AS_OF.isoformat()}T09:35:00-04:00",
    }
    broker = FakeBroker(existing_orders=[already_traded])
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "max_daily_notional_usd" in notes[0]
    assert broker.session.posts() == []


def test_submit_failure_mid_batch_aborts_nonzero(env: SimpleNamespace) -> None:
    broker = FakeBroker(post_fail_after=1)
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "failed after 1 of 2" in notes[0]
    assert "submitted orders are live" in notes[0]


def test_missing_reference_price_aborts_nonzero(env: SimpleNamespace) -> None:
    # Held symbol with no store bins and no snapshot price: nothing can price
    # its exit order.
    broker = FakeBroker(positions=[position_row("ZZZZ", 5, None)])
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "no reference price" in notes[0]
    assert "ZZZZ" in notes[0]
    assert broker.session.posts() == []


def test_notify_failure_never_masks_the_outcome(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    broker = FakeBroker(trading_days=set())

    def broken_notify(_text: str) -> None:
        raise RuntimeError("slack down")

    assert run_rebalance(
        broker.client(),
        broken_notify,
        as_of=AS_OF,
        db_path=env.db_path,
        store_path=env.store,
        limits=env.limits,
        breaker=env.breaker,
    ) == 1
    err = capsys.readouterr().err
    assert "Slack notification failed" in err
    assert "market closed" in err


# ---------------------------------------------------------------- daily summary (US-035)


def notion_ledger(responses: list[NotionResponse]) -> tuple[TradeLedger, NotionSession]:
    session = NotionSession(responses)
    return TradeLedger(NotionClient(session=session), "db-trade-ledger"), session


def notion_page(page_id: str) -> NotionResponse:
    return NotionResponse(200, {"object": "page", "id": page_id})


def test_traded_day_posts_daily_summary(env: SimpleNamespace) -> None:
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    assert len(notes) == 1
    assert "daily rebalance summary (2026-07-09)" in notes[0]
    assert "account equity: $100,000.00" in notes[0]
    assert "orders placed: 2" in notes[0]
    assert "2/2 orders filled" in notes[0]
    assert "gate/breaker rejections: none" in notes[0]


def test_no_trade_day_summary_includes_equity(env: SimpleNamespace) -> None:
    broker = FakeBroker(
        positions=[position_row("AAPL", 250, 200.0), position_row("MSFT", 125, 400.0)]
    )
    notes: list[str] = []
    assert run(env, broker, notes) == 0
    assert "orders placed: 0" in notes[0]
    assert "account equity: $100,000.00" in notes[0]


def test_gate_rejection_day_summary_lists_rejections_and_equity(
    env: SimpleNamespace,
) -> None:
    tight = Limits(
        max_order_notional_usd=1_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, limits=tight) == 1
    assert len(notes) == 1
    assert "gate/breaker rejections:" in notes[0]
    assert "max_order_notional_usd" in notes[0]
    assert "account equity: $100,000.00" in notes[0]
    assert "orders placed: 0" in notes[0]
    assert broker.session.posts() == []


def test_breaker_trip_day_summary_lists_the_trip(env: SimpleNamespace) -> None:
    env.hwm_file.write_text(json.dumps({"high_water_mark": 200_000.0}) + "\n")
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes) == 1
    assert "gate/breaker rejections:" in notes[0]
    assert "max_drawdown_pct" in notes[0]
    assert "account equity: $100,000.00" in notes[0]


def test_format_daily_summary_from_fixture_fill_set() -> None:
    submitted = [
        make_order(id="a", status="accepted", filled_qty=0.0, filled_avg_price=None),
        make_order(
            id="b", symbol="MSFT", status="accepted", filled_qty=0.0, filled_avg_price=None
        ),
    ]
    final = [
        make_order(id="a", status="filled", filled_qty=10.0, filled_avg_price=100.0),
        make_order(id="b", symbol="MSFT", status="rejected", filled_qty=0.0),
    ]
    text = format_daily_summary(AS_OF, 123_456.78, submitted, final)
    assert "daily rebalance summary (2026-07-09)" in text
    assert "account equity: $123,456.78" in text
    assert "orders placed: 2" in text
    assert "1/2 orders filled" in text
    assert "buy 10 AAPL: filled @ $100.00" in text
    assert "MSFT: rejected" in text
    assert "gate/breaker rejections: none" in text


def test_format_daily_summary_rejections_and_ledger_warnings() -> None:
    text = format_daily_summary(
        AS_OF,
        50_000.0,
        [],
        [],
        rejections=["max_order_notional_usd: order too big"],
        no_trade_note="order gate rejected the batch — nothing submitted",
        ledger_failures=["record_submitted buy AAPL: boom"],
    )
    assert "orders placed: 0" in text
    assert "order gate rejected the batch" in text
    assert "gate/breaker rejections:\n  max_order_notional_usd: order too big" in text
    assert "WARNING: Trade Ledger write failed — record_submitted buy AAPL: boom" in text


# ---------------------------------------------------------------- trade ledger (US-035)


def test_traded_day_writes_ledger_rows_and_final_fills(env: SimpleNamespace) -> None:
    # 2 orders -> 2 creates at submit time + 2 updates after the fill poll.
    ledger, session = notion_ledger(
        [notion_page("pg-1"), notion_page("pg-2"), notion_page("pg-1"), notion_page("pg-2")]
    )
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, ledger=ledger) == 0
    assert ledger.failures == []

    creates = [c for c in session.calls if c["method"] == "POST"]
    updates = [c for c in session.calls if c["method"] == "PATCH"]
    assert len(creates) == 2
    assert len(updates) == 2

    first = creates[0]["json"]
    assert first["parent"] == {"type": "database_id", "database_id": "db-trade-ledger"}
    props = first["properties"]
    assert props["Order"]["title"][0]["text"]["content"] == "2026-07-09 BUY 250 AAPL"
    assert props["Order ID"]["rich_text"][0]["text"]["content"] == "ord-1"
    assert props["Side"] == {"select": {"name": "buy"}}
    assert props["Qty"] == {"number": 250.0}
    assert props["Limit Price"] == {"number": 201.0}
    assert props["Status"] == {"select": {"name": "submitted"}}
    assert creates[1]["json"]["properties"]["Order"]["title"][0]["text"]["content"] == (
        "2026-07-09 BUY 125 MSFT"
    )

    assert updates[0]["url"].endswith("/v1/pages/pg-1")
    final_props = updates[0]["json"]["properties"]
    assert final_props["Status"] == {"select": {"name": "filled"}}
    assert final_props["Filled Qty"] == {"number": 250.0}
    assert final_props["Filled Avg Price"] == {"number": 201.0}


def test_mid_batch_submit_failure_still_records_live_orders(env: SimpleNamespace) -> None:
    ledger, session = notion_ledger([notion_page("pg-1")])
    broker = FakeBroker(post_fail_after=1)
    notes: list[str] = []
    assert run(env, broker, notes, ledger=ledger) == 1
    creates = [c for c in session.calls if c["method"] == "POST"]
    assert len(creates) == 1  # the one order that went in has its row
    assert creates[0]["json"]["properties"]["Symbol"]["rich_text"][0]["text"]["content"] == "AAPL"


def test_ledger_outage_never_breaks_the_run_and_is_surfaced(env: SimpleNamespace) -> None:
    responses = [NotionResponse(400, {"message": "boom"}) for _ in range(4)]
    ledger, _session = notion_ledger(responses)
    broker = FakeBroker()
    notes: list[str] = []
    assert run(env, broker, notes, ledger=ledger) == 0  # trade still completes
    assert "2/2 orders filled" in notes[0]
    assert "WARNING: Trade Ledger write failed" in notes[0]
    assert len(ledger.failures) == 4  # 2 failed creates + 2 failed final creates


# ---------------------------------------------------------------- helpers


def test_submitted_market_date_converts_utc_to_eastern() -> None:
    # 01:30 UTC is the previous evening in New York.
    early = make_order(submitted_at="2026-07-09T01:30:00Z")
    assert submitted_market_date(early) == dt.date(2026, 7, 8)
    midday = make_order(submitted_at="2026-07-09T13:30:00.000000Z")
    assert submitted_market_date(midday) == dt.date(2026, 7, 9)
    assert submitted_market_date(make_order(submitted_at=None)) is None
    assert submitted_market_date(make_order(submitted_at="not-a-date")) is None


def test_orders_submitted_on_filters_by_market_day() -> None:
    today = make_order(id="a", submitted_at="2026-07-09T13:30:00Z")
    yesterday_utc = make_order(id="b", submitted_at="2026-07-09T01:30:00Z")
    assert orders_submitted_on([today, yesterday_utc], AS_OF) == [today]


def test_day_traded_notional_sums_fills_only() -> None:
    filled = make_order(filled_qty=10.0, filled_avg_price=100.0)
    unfilled = make_order(id="u", status="accepted", filled_qty=0.0, filled_avg_price=None)
    assert day_traded_notional([filled, unfilled]) == pytest.approx(1000.0)


def test_latest_store_price_divides_close_by_factor(tmp_path: Path) -> None:
    write_bins(tmp_path, "AAPL", [50.0, 100.0], [0.5, 0.5])
    assert latest_store_price(tmp_path, "AAPL") == pytest.approx(200.0)
    assert latest_store_price(tmp_path, "MSFT") is None


def test_build_reference_prices_prefers_store_over_snapshot(tmp_path: Path) -> None:
    write_bins(tmp_path, "AAPL", [200.0], [1.0])
    from execution.alpaca_client import Position

    held = [
        Position(
            symbol="AAPL",
            qty=1,
            side="long",
            avg_entry_price=1.0,
            current_price=999.0,
            market_value=999.0,
        )
    ]
    prices = build_reference_prices(tmp_path, ["AAPL"], held)
    assert prices == {"AAPL": pytest.approx(200.0)}


def test_strategy_params_from_promoted_config() -> None:
    from execution.signal import StrategyParams

    assert _strategy_params({"topk": 5, "n_drop": 2}) == StrategyParams(topk=5, n_drop=2)
    assert _strategy_params({"universe": "us_liquid"}) is None
    with pytest.raises(RebalanceError, match="non-integer"):
        _strategy_params({"topk": "5", "n_drop": 2})
    with pytest.raises(RebalanceError, match="non-integer"):
        _strategy_params({"topk": True, "n_drop": 2})


def test_poll_fills_rejects_nonpositive_interval() -> None:
    broker = FakeBroker()
    with pytest.raises(ValueError, match="interval_seconds"):
        poll_fills(broker.client(), ["x"], interval_seconds=0)


def test_fill_summary_reports_partial_fills() -> None:
    submitted = [
        make_order(id="a", status="accepted", filled_qty=0.0, filled_avg_price=None),
        make_order(id="b", symbol="MSFT"),
    ]
    final = [
        make_order(id="a", status="accepted", filled_qty=0.0, filled_avg_price=None),
        make_order(id="b", symbol="MSFT", status="filled", filled_qty=10.0),
    ]
    text = fill_summary(submitted, final)
    assert text.startswith("1/2 orders filled")
    assert "AAPL: accepted (0 filled)" in text
    assert "MSFT: filled" in text

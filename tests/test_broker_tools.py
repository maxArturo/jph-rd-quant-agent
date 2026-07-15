"""US-046: read-only broker visibility tools (check_account/check_orders/check_pnl).

Drives the REAL ConversationCore with FakeClient scripts over a StubBroker —
the tool result fed back to the model is asserted from the recorded second
stream call, the same way test_trading_halt.py proves the halt tools.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from execution.alpaca_client import (
    Account,
    AlpacaError,
    Order,
    PortfolioEntry,
    PortfolioHistory,
    Position,
)
from orchestrator.conversation import (
    ConversationCore,
    format_account_report,
    format_orders_report,
    format_pnl_report,
)
from orchestrator.llm import ModelRouter
from orchestrator.state import StateStore
from tests.test_conversation import THREAD, RecordingSay, StubLauncher
from tests.test_llm import FakeClient, message, text_block, tool_use_block
from tests.test_trading_halt import make_breaker

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
    Position("MSFT", 125.0, "long", 402.0, 410.0, 51_250.0, 1_000.0, 0.0199),
    Position("AAPL", 50.0, "long", 201.0, 200.0, 10_000.0, -50.0, -0.005),
]

ORDERS = [
    Order(
        id="ord-2",
        client_order_id="rdq-2026-07-14-buy-MSFT",
        symbol="MSFT",
        qty=125.0,
        notional=None,
        side="buy",
        order_type="limit",
        time_in_force="day",
        limit_price=402.0,
        status="filled",
        filled_qty=125.0,
        filled_avg_price=401.5,
        submitted_at="2026-07-14T10:30:00Z",
    ),
    Order(
        id="ord-1",
        client_order_id="rdq-2026-07-14-buy-AAPL",
        symbol="AAPL",
        qty=50.0,
        notional=None,
        side="buy",
        order_type="limit",
        time_in_force="day",
        limit_price=201.0,
        status="accepted",
        filled_qty=0.0,
        filled_avg_price=None,
        submitted_at="2026-07-14T10:30:00Z",
    ),
]

HISTORY = PortfolioHistory(
    timeframe="1D",
    base_value=100_000.0,
    entries=[
        PortfolioEntry(dt.date(2026, 7, 13), 100_000.0, 0.0, 0.0),
        PortfolioEntry(dt.date(2026, 7, 14), 101_250.5, 1_250.5, 0.012505),
    ],
)


class StubBroker:
    """Records read calls; raises when primed with an error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._error = error

    def _maybe_raise(self) -> None:
        if self._error is not None:
            raise self._error

    def get_account(self) -> Account:
        self.calls.append({"method": "get_account"})
        self._maybe_raise()
        return ACCOUNT

    def get_positions(self) -> list[Position]:
        self.calls.append({"method": "get_positions"})
        self._maybe_raise()
        return POSITIONS

    def list_orders(
        self,
        status: str = "open",
        limit: int | None = None,
        symbols: list[str] | None = None,
        after: str | None = None,
        until: str | None = None,
    ) -> list[Order]:
        self.calls.append({"method": "list_orders", "status": status, "limit": limit})
        self._maybe_raise()
        return ORDERS

    def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> PortfolioHistory:
        self.calls.append(
            {"method": "get_portfolio_history", "period": period, "timeframe": timeframe}
        )
        self._maybe_raise()
        return HISTORY


def make_core(
    tmp_path: Path, client: FakeClient, broker: StubBroker
) -> ConversationCore:
    return ConversationCore(
        store=StateStore(db_path=tmp_path / "conv.sqlite"),
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        breaker=make_breaker(tmp_path),
        broker=broker,
    )


def tool_script(name: str, args: dict[str, Any], reply: str = "Here you go.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_1", name, args)]),
        message("end_turn", [text_block(reply)]),
    ]


def tool_result(client: FakeClient) -> dict[str, Any]:
    """The single tool_result block fed back after the scripted tool call."""
    (block,) = client.stream_calls[1]["messages"][-1]["content"]
    return block


# ------------------------------------------------------------------ tools


def test_check_account_reports_snapshot_and_positions(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_account", {}))
    broker = StubBroker()
    core = make_core(tmp_path, client, broker)
    say = RecordingSay()

    core.handle_message(THREAD, "how is the account doing?", say)

    offered = {tool["name"] for tool in client.stream_calls[0]["tools"]}
    assert {"check_account", "check_orders", "check_pnl"} <= offered
    assert [c["method"] for c in broker.calls] == ["get_account", "get_positions"]
    result = tool_result(client)
    assert "is_error" not in result  # only set on tool failure
    report = result["content"]
    assert "equity: $101,250.50" in report
    assert "since previous close: +$1,250.50 (+1.25%)" in report
    assert "AAPL: 50 @ avg $201.00" in report
    assert "unrealized +$1,000.00 (+1.99%)" in report
    assert "trading: active" in report
    assert say.calls[-1]["text"] == "Here you go."


def test_check_account_reports_halt_state(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_account", {}))
    core = make_core(tmp_path, client, StubBroker())
    core._breaker.halt("volatility spike")  # noqa: SLF001 - drive the real breaker

    core.handle_message(THREAD, "status?", RecordingSay())

    assert "trading: HALTED — volatility spike" in tool_result(client)["content"]


def test_check_orders_passes_filters_and_formats_fills(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=tool_script("check_orders", {"status": "all", "limit": 5})
    )
    broker = StubBroker()
    core = make_core(tmp_path, client, broker)

    core.handle_message(THREAD, "did the orders get executed?", RecordingSay())

    assert broker.calls == [{"method": "list_orders", "status": "all", "limit": 5}]
    report = tool_result(client)["content"]
    assert "2 all order(s), newest first:" in report
    assert "buy 125 MSFT @ limit $402.00 — filled 125 @ $401.50" in report
    assert "buy 50 AAPL @ limit $201.00 — accepted" in report


def test_check_orders_defaults_and_clamps_limit(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_orders", {"limit": 999}))
    broker = StubBroker()
    core = make_core(tmp_path, client, broker)

    core.handle_message(THREAD, "orders?", RecordingSay())

    assert broker.calls == [{"method": "list_orders", "status": "all", "limit": 50}]


def test_check_pnl_reports_period_totals(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_pnl", {"period": "1W"}))
    broker = StubBroker()
    core = make_core(tmp_path, client, broker)

    core.handle_message(THREAD, "how did we do this week?", RecordingSay())

    assert broker.calls == [
        {"method": "get_portfolio_history", "period": "1W", "timeframe": "1D"}
    ]
    report = tool_result(client)["content"]
    assert "portfolio P/L over 1W" in report
    assert "period total: +$1,250.50 (+1.25%)" in report
    assert "2026-07-14: equity $101,250.50, day +$1,250.50 (+1.25%)" in report


def test_broker_error_becomes_error_tool_result_not_a_crash(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=tool_script(
            "check_account", {}, reply="Alpaca is unreachable right now."
        )
    )
    broker = StubBroker(error=AlpacaError("HTTP 500 from /v2/account"))
    core = make_core(tmp_path, client, broker)
    say = RecordingSay()

    core.handle_message(THREAD, "account?", say)

    result = tool_result(client)
    assert result["is_error"] is True
    assert "HTTP 500" in result["content"]
    assert say.calls[-1]["text"] == "Alpaca is unreachable right now."


# ------------------------------------------------------------- formatters


def test_account_report_flat_account_without_optional_fields() -> None:
    bare = Account(
        id="a", status="ACTIVE", currency="USD",
        equity=100_000.0, cash=100_000.0, buying_power=200_000.0,
    )
    report = format_account_report(bare, [], "active")
    assert "positions: none (flat)" in report
    assert "since previous close" not in report


def test_orders_report_empty() -> None:
    assert format_orders_report([], "open") == "no open orders found on the paper account"


def test_pnl_report_without_equity_values() -> None:
    empty = PortfolioHistory(timeframe="1D", base_value=None, entries=[])
    assert "no portfolio history" in format_pnl_report(empty, "1M")

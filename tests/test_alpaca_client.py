"""Unit tests for execution/alpaca_client.py (mocked HTTP) plus a live proxy smoke test."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
from typing import Any

import pytest

from execution.alpaca_client import (
    BASE_URL,
    Account,
    AlpacaAuthError,
    AlpacaClient,
    AlpacaError,
    AlpacaRateLimitError,
    CalendarDay,
    CancelledOrder,
    Order,
    PortfolioEntry,
    Position,
)

ACCOUNT_ROW = {
    "id": "b3f2a9c1-0000-4000-8000-000000000001",
    "status": "ACTIVE",
    "currency": "USD",
    "equity": "100000.25",
    "cash": "40000.5",
    "buying_power": "80001.0",
}

ORDER_ROW = {
    "id": "ord-1",
    "client_order_id": "rdq-1",
    "symbol": "AAPL",
    "qty": "10",
    "notional": None,
    "side": "buy",
    "type": "limit",
    "time_in_force": "day",
    "limit_price": "199.5",
    "status": "accepted",
    "filled_qty": "0",
    "filled_avg_price": None,
    "submitted_at": "2026-07-09T13:30:00.000000Z",
}


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeSession:
    """Returns queued responses and records every request() call."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {"method": method, "url": url, "params": params, "json": json, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError("FakeSession ran out of queued responses")
        return self._responses.pop(0)


def make_client(
    responses: list[FakeResponse], **kwargs: Any
) -> tuple[AlpacaClient, FakeSession, list[float]]:
    session = FakeSession(responses)
    sleeps: list[float] = []
    client = AlpacaClient(session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


class TestPaperOnlyGuard:
    def test_default_base_url_is_paper(self) -> None:
        client, _, _ = make_client([])
        assert client.base_url == "https://paper-api.alpaca.markets"
        assert "paper-api" in BASE_URL

    def test_live_host_is_refused(self) -> None:
        with pytest.raises(ValueError, match="paper-only"):
            AlpacaClient(base_url="https://api.alpaca.markets")

    def test_custom_non_live_base_url_allowed(self) -> None:
        client, _, _ = make_client([], base_url="http://127.0.0.1:8123/")
        assert client.base_url == "http://127.0.0.1:8123"

    def test_no_apca_headers_in_source(self) -> None:
        source = (
            pathlib.Path(__file__).resolve().parents[1] / "execution" / "alpaca_client.py"
        ).read_text()
        assert "APCA-API-KEY-ID" not in source.replace(
            "APCA-API-KEY-ID / APCA-API-SECRET-KEY header", ""
        )
        assert "headers=" not in source  # proxy injects credentials; client sends none


class TestAccount:
    def test_get_account_parses_snapshot(self) -> None:
        client, session, _ = make_client([FakeResponse(200, ACCOUNT_ROW)])
        account = client.get_account()
        assert session.calls == [
            {
                "method": "GET",
                "url": f"{BASE_URL}/v2/account",
                "params": None,
                "json": None,
                "timeout": 30.0,
            }
        ]
        assert account == Account(
            id="b3f2a9c1-0000-4000-8000-000000000001",
            status="ACTIVE",
            currency="USD",
            equity=100000.25,
            cash=40000.5,
            buying_power=80001.0,
        )

    def test_unparseable_numeric_field_raises(self) -> None:
        row = dict(ACCOUNT_ROW, equity="not-a-number")
        client, _, _ = make_client([FakeResponse(200, row)])
        with pytest.raises(AlpacaError, match="equity"):
            client.get_account()

    def test_visibility_fields_parse_when_present(self) -> None:
        row = dict(
            ACCOUNT_ROW,
            last_equity="99500.75",
            long_market_value="59999.75",
            short_market_value="0",
        )
        client, _, _ = make_client([FakeResponse(200, row)])
        account = client.get_account()
        assert account.last_equity == 99500.75
        assert account.long_market_value == 59999.75
        assert account.short_market_value == 0.0

    def test_visibility_fields_default_to_none(self) -> None:
        client, _, _ = make_client([FakeResponse(200, ACCOUNT_ROW)])
        account = client.get_account()
        assert account.last_equity is None
        assert account.long_market_value is None
        assert account.short_market_value is None


class TestPositions:
    def test_get_positions_parses_long_and_short(self) -> None:
        rows = [
            {
                "symbol": "AAPL",
                "qty": "10",
                "side": "long",
                "avg_entry_price": "190.25",
                "current_price": "200.5",
                "market_value": "2005",
            },
            {
                "symbol": "TSLA",
                "qty": "-5",
                "side": "short",
                "avg_entry_price": "300",
                "current_price": None,
                "market_value": None,
            },
        ]
        client, session, _ = make_client([FakeResponse(200, rows)])
        positions = client.get_positions()
        assert session.calls[0]["url"] == f"{BASE_URL}/v2/positions"
        assert positions == [
            Position("AAPL", 10.0, "long", 190.25, 200.5, 2005.0),
            Position("TSLA", -5.0, "short", 300.0, None, None),
        ]

    def test_unrealized_pl_fields_parse_when_present(self) -> None:
        row = {
            "symbol": "AAPL",
            "qty": "10",
            "side": "long",
            "avg_entry_price": "190.25",
            "current_price": "200.5",
            "market_value": "2005",
            "unrealized_pl": "102.5",
            "unrealized_plpc": "0.0539",
        }
        client, _, _ = make_client([FakeResponse(200, [row])])
        (position,) = client.get_positions()
        assert position.unrealized_pl == 102.5
        assert position.unrealized_plpc == 0.0539

    def test_flat_account_returns_empty_list(self) -> None:
        client, _, _ = make_client([FakeResponse(200, [])])
        assert client.get_positions() == []

    def test_non_list_payload_raises(self) -> None:
        client, _, _ = make_client([FakeResponse(200, {"message": "oops"})])
        with pytest.raises(AlpacaError, match="expected a JSON list"):
            client.get_positions()


class TestListOrders:
    def test_default_status_open(self) -> None:
        client, session, _ = make_client([FakeResponse(200, [ORDER_ROW])])
        orders = client.list_orders()
        assert session.calls[0]["params"] == {"status": "open"}
        assert orders == [
            Order(
                id="ord-1",
                client_order_id="rdq-1",
                symbol="AAPL",
                qty=10.0,
                notional=None,
                side="buy",
                order_type="limit",
                time_in_force="day",
                limit_price=199.5,
                status="accepted",
                filled_qty=0.0,
                filled_avg_price=None,
                submitted_at="2026-07-09T13:30:00.000000Z",
            )
        ]

    def test_filters_passed_as_params(self) -> None:
        client, session, _ = make_client([FakeResponse(200, [])])
        client.list_orders(status="closed", limit=50, symbols=["AAPL", "MSFT"])
        assert session.calls[0]["params"] == {
            "status": "closed",
            "limit": "50",
            "symbols": "AAPL,MSFT",
        }

    def test_after_until_bounds_passed_as_params(self) -> None:
        client, session, _ = make_client([FakeResponse(200, [])])
        client.list_orders(
            status="all", after="2026-07-09T04:00:00Z", until="2026-07-10T04:00:00Z"
        )
        assert session.calls[0]["params"] == {
            "status": "all",
            "after": "2026-07-09T04:00:00Z",
            "until": "2026-07-10T04:00:00Z",
        }


class TestPlaceOrder:
    def test_limit_order_payload(self) -> None:
        client, session, _ = make_client([FakeResponse(200, ORDER_ROW)])
        order = client.place_order(
            "AAPL", qty=10, side="buy", limit_price=199.5, client_order_id="rdq-1"
        )
        assert session.calls == [
            {
                "method": "POST",
                "url": f"{BASE_URL}/v2/orders",
                "params": None,
                "json": {
                    "symbol": "AAPL",
                    "qty": "10",
                    "side": "buy",
                    "type": "limit",
                    "time_in_force": "day",
                    "limit_price": "199.5",
                    "client_order_id": "rdq-1",
                },
                "timeout": 30.0,
            }
        ]
        assert order.id == "ord-1"
        assert order.status == "accepted"

    def test_market_order_payload_omits_limit_price(self) -> None:
        row = dict(ORDER_ROW, type="market", limit_price=None)
        client, session, _ = make_client([FakeResponse(200, row)])
        client.place_order("AAPL", qty=2.5, side="sell", order_type="market")
        assert session.calls[0]["json"] == {
            "symbol": "AAPL",
            "qty": "2.5",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"qty": 0, "side": "buy", "limit_price": 1.0}, "qty must be positive"),
            ({"qty": -1, "side": "buy", "limit_price": 1.0}, "qty must be positive"),
            ({"qty": 1, "side": "hold", "limit_price": 1.0}, "side must be one of"),
            ({"qty": 1, "side": "buy", "order_type": "stop"}, "order_type must be one of"),
            ({"qty": 1, "side": "buy"}, "limit orders require limit_price"),
            (
                {"qty": 1, "side": "buy", "order_type": "market", "limit_price": 1.0},
                "market orders must not set limit_price",
            ),
        ],
    )
    def test_invalid_order_rejected_before_any_http(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        client, session, _ = make_client([])
        with pytest.raises(ValueError, match=match):
            client.place_order("AAPL", **kwargs)
        assert session.calls == []


class TestCancelOrder:
    def test_cancel_returns_none_on_204(self) -> None:
        client, session, _ = make_client([FakeResponse(204)])
        assert client.cancel_order("ord-1") is None
        assert session.calls == [
            {
                "method": "DELETE",
                "url": f"{BASE_URL}/v2/orders/ord-1",
                "params": None,
                "json": None,
                "timeout": 30.0,
            }
        ]

    def test_uncancelable_order_raises_with_detail(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(422, {"code": 42210000, "message": "order is not cancelable"})]
        )
        with pytest.raises(AlpacaError, match="not cancelable"):
            client.cancel_order("ord-1")


class TestCancelAllOrders:
    def test_parses_207_multi_status_body(self) -> None:
        client, session, _ = make_client(
            [FakeResponse(207, [{"id": "ord-1", "status": 200}, {"id": "ord-2", "status": 500}])]
        )
        cancelled = client.cancel_all_orders()
        assert cancelled == [
            CancelledOrder(id="ord-1", status=200),
            CancelledOrder(id="ord-2", status=500),
        ]
        assert session.calls[0]["method"] == "DELETE"
        assert session.calls[0]["url"] == f"{BASE_URL}/v2/orders"

    def test_nothing_to_cancel(self) -> None:
        client, _, _ = make_client([FakeResponse(207, [])])
        assert client.cancel_all_orders() == []

    def test_204_body_means_empty(self) -> None:
        client, _, _ = make_client([FakeResponse(204)])
        assert client.cancel_all_orders() == []


class TestClosePosition:
    def test_returns_the_liquidation_order(self) -> None:
        client, session, _ = make_client(
            [FakeResponse(200, dict(ORDER_ROW, id="close-1", side="sell", type="market"))]
        )
        order = client.close_position("AAPL")
        assert order.id == "close-1"
        assert order.side == "sell"
        assert order.order_type == "market"
        assert session.calls == [
            {
                "method": "DELETE",
                "url": f"{BASE_URL}/v2/positions/AAPL",
                "params": None,
                "json": None,
                "timeout": 30.0,
            }
        ]

    def test_no_such_position_raises_with_detail(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(404, {"code": 40410000, "message": "position does not exist"})]
        )
        with pytest.raises(AlpacaError, match="position does not exist"):
            client.close_position("AAPL")


class TestGetCalendar:
    def test_parses_trading_days_and_passes_range(self) -> None:
        import datetime as dt

        client, session, _ = make_client(
            [FakeResponse(200, [{"date": "2026-07-09", "open": "09:30", "close": "16:00"}])]
        )
        days = client.get_calendar(dt.date(2026, 7, 9), dt.date(2026, 7, 9))
        assert days == [CalendarDay(date="2026-07-09", open="09:30", close="16:00")]
        assert session.calls[0]["url"] == f"{BASE_URL}/v2/calendar"
        assert session.calls[0]["params"] == {"start": "2026-07-09", "end": "2026-07-09"}

    def test_closed_day_returns_empty_list(self) -> None:
        import datetime as dt

        client, _, _ = make_client([FakeResponse(200, [])])
        assert client.get_calendar(dt.date(2026, 7, 4), dt.date(2026, 7, 4)) == []


class TestPortfolioHistory:
    # 2026-07-13/14 pre-open (13:30 UTC = 09:30 Eastern) daily points.
    HISTORY_ROW = {
        "timestamp": [1783949400, 1784035800],
        "equity": [100000.0, 100250.5],
        "profit_loss": [0.0, 250.5],
        "profit_loss_pct": [0.0, 0.002505],
        "base_value": 100000.0,
        "timeframe": "1D",
    }

    def test_parses_daily_entries_with_market_dates(self) -> None:
        client, session, _ = make_client([FakeResponse(200, self.HISTORY_ROW)])
        history = client.get_portfolio_history(period="1W")
        assert session.calls[0]["url"] == f"{BASE_URL}/v2/account/portfolio/history"
        assert session.calls[0]["params"] == {"period": "1W", "timeframe": "1D"}
        assert history.timeframe == "1D"
        assert history.base_value == 100000.0
        assert [e.date.isoformat() for e in history.entries] == ["2026-07-13", "2026-07-14"]
        assert history.entries[1] == PortfolioEntry(
            date=dt.date(2026, 7, 14),
            equity=100250.5,
            profit_loss=250.5,
            profit_loss_pct=0.002505,
        )

    def test_null_values_and_ragged_columns_become_none(self) -> None:
        row = {
            "timestamp": [1783949400, None, 1784035800],
            "equity": [None, 100000.0, 100250.5],
            "profit_loss": [0.0],
            "profit_loss_pct": None,
            "base_value": None,
            "timeframe": "1D",
        }
        client, _, _ = make_client([FakeResponse(200, row)])
        history = client.get_portfolio_history()
        # The null-timestamp point is dropped; missing columns read as None.
        assert len(history.entries) == 2
        assert history.entries[0].equity is None
        assert history.entries[0].profit_loss == 0.0
        assert history.entries[1].profit_loss is None
        assert history.entries[1].profit_loss_pct is None
        assert history.base_value is None

    def test_non_dict_payload_raises(self) -> None:
        client, _, _ = make_client([FakeResponse(200, [1, 2])])
        with pytest.raises(AlpacaError, match="expected a JSON object"):
            client.get_portfolio_history()


class TestErrorsAndRetries:
    def test_401_names_identity_and_setup_script(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(401, {"code": 40110000, "message": "access key verification failed"})]
        )
        with pytest.raises(AlpacaAuthError) as excinfo:
            client.get_account()
        message = str(excinfo.value)
        assert "rdq-exec-paper" in message
        assert "ops/setup_onecli.sh" in message
        assert "access key verification failed" in message

    def test_403_also_raises_auth_error(self) -> None:
        client, _, _ = make_client([FakeResponse(403, {"message": "forbidden"})])
        with pytest.raises(AlpacaAuthError):
            client.get_positions()

    def test_403_with_401_family_code_is_still_auth(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(403, {"code": 40110000, "message": "access key verification failed"})]
        )
        with pytest.raises(AlpacaAuthError):
            client.get_positions()

    def test_403_business_rejection_is_not_an_auth_error(self) -> None:
        # 40310000 "insufficient buying power" is a 403 on a validly-authed
        # request (the 2026-07-23 abort): it must NOT tell the operator to
        # re-vault working credentials.
        client, _, _ = make_client(
            [FakeResponse(403, {"code": 40310000, "message": "insufficient buying power"})]
        )
        with pytest.raises(AlpacaError) as excinfo:
            client.get_positions()
        assert not isinstance(excinfo.value, AlpacaAuthError)
        message = str(excinfo.value)
        assert "insufficient buying power" in message
        assert "ops/setup_onecli.sh" not in message

    def test_429_honors_retry_after_then_succeeds(self) -> None:
        client, session, sleeps = make_client(
            [
                FakeResponse(429, headers={"Retry-After": "3"}),
                FakeResponse(200, ACCOUNT_ROW),
            ]
        )
        account = client.get_account()
        assert account.id == ACCOUNT_ROW["id"]
        assert sleeps == [3.0]
        assert len(session.calls) == 2

    def test_429_exponential_fallback_without_header(self) -> None:
        client, _, sleeps = make_client(
            [FakeResponse(429), FakeResponse(429), FakeResponse(200, ACCOUNT_ROW)]
        )
        client.get_account()
        assert sleeps == [1.0, 2.0]

    def test_429_exhaustion_raises_rate_limit_error(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(429) for _ in range(5)], max_retries=4
        )
        with pytest.raises(AlpacaRateLimitError, match="after 4 retries"):
            client.get_account()

    def test_5xx_is_never_retried(self) -> None:
        client, session, sleeps = make_client(
            [FakeResponse(500, {"message": "internal server error"})]
        )
        with pytest.raises(AlpacaError, match="HTTP 500"):
            client.get_account()
        assert sleeps == []
        assert len(session.calls) == 1

    def test_error_detail_falls_back_to_text(self) -> None:
        client, _, _ = make_client([FakeResponse(500, text="plain body")])
        with pytest.raises(AlpacaError, match="plain body"):
            client.get_account()


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RDQ_LIVE_TESTS") != "1",
    reason="live proxy smoke; run with RDQ_LIVE_TESTS=1 under `onecli run --agent rdq-exec-paper`",
)
class TestLiveSmoke:
    def test_get_account_returns_account_id(self) -> None:
        client = AlpacaClient()
        account = client.get_account()
        assert account.id
        assert account.currency == "USD"
        assert account.equity > 0

"""Unit tests for ops/flatten.py (mocked Alpaca) plus a live proxy smoke test."""

from __future__ import annotations

import os
import pathlib
from typing import Any
from urllib.parse import urlparse

import pytest

from execution.alpaca_client import AlpacaClient
from ops.flatten import main, run_flatten


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = ""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def order_row(order_id: str, symbol: str, side: str, status: str = "accepted") -> dict[str, Any]:
    return {
        "id": order_id,
        "client_order_id": f"cid-{order_id}",
        "symbol": symbol,
        "qty": "5",
        "notional": None,
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "100",
        "status": status,
        "filled_qty": "0",
        "filled_avg_price": None,
        "submitted_at": "2026-07-09T13:30:00Z",
    }


def position_row(symbol: str, qty: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "qty": str(qty),
        "side": "long" if qty >= 0 else "short",
        "avg_entry_price": "100",
        "current_price": "101",
        "market_value": str(qty * 101),
    }


class FlattenBroker:
    """Routed mock of the flatten-relevant Alpaca API, driven via the REAL client.

    ``cancel_settle_polls`` / ``close_settle_polls`` = how many GET polls
    after the respective DELETE still show the old state before it settles
    (None = never settles, the timeout path).
    """

    def __init__(
        self,
        open_orders: list[dict[str, Any]] | None = None,
        positions: list[dict[str, Any]] | None = None,
        cancel_settle_polls: int | None = 0,
        close_settle_polls: int | None = 0,
    ) -> None:
        self.open_orders = list(open_orders or [])
        self.positions = list(positions or [])
        self.cancel_settle_polls = cancel_settle_polls
        self.close_settle_polls = close_settle_polls
        self.cancel_all_called = False
        self.closed_symbols: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        path = urlparse(url).path
        self.calls.append((method, path))
        if (method, path) == ("GET", "/v2/orders"):
            if self.cancel_all_called and self.cancel_settle_polls is not None:
                if self.cancel_settle_polls <= 0:
                    return FakeResponse(200, [])
                self.cancel_settle_polls -= 1
            return FakeResponse(200, self.open_orders)
        if (method, path) == ("DELETE", "/v2/orders"):
            self.cancel_all_called = True
            return FakeResponse(
                207, [{"id": row["id"], "status": 200} for row in self.open_orders]
            )
        if (method, path) == ("GET", "/v2/positions"):
            if len(self.closed_symbols) == len(self.positions) and self.positions:
                if self.close_settle_polls is None:
                    return FakeResponse(200, self.positions)
                if self.close_settle_polls <= 0:
                    return FakeResponse(200, [])
                self.close_settle_polls -= 1
            return FakeResponse(200, self.positions)
        if method == "DELETE" and path.startswith("/v2/positions/"):
            symbol = path.rsplit("/", 1)[1]
            self.closed_symbols.append(symbol)
            row = next(p for p in self.positions if p["symbol"] == symbol)
            close_side = "sell" if float(row["qty"]) >= 0 else "buy"
            return FakeResponse(200, order_row(f"close-{symbol}", symbol, close_side))
        raise AssertionError(f"unexpected request: {method} {path}")


def run(broker: FlattenBroker, **kwargs: Any) -> tuple[int, list[str], list[float]]:
    client = AlpacaClient(session=broker)
    lines: list[str] = []
    sleeps: list[float] = []
    code = run_flatten(
        client,
        out=lines.append,
        sleep=sleeps.append,
        timeout_seconds=kwargs.pop("timeout_seconds", 10.0),
        poll_interval_seconds=kwargs.pop("poll_interval_seconds", 1.0),
    )
    return code, lines, sleeps


class TestCancelCloseVerifySequence:
    def test_full_sequence_cancels_then_closes_then_verifies(self) -> None:
        broker = FlattenBroker(
            open_orders=[order_row("o1", "AAPL", "buy")],
            positions=[position_row("AAPL", 5), position_row("TSLA", -3)],
        )
        code, lines, _ = run(broker)
        assert code == 0
        # Cancel happened, and strictly before any liquidation.
        assert broker.cancel_all_called
        cancel_idx = broker.calls.index(("DELETE", "/v2/orders"))
        close_idxs = [
            i for i, c in enumerate(broker.calls) if c[0] == "DELETE" and "positions" in c[1]
        ]
        assert close_idxs and min(close_idxs) > cancel_idx
        # Every position was closed (the short too), then verified empty.
        assert sorted(broker.closed_symbols) == ["AAPL", "TSLA"]
        assert broker.calls[-1] == ("GET", "/v2/positions")
        assert lines[-1] == "OK: /v2/positions confirmed empty — account is flat"

    def test_short_position_closed_with_buy(self) -> None:
        broker = FlattenBroker(positions=[position_row("TSLA", -3)])
        code, lines, _ = run(broker)
        assert code == 0
        assert any("closing TSLA: short -3.0 -> buy order close-TSLA" in line for line in lines)

    def test_already_flat_account_makes_no_deletes(self) -> None:
        broker = FlattenBroker()
        code, lines, sleeps = run(broker)
        assert code == 0
        assert all(method != "DELETE" for method, _ in broker.calls)
        assert sleeps == []
        assert "no open orders" in lines
        assert "no open positions" in lines

    def test_slow_cancel_is_polled_until_settled(self) -> None:
        broker = FlattenBroker(open_orders=[order_row("o1", "AAPL", "buy")], cancel_settle_polls=2)
        code, lines, sleeps = run(broker)
        assert code == 0
        assert len(sleeps) == 2
        assert any("open order(s) remaining" in line for line in lines)

    def test_orders_never_cancelling_aborts_before_liquidation(self) -> None:
        broker = FlattenBroker(
            open_orders=[order_row("o1", "AAPL", "buy")],
            positions=[position_row("AAPL", 5)],
            cancel_settle_polls=None,
        )
        code, lines, _ = run(broker, timeout_seconds=3.0)
        assert code == 1
        assert broker.closed_symbols == []
        assert any("still open" in line for line in lines)

    def test_positions_never_emptying_exits_1_naming_symbols(self) -> None:
        broker = FlattenBroker(
            positions=[position_row("AAPL", 5), position_row("MSFT", 2)],
            close_settle_polls=None,
        )
        code, lines, sleeps = run(broker, timeout_seconds=3.0)
        assert code == 1
        assert sorted(broker.closed_symbols) == ["AAPL", "MSFT"]
        assert len(sleeps) == 3
        assert "AAPL, MSFT" in lines[-1]
        assert "market is closed" in lines[-1]


class TestCli:
    def test_auth_failure_exits_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class AuthFailSession:
            def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
                return FakeResponse(403, {"message": "forbidden"})

        monkeypatch.setattr(
            "ops.flatten.AlpacaClient", lambda: AlpacaClient(session=AuthFailSession())
        )
        assert main([]) == 2

    def test_rejects_non_positive_intervals(self) -> None:
        with pytest.raises(SystemExit):
            main(["--poll-interval-seconds", "0"])


class TestRunbook:
    def test_runbook_covers_required_procedures(self) -> None:
        runbook = (pathlib.Path(__file__).resolve().parents[1] / "ops" / "runbook.md").read_text()
        assert "Pause the research loop" in runbook
        assert "Halt the rebalancer" in runbook
        assert "Flatten positions" in runbook
        assert "ops.flatten" in runbook
        assert "Rotate keys via OneCLI" in runbook
        assert "tailscale serve status" in runbook


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RDQ_LIVE_TESTS") != "1",
    reason="live proxy smoke test; set RDQ_LIVE_TESTS=1 and run under onecli",
)
class TestLiveSmoke:
    def test_flatten_confirms_flat_paper_account(self) -> None:
        client = AlpacaClient()
        positions = client.get_positions()
        if positions:
            pytest.skip(
                "paper account holds positions — refusing to liquidate a live book "
                "from the test suite; run `python -m ops.flatten` deliberately instead"
            )
        lines: list[str] = []
        assert run_flatten(client, out=lines.append) == 0
        assert lines[-1] == "OK: /v2/positions confirmed empty — account is flat"

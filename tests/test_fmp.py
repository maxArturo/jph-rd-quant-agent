"""Unit tests for data/fmp.py (mocked HTTP) plus a live proxy smoke test."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pytest

from data.fmp import (
    BASE_URL,
    Dividend,
    EodBar,
    FmpAuthError,
    FmpClient,
    FmpError,
    FmpRateLimitError,
    Split,
)


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
        return self._payload


class FakeSession:
    """Returns queued responses and records every GET call."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeSession ran out of queued responses")
        return self._responses.pop(0)


def make_client(
    responses: list[FakeResponse], **kwargs: Any
) -> tuple[FmpClient, FakeSession, list[float]]:
    session = FakeSession(responses)
    sleeps: list[float] = []
    client = FmpClient(session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


BAR_ROWS = [
    {
        "symbol": "AAPL",
        "date": "2026-05-04",
        "open": 210.0,
        "high": 212.5,
        "low": 208.0,
        "close": 211.0,
        "volume": 50_000_000,
    },
    {
        "symbol": "AAPL",
        "date": "2026-05-01",
        "open": 205.0,
        "high": 209.0,
        "low": 204.0,
        "close": 208.5,
        "volume": 48_000_000,
    },
]


class TestDateWindowing:
    def test_from_to_params_sent(self) -> None:
        client, session, _ = make_client([FakeResponse(200, BAR_ROWS)])
        client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        (call,) = session.calls
        assert call["url"] == f"{BASE_URL}/historical-price-eod/full"
        assert call["params"] == {"symbol": "AAPL", "from": "2026-05-01", "to": "2026-05-31"}

    def test_date_objects_accepted(self) -> None:
        client, session, _ = make_client([FakeResponse(200, [])])
        client.get_eod_bars("AAPL", date(2026, 5, 1), date(2026, 5, 31))
        assert session.calls[0]["params"]["from"] == "2026-05-01"
        assert session.calls[0]["params"]["to"] == "2026-05-31"

    def test_start_after_end_rejected(self) -> None:
        client, session, _ = make_client([])
        with pytest.raises(ValueError, match="after end"):
            client.get_eod_bars("AAPL", "2026-06-01", "2026-05-01")
        assert session.calls == []  # rejected before any HTTP

    def test_malformed_date_rejected(self) -> None:
        client, _, _ = make_client([])
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            client.get_eod_bars("AAPL", "05/01/2026", "2026-05-31")

    def test_bars_parsed_and_sorted_ascending(self) -> None:
        client, _, _ = make_client([FakeResponse(200, BAR_ROWS)])
        bars = client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert bars == [
            EodBar("AAPL", date(2026, 5, 1), 205.0, 209.0, 204.0, 208.5, 48_000_000.0),
            EodBar("AAPL", date(2026, 5, 4), 210.0, 212.5, 208.0, 211.0, 50_000_000.0),
        ]


class TestRateLimitBackoff:
    def test_429_honors_retry_after_then_succeeds(self) -> None:
        client, session, sleeps = make_client(
            [
                FakeResponse(429, headers={"Retry-After": "7"}),
                FakeResponse(200, BAR_ROWS),
            ]
        )
        bars = client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert len(bars) == 2
        assert sleeps == [7.0]
        assert len(session.calls) == 2

    def test_429_without_retry_after_backs_off_exponentially(self) -> None:
        client, _, sleeps = make_client(
            [FakeResponse(429), FakeResponse(429), FakeResponse(200, [])]
        )
        client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert sleeps == [2.0, 4.0]

    def test_429_exhausts_retry_budget(self) -> None:
        client, session, sleeps = make_client([FakeResponse(429)] * 3, max_retries=2)
        with pytest.raises(FmpRateLimitError, match="429"):
            client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert len(session.calls) == 3  # initial try + 2 retries
        assert len(sleeps) == 2

    def test_non_numeric_retry_after_falls_back_to_exponential(self) -> None:
        client, _, sleeps = make_client(
            [
                FakeResponse(429, headers={"Retry-After": "Wed, 08 Jul 2026 21:00:00 GMT"}),
                FakeResponse(200, []),
            ]
        )
        client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert sleeps == [2.0]


class TestAuthErrors:
    @pytest.mark.parametrize("status", [401, 403])
    def test_actionable_missing_secret_message(self, status: int) -> None:
        client, _, _ = make_client(
            [FakeResponse(status, text='{"Error Message": "Invalid API KEY."}')]
        )
        with pytest.raises(FmpAuthError) as excinfo:
            client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        message = str(excinfo.value)
        assert "onecli run --agent rdq-research" in message
        assert "ops/setup_onecli.sh" in message
        assert "financialmodelingprep.com" in message
        assert str(status) in message

    def test_other_http_error_raises_fmp_error(self) -> None:
        client, _, _ = make_client([FakeResponse(500, text="boom")])
        with pytest.raises(FmpError, match="HTTP 500"):
            client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")

    def test_non_list_payload_raises(self) -> None:
        client, _, _ = make_client([FakeResponse(200, {"Error Message": "nope"})])
        with pytest.raises(FmpError, match="expected a JSON list"):
            client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")


class TestSplitsAndDividends:
    def test_splits_endpoint_and_parsing(self) -> None:
        rows = [
            {"symbol": "NVDA", "date": "2024-06-10", "numerator": 10, "denominator": 1},
            {"symbol": "NVDA", "date": "2021-07-20", "numerator": 4, "denominator": 1},
        ]
        client, session, _ = make_client([FakeResponse(200, rows)])
        splits = client.get_splits("NVDA")
        assert session.calls[0]["url"] == f"{BASE_URL}/splits"
        assert session.calls[0]["params"] == {"symbol": "NVDA"}
        assert splits == [
            Split("NVDA", date(2021, 7, 20), 4.0, 1.0),
            Split("NVDA", date(2024, 6, 10), 10.0, 1.0),
        ]
        assert splits[1].ratio == 10.0

    def test_dividends_endpoint_and_parsing(self) -> None:
        rows = [
            {"symbol": "AAPL", "date": "2026-05-11", "dividend": 0.26},
            {"symbol": "AAPL", "date": "2026-02-10", "dividend": 0.25},
        ]
        client, session, _ = make_client([FakeResponse(200, rows)])
        dividends = client.get_dividends("AAPL")
        assert session.calls[0]["url"] == f"{BASE_URL}/dividends"
        assert dividends == [
            Dividend("AAPL", date(2026, 2, 10), 0.25),
            Dividend("AAPL", date(2026, 5, 11), 0.26),
        ]


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RDQ_LIVE_TESTS") != "1",
    reason="live proxy smoke; run with RDQ_LIVE_TESTS=1 under `onecli run --agent rdq-research`",
)
class TestLiveSmoke:
    def test_aapl_one_month_window_returns_bars(self) -> None:
        client = FmpClient()
        bars = client.get_eod_bars("AAPL", "2026-05-01", "2026-05-31")
        assert len(bars) > 15
        assert all(bar.symbol == "AAPL" for bar in bars)
        assert all(bar.close > 0 for bar in bars)
        assert bars == sorted(bars, key=lambda bar: bar.date)
        assert all(date(2026, 5, 1) <= bar.date <= date(2026, 5, 31) for bar in bars)

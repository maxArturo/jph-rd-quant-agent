"""Unit tests for data/fmp.py (mocked HTTP) plus a live proxy smoke test."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pytest

from data.fmp import (
    BASE_URL,
    TREASURY_CHUNK_DAYS,
    CommodityEod,
    Dividend,
    EodBar,
    FmpAuthError,
    FmpClient,
    FmpError,
    FmpRateLimitError,
    Split,
    TreasuryCurve,
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


# Recorded from live /stable/historical-price-eod/light responses (US-064 probe,
# docs/decisions.md 2026-08-24): rows are {symbol, date, price, volume} — no OHLC.
# Newest-first, matching the FMP list-endpoint quirk.
COMMODITY_ROWS = [
    {"symbol": "BZUSD", "date": "2025-01-06", "price": 76.30, "volume": 31_542},
    {"symbol": "BZUSD", "date": "2025-01-03", "price": 76.51, "volume": 28_907},
    {"symbol": "BZUSD", "date": "2025-01-02", "price": 75.93, "volume": 26_412},
]


def treasury_row(iso_date: str, year10: float = 4.60) -> dict[str, Any]:
    """A /stable/treasury-rates row as recorded by the US-064 probe."""
    return {
        "date": iso_date,
        "month1": 4.44,
        "month2": 4.42,
        "month3": 4.35,
        "month6": 4.25,
        "year1": 4.17,
        "year2": 4.28,
        "year3": 4.35,
        "year5": 4.41,
        "year7": 4.48,
        "year10": year10,
        "year20": 4.86,
        "year30": 4.82,
    }


class TestCommodityEod:
    def test_light_endpoint_and_params(self) -> None:
        client, session, _ = make_client([FakeResponse(200, COMMODITY_ROWS)])
        client.get_commodity_eod("BZUSD", "2025-01-02", "2025-01-31")
        (call,) = session.calls
        assert call["url"] == f"{BASE_URL}/historical-price-eod/light"
        assert call["params"] == {"symbol": "BZUSD", "from": "2025-01-02", "to": "2025-01-31"}

    def test_rows_parsed_and_sorted_ascending(self) -> None:
        client, _, _ = make_client([FakeResponse(200, COMMODITY_ROWS)])
        points = client.get_commodity_eod("BZUSD", "2025-01-02", "2025-01-31")
        assert points == [
            CommodityEod("BZUSD", date(2025, 1, 2), 75.93, 26_412.0),
            CommodityEod("BZUSD", date(2025, 1, 3), 76.51, 28_907.0),
            CommodityEod("BZUSD", date(2025, 1, 6), 76.30, 31_542.0),
        ]

    def test_missing_volume_defaults_to_zero(self) -> None:
        rows = [{"symbol": "DXUSD", "date": "2025-01-02", "price": 108.44}]
        client, _, _ = make_client([FakeResponse(200, rows)])
        (point,) = client.get_commodity_eod("DXUSD", "2025-01-02", "2025-01-02")
        assert point.volume == 0.0

    def test_start_after_end_rejected(self) -> None:
        client, session, _ = make_client([])
        with pytest.raises(ValueError, match="after end"):
            client.get_commodity_eod("BZUSD", "2025-02-01", "2025-01-01")
        assert session.calls == []

    def test_429_retries_with_backoff(self) -> None:
        client, session, sleeps = make_client(
            [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, COMMODITY_ROWS)]
        )
        points = client.get_commodity_eod("BZUSD", "2025-01-02", "2025-01-31")
        assert len(points) == 3
        assert sleeps == [3.0]
        assert len(session.calls) == 2

    def test_auth_error_typed(self) -> None:
        client, _, _ = make_client([FakeResponse(401, text="denied")])
        with pytest.raises(FmpAuthError, match="onecli run --agent rdq-research"):
            client.get_commodity_eod("BZUSD", "2025-01-02", "2025-01-31")

    def test_non_list_payload_raises(self) -> None:
        client, _, _ = make_client([FakeResponse(200, {"Error Message": "nope"})])
        with pytest.raises(FmpError, match="expected a JSON list"):
            client.get_commodity_eod("BZUSD", "2025-01-02", "2025-01-31")


class TestTreasuryRates:
    def test_short_window_single_request(self) -> None:
        client, session, _ = make_client([FakeResponse(200, [treasury_row("2025-01-02")])])
        curves = client.get_treasury_rates("2025-01-02", "2025-01-31")
        (call,) = session.calls
        assert call["url"] == f"{BASE_URL}/treasury-rates"
        assert call["params"] == {"from": "2025-01-02", "to": "2025-01-31"}
        assert curves == [
            TreasuryCurve(
                date=date(2025, 1, 2),
                month1=4.44,
                month2=4.42,
                month3=4.35,
                month6=4.25,
                year1=4.17,
                year2=4.28,
                year3=4.35,
                year5=4.41,
                year7=4.48,
                year10=4.60,
                year20=4.86,
                year30=4.82,
            )
        ]

    def test_wide_window_chunks_at_90_calendar_days(self) -> None:
        # 2025-01-02 -> 2025-05-01 is 120 calendar days: two chunks expected.
        client, session, _ = make_client(
            [
                FakeResponse(200, [treasury_row("2025-04-01"), treasury_row("2025-01-02")]),
                FakeResponse(200, [treasury_row("2025-05-01"), treasury_row("2025-04-02")]),
            ]
        )
        curves = client.get_treasury_rates("2025-01-02", "2025-05-01")
        assert [call["params"] for call in session.calls] == [
            {"from": "2025-01-02", "to": "2025-04-01"},  # 90 calendar days inclusive
            {"from": "2025-04-02", "to": "2025-05-01"},
        ]
        # Merged result is contiguous across the chunk boundary, ascending.
        assert [curve.date for curve in curves] == [
            date(2025, 1, 2),
            date(2025, 4, 1),
            date(2025, 4, 2),
            date(2025, 5, 1),
        ]

    def test_chunk_size_stays_at_or_under_cap(self) -> None:
        responses = [FakeResponse(200, []) for _ in range(5)]
        client, session, _ = make_client(responses)
        client.get_treasury_rates("2025-01-02", "2026-01-02")
        assert len(session.calls) > 1
        covered_start = None
        for call in session.calls:
            start = date.fromisoformat(call["params"]["from"])
            end = date.fromisoformat(call["params"]["to"])
            assert (end - start).days + 1 <= TREASURY_CHUNK_DAYS
            if covered_start is not None:
                assert start == covered_start  # chunks are contiguous, no gaps
            covered_start = end + timedelta(days=1)
        assert session.calls[0]["params"]["from"] == "2025-01-02"
        assert session.calls[-1]["params"]["to"] == "2026-01-02"

    def test_boundary_duplicate_dates_deduplicated(self) -> None:
        # Both chunks report the boundary date 2025-04-01: exactly one row survives.
        client, _, _ = make_client(
            [
                FakeResponse(200, [treasury_row("2025-04-01", year10=4.10)]),
                FakeResponse(200, [treasury_row("2025-04-02"), treasury_row("2025-04-01")]),
            ]
        )
        curves = client.get_treasury_rates("2025-01-02", "2025-05-01")
        assert [curve.date for curve in curves] == [date(2025, 4, 1), date(2025, 4, 2)]

    def test_missing_tenor_is_none(self) -> None:
        row = treasury_row("2025-01-02")
        del row["month2"]
        row["year20"] = None
        client, _, _ = make_client([FakeResponse(200, [row])])
        (curve,) = client.get_treasury_rates("2025-01-02", "2025-01-31")
        assert curve.month2 is None
        assert curve.year20 is None
        assert curve.year10 == 4.60

    def test_start_after_end_rejected(self) -> None:
        client, session, _ = make_client([])
        with pytest.raises(ValueError, match="after end"):
            client.get_treasury_rates("2025-02-01", "2025-01-01")
        assert session.calls == []

    def test_http_error_mid_chunk_raises(self) -> None:
        client, _, _ = make_client(
            [FakeResponse(200, [treasury_row("2025-04-01")]), FakeResponse(500, text="boom")]
        )
        with pytest.raises(FmpError, match="HTTP 500"):
            client.get_treasury_rates("2025-01-02", "2025-05-01")

    def test_429_retries_with_backoff(self) -> None:
        client, _, sleeps = make_client(
            [FakeResponse(429), FakeResponse(200, [treasury_row("2025-01-02")])]
        )
        curves = client.get_treasury_rates("2025-01-02", "2025-01-31")
        assert len(curves) == 1
        assert sleeps == [2.0]


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

    def test_brent_one_month_window_returns_prices(self) -> None:
        client = FmpClient()
        points = client.get_commodity_eod("BZUSD", "2026-05-01", "2026-05-31")
        assert len(points) > 15
        assert all(point.price > 0 for point in points)
        assert points == sorted(points, key=lambda point: point.date)

    def test_treasury_rates_chunked_over_120_day_window(self) -> None:
        # >90 calendar days forces chunking; the merged result must span the
        # whole window (a truncated fetch would only cover the trailing chunk).
        client = FmpClient()
        curves = client.get_treasury_rates("2026-01-02", "2026-05-01")
        assert len(curves) > 70  # ~80 trading days in the window
        assert curves[0].date < date(2026, 2, 1)  # head survived the merge
        assert curves[-1].date > date(2026, 4, 15)
        assert all(curve.year10 is not None and curve.year10 > 0 for curve in curves)
        dates = [curve.date for curve in curves]
        assert dates == sorted(set(dates))  # ascending, boundary dates deduped

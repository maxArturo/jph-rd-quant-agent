"""US-012: split/dividend adjustment factor computation (data/adjust.py)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from data.adjust import AdjustmentError, adjusted_closes, adjustment_factors
from data.fmp import Dividend, EodBar, Split


def make_bar(day: date, close: float, symbol: str = "TEST") -> EodBar:
    return EodBar(
        symbol=symbol,
        date=day,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000.0,
    )


def make_bars(start: date, closes: list[float], symbol: str = "TEST") -> list[EodBar]:
    return [
        make_bar(start + timedelta(days=offset), close, symbol)
        for offset, close in enumerate(closes)
    ]


class TestNoEvents:
    def test_no_event_ticker_yields_factor_one_throughout(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 101.0, 102.0, 103.0])
        factors = adjustment_factors(bars)
        assert factors == {bar.date: 1.0 for bar in bars}

    def test_empty_bars_yield_empty_factors(self) -> None:
        assert adjustment_factors([]) == {}


class TestSplits:
    def test_ten_for_one_split_produces_point_one_factor_step(self) -> None:
        bars = make_bars(date(2024, 1, 1), [1000.0, 1010.0, 101.0, 102.0])
        split = Split(symbol="TEST", date=date(2024, 1, 3), numerator=10.0, denominator=1.0)
        factors = adjustment_factors(bars, splits=[split])
        assert factors[date(2024, 1, 1)] == pytest.approx(0.1)
        assert factors[date(2024, 1, 2)] == pytest.approx(0.1)
        # Days on/after the ex-date are unadjusted.
        assert factors[date(2024, 1, 3)] == 1.0
        assert factors[date(2024, 1, 4)] == 1.0

    def test_consecutive_splits_compound(self) -> None:
        bars = make_bars(date(2024, 1, 1), [400.0, 200.0, 201.0, 100.0, 101.0])
        splits = [
            Split(symbol="TEST", date=date(2024, 1, 2), numerator=2.0, denominator=1.0),
            Split(symbol="TEST", date=date(2024, 1, 4), numerator=2.0, denominator=1.0),
        ]
        factors = adjustment_factors(bars, splits=splits)
        assert factors[date(2024, 1, 1)] == pytest.approx(0.25)
        assert factors[date(2024, 1, 2)] == pytest.approx(0.5)
        assert factors[date(2024, 1, 3)] == pytest.approx(0.5)
        assert factors[date(2024, 1, 4)] == 1.0

    def test_reverse_split_raises_factor(self) -> None:
        # 1:5 reverse split: numerator/denominator = 0.2, factor before = 5.0.
        bars = make_bars(date(2024, 1, 1), [2.0, 10.0])
        split = Split(symbol="TEST", date=date(2024, 1, 2), numerator=1.0, denominator=5.0)
        factors = adjustment_factors(bars, splits=[split])
        assert factors[date(2024, 1, 1)] == pytest.approx(5.0)

    def test_non_positive_split_ratio_raises(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 100.0])
        split = Split(symbol="TEST", date=date(2024, 1, 2), numerator=0.0, denominator=1.0)
        with pytest.raises(AdjustmentError, match="non-positive split ratio"):
            adjustment_factors(bars, splits=[split])


class TestDividends:
    def test_dividend_adjusts_proportionally_to_prev_close(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 99.0])
        dividend = Dividend(symbol="TEST", date=date(2024, 1, 2), dividend=2.0)
        factors = adjustment_factors(bars, dividends=[dividend])
        # prev_close = 100.0 on Jan 1 -> (100 - 2) / 100 = 0.98 before ex-date.
        assert factors[date(2024, 1, 1)] == pytest.approx(0.98)
        assert factors[date(2024, 1, 2)] == 1.0

    def test_dividend_uses_last_close_strictly_before_ex_date(self) -> None:
        # Ex-date falls on a non-bar day (weekend); prev_close is Jan 1's 50.0.
        bars = [make_bar(date(2024, 1, 1), 50.0), make_bar(date(2024, 1, 5), 49.0)]
        dividend = Dividend(symbol="TEST", date=date(2024, 1, 3), dividend=1.0)
        factors = adjustment_factors(bars, dividends=[dividend])
        assert factors[date(2024, 1, 1)] == pytest.approx((50.0 - 1.0) / 50.0)
        assert factors[date(2024, 1, 5)] == 1.0

    def test_multiple_dividends_compound(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 100.0, 100.0])
        dividends = [
            Dividend(symbol="TEST", date=date(2024, 1, 2), dividend=1.0),
            Dividend(symbol="TEST", date=date(2024, 1, 3), dividend=2.0),
        ]
        factors = adjustment_factors(bars, dividends=dividends)
        assert factors[date(2024, 1, 1)] == pytest.approx(0.99 * 0.98)
        assert factors[date(2024, 1, 2)] == pytest.approx(0.98)
        assert factors[date(2024, 1, 3)] == 1.0

    def test_dividend_gte_prev_close_raises(self) -> None:
        bars = make_bars(date(2024, 1, 1), [1.0, 1.0])
        dividend = Dividend(symbol="TEST", date=date(2024, 1, 2), dividend=1.5)
        with pytest.raises(AdjustmentError, match="previous close"):
            adjustment_factors(bars, dividends=[dividend])

    def test_negative_dividend_raises(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 100.0])
        dividend = Dividend(symbol="TEST", date=date(2024, 1, 2), dividend=-0.5)
        with pytest.raises(AdjustmentError, match="negative dividend"):
            adjustment_factors(bars, dividends=[dividend])


class TestCombinedAndEdgeCases:
    def test_split_and_dividend_compound(self) -> None:
        bars = make_bars(date(2024, 1, 1), [100.0, 100.0, 50.0, 50.0])
        dividend = Dividend(symbol="TEST", date=date(2024, 1, 2), dividend=2.0)
        split = Split(symbol="TEST", date=date(2024, 1, 3), numerator=2.0, denominator=1.0)
        factors = adjustment_factors(bars, splits=[split], dividends=[dividend])
        assert factors[date(2024, 1, 1)] == pytest.approx(0.98 * 0.5)
        assert factors[date(2024, 1, 2)] == pytest.approx(0.5)
        assert factors[date(2024, 1, 3)] == 1.0

    def test_events_outside_window_are_ignored(self) -> None:
        bars = make_bars(date(2024, 6, 1), [100.0, 101.0, 102.0])
        events_outside = [
            # On the first bar date: affects only days before the window.
            Split(symbol="TEST", date=date(2024, 6, 1), numerator=2.0, denominator=1.0),
            # After the last bar: announced/future ex-date, must not adjust.
            Split(symbol="TEST", date=date(2024, 7, 1), numerator=10.0, denominator=1.0),
        ]
        dividends_outside = [Dividend(symbol="TEST", date=date(2024, 8, 1), dividend=1.0)]
        factors = adjustment_factors(bars, splits=events_outside, dividends=dividends_outside)
        assert all(factor == 1.0 for factor in factors.values())

    def test_unsorted_bars_and_events_are_handled(self) -> None:
        bars = make_bars(date(2024, 1, 1), [1000.0, 101.0, 102.0])
        split = Split(symbol="TEST", date=date(2024, 1, 2), numerator=10.0, denominator=1.0)
        factors = adjustment_factors(list(reversed(bars)), splits=[split])
        assert factors[date(2024, 1, 1)] == pytest.approx(0.1)
        assert factors[date(2024, 1, 2)] == 1.0

    def test_duplicate_bar_dates_raise(self) -> None:
        bars = [make_bar(date(2024, 1, 1), 100.0), make_bar(date(2024, 1, 1), 101.0)]
        with pytest.raises(AdjustmentError, match="duplicate bar dates"):
            adjustment_factors(bars)

    def test_adjusted_closes_returns_ascending_scaled_series(self) -> None:
        bars = make_bars(date(2024, 1, 1), [1000.0, 101.0])
        split = Split(symbol="TEST", date=date(2024, 1, 2), numerator=10.0, denominator=1.0)
        series = adjusted_closes(list(reversed(bars)), splits=[split])
        assert series == [
            (date(2024, 1, 1), pytest.approx(100.0)),
            (date(2024, 1, 2), pytest.approx(101.0)),
        ]


# Real NVDA raw closes around the 2024-06-10 10:1 split (approximate values;
# the raw series cliffs ~-90% across the split, the adjusted one must not).
NVDA_RAW_CLOSES = [
    (date(2024, 6, 3), 1150.00),
    (date(2024, 6, 4), 1164.37),
    (date(2024, 6, 5), 1224.40),
    (date(2024, 6, 6), 1209.98),
    (date(2024, 6, 7), 1208.88),
    (date(2024, 6, 10), 121.79),  # first post-split close
    (date(2024, 6, 11), 120.91),
    (date(2024, 6, 12), 125.20),
    (date(2024, 6, 13), 129.61),
    (date(2024, 6, 14), 131.88),
]
NVDA_SPLIT = Split(symbol="NVDA", date=date(2024, 6, 10), numerator=10.0, denominator=1.0)


def daily_pct_moves(series: list[tuple[date, float]]) -> list[float]:
    return [
        abs(series[i][1] / series[i - 1][1] - 1.0)
        for i in range(1, len(series))
    ]


class TestNvdaSplitFixture:
    def test_raw_closes_cliff_across_split_date(self) -> None:
        raw = [(day, close) for day, close in NVDA_RAW_CLOSES]
        assert max(daily_pct_moves(raw)) > 0.20

    def test_adjusted_closes_show_no_cliff_across_split_date(self) -> None:
        bars = [make_bar(day, close, "NVDA") for day, close in NVDA_RAW_CLOSES]
        adjusted = adjusted_closes(bars, splits=[NVDA_SPLIT])
        assert max(daily_pct_moves(adjusted)) < 0.20

    def test_pre_split_factor_is_point_one(self) -> None:
        bars = [make_bar(day, close, "NVDA") for day, close in NVDA_RAW_CLOSES]
        factors = adjustment_factors(bars, splits=[NVDA_SPLIT])
        assert factors[date(2024, 6, 7)] == pytest.approx(0.1)
        assert factors[date(2024, 6, 10)] == 1.0

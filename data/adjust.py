"""Split/dividend price adjustment factors.

FMP closes are RAW (unadjusted); Qlib needs backward-adjusted prices. This
module turns the splits + dividends returned by data/fmp.py into a per-day
multiplicative adjustment factor for a ticker's bar window:

    adjusted_close(day) = raw_close(day) * factor(day)

Convention (standard backward adjustment):
- The factor on the last bar of the window is 1.0; walking backwards in time,
  crossing an event's ex-date multiplies the running factor.
- A split with ratio r (10:1 -> r = 10) multiplies the factor for all days
  strictly before its ex-date by 1/r (the 0.1 "factor step").
- A cash dividend D with ex-date d multiplies the factor for all days strictly
  before d by (prev_close - D) / prev_close, where prev_close is the raw close
  on the last bar before d (proportional adjustment).
- Events dated outside the bar window are ignored: on/before the first bar
  they affect no bar in the window; after the last bar they are typically
  announced-but-future ex-dates and must not adjust today's prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from data.fmp import Dividend, EodBar, Split


class AdjustmentError(ValueError):
    """Raised when event or bar data cannot produce a valid factor series."""


def adjustment_factors(
    bars: Sequence[EodBar],
    splits: Sequence[Split] = (),
    dividends: Sequence[Dividend] = (),
) -> dict[date, float]:
    """Per-day adjustment factor for every bar date, keyed by date.

    Accepts bars/events in any order. Raises AdjustmentError on duplicate bar
    dates, non-positive split ratios, or a dividend >= its previous close
    (which would produce a non-positive price).
    """
    if not bars:
        return {}
    ordered = sorted(bars, key=lambda bar: bar.date)
    if len({bar.date for bar in ordered}) != len(ordered):
        raise AdjustmentError(f"duplicate bar dates for {ordered[0].symbol}")

    last_date = ordered[-1].date
    events: list[Split | Dividend] = sorted(
        (event for event in [*splits, *dividends] if event.date <= last_date),
        key=lambda event: event.date,
        reverse=True,
    )

    factor = 1.0
    factors: dict[date, float] = {}
    index = 0
    for bar in reversed(ordered):
        # Consume every event with ex-date after this bar: `bar` is the last
        # bar strictly before those ex-dates, so its raw close is the
        # dividend's prev_close.
        while index < len(events) and events[index].date > bar.date:
            event = events[index]
            if isinstance(event, Split):
                if event.ratio <= 0:
                    raise AdjustmentError(
                        f"non-positive split ratio {event.ratio} on {event.date} "
                        f"for {event.symbol}"
                    )
                factor /= event.ratio
            else:
                if event.dividend < 0:
                    raise AdjustmentError(
                        f"negative dividend {event.dividend} on {event.date} "
                        f"for {event.symbol}"
                    )
                if bar.close <= event.dividend:
                    raise AdjustmentError(
                        f"dividend {event.dividend} on {event.date} for {event.symbol} "
                        f">= previous close {bar.close} on {bar.date}"
                    )
                factor *= (bar.close - event.dividend) / bar.close
            index += 1
        factors[bar.date] = factor
    return factors


def adjusted_closes(
    bars: Sequence[EodBar],
    splits: Sequence[Split] = (),
    dividends: Sequence[Dividend] = (),
) -> list[tuple[date, float]]:
    """(date, adjusted_close) ascending by date: raw close * that day's factor."""
    factors = adjustment_factors(bars, splits, dividends)
    return [
        (bar.date, bar.close * factors[bar.date])
        for bar in sorted(bars, key=lambda bar: bar.date)
    ]

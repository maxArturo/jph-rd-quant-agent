"""Incremental refresh of an existing Qlib US store from FMP (US-036).

Pulls only the bars each ticker is missing (since its own last stored date),
refetches the full split/dividend history, recomputes adjustment factors over
the merged series, and rebuilds the store through build_store's
temp -> validate -> atomic-swap path. Custom instruments files written by
data/make_universe.py are carried across the rebuild with their date spans
refreshed, inside the same atomic swap.

Raw bars are recovered from the store itself (the field conventions make raw
values recoverable: raw price = stored adjusted price / factor, raw volume =
stored volume * factor), so a refresh needs no full FMP re-backfill and stays
correct when a NEW split or dividend lands between refreshes — the whole
factor series is recomputed, re-scaling history exactly like a fresh build.
Round-tripping through the float32 bins costs ~1e-7 relative noise per
rebuild; negligible against price data.

Idempotency: when no ticker has anything new to pull (window empty, or FMP
returns no bars — weekend, holiday), the store is left byte-for-byte
untouched and the CLI exits 0 with an "already current" notice.

The default --end is *yesterday* in America/New_York, never today: during an
open session FMP's EOD endpoint can return a partial bar for today, which
must not be stored as a settled close. The pre-open refresh timer only ever
needs the previous session's bar.

Run under `onecli run --agent rdq-exec-paper` (or rdq-research) so the proxy
injects the FMP key.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from data.adjust import AdjustmentError
from data.build_store import (
    DEFAULT_STORE_PATH,
    FIELDS,
    FREQ,
    MARKET_ALL,
    BuildError,
    TickerBundle,
    build_store,
)
from data.fmp import DateLike, EodBar, FmpClient, FmpError, _to_iso_date

MARKET_TZ = ZoneInfo("America/New_York")


class RefreshError(RuntimeError):
    """Raised when the existing store cannot be read or safely refreshed."""


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one refresh: whether the store was rebuilt and what changed."""

    updated: bool
    last_date_before: date
    last_date_after: date
    new_bars: dict[str, int]  # symbol -> number of appended bars


def default_end() -> date:
    """Yesterday in America/New_York — the last date whose bar can be settled."""
    return datetime.now(MARKET_TZ).date() - timedelta(days=1)


# ---------------------------------------------------------------------------
# Reading the existing store back into raw bars


def read_calendar(store: Path) -> list[date]:
    path = store / "calendars" / f"{FREQ}.txt"
    if not path.exists():
        raise RefreshError(
            f"no store at {store} (missing {path}); build one first with data/build_store.py"
        )
    days = [date.fromisoformat(line) for line in path.read_text().splitlines() if line.strip()]
    if not days:
        raise RefreshError(f"store calendar {path} is empty")
    return days


def read_all_symbols(store: Path) -> list[str]:
    path = store / "instruments" / f"{MARKET_ALL}.txt"
    if not path.exists():
        raise RefreshError(f"store at {store} has no instruments file {path}")
    symbols = [line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()]
    if not symbols:
        raise RefreshError(f"instruments file {path} lists no tickers")
    return symbols


def read_universes(store: Path) -> dict[str, list[str]]:
    """Every instruments file except all.txt, as name -> ordered ticker list."""
    universes: dict[str, list[str]] = {}
    for path in sorted((store / "instruments").glob("*.txt")):
        if path.stem == MARKET_ALL:
            continue
        universes[path.stem] = [
            line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()
        ]
    return universes


def _read_field(feature_dir: Path, field: str) -> tuple[int, np.ndarray]:
    path = feature_dir / f"{field}.{FREQ}.bin"
    if not path.exists():
        raise RefreshError(f"missing feature file {path}")
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        raise RefreshError(f"{path} has no values")
    return int(data[0]), data[1:]


def read_raw_bars(store: Path, symbol: str, calendar: list[date]) -> tuple[EodBar, ...]:
    """Reconstruct the raw (unadjusted) bars for one ticker from its bins."""
    feature_dir = store / "features" / symbol.lower()
    arrays: dict[str, np.ndarray] = {}
    start_index = -1
    span = -1
    for field in FIELDS:
        index, values = _read_field(feature_dir, field)
        if start_index == -1:
            start_index, span = index, len(values)
        elif index != start_index or len(values) != span:
            raise RefreshError(f"{symbol} feature bins disagree on span; store is corrupt")
        arrays[field] = values
    if start_index < 0 or start_index + span > len(calendar):
        raise RefreshError(f"{symbol} span exceeds the store calendar; store is corrupt")
    bars: list[EodBar] = []
    for i in range(span):
        day = calendar[start_index + i]
        close = float(arrays["close"][i])
        factor = float(arrays["factor"][i])
        if math.isnan(close) or math.isnan(factor) or factor <= 0:
            raise RefreshError(
                f"{symbol} has NaN/invalid close or factor on {day.isoformat()}; "
                "refusing to refresh a corrupt store"
            )
        bars.append(
            EodBar(
                symbol=symbol,
                date=day,
                open=float(arrays["open"][i]) / factor,
                high=float(arrays["high"][i]) / factor,
                low=float(arrays["low"][i]) / factor,
                close=close / factor,
                volume=float(arrays["volume"][i]) * factor,
            )
        )
    return tuple(bars)


# ---------------------------------------------------------------------------
# Refresh


def refresh_store(
    store: Path, client: FmpClient, end: DateLike | None = None
) -> RefreshResult:
    """Pull bars since each ticker's last stored date and rebuild if anything landed."""
    store = store.expanduser()
    end_date = date.fromisoformat(_to_iso_date(end if end is not None else default_end(), "end"))
    calendar = read_calendar(store)
    symbols = read_all_symbols(store)
    universes = read_universes(store)
    existing = {symbol: read_raw_bars(store, symbol, calendar) for symbol in symbols}
    last_before = calendar[-1]

    new_bars: dict[str, list[EodBar]] = {}
    for symbol in symbols:
        last = existing[symbol][-1].date
        window_start = last + timedelta(days=1)
        if window_start > end_date:
            continue
        fetched = client.get_eod_bars(symbol, window_start, end_date)
        fresh = sorted(
            (bar for bar in fetched if last < bar.date <= end_date), key=lambda b: b.date
        )
        if fresh:
            new_bars[symbol] = fresh

    if not new_bars:
        return RefreshResult(False, last_before, last_before, {})

    bundles = [
        TickerBundle(
            symbol=symbol,
            bars=existing[symbol] + tuple(new_bars.get(symbol, ())),
            splits=tuple(client.get_splits(symbol)),
            dividends=tuple(client.get_dividends(symbol)),
        )
        for symbol in symbols
    ]
    build_store(bundles, store, extra_instruments=universes)
    last_after = max(bundle.bars[-1].date for bundle in bundles)
    return RefreshResult(
        True, last_before, last_after, {s: len(bars) for s, bars in new_bars.items()}
    )


# ---------------------------------------------------------------------------
# Extending the store with new tickers (on-demand universe backfill)


@dataclass(frozen=True)
class ExtendResult:
    """Outcome of one extend: bars added per new symbol, symbols excluded."""

    added: dict[str, int]  # new symbol -> number of bars written
    missing: tuple[str, ...]  # requested symbols FMP returned no bars for
    gapped: tuple[str, ...] = ()  # symbols whose FMP bars have mid-series holes


def extend_store(
    store: Path, client: FmpClient, symbols: Sequence[str], end: DateLike | None = None
) -> ExtendResult:
    """Add new tickers to an existing store via full-history FMP backfill.

    Existing tickers are carried across unchanged (raw bars recovered from
    the bins); each requested symbol is fetched from the store's first
    calendar date through ``end`` (default: the store's own last calendar
    date, so new tickers never run ahead of the rest — the nightly refresh
    advances everyone together) and merged through build_store's atomic
    temp -> validate -> swap path, custom universes preserved. Split and
    dividend history is refetched for every ticker because the rebuild
    recomputes all adjustment factors — same trade-off refresh_store makes.

    The store calendar is an invariant: bars on days the store has never
    seen (foreign-venue history on US holidays — dual listings like GLXY
    carry TSX days) are dropped, so an extend can never add calendar days
    or punch NaN holes into existing tickers' spans. A symbol whose
    surviving bars still have holes inside their own span (vendor gaps,
    long halts) would fail store validation, so it is excluded and
    reported in ``gapped`` rather than poisoning the whole batch.

    Symbols FMP has no bars for are reported in ``missing`` and written
    nowhere. Symbols already in the store are skipped (added=0, not missing).
    """
    store = store.expanduser()
    calendar = read_calendar(store)
    calendar_days = set(calendar)
    existing_symbols = read_all_symbols(store)
    universes = read_universes(store)
    end_date = date.fromisoformat(_to_iso_date(end if end is not None else calendar[-1], "end"))

    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    new_symbols = [s for s in requested if s not in set(existing_symbols)]
    if not new_symbols:
        return ExtendResult(added={}, missing=())

    fetched: dict[str, list[EodBar]] = {}
    missing: list[str] = []
    gapped: list[str] = []
    for symbol in new_symbols:
        bars = sorted(
            (
                b
                for b in client.get_eod_bars(symbol, calendar[0], end_date)
                if b.date <= end_date and b.date in calendar_days
            ),
            key=lambda b: b.date,
        )
        if not bars:
            missing.append(symbol)
            continue
        bar_days = {b.date for b in bars}
        span_days = [d for d in calendar if bars[0].date <= d <= bars[-1].date]
        if any(day not in bar_days for day in span_days):
            gapped.append(symbol)
            continue
        fetched[symbol] = bars

    if not fetched:
        return ExtendResult(added={}, missing=tuple(missing), gapped=tuple(gapped))

    existing = {symbol: read_raw_bars(store, symbol, calendar) for symbol in existing_symbols}
    bundles = [
        TickerBundle(
            symbol=symbol,
            bars=existing[symbol],
            splits=tuple(client.get_splits(symbol)),
            dividends=tuple(client.get_dividends(symbol)),
        )
        for symbol in existing_symbols
    ] + [
        TickerBundle(
            symbol=symbol,
            bars=tuple(bars),
            splits=tuple(client.get_splits(symbol)),
            dividends=tuple(client.get_dividends(symbol)),
        )
        for symbol, bars in fetched.items()
    ]
    build_store(bundles, store, extra_instruments=universes)
    return ExtendResult(
        added={symbol: len(bars) for symbol, bars in fetched.items()},
        missing=tuple(missing),
        gapped=tuple(gapped),
    )


# ---------------------------------------------------------------------------
# CLI


def main(argv: Any = None, client: FmpClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh the Qlib US store from FMP "
        "(run under `onecli run --agent rdq-exec-paper` so the proxy injects the FMP key)."
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE_PATH,
        help=f"store directory to refresh (default {DEFAULT_STORE_PATH})",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="last bar date to pull, YYYY-MM-DD (default: yesterday in America/New_York — "
        "never today; an in-progress session would land a partial bar)",
    )
    parser.add_argument(
        "--add-tickers",
        default=None,
        metavar="SYM,SYM,...",
        help="extend the store with these new tickers (full-history backfill aligned to the"
        " store's calendar) instead of refreshing existing ones",
    )
    args = parser.parse_args(argv)
    fmp = client if client is not None else FmpClient()
    if args.add_tickers is not None:
        try:
            extended = extend_store(
                Path(args.store), fmp, args.add_tickers.split(","), args.end
            )
        except (RefreshError, BuildError, FmpError, AdjustmentError, ValueError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        for symbol, count in sorted(extended.added.items()):
            print(f"added {symbol}: {count} bars")
        failed = False
        if extended.missing:
            print(
                f"ERROR: FMP has no data for: {' '.join(extended.missing)}", file=sys.stderr
            )
            failed = True
        if extended.gapped:
            print(
                "ERROR: FMP bars have mid-series gaps (excluded) for: "
                f"{' '.join(extended.gapped)}",
                file=sys.stderr,
            )
            failed = True
        if failed:
            return 1
        if not extended.added:
            print("nothing to do: every requested ticker is already in the store")
        return 0
    try:
        result = refresh_store(Path(args.store), fmp, args.end)
    except (RefreshError, BuildError, FmpError, AdjustmentError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.updated:
        total = sum(result.new_bars.values())
        print(
            f"store refreshed at {Path(args.store).expanduser()}: +{total} bars across "
            f"{len(result.new_bars)} tickers "
            f"({result.last_date_before.isoformat()} -> {result.last_date_after.isoformat()})"
        )
    else:
        print(f"store already current (last date {result.last_date_before.isoformat()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

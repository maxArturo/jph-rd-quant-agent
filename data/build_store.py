"""Qlib US bin-store builder: FMP bars -> adjusted OHLCV -> dump_bin layout.

Turns a ticker list into a Qlib data store (calendars/day.txt,
instruments/all.txt, features/<sym>/<field>.day.bin) at the target path
(default ~/.qlib/qlib_data/us_data).

Reliability model:
- The FMP backfill is checkpointed per ticker (JSON under <output>.checkpoint/):
  a crash mid-list resumes on rerun without refetching or duplicating tickers.
  A checkpoint is only reused when its (start, end) window matches.
- The store is written to a temp dir next to the target, validated, then
  atomically swapped in. A failed build never leaves a partial store behind.

Field conventions (Qlib backward adjustment, see data/adjust.py):
- factor(day) per data/adjust.py; open/high/low/close stored ADJUSTED
  (raw * factor); volume stored raw / factor; the raw close is recoverable
  as close / factor.
- Market broadcast series ($mkt_*, US-066) are stored RAW on every
  instrument's row — identical value across instruments per date, implicit
  factor 1, never touched by the ticker's adjustment math. Equity days with
  no market observation forward-fill from the last observation; days before
  a series' first observation are NaN.
- Bin format matches qlib FileFeatureStorage: little-endian float32 array
  whose first element is the calendar index of the ticker's first bar.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from data.adjust import AdjustmentError, adjustment_factors
from data.fmp import DateLike, Dividend, EodBar, FmpClient, FmpError, Split, _to_iso_date

FREQ = "day"
FIELDS = ("open", "high", "low", "close", "volume", "factor")
MARKET_ALL = "all"
DEFAULT_STORE_PATH = "~/.qlib/qlib_data/us_data"

# Market-level broadcast series (US-066): one value per date, written RAW onto
# every instrument's row ($mkt_* fields) and never touched by the ticker's
# $factor adjustment. Commodity fields map to FMP /historical-price-eod/light
# symbols; mkt_y10 is the year10 tenor of /stable/treasury-rates.
COMMODITY_SYMBOLS: dict[str, str] = {
    "mkt_brent": "BZUSD",
    "mkt_wti": "CLUSD",
    "mkt_heatoil": "HOUSD",
    "mkt_natgas": "NGUSD",
    "mkt_gasoline": "RBUSD",
    "mkt_gold": "GCUSD",
    "mkt_dxy": "DXUSD",
}
TREASURY_FIELD = "mkt_y10"
MARKET_FIELDS = (*COMMODITY_SYMBOLS, TREASURY_FIELD)
# Canonical backfill start for the market series (probe-verified 2026-08-24,
# docs/decisions.md US-064) — FMP coverage before this date is unverified.
MARKET_SERIES_START = date(2025, 1, 2)
# Each series must directly observe at least this share of the trading days
# since its own first observation, or the build fails loud (forward-fill is
# for the odd commodity holiday, not for masking a broken feed).
MARKET_COVERAGE_MIN = 0.99

MarketSeriesMap = Mapping[str, Sequence[tuple[date, float]]]


class BuildError(RuntimeError):
    """Raised when the store cannot be built from the fetched data."""


class StoreValidationError(BuildError):
    """Raised when a freshly written store fails validation (no swap happens)."""


@dataclass(frozen=True)
class TickerBundle:
    """Everything fetched for one ticker: raw bars plus adjustment events."""

    symbol: str
    bars: tuple[EodBar, ...]
    splits: tuple[Split, ...]
    dividends: tuple[Dividend, ...]


FetchFn = Callable[[str], TickerBundle]


def fetch_bundle(client: FmpClient, symbol: str, start: DateLike, end: DateLike) -> TickerBundle:
    """Fetch bars + splits + dividends for one ticker through the FMP client."""
    return TickerBundle(
        symbol=symbol,
        bars=tuple(client.get_eod_bars(symbol, start, end)),
        splits=tuple(client.get_splits(symbol)),
        dividends=tuple(client.get_dividends(symbol)),
    )


def fetch_market_series(
    client: FmpClient, start: DateLike, end: DateLike
) -> dict[str, tuple[tuple[date, float], ...]]:
    """Fetch every $mkt_* series as (date, value) observations through FMP.

    Commodity prices come from get_commodity_eod per COMMODITY_SYMBOLS; mkt_y10
    is the year10 tenor of get_treasury_rates (days FMP reports no 10y value for
    are skipped — the store build forward-fills over them).
    """
    series: dict[str, tuple[tuple[date, float], ...]] = {}
    for field, symbol in COMMODITY_SYMBOLS.items():
        rows = client.get_commodity_eod(symbol, start, end)
        series[field] = tuple((row.date, row.price) for row in rows)
    curves = client.get_treasury_rates(start, end)
    series[TREASURY_FIELD] = tuple(
        (curve.date, curve.year10) for curve in curves if curve.year10 is not None
    )
    return series


# ---------------------------------------------------------------------------
# Checkpointing


def _bundle_to_json(bundle: TickerBundle, start_iso: str, end_iso: str) -> dict[str, Any]:
    return {
        "symbol": bundle.symbol,
        "start": start_iso,
        "end": end_iso,
        "bars": [
            [b.date.isoformat(), b.open, b.high, b.low, b.close, b.volume] for b in bundle.bars
        ],
        "splits": [[s.date.isoformat(), s.numerator, s.denominator] for s in bundle.splits],
        "dividends": [[d.date.isoformat(), d.dividend] for d in bundle.dividends],
    }


def _bundle_from_json(payload: dict[str, Any]) -> TickerBundle:
    symbol = str(payload["symbol"])
    return TickerBundle(
        symbol=symbol,
        bars=tuple(
            EodBar(symbol, date.fromisoformat(r[0]), r[1], r[2], r[3], r[4], r[5])
            for r in payload["bars"]
        ),
        splits=tuple(
            Split(symbol, date.fromisoformat(r[0]), r[1], r[2]) for r in payload["splits"]
        ),
        dividends=tuple(
            Dividend(symbol, date.fromisoformat(r[0]), r[1]) for r in payload["dividends"]
        ),
    )


def _load_checkpoint(path: Path, start_iso: str, end_iso: str) -> TickerBundle | None:
    """Return the checkpointed bundle, or None if absent/window-mismatched/corrupt."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("start") != start_iso or payload.get("end") != end_iso:
            return None
        return _bundle_from_json(payload)
    except (ValueError, KeyError, IndexError, TypeError):
        return None  # corrupt checkpoint: refetch


def _write_checkpoint(path: Path, bundle: TickerBundle, start_iso: str, end_iso: str) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_bundle_to_json(bundle, start_iso, end_iso)))
    os.replace(tmp, path)


def backfill(
    symbols: Sequence[str],
    fetch: FetchFn,
    checkpoint_dir: Path,
    start: DateLike,
    end: DateLike,
) -> list[TickerBundle]:
    """Fetch every symbol, checkpointing each ticker as it lands.

    Already-checkpointed tickers (same date window) are not refetched, so a
    rerun after a crash continues where the previous run stopped.
    """
    start_iso = _to_iso_date(start, "start")
    end_iso = _to_iso_date(end, "end")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bundles: list[TickerBundle] = []
    for symbol in symbols:
        ckpt = checkpoint_dir / f"{symbol}.json"
        bundle = _load_checkpoint(ckpt, start_iso, end_iso)
        if bundle is None:
            bundle = fetch(symbol)
            _write_checkpoint(ckpt, bundle, start_iso, end_iso)
        bundles.append(bundle)
    return bundles


# ---------------------------------------------------------------------------
# Store writing


def _feature_series(bundle: TickerBundle) -> dict[str, list[tuple[date, float]]]:
    """Per-field (date, value) series for one ticker, adjusted per data/adjust.py."""
    factors = adjustment_factors(bundle.bars, bundle.splits, bundle.dividends)
    series: dict[str, list[tuple[date, float]]] = {field: [] for field in FIELDS}
    for bar in sorted(bundle.bars, key=lambda b: b.date):
        factor = factors[bar.date]
        if factor <= 0:
            raise BuildError(f"non-positive adjustment factor {factor} for {bundle.symbol}")
        series["open"].append((bar.date, bar.open * factor))
        series["high"].append((bar.date, bar.high * factor))
        series["low"].append((bar.date, bar.low * factor))
        series["close"].append((bar.date, bar.close * factor))
        series["volume"].append((bar.date, bar.volume / factor))
        series["factor"].append((bar.date, factor))
    return series


def _market_matrix(
    calendar: Sequence[date], name: str, observations: Sequence[tuple[date, float]]
) -> np.ndarray:
    """Full-calendar values for one market series, ready to broadcast.

    Trading days with no observation of their own (commodity holiday) take the
    last observation — including off-calendar ones (some series print on
    weekends). Days before the series' first observation stay NaN. Direct
    coverage below MARKET_COVERAGE_MIN of the trading days since the first
    observation fails the build loudly with the series named.
    """
    if name in FIELDS or not name.isidentifier() or name != name.lower():
        raise BuildError(
            f"invalid market series name {name!r}: must be a lowercase identifier "
            f"(no '$' prefix) and must not collide with the per-ticker fields {FIELDS}"
        )
    if not observations:
        raise BuildError(f"market series {name} has no observations")
    by_date: dict[date, float] = {}
    for day, value in observations:
        if not math.isfinite(value):
            raise BuildError(
                f"market series {name} has a non-finite value on {day.isoformat()}"
            )
        by_date[day] = value
    obs_days = sorted(by_date)
    span_days = [d for d in calendar if d >= obs_days[0]]
    if not span_days:
        raise BuildError(
            f"market series {name} starts {obs_days[0].isoformat()}, "
            "after the store calendar ends"
        )
    covered = sum(1 for d in span_days if d in by_date)
    coverage = covered / len(span_days)
    if coverage < MARKET_COVERAGE_MIN:
        raise StoreValidationError(
            f"market series {name} directly covers {covered}/{len(span_days)} trading days "
            f"({coverage:.1%}) since {obs_days[0].isoformat()}; "
            f"minimum is {MARKET_COVERAGE_MIN:.0%}"
        )
    values = np.full(len(calendar), np.nan)
    for i, day in enumerate(calendar):
        pos = bisect.bisect_right(obs_days, day) - 1
        if pos >= 0:
            values[i] = by_date[obs_days[pos]]
    return values


def _write_bin(path: Path, start_index: int, values: Sequence[float] | np.ndarray) -> None:
    np.hstack([np.array([start_index], dtype="<f"), np.asarray(values, dtype="<f")]).astype(
        "<f"
    ).tofile(path)


def build_store(
    bundles: Sequence[TickerBundle],
    target: Path,
    extra_instruments: Mapping[str, Sequence[tuple[str, str, str]]] | None = None,
    market_series: MarketSeriesMap | None = None,
) -> None:
    """Write a Qlib bin store for the bundles: temp dir -> validate -> swap.

    extra_instruments maps additional universe names to their (symbol, start,
    end) rows — data/refresh.py uses this to carry make_universe files across
    a rebuild inside the same atomic swap. Rows are written verbatim (the
    caller owns span semantics: refresh.refresh_universe_spans advances only
    ends that tracked their ticker's end), so multi-span point-in-time
    universes survive a rebuild intact.

    market_series maps broadcast field names (e.g. "mkt_brent", no '$') to
    their (date, value) observations. Each series is forward-filled onto the
    equity calendar (see _market_matrix) and written RAW into every
    instrument's feature dir — identical value across instruments per date,
    never touched by the ticker's $factor adjustment.
    """
    if not bundles:
        raise BuildError("no tickers to build a store from")
    for bundle in bundles:
        if not bundle.bars:
            raise BuildError(f"no bars fetched for {bundle.symbol}; refusing to build")
    seen: set[str] = set()
    for bundle in bundles:
        if bundle.symbol in seen:
            raise BuildError(f"duplicate ticker {bundle.symbol} in bundle list")
        seen.add(bundle.symbol)

    calendar = sorted({bar.date for bundle in bundles for bar in bundle.bars})
    positions = {day: idx for idx, day in enumerate(calendar)}
    market_matrix = {
        name: _market_matrix(calendar, name, observations)
        for name, observations in (market_series or {}).items()
    }

    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f"{target.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        (tmp / "calendars").mkdir(parents=True)
        (tmp / "instruments").mkdir()
        (tmp / "calendars" / f"{FREQ}.txt").write_text(
            "".join(f"{day.isoformat()}\n" for day in calendar)
        )
        instrument_lines = []
        spans: dict[str, tuple[date, date]] = {}
        for bundle in bundles:
            ordered = sorted(bundle.bars, key=lambda b: b.date)
            first, last = ordered[0].date, ordered[-1].date
            spans[bundle.symbol] = (first, last)
            instrument_lines.append(f"{bundle.symbol}\t{first.isoformat()}\t{last.isoformat()}\n")
            feature_dir = tmp / "features" / bundle.symbol.lower()
            feature_dir.mkdir(parents=True)
            start_index = positions[first]
            span = positions[last] - start_index + 1
            for field, points in _feature_series(bundle).items():
                values = np.full(span, np.nan)
                for day, value in points:
                    values[positions[day] - start_index] = value
                _write_bin(feature_dir / f"{field}.{FREQ}.bin", start_index, values)
            for name, matrix in market_matrix.items():
                _write_bin(
                    feature_dir / f"{name}.{FREQ}.bin",
                    start_index,
                    matrix[start_index : start_index + span],
                )
        (tmp / "instruments" / f"{MARKET_ALL}.txt").write_text("".join(instrument_lines))
        for name, universe_rows in (extra_instruments or {}).items():
            if name == MARKET_ALL:
                raise BuildError(f"universe name {name!r} is reserved for the full store")
            unknown = sorted({s for s, _, _ in universe_rows if s not in spans})
            if unknown:
                raise BuildError(
                    f"universe {name!r} references tickers not in the store: {unknown}"
                )
            (tmp / "instruments" / f"{name}.txt").write_text(
                "".join(f"{s}\t{start}\t{end}\n" for s, start, end in universe_rows)
            )
        validate_store(
            tmp, [bundle.symbol for bundle in bundles], market_fields=tuple(market_matrix)
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _swap_into_place(tmp, target)


def _swap_into_place(tmp: Path, target: Path) -> None:
    backup = target.parent / f"{target.name}.old"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    tmp.rename(target)
    if backup.exists():
        shutil.rmtree(backup)


# ---------------------------------------------------------------------------
# Validation


def validate_store(
    store_dir: Path, symbols: Sequence[str], market_fields: Sequence[str] = ()
) -> None:
    """Assert the store at store_dir is complete and readable for symbols.

    Checks calendar ordering, instruments coverage, per-field bin presence,
    index bounds, and that no ticker has a NaN close/factor inside its own
    [first, last] span (a mid-series gap means bad source data - fail loudly
    rather than ship a store Qlib will silently propagate NaNs from).
    market_fields lists broadcast series every instrument must also carry a
    bin for (their NaN heads before the series' first observation are legal).
    """
    calendar_path = store_dir / "calendars" / f"{FREQ}.txt"
    if not calendar_path.exists():
        raise StoreValidationError(f"missing calendar file {calendar_path}")
    days = [date.fromisoformat(line) for line in calendar_path.read_text().splitlines() if line]
    if not days:
        raise StoreValidationError("calendar is empty")
    if any(b <= a for a, b in zip(days, days[1:], strict=False)):
        raise StoreValidationError("calendar dates are not strictly ascending")

    instruments_path = store_dir / "instruments" / f"{MARKET_ALL}.txt"
    if not instruments_path.exists():
        raise StoreValidationError(f"missing instruments file {instruments_path}")
    listed = {
        line.split("\t")[0] for line in instruments_path.read_text().splitlines() if line.strip()
    }
    missing = set(symbols) - listed
    extra = listed - set(symbols)
    if missing or extra:
        raise StoreValidationError(
            f"instruments mismatch: missing={sorted(missing)} unexpected={sorted(extra)}"
        )

    for symbol in symbols:
        feature_dir = store_dir / "features" / symbol.lower()
        for field in (*FIELDS, *market_fields):
            bin_path = feature_dir / f"{field}.{FREQ}.bin"
            if not bin_path.exists():
                raise StoreValidationError(f"missing feature file {bin_path}")
            data = np.fromfile(bin_path, dtype="<f")
            if len(data) < 2:
                raise StoreValidationError(f"{bin_path} has no values")
            start_index = int(data[0])
            if start_index < 0 or start_index + len(data) - 1 > len(days):
                raise StoreValidationError(
                    f"{bin_path} index range [{start_index}, {start_index + len(data) - 2}] "
                    f"exceeds calendar length {len(days)}"
                )
            if field in ("close", "factor") and any(math.isnan(v) for v in data[1:]):
                raise StoreValidationError(
                    f"{symbol} has NaN {field} inside its date span (mid-series gap in "
                    "source bars); refusing to ship a gapped store"
                )


# ---------------------------------------------------------------------------
# CLI


def resolve_tickers(tickers: str | None, tickers_file: str | None) -> list[str]:
    """Uppercased, de-duplicated (order-preserving) ticker list from CLI args."""
    if (tickers is None) == (tickers_file is None):
        raise BuildError("provide exactly one of --tickers or --tickers-file")
    if tickers is not None:
        raw = tickers.split(",")
    else:
        raw = Path(tickers_file or "").read_text().split()
    out: list[str] = []
    for item in raw:
        symbol = item.strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    if not out:
        raise BuildError("ticker list is empty")
    return out


def build_from_fmp(
    symbols: Sequence[str],
    start: DateLike,
    end: DateLike,
    output: Path,
    checkpoint_dir: Path | None = None,
    client: FmpClient | None = None,
    market_start: DateLike | None = None,
) -> None:
    """Backfill symbols from FMP (checkpointed) and build the store at output.

    When market_start is given, every $mkt_* series is also fetched over
    [market_start, end] and broadcast into the store.
    """
    output = output.expanduser()
    if checkpoint_dir is None:
        checkpoint_dir = output.parent / f"{output.name}.checkpoint"
    fmp = client if client is not None else FmpClient()
    bundles = backfill(
        symbols, lambda s: fetch_bundle(fmp, s, start, end), checkpoint_dir, start, end
    )
    market = fetch_market_series(fmp, market_start, end) if market_start is not None else None
    build_store(bundles, output, market_series=market)


def main(argv: Sequence[str] | None = None, client: FmpClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Qlib US bin store from FMP EOD data "
        "(run under `onecli run --agent rdq-research` so the proxy injects the FMP key)."
    )
    parser.add_argument("--tickers", help="comma-separated ticker list, e.g. AAPL,MSFT")
    parser.add_argument("--tickers-file", help="file with one ticker per line")
    parser.add_argument("--start", required=True, help="backfill start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="backfill end date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=DEFAULT_STORE_PATH,
        help=f"store directory (default {DEFAULT_STORE_PATH})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="per-ticker fetch checkpoints (default <output>.checkpoint)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard existing checkpoints and refetch everything",
    )
    parser.add_argument(
        "--market-start",
        default=None,
        metavar="YYYY-MM-DD",
        help="also fetch the $mkt_* market series from this date and broadcast them "
        f"into the store (canonical start {MARKET_SERIES_START.isoformat()}; "
        "default: no market series)",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else output.parent / f"{output.name}.checkpoint"
    )
    try:
        symbols = resolve_tickers(args.tickers, args.tickers_file)
        if args.fresh and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        build_from_fmp(
            symbols,
            args.start,
            args.end,
            output,
            checkpoint_dir,
            client,
            market_start=args.market_start,
        )
    except (BuildError, FmpError, AdjustmentError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"store built at {output} ({len(symbols)} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

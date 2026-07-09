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
- Bin format matches qlib FileFeatureStorage: little-endian float32 array
  whose first element is the calendar index of the ticker's first bar.
"""

from __future__ import annotations

import argparse
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


def _write_bin(path: Path, start_index: int, values: Sequence[float] | np.ndarray) -> None:
    np.hstack([np.array([start_index], dtype="<f"), np.asarray(values, dtype="<f")]).astype(
        "<f"
    ).tofile(path)


def build_store(
    bundles: Sequence[TickerBundle],
    target: Path,
    extra_instruments: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Write a Qlib bin store for the bundles: temp dir -> validate -> swap.

    extra_instruments maps additional universe names to their ticker lists
    (data/refresh.py uses this to carry make_universe files across a rebuild);
    each file is written with spans refreshed from the new bundles, inside the
    same atomic swap as the rest of the store.
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
        (tmp / "instruments" / f"{MARKET_ALL}.txt").write_text("".join(instrument_lines))
        for name, universe_symbols in (extra_instruments or {}).items():
            if name == MARKET_ALL:
                raise BuildError(f"universe name {name!r} is reserved for the full store")
            unknown = [s for s in universe_symbols if s not in spans]
            if unknown:
                raise BuildError(
                    f"universe {name!r} references tickers not in the store: {sorted(unknown)}"
                )
            (tmp / "instruments" / f"{name}.txt").write_text(
                "".join(
                    f"{s}\t{spans[s][0].isoformat()}\t{spans[s][1].isoformat()}\n"
                    for s in universe_symbols
                )
            )
        validate_store(tmp, [bundle.symbol for bundle in bundles])
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


def validate_store(store_dir: Path, symbols: Sequence[str]) -> None:
    """Assert the store at store_dir is complete and readable for symbols.

    Checks calendar ordering, instruments coverage, per-field bin presence,
    index bounds, and that no ticker has a NaN close/factor inside its own
    [first, last] span (a mid-series gap means bad source data - fail loudly
    rather than ship a store Qlib will silently propagate NaNs from).
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
        for field in FIELDS:
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
) -> None:
    """Backfill symbols from FMP (checkpointed) and build the store at output."""
    output = output.expanduser()
    if checkpoint_dir is None:
        checkpoint_dir = output.parent / f"{output.name}.checkpoint"
    fmp = client if client is not None else FmpClient()
    bundles = backfill(
        symbols, lambda s: fetch_bundle(fmp, s, start, end), checkpoint_dir, start, end
    )
    build_store(bundles, output)


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
        build_from_fmp(symbols, args.start, args.end, output, checkpoint_dir, client)
    except (BuildError, FmpError, AdjustmentError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"store built at {output} ({len(symbols)} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

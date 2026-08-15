"""Universe generator: writes instruments/<name>.txt into an existing Qlib store.

Universes control which tickers a research run ranks over (RD-Agent(Q) is
cross-sectional). Built-in universes live in data/config.yaml:
- us_liquid: liquidity/price-filtered broad default (min ADV + min price,
  evaluated against the store itself; defaults to all store tickers)
- sp500: benchmarking list from a committed constituent snapshot

Two membership modes (per-universe ``mode:`` key, ``--mode`` CLI override):
- last_window (legacy): filter on the store's most recent ADV window and last
  raw price; kept tickers get their full all.txt span. Frozen/promoted
  snapshot universes stay on this mode.
- pit (point-in-time, us_liquid's default): membership is re-evaluated on the
  first trading day of every month from trailing ADV/price as of that date,
  with one-period entry/exit hysteresis (a flip needs the opposite signal on
  two consecutive monthly evaluations, so a single-month dip or spike never
  churns membership). Output may hold multiple SYMBOL\\tstart\\tend rows per
  symbol - qlib's native span semantics.

Tickers requested but absent from the store are a hard error (printed gap
list, nonzero exit) - build the store first with data/build_store.py.

Filter math notes: the store keeps close ADJUSTED (raw * factor) and volume
raw / factor, so close * volume is exactly the RAW daily dollar volume; the
raw price on any day is close / factor (see data/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.build_store import DEFAULT_STORE_PATH, FREQ, MARKET_ALL, BuildError, resolve_tickers

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_ADV_WINDOW = 20
MODE_PIT = "pit"
MODE_LAST_WINDOW = "last_window"
MODES = (MODE_PIT, MODE_LAST_WINDOW)
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class UniverseError(RuntimeError):
    """Raised when a universe cannot be generated."""


@dataclass(frozen=True)
class UniverseConfig:
    """Resolved config for one universe (built-in from config.yaml, or bare custom)."""

    name: str
    builtin: bool = False
    min_adv_usd: float | None = None
    min_price: float | None = None
    adv_window: int = DEFAULT_ADV_WINDOW
    tickers_file: Path | None = None
    mode: str = MODE_LAST_WINDOW

    @property
    def has_filters(self) -> bool:
        return self.min_adv_usd is not None or self.min_price is not None


@dataclass(frozen=True)
class Rejection:
    """A ticker dropped by a liquidity filter, with the human-readable reason."""

    symbol: str
    reason: str


def resolve_config(name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> UniverseConfig:
    """Load the named universe from config.yaml; unknown names are bare custom universes."""
    import yaml  # lazy: keeps offline import cost off non-CLI users

    if not config_path.exists():
        raise UniverseError(f"universe config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise UniverseError(f"unparseable universe config {config_path}: {exc}") from exc
    universes = payload.get("universes") if isinstance(payload, dict) else None
    if not isinstance(universes, dict):
        raise UniverseError(f"{config_path} must contain a top-level 'universes' mapping")
    entry = universes.get(name)
    if entry is None:
        return UniverseConfig(name=name)
    if not isinstance(entry, dict):
        raise UniverseError(f"universe '{name}' in {config_path} must be a mapping")
    tickers_file = entry.get("tickers_file")
    mode = str(entry.get("mode", MODE_LAST_WINDOW))
    if mode not in MODES:
        raise UniverseError(
            f"universe '{name}' in {config_path} has unknown mode {mode!r}: "
            f"use one of {', '.join(MODES)}"
        )
    return UniverseConfig(
        name=name,
        builtin=True,
        min_adv_usd=float(entry["min_adv_usd"]) if "min_adv_usd" in entry else None,
        min_price=float(entry["min_price"]) if "min_price" in entry else None,
        adv_window=int(entry.get("adv_window", DEFAULT_ADV_WINDOW)),
        tickers_file=config_path.parent / str(tickers_file) if tickers_file else None,
        mode=mode,
    )


def read_instrument_spans(store: Path) -> dict[str, tuple[str, str]]:
    """SYMBOL -> (start, end) from the store's master instruments/all.txt."""
    path = store / "instruments" / f"{MARKET_ALL}.txt"
    if not path.exists():
        raise UniverseError(
            f"store has no instruments file at {path} - build it first (data/build_store.py)"
        )
    spans: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise UniverseError(f"malformed instruments line in {path}: {line!r}")
        spans[parts[0]] = (parts[1], parts[2])
    if not spans:
        raise UniverseError(f"instruments file {path} is empty")
    return spans


def _read_bin(path: Path) -> tuple[int, np.ndarray]:
    """(calendar index of the first bar, float64 values) from one qlib bin file."""
    if not path.exists():
        raise UniverseError(f"missing feature file {path}")
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        raise UniverseError(f"feature file {path} has no values")
    return int(data[0]), data[1:].astype(np.float64)


def _load_liquidity_series(store: Path, symbol: str) -> tuple[int, np.ndarray, np.ndarray]:
    """(calendar offset, raw daily dollar volume per bar, raw close per bar)."""
    feature_dir = store / "features" / symbol.lower()
    close_offset, close = _read_bin(feature_dir / f"close.{FREQ}.bin")
    volume_offset, volume = _read_bin(feature_dir / f"volume.{FREQ}.bin")
    factor_offset, factor = _read_bin(feature_dir / f"factor.{FREQ}.bin")
    if not (
        close_offset == volume_offset == factor_offset
        and len(close) == len(volume) == len(factor)
    ):
        raise UniverseError(f"feature alignment mismatch for {symbol} in {store}")
    # adjusted close * (raw volume / factor) == raw dollar volume; close / factor == raw price
    return close_offset, close * volume, close / factor


def liquidity_stats(store: Path, symbol: str, adv_window: int) -> tuple[float, float]:
    """(average daily dollar volume USD over the last adv_window bars, last raw price)."""
    _, dollar, raw_price = _load_liquidity_series(store, symbol)
    adv = float(np.nanmean(dollar[-adv_window:]))
    return adv, float(raw_price[-1])


def read_calendar(store: Path) -> list[str]:
    """Trading days (ISO strings, ascending) from the store's day calendar."""
    path = store / "calendars" / f"{FREQ}.txt"
    if not path.exists():
        raise UniverseError(
            f"store has no calendar at {path} - build it first (data/build_store.py)"
        )
    days = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not days:
        raise UniverseError(f"calendar {path} is empty")
    return days


def month_start_indices(calendar: Sequence[str]) -> list[int]:
    """Calendar index of each month's first trading day - the PIT evaluation dates."""
    indices: list[int] = []
    previous = ""
    for i, day in enumerate(calendar):
        month = day[:7]
        if month != previous:
            indices.append(i)
            previous = month
    return indices


def pit_membership_spans(
    store: Path,
    symbol: str,
    calendar: Sequence[str],
    eval_indices: Sequence[int],
    config: UniverseConfig,
    ticker_end: str,
) -> list[tuple[str, str]]:
    """Point-in-time membership spans for one ticker, as (start, end) ISO pairs.

    On each evaluation date the ticker passes when trailing ADV (up to
    adv_window bars ending on that date; shorter early history uses what
    exists, matching last_window semantics) and that day's raw price clear the
    thresholds. One-period hysteresis: membership flips only when the opposite
    signal repeats on two consecutive evaluations - a member's span starts at
    the confirming (second passing) evaluation date and ends the trading day
    before a confirmed exit. Evaluations before the ticker's first bar are
    skipped; evaluations after its last bar fail, and every span end is
    clamped to the ticker's own all.txt end so delistings terminate spans.
    """
    offset, dollar, raw_price = _load_liquidity_series(store, symbol)
    bars = len(dollar)

    def _passes(cal_idx: int) -> bool | None:
        bar_idx = cal_idx - offset
        if bar_idx < 0:
            return None  # not yet listed: not an evaluation at all
        if bar_idx >= bars:
            return False  # no bar on this evaluation date (delisted)
        window = dollar[max(0, bar_idx - config.adv_window + 1) : bar_idx + 1]
        if config.min_adv_usd is not None and float(np.nanmean(window)) < config.min_adv_usd:
            return False
        if config.min_price is not None and float(raw_price[bar_idx]) < config.min_price:
            return False
        return True

    spans: list[tuple[str, str]] = []
    state: bool | None = None  # None until the first evaluation after listing
    streak = 0  # consecutive evaluations disagreeing with state
    open_start: str | None = None
    for cal_idx in eval_indices:
        result = _passes(cal_idx)
        if result is None:
            continue
        if state is None:
            state = result
            if result:
                open_start = calendar[cal_idx]
            continue
        if result == state:
            streak = 0
            continue
        streak += 1
        if streak < 2:
            continue
        state, streak = result, 0
        if result:
            open_start = calendar[cal_idx]
        else:
            assert open_start is not None
            spans.append((open_start, min(calendar[cal_idx - 1], ticker_end)))
            open_start = None
    if open_start is not None:
        spans.append((open_start, min(calendar[-1], ticker_end)))
    return spans


def compute_pit_rows(
    store: Path,
    symbols: Sequence[str],
    config: UniverseConfig,
    spans: dict[str, tuple[str, str]],
    calendar: Sequence[str],
) -> tuple[list[tuple[str, str, str]], list[Rejection]]:
    """(instrument rows, rejects-with-reason) under monthly PIT evaluation."""
    eval_indices = month_start_indices(calendar)
    rows: list[tuple[str, str, str]] = []
    rejected: list[Rejection] = []
    for symbol in symbols:
        member = pit_membership_spans(
            store, symbol, calendar, eval_indices, config, spans[symbol][1]
        )
        if member:
            rows.extend((symbol, start, end) for start, end in member)
        else:
            rejected.append(
                Rejection(symbol, "never met ADV/price thresholds at a monthly evaluation")
            )
    return rows, rejected


def apply_filters(
    store: Path, symbols: Sequence[str], config: UniverseConfig
) -> tuple[list[str], list[Rejection]]:
    """Split symbols into (kept, rejected-with-reason) under the config's filters."""
    if not config.has_filters:
        return list(symbols), []
    kept: list[str] = []
    rejected: list[Rejection] = []
    for symbol in symbols:
        adv, price = liquidity_stats(store, symbol, config.adv_window)
        if config.min_adv_usd is not None and adv < config.min_adv_usd:
            rejected.append(
                Rejection(symbol, f"ADV ${adv:,.0f} < min ${config.min_adv_usd:,.0f}")
            )
        elif config.min_price is not None and price < config.min_price:
            rejected.append(
                Rejection(symbol, f"price ${price:.2f} < min ${config.min_price:.2f}")
            )
        else:
            kept.append(symbol)
    return kept, rejected


def write_instrument_rows(
    store: Path, name: str, rows: Sequence[tuple[str, str, str]]
) -> Path:
    """Write instruments/<name>.txt (SYMBOL\\tstart\\tend rows, sorted).

    Multiple rows per symbol are allowed - qlib treats each as a membership
    span (the PIT mode's output shape).
    """
    if name == MARKET_ALL:
        raise UniverseError(f"universe name '{MARKET_ALL}' is reserved for the master list")
    if not _NAME_RE.fullmatch(name):
        raise UniverseError(
            f"invalid universe name {name!r}: use lowercase letters, digits, underscores"
        )
    if not rows:
        raise UniverseError("universe is empty (all tickers filtered out); nothing written")
    path = store / "instruments" / f"{name}.txt"
    path.write_text("".join(f"{s}\t{start}\t{end}\n" for s, start, end in sorted(rows)))
    return path


def write_instruments_file(
    store: Path, name: str, symbols: Sequence[str], spans: dict[str, tuple[str, str]]
) -> Path:
    """Write instruments/<name>.txt with each symbol's full master span (legacy shape)."""
    return write_instrument_rows(
        store, name, [(s, spans[s][0], spans[s][1]) for s in symbols]
    )


def _resolve_requested(
    args_tickers: str | None,
    args_tickers_file: str | None,
    config: UniverseConfig,
    spans: dict[str, tuple[str, str]],
) -> list[str]:
    """Ticker list precedence: CLI args > config tickers_file > all store tickers (built-ins)."""
    if args_tickers is not None or args_tickers_file is not None:
        return resolve_tickers(args_tickers, args_tickers_file)
    if config.tickers_file is not None:
        if not config.tickers_file.exists():
            raise UniverseError(
                f"tickers file for '{config.name}' not found: {config.tickers_file}"
            )
        return resolve_tickers(None, str(config.tickers_file))
    if config.builtin:
        return sorted(spans)
    raise UniverseError(
        f"universe '{config.name}' is not built-in; provide --tickers or --tickers-file"
    )


def make_universe(
    name: str,
    store: Path,
    tickers: str | None = None,
    tickers_file: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    mode: str | None = None,
) -> Path:
    """Generate instruments/<name>.txt in the store; raises UniverseError on any problem.

    mode overrides the universe's configured membership mode (see module
    docstring); None keeps the config's choice (last_window unless set).
    """
    config = resolve_config(name, config_path)
    effective_mode = mode if mode is not None else config.mode
    if effective_mode not in MODES:
        raise UniverseError(
            f"unknown universe mode {effective_mode!r}: use one of {', '.join(MODES)}"
        )
    store = store.expanduser()
    spans = read_instrument_spans(store)
    requested = _resolve_requested(tickers, tickers_file, config, spans)
    gaps = [s for s in requested if s not in spans]
    if gaps:
        raise UniverseError(
            f"{len(gaps)} ticker(s) absent from the store at {store} - backfill them with "
            f"data/build_store.py first: {' '.join(gaps)}"
        )
    if effective_mode == MODE_PIT:
        if not config.has_filters:
            raise UniverseError(
                f"point-in-time mode needs liquidity filters, but universe '{name}' "
                "defines none (min_adv_usd / min_price) - use --mode last_window"
            )
        rows, rejected = compute_pit_rows(
            store, requested, config, spans, read_calendar(store)
        )
        for rejection in rejected:
            print(f"filtered {rejection.symbol}: {rejection.reason}")
        path = write_instrument_rows(store, name, rows)
        kept_count = len({row[0] for row in rows})
        print(
            f"universe '{name}' written to {path} (point-in-time: {kept_count} tickers, "
            f"{len(rows)} membership spans, {len(rejected)} filtered)"
        )
        return path
    kept, rejected = apply_filters(store, requested, config)
    for rejection in rejected:
        print(f"filtered {rejection.symbol}: {rejection.reason}")
    path = write_instruments_file(store, name, kept, spans)
    print(f"universe '{name}' written to {path} ({len(kept)} tickers, {len(rejected)} filtered)")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a universe instruments file into an existing Qlib store."
    )
    parser.add_argument(
        "--name", required=True, help="universe name (built-in: us_liquid, sp500; or custom)"
    )
    parser.add_argument("--tickers", help="comma-separated ticker list, e.g. AAPL,MSFT")
    parser.add_argument("--tickers-file", help="file with one ticker per line")
    parser.add_argument(
        "--store", default=DEFAULT_STORE_PATH, help=f"Qlib store dir (default {DEFAULT_STORE_PATH})"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="universe config yaml (default data/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="membership mode override: pit (monthly point-in-time spans) or "
        "last_window (legacy: filter on the most recent window only)",
    )
    args = parser.parse_args(argv)
    try:
        make_universe(
            name=args.name,
            store=Path(args.store),
            tickers=args.tickers,
            tickers_file=args.tickers_file,
            config_path=Path(args.config).expanduser(),
            mode=args.mode,
        )
    except (UniverseError, BuildError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Universe generator: writes instruments/<name>.txt into an existing Qlib store.

Universes control which tickers a research run ranks over (RD-Agent(Q) is
cross-sectional). Built-in universes live in data/config.yaml:
- us_liquid: liquidity/price-filtered broad default (min ADV + min price,
  evaluated against the store itself; defaults to all store tickers)
- sp500: benchmarking list from a committed constituent snapshot

Tickers requested but absent from the store are a hard error (printed gap
list, nonzero exit) - build the store first with data/build_store.py.

Filter math notes: the store keeps close ADJUSTED (raw * factor) and volume
raw / factor, so close * volume is exactly the RAW daily dollar volume; the
raw price on a ticker's last day is close / factor (see data/CLAUDE.md).
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
    return UniverseConfig(
        name=name,
        builtin=True,
        min_adv_usd=float(entry["min_adv_usd"]) if "min_adv_usd" in entry else None,
        min_price=float(entry["min_price"]) if "min_price" in entry else None,
        adv_window=int(entry.get("adv_window", DEFAULT_ADV_WINDOW)),
        tickers_file=config_path.parent / str(tickers_file) if tickers_file else None,
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


def _read_bin(path: Path) -> np.ndarray:
    """Feature values (calendar-index header stripped) from one qlib bin file."""
    if not path.exists():
        raise UniverseError(f"missing feature file {path}")
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        raise UniverseError(f"feature file {path} has no values")
    return data[1:].astype(np.float64)


def liquidity_stats(store: Path, symbol: str, adv_window: int) -> tuple[float, float]:
    """(average daily dollar volume USD over the last adv_window bars, last raw price)."""
    feature_dir = store / "features" / symbol.lower()
    close = _read_bin(feature_dir / f"close.{FREQ}.bin")
    volume = _read_bin(feature_dir / f"volume.{FREQ}.bin")
    factor = _read_bin(feature_dir / f"factor.{FREQ}.bin")
    if not (len(close) == len(volume) == len(factor)):
        raise UniverseError(f"feature length mismatch for {symbol} in {store}")
    dollar = close * volume  # adjusted close * (raw volume / factor) == raw dollar volume
    adv = float(np.nanmean(dollar[-adv_window:]))
    last_raw_price = float(close[-1] / factor[-1])
    return adv, last_raw_price


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


def write_instruments_file(
    store: Path, name: str, symbols: Sequence[str], spans: dict[str, tuple[str, str]]
) -> Path:
    """Write instruments/<name>.txt (SYMBOL\\tstart\\tend rows, sorted by symbol)."""
    if name == MARKET_ALL:
        raise UniverseError(f"universe name '{MARKET_ALL}' is reserved for the master list")
    if not _NAME_RE.fullmatch(name):
        raise UniverseError(
            f"invalid universe name {name!r}: use lowercase letters, digits, underscores"
        )
    if not symbols:
        raise UniverseError("universe is empty (all tickers filtered out); nothing written")
    path = store / "instruments" / f"{name}.txt"
    path.write_text(
        "".join(f"{s}\t{spans[s][0]}\t{spans[s][1]}\n" for s in sorted(symbols))
    )
    return path


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
) -> Path:
    """Generate instruments/<name>.txt in the store; raises UniverseError on any problem."""
    config = resolve_config(name, config_path)
    store = store.expanduser()
    spans = read_instrument_spans(store)
    requested = _resolve_requested(tickers, tickers_file, config, spans)
    gaps = [s for s in requested if s not in spans]
    if gaps:
        raise UniverseError(
            f"{len(gaps)} ticker(s) absent from the store at {store} - backfill them with "
            f"data/build_store.py first: {' '.join(gaps)}"
        )
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
    args = parser.parse_args(argv)
    try:
        make_universe(
            name=args.name,
            store=Path(args.store),
            tickers=args.tickers,
            tickers_file=args.tickers_file,
            config_path=Path(args.config).expanduser(),
        )
    except (UniverseError, BuildError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

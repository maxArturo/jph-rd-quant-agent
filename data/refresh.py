"""Incremental refresh of an existing Qlib US store from FMP (US-036).

Pulls only the bars each ticker is missing (since its own last stored date),
refetches the full split/dividend history, recomputes adjustment factors over
the merged series, and rebuilds the store through build_store's
temp -> validate -> atomic-swap path. Custom instruments files written by
data/make_universe.py are carried across the rebuild inside the same atomic
swap, membership spans preserved: a span that ended at its ticker's last bar
follows the ticker's new end, while spans closed earlier (point-in-time
exits, US-023) are kept verbatim.

Raw bars are recovered from the store itself (the field conventions make raw
values recoverable: raw price = stored adjusted price / factor, raw volume =
stored volume * factor), so a refresh needs no full FMP re-backfill and stays
correct when a NEW split or dividend lands between refreshes — the whole
factor series is recomputed, re-scaling history exactly like a fresh build.
Round-tripping through the float32 bins costs ~1e-7 relative noise per
rebuild; negligible against price data.

Market broadcast fields ($mkt_*, US-067): any mkt_* bins the store carries
are read back as (date, value) observations (skipping the raw-recovery /
adjustment math entirely — implicit factor 1), each series is pulled since
the store's last date under the same --end rule as equity bars, and the
merged series ride the same temp -> validate -> atomic-swap rebuild. A
series whose FMP fetch fails is forward-filled from its last stored value
and reported as a warning (Slack line from the CLI) — a market-data outage
must never block the pre-open refresh -> predict -> rebalance chain.
--market-start backfills any of the canonical MARKET_FIELDS the store does
not carry yet (one-time introduction; from then on the nightly refresh
advances them incrementally).

News counts ($news_ct_1d, US-073): when the store carries news_ct_1d bins,
each refreshed trading day pulls yesterday's articles per ticker (same --end
rule), appends them to the news archive (data/build_news.py layout — the
checkpoint's end advances, its start is kept), and carries the per-ticker
count bins through the same rebuild (raw, implicit factor 1). A ticker whose
news fetch fails gets an explicit 0 count for the new day(s) plus a warning
(Slack line from the CLI) — a news outage must never block the pre-open
refresh -> predict -> rebalance chain. --news-start is the one-time
introduction: it builds the series from the archive for every store ticker
(backfilling any ticker the archive has never seen) and forces a rebuild.

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
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from data.adjust import AdjustmentError
from data.build_store import (
    COMMODITY_SYMBOLS,
    DEFAULT_STORE_PATH,
    FIELDS,
    FREQ,
    MARKET_ALL,
    MARKET_FIELDS,
    MARKET_SERIES_START,
    NEWS_FIELD,
    TREASURY_FIELD,
    BuildError,
    TickerBundle,
    build_store,
)
from data.fmp import DateLike, EodBar, FmpClient, FmpError, NewsArticle, _to_iso_date

MARKET_TZ = ZoneInfo("America/New_York")

# Daily news ingest pace. Faster than the bulk-backfill's 3 req/s (recorded
# US-071) because the 04:30 ET refresh must finish before the 04:45 pred
# refresh: ~590 tickers x ~2 requests at 8 req/s is ~2.5 min, still well
# under FMP's 750 req/min plan limit.
REFRESH_NEWS_THROTTLE_RPS = 8.0

# fetch_pages(symbol, start_iso, end_iso) -> pages of articles
# (FmpClient.iter_stock_news_pages contract; see data/build_news.py).
NewsFetchPages = Callable[[str, str, str], Iterator[list[NewsArticle]]]


class RefreshError(RuntimeError):
    """Raised when the existing store cannot be read or safely refreshed."""


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one refresh: whether the store was rebuilt and what changed."""

    updated: bool
    last_date_before: date
    last_date_after: date
    new_bars: dict[str, int]  # symbol -> number of appended bars
    warnings: tuple[str, ...] = ()  # degraded-but-not-fatal notices (market/news outages)
    market_introduced: tuple[str, ...] = ()  # $mkt_* fields backfilled this run
    news_introduced: bool = False  # $news_ct_1d built from the archive this run


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


def read_universes(store: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Every instruments file except all.txt, as name -> ordered (symbol, start, end) rows.

    Full rows, not just symbols: point-in-time universes (data/make_universe.py
    ``mode: pit``) carry multiple membership spans per symbol, which a rebuild
    must preserve — re-deriving spans from all.txt would silently flatten them
    back to full-history membership (the exact bias US-023 removed).
    """
    universes: dict[str, list[tuple[str, str, str]]] = {}
    for path in sorted((store / "instruments").glob("*.txt")):
        if path.stem == MARKET_ALL:
            continue
        rows: list[tuple[str, str, str]] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise RefreshError(f"malformed instruments line in {path}: {line!r}")
            rows.append((parts[0], parts[1], parts[2]))
        universes[path.stem] = rows
    return universes


def refresh_universe_spans(
    universes: dict[str, list[tuple[str, str, str]]],
    old_ends: dict[str, date],
    new_ends: dict[str, date],
) -> dict[str, list[tuple[str, str, str]]]:
    """Advance each row's end only when it tracked its ticker's end before the refresh.

    A span ending at (or somehow past) the ticker's pre-refresh last bar is
    "open" — the ticker was still a member at store end — so it follows the
    ticker's new end. A span closed earlier (a PIT exit) is history and is
    carried verbatim. For legacy full-span files this reproduces the old
    rewrite-from-all.txt behavior exactly (starts never move on a refresh).
    """
    refreshed: dict[str, list[tuple[str, str, str]]] = {}
    for name, rows in universes.items():
        refreshed[name] = [
            (
                symbol,
                start,
                new_ends[symbol].isoformat()
                if symbol in old_ends and end >= old_ends[symbol].isoformat()
                else end,
            )
            for symbol, start, end in rows
        ]
    return refreshed


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
# Market broadcast fields ($mkt_*): read-back and incremental pull


def read_market_fields(store: Path, symbols: Sequence[str]) -> tuple[str, ...]:
    """Sorted union of mkt_* broadcast fields present in the store's feature bins."""
    fields: set[str] = set()
    suffix = f".{FREQ}.bin"
    for symbol in symbols:
        feature_dir = store / "features" / symbol.lower()
        fields.update(p.name[: -len(suffix)] for p in feature_dir.glob(f"mkt_*{suffix}"))
    return tuple(sorted(fields))


def read_market_series(
    store: Path, symbols: Sequence[str], calendar: list[date], fields: Sequence[str]
) -> dict[str, list[tuple[date, float]]]:
    """Recover each stored $mkt_* series as (date, value) observations.

    Bins hold forward-filled values, so every stored day reads back as a
    direct observation (the original off-calendar prints are not
    recoverable, and don't need to be); the NaN head before a series' first
    observation is dropped. Values are identical across instruments, so
    overlaying every symbol's bin reconstructs the full calendar span even
    when no single ticker spans the whole store.
    """
    series: dict[str, list[tuple[date, float]]] = {}
    for name in fields:
        values = np.full(len(calendar), np.nan)
        for symbol in symbols:
            path = store / "features" / symbol.lower() / f"{name}.{FREQ}.bin"
            if not path.exists():
                raise RefreshError(
                    f"{symbol} is missing market bin {path}; store is corrupt"
                )
            data = np.fromfile(path, dtype="<f")
            if len(data) < 2:
                raise RefreshError(f"{path} has no values")
            start_index = int(data[0])
            points = data[1:]
            if start_index < 0 or start_index + len(points) > len(calendar):
                raise RefreshError(
                    f"{symbol} market bin {name} exceeds the store calendar; store is corrupt"
                )
            segment = values[start_index : start_index + len(points)]
            mask = np.isnan(segment)
            segment[mask] = points[mask]
        observations = [
            (day, float(value))
            for day, value in zip(calendar, values, strict=True)
            if not math.isnan(value)
        ]
        if not observations:
            raise RefreshError(f"market series {name} in the store holds no values")
        series[name] = observations
    return series


def _fetch_market_observations(
    client: FmpClient, name: str, start: date, end: date
) -> list[tuple[date, float]]:
    """One canonical $mkt_* series as (date, value) rows over [start, end]."""
    if name == TREASURY_FIELD:
        return [
            (curve.date, curve.year10)
            for curve in client.get_treasury_rates(start, end)
            if curve.year10 is not None
        ]
    rows = client.get_commodity_eod(COMMODITY_SYMBOLS[name], start, end)
    return [(row.date, row.price) for row in rows]


def _pull_market_series(
    client: FmpClient,
    stored: dict[str, list[tuple[date, float]]],
    pull_stored: bool,
    introduce: Sequence[str],
    last_before: date,
    end_date: date,
    market_start: date | None,
) -> tuple[dict[str, list[tuple[date, float]]], list[str], list[str]]:
    """Advance stored series and backfill introduced ones; degrade per series.

    Returns (merged series map, warnings, fields actually introduced). A
    series whose fetch fails keeps its stored observations — the rebuild's
    forward-fill covers the new days — and is reported as a warning instead
    of raising: a market-data outage must never block the refresh chain.
    """
    merged = {name: list(observations) for name, observations in stored.items()}
    warnings: list[str] = []
    introduced: list[str] = []
    windows: list[tuple[str, date]] = []
    if pull_stored:
        windows += [(name, last_before + timedelta(days=1)) for name in stored]
    if market_start is not None:
        windows += [(name, market_start) for name in introduce]
    for name, window_start in windows:
        if name not in MARKET_FIELDS:
            warnings.append(
                f"unknown market series {name} in the store (no FMP mapping); "
                "carried forward-filled"
            )
            continue
        if window_start > end_date:
            continue
        try:
            fetched = _fetch_market_observations(client, name, window_start, end_date)
        except FmpError as exc:
            if name in stored:
                warnings.append(
                    f"market series {name}: FMP fetch failed ({exc}); "
                    "forward-filled from the last stored value"
                )
            else:
                warnings.append(f"market series {name}: FMP fetch failed ({exc}); not introduced")
            continue
        fresh = sorted((d, v) for d, v in fetched if window_start <= d <= end_date)
        if name in stored:
            merged[name] += fresh
        elif fresh:
            merged[name] = fresh
            introduced.append(name)
        else:
            warnings.append(
                f"market series {name}: FMP returned no rows over "
                f"{window_start.isoformat()}..{end_date.isoformat()}; not introduced"
            )
    return merged, warnings, introduced


# ---------------------------------------------------------------------------
# Per-ticker raw series ($news_ct_1d, US-073): read-back and daily ingest


def read_ticker_fields(store: Path, symbols: Sequence[str]) -> tuple[str, ...]:
    """Sorted union of per-ticker raw series bins (news_ct_1d, ...) in the store."""
    known = set(FIELDS)
    suffix = f".{FREQ}.bin"
    fields: set[str] = set()
    for symbol in symbols:
        for path in (store / "features" / symbol.lower()).glob(f"*{suffix}"):
            name = path.name[: -len(suffix)]
            if name not in known and not name.startswith("mkt_"):
                fields.add(name)
    return tuple(sorted(fields))


def read_ticker_series(
    store: Path, symbols: Sequence[str], calendar: list[date], fields: Sequence[str]
) -> dict[str, dict[str, list[tuple[date, float]]]]:
    """Recover each per-ticker raw series as per-symbol (date, value) observations.

    NaN days carry no observation (they mean "outside the series' coverage" —
    the rebuild writes them back as NaN); a symbol whose bin is all NaN gets
    no entry, so its rebuilt bin comes back all-NaN too. A missing bin on any
    symbol is corruption: build_store writes one per instrument.
    """
    series: dict[str, dict[str, list[tuple[date, float]]]] = {}
    for name in fields:
        per_symbol: dict[str, list[tuple[date, float]]] = {}
        for symbol in symbols:
            path = store / "features" / symbol.lower() / f"{name}.{FREQ}.bin"
            if not path.exists():
                raise RefreshError(
                    f"{symbol} is missing ticker-series bin {path}; store is corrupt"
                )
            data = np.fromfile(path, dtype="<f")
            if len(data) < 2:
                raise RefreshError(f"{path} has no values")
            start_index = int(data[0])
            points = data[1:]
            if start_index < 0 or start_index + len(points) > len(calendar):
                raise RefreshError(
                    f"{symbol} ticker bin {name} exceeds the store calendar; store is corrupt"
                )
            observations = [
                (calendar[start_index + i], float(value))
                for i, value in enumerate(points)
                if not math.isnan(float(value))
            ]
            if observations:
                per_symbol[symbol] = observations
        series[name] = per_symbol
    return series


def _pull_news_series(
    fetch_pages: NewsFetchPages,
    news_root: Path | str | None,
    stored_ticker: dict[str, dict[str, list[tuple[date, float]]]],
    symbols: Sequence[str],
    existing_ends: dict[str, date],
    new_bars: dict[str, list[EodBar]],
    calendar: list[date],
    last_before: date,
    end_date: date,
    news_start: date | None,
    sleep: Callable[[float], None],
    throttle_rps: float,
) -> tuple[dict[str, dict[str, list[tuple[date, float]]]], list[str], bool]:
    """Advance the news archive and merge $news_ct_1d observations; degrade per ticker.

    Returns (merged ticker-series map, warnings, news-introduced flag). A
    ticker whose news fetch fails gets explicit 0 counts for its new days —
    a news outage must never block the refresh chain. When ``news_start`` is
    set and the store has no news bins, the series is built from the archive
    for every store ticker (introduction), backfilling tickers the archive
    has never seen; an introduction failure degrades to a warning.
    """
    from data import build_news

    series = {name: {s: list(obs) for s, obs in per.items()} for name, per in stored_ticker.items()}
    warnings: list[str] = []
    news_present = NEWS_FIELD in series
    introduce = news_start is not None and not news_present
    if new_bars:
        for name in series:
            if name != NEWS_FIELD:
                warnings.append(
                    f"unknown ticker series {name} in the store (no ingest path); "
                    "new days left empty"
                )
    if not news_present and not introduce:
        return series, warnings, False

    root = Path(news_root if news_root is not None else build_news.DEFAULT_NEWS_ROOT).expanduser()
    new_days = sorted({b.date for bars in new_bars.values() for b in bars if b.date > last_before})
    final_days = calendar + new_days

    if news_present and new_bars:
        per_symbol = series[NEWS_FIELD]
        failed: list[str] = []
        for symbol in symbols:
            fresh = new_bars.get(symbol)
            if not fresh:
                continue  # span does not advance; nothing to append
            old_end = existing_ends[symbol]
            sym_end = fresh[-1].date
            try:
                build_news.advance_ticker(
                    fetch_pages, symbol, end_date, root, sleep=sleep, throttle_rps=throttle_rps
                )
                counts = [
                    (day, float(count))
                    for day, count in build_news.daily_counts(
                        root, symbol, final_days, old_end, sym_end
                    )
                    if day > old_end
                ]
            except Exception:  # noqa: BLE001 — news must never block the refresh chain
                failed.append(symbol)
                counts = [(day, 0.0) for day in final_days if old_end < day <= sym_end]
            per_symbol[symbol] = per_symbol.get(symbol, []) + counts
        if failed:
            shown = ", ".join(failed[:5]) + (", ..." if len(failed) > 5 else "")
            warnings.append(
                f"news ingest failed for {len(failed)} ticker(s) ({shown}); "
                "$news_ct_1d recorded as 0 for the new day(s)"
            )

    introduced = False
    if introduce and news_start is not None:
        final_end = final_days[-1]
        try:
            for symbol in symbols:
                window = build_news.checkpoint_window(root, symbol)
                if window is None or window[0] > news_start:
                    build_news.backfill_ticker(
                        fetch_pages,
                        symbol,
                        news_start,
                        end_date,
                        root,
                        sleep=sleep,
                        throttle_rps=throttle_rps,
                    )
                else:
                    build_news.advance_ticker(
                        fetch_pages, symbol, end_date, root, sleep=sleep, throttle_rps=throttle_rps
                    )
            series[NEWS_FIELD] = dict(
                build_news.read_news_series(root, symbols, final_days, news_start, final_end)
            )
            introduced = True
        except (FmpError, build_news.NewsBuildError, OSError) as exc:
            warnings.append(f"news series not introduced: {exc}")
    return series, warnings, introduced


# ---------------------------------------------------------------------------
# Refresh


def refresh_store(
    store: Path,
    client: FmpClient,
    end: DateLike | None = None,
    market_start: DateLike | None = None,
    news_start: DateLike | None = None,
    news_root: Path | str | None = None,
    news_fetch_pages: NewsFetchPages | None = None,
    news_sleep: Callable[[float], None] = time.sleep,
    news_throttle: float = REFRESH_NEWS_THROTTLE_RPS,
) -> RefreshResult:
    """Pull bars since each ticker's last stored date and rebuild if anything landed.

    Any $mkt_* fields the store carries are pulled forward over the same
    window and rebuilt alongside (raw broadcast, implicit factor 1 — the
    adjustment math never sees them). market_start additionally backfills
    the canonical MARKET_FIELDS the store lacks, from that date — the
    one-time introduction path; it forces a rebuild even when no equity bar
    is new.

    News counts work the same way per ticker: stored news_ct_1d bins are read
    back, yesterday's articles are pulled into the archive under ``news_root``
    (default data/build_news.DEFAULT_NEWS_ROOT), and each refreshed ticker's
    new days get their archive-derived counts (0 on failure, with a warning).
    news_start is the one-time introduction from the archive.
    """
    store = store.expanduser()
    end_date = date.fromisoformat(_to_iso_date(end if end is not None else default_end(), "end"))
    calendar = read_calendar(store)
    symbols = read_all_symbols(store)
    universes = read_universes(store)
    existing = {symbol: read_raw_bars(store, symbol, calendar) for symbol in symbols}
    last_before = calendar[-1]
    market_fields = read_market_fields(store, symbols)
    stored_market = read_market_series(store, symbols, calendar, market_fields)
    stored_ticker = read_ticker_series(
        store, symbols, calendar, read_ticker_fields(store, symbols)
    )
    introduce: tuple[str, ...] = ()
    market_start_date: date | None = None
    if market_start is not None:
        market_start_date = date.fromisoformat(_to_iso_date(market_start, "market-start"))
        introduce = tuple(name for name in MARKET_FIELDS if name not in stored_market)
    news_start_date: date | None = None
    if news_start is not None:
        news_start_date = date.fromisoformat(_to_iso_date(news_start, "news-start"))
    introduce_news = news_start_date is not None and NEWS_FIELD not in stored_ticker

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

    if not new_bars and not introduce and not introduce_news:
        return RefreshResult(False, last_before, last_before, {})

    # Stored series only advance when the equity calendar does: without new
    # bars there is no new trading day to place a market value on.
    market_series, warnings, introduced = _pull_market_series(
        client,
        stored_market,
        pull_stored=bool(new_bars),
        introduce=introduce,
        last_before=last_before,
        end_date=end_date,
        market_start=market_start_date,
    )
    ticker_series, news_warnings, news_introduced = _pull_news_series(
        news_fetch_pages if news_fetch_pages is not None else client.iter_stock_news_pages,
        news_root,
        stored_ticker,
        symbols,
        {symbol: bars[-1].date for symbol, bars in existing.items()},
        new_bars,
        calendar,
        last_before,
        end_date,
        news_start_date if introduce_news else None,
        sleep=news_sleep,
        throttle_rps=news_throttle,
    )
    warnings += news_warnings
    if not new_bars and not introduced and not news_introduced:
        # Introduction was requested but nothing landed (outage/empty): the
        # rebuild would be a byte-identical rewrite, so skip it.
        return RefreshResult(False, last_before, last_before, {}, warnings=tuple(warnings))

    bundles = [
        TickerBundle(
            symbol=symbol,
            bars=existing[symbol] + tuple(new_bars.get(symbol, ())),
            splits=tuple(client.get_splits(symbol)),
            dividends=tuple(client.get_dividends(symbol)),
        )
        for symbol in symbols
    ]
    old_ends = {symbol: bars[-1].date for symbol, bars in existing.items()}
    new_ends = {bundle.symbol: bundle.bars[-1].date for bundle in bundles}
    build_store(
        bundles,
        store,
        extra_instruments=refresh_universe_spans(universes, old_ends, new_ends),
        market_series=market_series or None,
        ticker_series=ticker_series or None,
    )
    last_after = max(bundle.bars[-1].date for bundle in bundles)
    return RefreshResult(
        True,
        last_before,
        last_after,
        {s: len(bars) for s, bars in new_bars.items()},
        warnings=tuple(warnings),
        market_introduced=tuple(introduced),
        news_introduced=news_introduced,
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
    Any $mkt_* fields the store carries are read back and rebuilt alongside,
    so new tickers get the broadcast bins over their own spans too. Per-ticker
    series (news_ct_1d) are carried for existing tickers; a new ticker's bin
    starts all-NaN (outside news coverage) until its archive backfill lands
    and a later rebuild picks it up.

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
    stored_market = read_market_series(
        store, existing_symbols, calendar, read_market_fields(store, existing_symbols)
    )
    stored_ticker = read_ticker_series(
        store, existing_symbols, calendar, read_ticker_fields(store, existing_symbols)
    )
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
    build_store(
        bundles,
        store,
        extra_instruments=universes,
        market_series=stored_market or None,
        ticker_series=stored_ticker or None,
    )
    return ExtendResult(
        added={symbol: len(bars) for symbol, bars in fetched.items()},
        missing=tuple(missing),
        gapped=tuple(gapped),
    )


# ---------------------------------------------------------------------------
# CLI


def _slack_notify(message: str) -> None:
    """Post to the ops channel (repo-.env slack_notifier; Slack bypasses the proxy)."""
    from execution.rebalance import slack_notifier

    slack_notifier()(message)


def main(
    argv: Any = None,
    client: FmpClient | None = None,
    notify: Callable[[str], None] | None = None,
    news_fetch_pages: NewsFetchPages | None = None,
    news_sleep: Callable[[float], None] = time.sleep,
) -> int:
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
    parser.add_argument(
        "--market-start",
        default=None,
        metavar="YYYY-MM-DD",
        help="backfill any canonical $mkt_* market series the store does not carry yet, "
        f"from this date (canonical start {MARKET_SERIES_START.isoformat()}); series "
        "already in the store always refresh incrementally without this flag",
    )
    parser.add_argument(
        "--news-start",
        default=None,
        metavar="YYYY-MM-DD",
        help="one-time $news_ct_1d introduction: build the per-ticker news-count series "
        "from the news archive starting at this date (canonical start 2025-01-02) when "
        "the store does not carry news bins yet; a store already carrying them always "
        "ingests yesterday's articles without this flag",
    )
    parser.add_argument(
        "--news-root",
        default=None,
        help="news archive + checkpoint root (default ~/rdq-data/news)",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="print refresh warnings to stderr only (no Slack notice)",
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
        result = refresh_store(
            Path(args.store),
            fmp,
            args.end,
            market_start=args.market_start,
            news_start=args.news_start,
            news_root=args.news_root,
            news_fetch_pages=news_fetch_pages,
            news_sleep=news_sleep,
        )
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
    if result.market_introduced:
        print(f"market series introduced: {', '.join(result.market_introduced)}")
    if result.news_introduced:
        print(f"news series introduced: {NEWS_FIELD}")
    for line in result.warnings:
        print(f"WARNING: {line}", file=sys.stderr)
    if result.warnings and not args.no_slack:
        message = ":warning: store refresh: " + "; ".join(result.warnings)
        try:
            (notify if notify is not None else _slack_notify)(message)
        except Exception as exc:  # noqa: BLE001 — a warning notice must never fail the refresh
            print(f"slack notice failed ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

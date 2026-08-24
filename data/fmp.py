"""Typed FMP (Financial Modeling Prep) client for EOD bars, splits, and dividends.

All requests are bare HTTPS: no apikey appears in code, env, or params. The
OneCLI proxy injects the query-param key when the process runs under
`onecli run --agent rdq-research` or `--agent rdq-exec-paper` (the identities
with the financialmodelingprep.com secret assignment).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import requests

BASE_URL = "https://financialmodelingprep.com/stable"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 2.0

# /stable/treasury-rates silently truncates windows wider than ~90 calendar
# days (HTTP 200, trailing ~3 months only) — proven by ops/probe_market_series.sh
# (docs/decisions.md 2026-08-24). get_treasury_rates chunks to stay under it.
TREASURY_CHUNK_DAYS = 90

# /stable/news/stock pagination (probe-verified 2026-08-24, docs/decisions.md
# US-071): pages are served newest-first and have PAGE-SIZE JITTER — a
# mid-stream page can return fewer than `limit` rows with more history behind
# it. ONLY an empty page marks end-of-history; never stop on a short page.
NEWS_PAGE_LIMIT = 250
# Backstop against a pagination loop that never empties (AAPL over the full
# 599-day window is 44 pages; nothing legitimate approaches this).
NEWS_MAX_PAGES = 500

DateLike = date | str


class FmpError(RuntimeError):
    """Base error for FMP client failures."""


class FmpAuthError(FmpError):
    """401/403 from FMP: the OneCLI proxy did not inject a valid key."""


class FmpRateLimitError(FmpError):
    """429 from FMP persisted beyond the retry budget."""


@dataclass(frozen=True)
class EodBar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Split:
    symbol: str
    date: date
    numerator: float
    denominator: float

    @property
    def ratio(self) -> float:
        """Shares multiplier: 10:1 split -> 10.0 (one old share becomes ten)."""
        return self.numerator / self.denominator


@dataclass(frozen=True)
class Dividend:
    symbol: str
    date: date  # ex-dividend date
    dividend: float


@dataclass(frozen=True)
class CommodityEod:
    """One /historical-price-eod/light row: price only, no OHLC on this endpoint."""

    symbol: str
    date: date
    price: float
    volume: float


@dataclass(frozen=True)
class TreasuryCurve:
    """One /treasury-rates row; tenors absent from the response are None."""

    date: date
    month1: float | None
    month2: float | None
    month3: float | None
    month6: float | None
    year1: float | None
    year2: float | None
    year3: float | None
    year5: float | None
    year7: float | None
    year10: float | None
    year20: float | None
    year30: float | None


@dataclass(frozen=True)
class NewsArticle:
    """One /news/stock row.

    ``published`` is US/Eastern wall-clock at second resolution exactly as FMP
    serves it (probe-verified 2026-08-24) — kept naive, never tz-converted.
    """

    symbol: str
    published: datetime
    publisher: str
    title: str
    url: str


_TREASURY_TENORS = (
    "month1",
    "month2",
    "month3",
    "month6",
    "year1",
    "year2",
    "year3",
    "year5",
    "year7",
    "year10",
    "year20",
    "year30",
)


def _to_iso_date(value: DateLike, name: str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD or datetime.date, got {value!r}") from exc


def _parse_date(value: Any, context: str) -> date:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise FmpError(f"unparseable date {value!r} in {context} response") from exc


def _window(start: DateLike, end: DateLike) -> tuple[str, str]:
    """Validate a [start, end] window and return it as ISO strings."""
    start_iso = _to_iso_date(start, "start")
    end_iso = _to_iso_date(end, "end")
    if start_iso > end_iso:
        raise ValueError(f"start {start_iso} is after end {end_iso}")
    return start_iso, end_iso


def _tenor_value(row: dict[str, Any], tenor: str) -> float | None:
    value = row.get(tenor)
    return None if value is None else float(value)


def _parse_published(value: Any) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise FmpError(f"unparseable publishedDate {value!r} in news/stock response") from exc


class FmpClient:
    """Minimal typed client for the FMP /stable API through the OneCLI proxy.

    429 responses are retried with Retry-After (or exponential) backoff;
    401/403 raise FmpAuthError with the fix spelled out.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep

    def get_eod_bars(self, symbol: str, start: DateLike, end: DateLike) -> list[EodBar]:
        """Daily raw (unadjusted) OHLCV bars for [start, end], ascending by date."""
        start_iso = _to_iso_date(start, "start")
        end_iso = _to_iso_date(end, "end")
        if start_iso > end_iso:
            raise ValueError(f"start {start_iso} is after end {end_iso}")
        rows = self._get_list(
            "/historical-price-eod/full",
            {"symbol": symbol, "from": start_iso, "to": end_iso},
        )
        bars = [
            EodBar(
                symbol=str(row.get("symbol", symbol)),
                date=_parse_date(row["date"], "historical-price-eod"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in rows
        ]
        return sorted(bars, key=lambda bar: bar.date)

    def get_splits(self, symbol: str) -> list[Split]:
        """All stock splits for symbol, ascending by date."""
        rows = self._get_list("/splits", {"symbol": symbol})
        splits = [
            Split(
                symbol=str(row.get("symbol", symbol)),
                date=_parse_date(row["date"], "splits"),
                numerator=float(row["numerator"]),
                denominator=float(row["denominator"]),
            )
            for row in rows
        ]
        return sorted(splits, key=lambda split: split.date)

    def get_dividends(self, symbol: str) -> list[Dividend]:
        """All cash dividends for symbol (ex-date, unadjusted amount), ascending by date."""
        rows = self._get_list("/dividends", {"symbol": symbol})
        dividends = [
            Dividend(
                symbol=str(row.get("symbol", symbol)),
                date=_parse_date(row["date"], "dividends"),
                dividend=float(row["dividend"]),
            )
            for row in rows
        ]
        return sorted(dividends, key=lambda dividend: dividend.date)

    def get_commodity_eod(self, symbol: str, start: DateLike, end: DateLike) -> list[CommodityEod]:
        """Daily EOD prices for a commodity/index symbol (e.g. BZUSD), ascending by date.

        Uses /historical-price-eod/light — rows carry price only (no OHLC).
        Commodities trade some NYSE holidays (DXUSD even has weekend rows);
        callers aligning to the equity calendar must select/forward-fill.
        """
        start_iso, end_iso = _window(start, end)
        rows = self._get_list(
            "/historical-price-eod/light",
            {"symbol": symbol, "from": start_iso, "to": end_iso},
        )
        points = [
            CommodityEod(
                symbol=str(row.get("symbol", symbol)),
                date=_parse_date(row["date"], "historical-price-eod/light"),
                price=float(row["price"]),
                volume=float(row.get("volume") or 0.0),
            )
            for row in rows
        ]
        return sorted(points, key=lambda point: point.date)

    def get_treasury_rates(self, start: DateLike, end: DateLike) -> list[TreasuryCurve]:
        """Daily treasury curve rows for [start, end], ascending by date.

        Transparently chunks requests to <=TREASURY_CHUNK_DAYS calendar days —
        the endpoint silently truncates wider windows (HTTP 200, trailing ~3
        months only) — and merges the chunks, deduplicating boundary dates.
        """
        start_iso, end_iso = _window(start, end)
        window_start = date.fromisoformat(start_iso)
        window_end = date.fromisoformat(end_iso)
        by_date: dict[date, TreasuryCurve] = {}
        chunk_start = window_start
        while chunk_start <= window_end:
            chunk_end = min(chunk_start + timedelta(days=TREASURY_CHUNK_DAYS - 1), window_end)
            rows = self._get_list(
                "/treasury-rates",
                {"from": chunk_start.isoformat(), "to": chunk_end.isoformat()},
            )
            for row in rows:
                row_date = _parse_date(row["date"], "treasury-rates")
                by_date[row_date] = TreasuryCurve(
                    date=row_date,
                    **{tenor: _tenor_value(row, tenor) for tenor in _TREASURY_TENORS},
                )
            chunk_start = chunk_end + timedelta(days=1)
        return [by_date[row_date] for row_date in sorted(by_date)]

    def iter_stock_news_pages(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        limit: int = NEWS_PAGE_LIMIT,
        max_pages: int = NEWS_MAX_PAGES,
    ) -> Iterator[list[NewsArticle]]:
        """Non-empty /news/stock pages for [start, end], newest-first as served.

        Termination rule (probe 2026-08-24): ONLY an empty page ends history —
        mid-stream pages can come back short of ``limit`` (page-size jitter)
        with more history behind them, so a short page must NOT stop the walk.
        The terminal empty page is fetched but not yielded.
        """
        start_iso, end_iso = _window(start, end)
        for page in range(max_pages):
            rows = self._get_list(
                "/news/stock",
                {
                    "symbols": symbol,
                    "from": start_iso,
                    "to": end_iso,
                    "limit": str(limit),
                    "page": str(page),
                },
            )
            if not rows:
                return
            yield [
                NewsArticle(
                    symbol=str(row.get("symbol", symbol)),
                    published=_parse_published(row["publishedDate"]),
                    publisher=str(row.get("publisher") or ""),
                    title=str(row.get("title") or ""),
                    url=str(row.get("url") or ""),
                )
                for row in rows
            ]
        raise FmpError(
            f"news/stock for {symbol} still returning articles after {max_pages} pages "
            f"({start_iso}..{end_iso}) — raise max_pages if the window is legitimately huge"
        )

    def get_stock_news(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        limit: int = NEWS_PAGE_LIMIT,
        max_pages: int = NEWS_MAX_PAGES,
    ) -> list[NewsArticle]:
        """All stock-news articles for [start, end], ascending by published time.

        Flattens iter_stock_news_pages, deduplicating (published, url) pairs
        that straddle a page boundary.
        """
        by_key: dict[tuple[datetime, str], NewsArticle] = {}
        for page in self.iter_stock_news_pages(symbol, start, end, limit, max_pages):
            for article in page:
                by_key[(article.published, article.url)] = article
        return [by_key[key] for key in sorted(by_key)]

    def _get_list(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        payload = self._get(path, params)
        if not isinstance(payload, list):
            raise FmpError(f"expected a JSON list from {path}, got: {str(payload)[:200]}")
        return payload

    def _get(self, path: str, params: dict[str, str]) -> Any:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            response = self.session.get(url, params=params, timeout=self.timeout)
            status = getattr(response, "status_code", None)
            if status == 429:
                if attempt >= self.max_retries:
                    raise FmpRateLimitError(
                        f"FMP kept returning 429 for {path} after "
                        f"{self.max_retries} retries; back off and retry later"
                    )
                self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue
            if status in (401, 403):
                raise FmpAuthError(
                    f"FMP returned {status} for {path}: no valid apikey was injected. "
                    "Run this process under `onecli run --agent rdq-research` (or "
                    "rdq-exec-paper for the nightly refresh) and make "
                    "sure the financialmodelingprep.com secret is vaulted and assigned "
                    "(ops/setup_onecli.sh, then verify with ops/check_onecli.sh). "
                    f"Body: {response.text[:200]}"
                )
            if status != 200:
                raise FmpError(f"FMP returned HTTP {status} for {path}: {response.text[:200]}")
            return response.json()

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After") if response.headers else None
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass  # non-numeric Retry-After (HTTP-date form): fall back to exponential
        return DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt)

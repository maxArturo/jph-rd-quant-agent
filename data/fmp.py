"""Typed FMP (Financial Modeling Prep) client for EOD bars, splits, and dividends.

All requests are bare HTTPS: no apikey appears in code, env, or params. The
OneCLI proxy injects the query-param key when the process runs under
`onecli run --agent rdq-research` or `--agent rdq-exec-paper` (the identities
with the financialmodelingprep.com secret assignment).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

BASE_URL = "https://financialmodelingprep.com/stable"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 2.0

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

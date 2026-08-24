"""Per-ticker daily news-attention counts from FMP stock news (US-072).

Point-in-time cutoff: an article counts toward the trading day whose close
could have "known" it — published at or before 16:00:00 US/Eastern on a
trading day counts toward THAT day; anything later (after the close, or on a
weekend/holiday) counts toward the NEXT trading day. FMP publishedDate is
already US/Eastern wall-clock at second resolution (probe-verified
2026-08-24, docs/decisions.md US-071), so bucketing is pure date/time
arithmetic on the store's trading calendar — no tz conversion, DST-safe by
construction.

Layout under the news root (default ~/rdq-data/news, outside the repo tree):
- archive/<SYM>/<YYYY-MM-DD>.json — raw records for articles PUBLISHED on that
  ET calendar date: [{"ts", "headline", "url", "publisher"}] ascending by ts.
  Date files are rewritten wholesale per fetch (stale window files removed
  first), so a crashed run's rerun overwrites — never duplicates. Sentiment
  scoring later reads this archive and never refetches.
- checkpoints/<SYM>.json — fetch-completion marker (build_store.py pattern):
  reused only when its (start, end) window matches; counts always derive from
  the archive, the checkpoint only says "this window is fully fetched".

Counts are 0 (not NaN) on no-news days: the series is gapless over the
trading days of the requested window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from data.build_store import DEFAULT_STORE_PATH
from data.fmp import DateLike, FmpClient, NewsArticle, _to_iso_date
from data.refresh import RefreshError, default_end, read_calendar

DEFAULT_NEWS_ROOT = "~/rdq-data/news"
# Article published strictly after this ET wall-clock time rolls to the next
# trading day (16:00:00 exactly still counts toward the same day's close).
PIT_CUTOFF = time(16, 0, 0)
# Canonical backfill start (probe-verified history depth, docs/decisions.md
# US-071 — mega-cap history reaches exactly this date, not plan-capped).
NEWS_BACKFILL_START = date(2025, 1, 2)
# Throttle recorded in docs/decisions.md US-071 (well under the plan's 750/min).
DEFAULT_THROTTLE_RPS = 3.0

# fetch_pages(symbol, start_iso, end_iso) -> pages of articles; each consumed
# page is one HTTP request, and the walk always ends by fetching one final
# empty page (FmpClient.iter_stock_news_pages contract).
FetchPagesFn = Callable[[str, str, str], Iterator[list[NewsArticle]]]


class NewsBuildError(RuntimeError):
    """Raised when the news backfill cannot proceed (bad args, bad store)."""


@dataclass(frozen=True)
class TickerNewsResult:
    """Outcome for one ticker: fetched fresh or resumed from its checkpoint."""

    symbol: str
    articles: int
    requests: int
    fetched: bool  # False when the (start, end)-matching checkpoint was reused
    error: str | None = None


def archive_dir(root: Path) -> Path:
    return root / "archive"


def checkpoint_dir(root: Path) -> Path:
    return root / "checkpoints"


# ---------------------------------------------------------------------------
# PIT bucketing and counts


def pit_bucket(published: datetime, trading_days: Sequence[date]) -> date | None:
    """Trading day the article counts toward, or None past the calendar end.

    ``trading_days`` must be ascending. Published at or before PIT_CUTOFF on a
    trading day -> that day; otherwise the next trading day strictly after the
    published date.
    """
    published_day = published.date()
    index = bisect_left(trading_days, published_day)
    is_trading_day = index < len(trading_days) and trading_days[index] == published_day
    if is_trading_day and published.time() <= PIT_CUTOFF:
        return published_day
    next_index = bisect_right(trading_days, published_day)
    return trading_days[next_index] if next_index < len(trading_days) else None


def read_archived_records(root: Path, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
    """Raw records archived for symbol with a published DATE inside [start, end]."""
    ticker_dir = archive_dir(root) / symbol
    if not ticker_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(ticker_dir.glob("*.json")):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            raise NewsBuildError(f"unexpected non-date archive file {path}") from None
        if start <= file_date <= end:
            records.extend(json.loads(path.read_text()))
    return records


def daily_counts(
    root: Path,
    symbol: str,
    trading_days: Sequence[date],
    start: date,
    end: date,
) -> list[tuple[date, int]]:
    """Gapless (trading day, article count) series over [start, end].

    Every trading day in the window appears; no-news days are 0. Articles
    whose PIT bucket falls past ``end`` (or past the calendar) are excluded
    here but stay in the archive — they count once a later window covers them.
    """
    window_days = [d for d in trading_days if start <= d <= end]
    counts = {d: 0 for d in window_days}
    for record in read_archived_records(root, symbol, start, end):
        published = datetime.strptime(str(record["ts"]), "%Y-%m-%d %H:%M:%S")
        bucket = pit_bucket(published, trading_days)
        if bucket is not None and bucket in counts:
            counts[bucket] += 1
    return [(d, counts[d]) for d in window_days]


# ---------------------------------------------------------------------------
# Archive + checkpoints


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _archive_articles(
    root: Path, symbol: str, articles: Iterable[NewsArticle], start: date, end: date
) -> int:
    """Rewrite the symbol's archive date-files for [start, end] wholesale.

    Existing date files inside the window are removed first so a rerun after a
    mid-ticker crash overwrites cleanly — no duplicates, no stale leftovers.
    Returns the number of archived records (deduplicated on (ts, url)).
    """
    ticker_dir = archive_dir(root) / symbol
    ticker_dir.mkdir(parents=True, exist_ok=True)
    for path in ticker_dir.glob("*.json"):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            raise NewsBuildError(f"unexpected non-date archive file {path}") from None
        if start <= file_date <= end:
            path.unlink()
    by_day: dict[date, dict[tuple[str, str], dict[str, str]]] = {}
    for article in articles:
        ts = article.published.isoformat(sep=" ")
        by_day.setdefault(article.published.date(), {})[(ts, article.url)] = {
            "ts": ts,
            "headline": article.title,
            "url": article.url,
            "publisher": article.publisher,
        }
    total = 0
    for day, by_key in by_day.items():
        records = [by_key[key] for key in sorted(by_key)]
        _write_json(ticker_dir / f"{day.isoformat()}.json", records)
        total += len(records)
    return total


def _load_checkpoint(path: Path, start_iso: str, end_iso: str) -> dict[str, Any] | None:
    """Return the checkpoint payload, or None if absent/window-mismatched/corrupt."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("start") != start_iso or payload.get("end") != end_iso:
            return None
        return {
            "articles": int(payload["articles"]),
            "requests": int(payload["requests"]),
        }
    except (ValueError, KeyError, TypeError):
        return None  # corrupt checkpoint: refetch


# ---------------------------------------------------------------------------
# Backfill


def backfill_ticker(
    fetch_pages: FetchPagesFn,
    symbol: str,
    start: DateLike,
    end: DateLike,
    root: Path,
    sleep: Callable[[float], None] = time_module.sleep,
    throttle_rps: float = DEFAULT_THROTTLE_RPS,
) -> TickerNewsResult:
    """Fetch, archive, and checkpoint one ticker's news for [start, end].

    A checkpoint whose (start, end) window matches short-circuits the fetch
    entirely (crash-resume without refetching). ``sleep`` runs once before
    every HTTP request (including the terminal empty page) at 1/throttle_rps.
    """
    start_iso = _to_iso_date(start, "start")
    end_iso = _to_iso_date(end, "end")
    ckpt_dir = checkpoint_dir(root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{symbol}.json"
    existing = _load_checkpoint(ckpt_path, start_iso, end_iso)
    if existing is not None:
        return TickerNewsResult(
            symbol=symbol,
            articles=existing["articles"],
            requests=existing["requests"],
            fetched=False,
        )
    pause = 1.0 / throttle_rps if throttle_rps > 0 else 0.0
    articles: list[NewsArticle] = []
    pages = fetch_pages(symbol, start_iso, end_iso)
    requests = 0
    while True:
        sleep(pause)
        requests += 1  # the next() below is exactly one HTTP request
        page = next(pages, None)
        if page is None:
            break
        articles.extend(page)
    archived = _archive_articles(
        root, symbol, articles, date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    )
    _write_json(
        ckpt_path,
        {
            "symbol": symbol,
            "start": start_iso,
            "end": end_iso,
            "articles": archived,
            "requests": requests,
        },
    )
    return TickerNewsResult(symbol=symbol, articles=archived, requests=requests, fetched=True)


def backfill_news(
    fetch_pages: FetchPagesFn,
    symbols: Sequence[str],
    start: DateLike,
    end: DateLike,
    root: Path,
    sleep: Callable[[float], None] = time_module.sleep,
    throttle_rps: float = DEFAULT_THROTTLE_RPS,
    log: Callable[[str], None] | None = None,
) -> list[TickerNewsResult]:
    """Backfill every symbol, checkpointing each as it lands.

    A per-ticker fetch failure is recorded on its result (error=...) and the
    walk continues — reruns resume from checkpoints, so retrying failures is
    just running the same command again.
    """
    results: list[TickerNewsResult] = []
    for symbol in symbols:
        try:
            result = backfill_ticker(
                fetch_pages, symbol, start, end, root, sleep=sleep, throttle_rps=throttle_rps
            )
        except Exception as exc:  # noqa: BLE001 - recorded, rerun resumes
            result = TickerNewsResult(
                symbol=symbol, articles=0, requests=0, fetched=True, error=str(exc)
            )
        results.append(result)
        if log is not None:
            if result.error is not None:
                log(f"FAIL  {symbol}  {result.error}")
            else:
                mode = "fetched" if result.fetched else "resumed"
                log(
                    f"OK    {symbol}  articles={result.articles} "
                    f"requests={result.requests} ({mode})"
                )
    return results


# ---------------------------------------------------------------------------
# CLI


def _universe_symbols(store: Path, name: str) -> list[str]:
    """Distinct symbols of instruments/<name>.txt (PIT files repeat symbols)."""
    path = store / "instruments" / f"{name}.txt"
    if not path.exists():
        raise NewsBuildError(f"store at {store} has no universe file {path}")
    symbols = {line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()}
    if not symbols:
        raise NewsBuildError(f"universe file {path} lists no tickers")
    return sorted(symbols)


def _client_fetch_pages(client: FmpClient) -> FetchPagesFn:
    def fetch(symbol: str, start_iso: str, end_iso: str) -> Iterator[list[NewsArticle]]:
        return client.iter_stock_news_pages(symbol, start_iso, end_iso)

    return fetch


def main(
    argv: Sequence[str] | None = None,
    fetch_pages: FetchPagesFn | None = None,
    sleep: Callable[[float], None] = time_module.sleep,
) -> int:
    parser = argparse.ArgumentParser(description="Backfill per-ticker daily news counts from FMP")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--tickers", help="comma-separated symbols to backfill")
    who.add_argument("--universe", help="store universe name (instruments/<name>.txt)")
    parser.add_argument("--start", default=NEWS_BACKFILL_START.isoformat())
    parser.add_argument(
        "--end", default=None, help="last published date, default yesterday America/New_York"
    )
    parser.add_argument("--root", default=DEFAULT_NEWS_ROOT, help="archive + checkpoint root")
    parser.add_argument("--store", default=DEFAULT_STORE_PATH, help="Qlib store (trading calendar)")
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE_RPS, help="requests/s")
    parser.add_argument(
        "--print-counts",
        metavar="SYM",
        help="after the backfill, print SYM's full daily count series (spot-checks)",
    )
    args = parser.parse_args(argv)
    try:
        store = Path(args.store).expanduser()
        root = Path(args.root).expanduser()
        trading_days = read_calendar(store)
        symbols = (
            [s for s in (t.strip() for t in args.tickers.split(",")) if s]
            if args.tickers
            else _universe_symbols(store, args.universe)
        )
        if not symbols:
            raise NewsBuildError("no tickers to backfill")
        start = date.fromisoformat(_to_iso_date(args.start, "start"))
        end = (
            date.fromisoformat(_to_iso_date(args.end, "end"))
            if args.end is not None
            else default_end()
        )
        fetch = fetch_pages if fetch_pages is not None else _client_fetch_pages(FmpClient())
        results = backfill_news(
            fetch,
            symbols,
            start,
            end,
            root,
            sleep=sleep,
            throttle_rps=args.throttle,
            log=print,
        )
        failures = [r for r in results if r.error is not None]
        for result in results:
            if result.error is not None:
                continue
            series = daily_counts(root, result.symbol, trading_days, start, end)
            nonzero = [(d, c) for d, c in series if c > 0]
            peak = max(series, key=lambda item: item[1]) if series else None
            peak_text = f"max={peak[0].isoformat()}:{peak[1]}" if peak else "max=n/a"
            print(
                f"COUNTS  {result.symbol}  trading_days={len(series)} "
                f"nonzero={len(nonzero)} {peak_text}"
            )
        if args.print_counts:
            for day, count in daily_counts(root, args.print_counts, trading_days, start, end):
                print(f"{args.print_counts}  {day.isoformat()}  {count}")
        total_articles = sum(r.articles for r in results)
        total_requests = sum(r.requests for r in results)
        print(
            f"news backfill: {len(results) - len(failures)}/{len(results)} ticker(s) ok, "
            f"articles={total_articles} requests={total_requests} "
            f"window={start.isoformat()}..{end.isoformat()} root={root}"
        )
        return 1 if failures else 0
    except (NewsBuildError, RefreshError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for data/build_news.py: PIT cutoff bucketing, checkpointed backfill,
archive layout, 0-fill daily counts, and the CLI. No live HTTP anywhere."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from data.build_news import (
    NEWS_BACKFILL_START,
    NewsBuildError,
    TickerNewsResult,
    archive_dir,
    backfill_news,
    backfill_ticker,
    checkpoint_dir,
    checkpoint_window,
    daily_counts,
    main,
    pit_bucket,
    read_news_series,
)
from data.build_store import TickerBundle, build_store
from data.fmp import NewsArticle
from tests.test_build_store import DAYS, make_bars


def _weekdays(start: date, end: date) -> list[date]:
    day, out = start, []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


# Weekday trading calendar around both 2025 DST transitions, with a synthetic
# holiday on Monday 2025-03-17. Spring-forward: Sun 2025-03-09 (02:00 ET does
# not exist); fall-back: Sun 2025-11-02 (01:xx ET happens twice).
HOLIDAY = date(2025, 3, 17)
TRADING = [
    d for d in _weekdays(date(2025, 3, 3), date(2025, 3, 21)) if d != HOLIDAY
] + _weekdays(date(2025, 10, 27), date(2025, 11, 7))


def ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def art(when: str, symbol: str = "AAPL", url: str | None = None) -> NewsArticle:
    return NewsArticle(
        symbol=symbol,
        published=ts(when),
        publisher="Reuters",
        title=f"headline {when}",
        url=url or f"https://news.example/{symbol}/{when.replace(' ', 'T')}",
    )


class FakeNewsFetcher:
    """fetch_pages stand-in serving canned articles in pages of 2."""

    def __init__(
        self, articles: dict[str, list[NewsArticle]], fail: frozenset[str] | set[str] = frozenset()
    ) -> None:
        self.articles = articles
        self.fail = set(fail)
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, symbol: str, start_iso: str, end_iso: str) -> Iterator[list[NewsArticle]]:
        self.calls.append((symbol, start_iso, end_iso))
        if symbol in self.fail:
            raise RuntimeError(f"simulated FMP outage for {symbol}")
        rows = self.articles.get(symbol, [])
        for i in range(0, len(rows), 2):
            yield rows[i : i + 2]


class TestPitBucket:
    def test_before_close_counts_same_day(self) -> None:
        assert pit_bucket(ts("2025-03-04 15:59:59"), TRADING) == date(2025, 3, 4)

    def test_exactly_1600_counts_same_day(self) -> None:
        assert pit_bucket(ts("2025-03-04 16:00:00"), TRADING) == date(2025, 3, 4)

    def test_one_second_after_close_rolls_to_next_trading_day(self) -> None:
        assert pit_bucket(ts("2025-03-04 16:00:01"), TRADING) == date(2025, 3, 5)

    def test_friday_evening_rolls_to_monday(self) -> None:
        assert pit_bucket(ts("2025-03-07 17:30:00"), TRADING) == date(2025, 3, 10)

    def test_saturday_rolls_to_monday(self) -> None:
        assert pit_bucket(ts("2025-03-08 09:00:00"), TRADING) == date(2025, 3, 10)

    def test_spring_forward_sunday_rolls_to_monday_unshifted(self) -> None:
        # 2025-03-09 is the ET spring-forward date; timestamps are already ET
        # wall-clock so bucketing must not apply any tz math around 02:00.
        assert pit_bucket(ts("2025-03-09 01:30:00"), TRADING) == date(2025, 3, 10)
        assert pit_bucket(ts("2025-03-09 03:30:00"), TRADING) == date(2025, 3, 10)

    def test_day_after_fall_back_keeps_wall_clock(self) -> None:
        # Monday after the 2025-11-02 fall-back: 15:59:59 must stay same-day
        # (a UTC-offset conversion would shift it across the cutoff).
        assert pit_bucket(ts("2025-11-03 15:59:59"), TRADING) == date(2025, 11, 3)
        assert pit_bucket(ts("2025-10-31 16:00:01"), TRADING) == date(2025, 11, 3)

    def test_holiday_rolls_forward(self) -> None:
        assert pit_bucket(ts("2025-03-17 10:00:00"), TRADING) == date(2025, 3, 18)

    def test_before_calendar_start_rolls_to_first_trading_day(self) -> None:
        assert pit_bucket(ts("2025-03-01 10:00:00"), TRADING) == date(2025, 3, 3)

    def test_past_calendar_end_is_none(self) -> None:
        assert pit_bucket(ts("2025-11-07 16:00:01"), TRADING) is None


class TestBackfill:
    def test_archive_keyed_by_ticker_and_date(self, tmp_path: Path) -> None:
        fetcher = FakeNewsFetcher(
            {
                "AAPL": [
                    art("2025-03-04 09:00:00"),
                    art("2025-03-04 17:15:30"),
                    art("2025-03-06 12:00:00"),
                ]
            }
        )
        result = backfill_ticker(
            fetcher, "AAPL", "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        assert result == TickerNewsResult(symbol="AAPL", articles=3, requests=3, fetched=True)
        ticker_dir = archive_dir(tmp_path) / "AAPL"
        assert sorted(p.name for p in ticker_dir.iterdir()) == [
            "2025-03-04.json",
            "2025-03-06.json",
        ]
        records = json.loads((ticker_dir / "2025-03-04.json").read_text())
        assert [r["ts"] for r in records] == ["2025-03-04 09:00:00", "2025-03-04 17:15:30"]
        assert set(records[0]) == {"ts", "headline", "url", "publisher"}
        assert records[0]["headline"] == "headline 2025-03-04 09:00:00"
        assert records[0]["publisher"] == "Reuters"
        assert records[0]["url"].startswith("https://news.example/AAPL/")

    def test_request_count_and_throttle_sleeps(self, tmp_path: Path) -> None:
        # 5 articles in pages of 2 -> 3 yielded pages + 1 terminal empty page.
        fetcher = FakeNewsFetcher(
            {"AAPL": [art(f"2025-03-0{d} 10:00:00") for d in (3, 4, 5, 6, 7)]}
        )
        sleeps: list[float] = []
        result = backfill_ticker(
            fetcher,
            "AAPL",
            "2025-03-03",
            "2025-03-14",
            tmp_path,
            sleep=sleeps.append,
            throttle_rps=2.0,
        )
        assert result.requests == 4
        assert result.articles == 5
        assert sleeps == [0.5, 0.5, 0.5, 0.5]  # one pause per HTTP request

    def test_crash_resume_skips_completed_ticker(self, tmp_path: Path) -> None:
        articles = {
            "AAPL": [art("2025-03-04 10:00:00")],
            "NVDA": [art("2025-03-05 11:00:00", symbol="NVDA")],
        }
        broken = FakeNewsFetcher(articles, fail={"NVDA"})
        results = backfill_news(
            broken, ["AAPL", "NVDA"], "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        assert results[0].error is None and results[0].fetched
        assert results[1].error is not None and "NVDA" in results[1].error
        assert not (checkpoint_dir(tmp_path) / "NVDA.json").exists()

        healthy = FakeNewsFetcher(articles)
        results = backfill_news(
            healthy, ["AAPL", "NVDA"], "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        assert [(r.symbol, r.fetched, r.error) for r in results] == [
            ("AAPL", False, None),  # checkpoint reused, not refetched
            ("NVDA", True, None),
        ]
        assert [c[0] for c in healthy.calls] == ["NVDA"]

    def test_window_change_refetches(self, tmp_path: Path) -> None:
        fetcher = FakeNewsFetcher({"AAPL": [art("2025-03-04 10:00:00")]})
        for _ in range(2):
            backfill_ticker(
                fetcher, "AAPL", "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
            )
        assert len(fetcher.calls) == 1  # same window: checkpoint reused
        result = backfill_ticker(
            fetcher, "AAPL", "2025-03-03", "2025-03-21", tmp_path, sleep=lambda _s: None
        )
        assert len(fetcher.calls) == 2 and result.fetched

    def test_corrupt_checkpoint_refetches(self, tmp_path: Path) -> None:
        fetcher = FakeNewsFetcher({"AAPL": [art("2025-03-04 10:00:00")]})
        checkpoint_dir(tmp_path).mkdir(parents=True)
        (checkpoint_dir(tmp_path) / "AAPL.json").write_text("{not json")
        result = backfill_ticker(
            fetcher, "AAPL", "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        assert result.fetched and len(fetcher.calls) == 1

    def test_rerun_after_mid_ticker_crash_never_duplicates(self, tmp_path: Path) -> None:
        # Simulate a crash after archive files landed but before the
        # checkpoint: the rerun refetches and rewrites wholesale.
        fetcher = FakeNewsFetcher(
            {"AAPL": [art("2025-03-04 10:00:00"), art("2025-03-04 11:00:00")]}
        )
        backfill_ticker(
            fetcher, "AAPL", "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        (checkpoint_dir(tmp_path) / "AAPL.json").unlink()
        result = backfill_ticker(
            fetcher, "AAPL", "2025-03-03", "2025-03-14", tmp_path, sleep=lambda _s: None
        )
        assert result.articles == 2
        records = json.loads((archive_dir(tmp_path) / "AAPL" / "2025-03-04.json").read_text())
        assert len(records) == 2


class TestDailyCounts:
    def _backfill(self, tmp_path: Path, articles: list[NewsArticle], end: str) -> None:
        fetcher = FakeNewsFetcher({"AAPL": articles})
        backfill_ticker(fetcher, "AAPL", "2025-03-03", end, tmp_path, sleep=lambda _s: None)

    def test_no_news_days_are_zero_and_series_is_gapless(self, tmp_path: Path) -> None:
        self._backfill(
            tmp_path,
            [art("2025-03-04 10:00:00"), art("2025-03-12 09:00:00")],
            "2025-03-21",
        )
        window_days = [d for d in TRADING if d <= date(2025, 3, 21)]
        series = daily_counts(tmp_path, "AAPL", TRADING, date(2025, 3, 3), date(2025, 3, 21))
        assert [d for d, _c in series] == window_days  # every trading day, in order
        counts = dict(series)
        assert counts[date(2025, 3, 4)] == 1
        assert counts[date(2025, 3, 12)] == 1
        zero_days = [d for d, c in series if c == 0]
        assert len(zero_days) == len(window_days) - 2
        assert all(isinstance(c, int) for _d, c in series)

    def test_cutoff_and_weekend_bucketing_end_to_end(self, tmp_path: Path) -> None:
        self._backfill(
            tmp_path,
            [
                art("2025-03-04 16:00:00"),  # boundary: same day
                art("2025-03-04 16:00:01"),  # next trading day
                art("2025-03-08 12:00:00"),  # Saturday -> Monday
                art("2025-03-09 02:30:00"),  # spring-forward Sunday -> Monday
            ],
            "2025-03-21",
        )
        counts = dict(daily_counts(tmp_path, "AAPL", TRADING, date(2025, 3, 3), date(2025, 3, 21)))
        assert counts[date(2025, 3, 4)] == 1
        assert counts[date(2025, 3, 5)] == 1
        assert counts[date(2025, 3, 10)] == 2

    def test_rollover_past_window_end_stays_archived(self, tmp_path: Path) -> None:
        # Published after the close of the window's last trading day: not
        # counted in this window, but counted once a later window covers the
        # next trading day (2025-03-17 is the synthetic holiday -> 03-18).
        self._backfill(tmp_path, [art("2025-03-14 17:00:00")], "2025-03-14")
        short = dict(daily_counts(tmp_path, "AAPL", TRADING, date(2025, 3, 3), date(2025, 3, 14)))
        assert sum(short.values()) == 0
        extended = dict(
            daily_counts(tmp_path, "AAPL", TRADING, date(2025, 3, 3), date(2025, 3, 21))
        )
        assert extended[date(2025, 3, 18)] == 1

    def test_missing_ticker_archive_is_all_zeros(self, tmp_path: Path) -> None:
        series = daily_counts(tmp_path, "GHOST", TRADING, date(2025, 3, 3), date(2025, 3, 14))
        assert len(series) > 0
        assert all(c == 0 for _d, c in series)


class TestReadNewsSeries:
    """read_news_series: the US-073 bridge from the archive to build_store's
    ticker_series= — explicit float zeros, checkpoint-coverage enforced."""

    START, END = date(2025, 3, 3), date(2025, 3, 21)

    def _backfill(self, root: Path, symbol: str, articles: list[NewsArticle]) -> None:
        fetcher = FakeNewsFetcher({symbol: articles})
        backfill_ticker(
            fetcher, symbol, self.START, self.END, root, sleep=lambda _s: None
        )

    def test_observations_are_float_counts_with_explicit_zeros(self, tmp_path: Path) -> None:
        self._backfill(tmp_path, "AAPL", [art("2025-03-04 10:00:00"), art("2025-03-04 11:00:00")])
        self._backfill(tmp_path, "MSFT", [art("2025-03-12 09:00:00", symbol="MSFT")])
        series = read_news_series(tmp_path, ["AAPL", "MSFT"], TRADING, self.START, self.END)
        assert set(series) == {"AAPL", "MSFT"}
        window_days = [d for d in TRADING if self.START <= d <= self.END]
        for symbol in ("AAPL", "MSFT"):
            assert [d for d, _v in series[symbol]] == window_days  # gapless
            assert all(isinstance(v, float) for _d, v in series[symbol])
        counts = dict(series["AAPL"])
        assert counts[date(2025, 3, 4)] == 2.0
        assert counts[date(2025, 3, 5)] == 0.0  # covered no-news day: explicit 0
        assert dict(series["MSFT"])[date(2025, 3, 12)] == 1.0

    def test_narrower_window_than_checkpoint_is_covered(self, tmp_path: Path) -> None:
        self._backfill(tmp_path, "AAPL", [art("2025-03-04 10:00:00")])
        series = read_news_series(
            tmp_path, ["AAPL"], TRADING, date(2025, 3, 4), date(2025, 3, 14)
        )
        assert [d for d, _v in series["AAPL"]] == [
            d for d in TRADING if date(2025, 3, 4) <= d <= date(2025, 3, 14)
        ]

    def test_never_backfilled_symbol_fails_loud(self, tmp_path: Path) -> None:
        self._backfill(tmp_path, "AAPL", [art("2025-03-04 10:00:00")])
        with pytest.raises(NewsBuildError, match="GHOST"):
            read_news_series(tmp_path, ["AAPL", "GHOST"], TRADING, self.START, self.END)

    def test_checkpoint_not_covering_window_fails_loud(self, tmp_path: Path) -> None:
        self._backfill(tmp_path, "AAPL", [art("2025-03-04 10:00:00")])
        assert checkpoint_window(tmp_path, "AAPL") == (self.START, self.END)
        with pytest.raises(NewsBuildError, match="AAPL"):
            read_news_series(
                tmp_path, ["AAPL"], TRADING, self.START, self.END + timedelta(days=1)
            )


@pytest.fixture()
def fixture_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    build_store([TickerBundle("AAPL", make_bars("AAPL"), (), ())], store)
    return store


class TestCli:
    def test_backfill_and_counts_output(
        self, tmp_path: Path, fixture_store: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetcher = FakeNewsFetcher(
            {
                "AAPL": [
                    art("2024-01-03 10:00:00"),
                    art("2024-01-03 11:00:00"),
                    art("2024-01-03 17:00:00"),  # after close -> 2024-01-04
                ]
            }
        )
        rc = main(
            [
                "--tickers",
                "AAPL",
                "--start",
                DAYS[0].isoformat(),
                "--end",
                DAYS[-1].isoformat(),
                "--root",
                str(tmp_path / "news"),
                "--store",
                str(fixture_store),
                "--print-counts",
                "AAPL",
            ],
            fetch_pages=fetcher,
            sleep=lambda _s: None,
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "OK    AAPL  articles=3 requests=3 (fetched)" in out
        assert "COUNTS  AAPL  trading_days=5 nonzero=2 max=2024-01-03:2" in out
        assert "AAPL  2024-01-03  2" in out
        assert "AAPL  2024-01-04  1" in out
        assert "AAPL  2024-01-08  0" in out
        assert "news backfill: 1/1 ticker(s) ok, articles=3 requests=3" in out

    def test_universe_mode_uses_store_instruments(
        self, tmp_path: Path, fixture_store: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetcher = FakeNewsFetcher({"AAPL": [art("2024-01-03 10:00:00")]})
        rc = main(
            [
                "--universe",
                "all",
                "--start",
                DAYS[0].isoformat(),
                "--end",
                DAYS[-1].isoformat(),
                "--root",
                str(tmp_path / "news"),
                "--store",
                str(fixture_store),
            ],
            fetch_pages=fetcher,
            sleep=lambda _s: None,
        )
        assert rc == 0
        assert [c[0] for c in fetcher.calls] == ["AAPL"]
        assert "1/1 ticker(s) ok" in capsys.readouterr().out

    def test_ticker_failure_exits_nonzero_but_continues(
        self, tmp_path: Path, fixture_store: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetcher = FakeNewsFetcher(
            {"AAPL": [art("2024-01-03 10:00:00")]}, fail={"NVDA"}
        )
        rc = main(
            [
                "--tickers",
                "NVDA,AAPL",
                "--start",
                DAYS[0].isoformat(),
                "--end",
                DAYS[-1].isoformat(),
                "--root",
                str(tmp_path / "news"),
                "--store",
                str(fixture_store),
            ],
            fetch_pages=fetcher,
            sleep=lambda _s: None,
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "FAIL  NVDA" in out
        assert "OK    AAPL" in out  # the failure did not stop the walk
        assert "news backfill: 1/2 ticker(s) ok" in out

    def test_missing_store_errors_loud(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "--tickers",
                "AAPL",
                "--root",
                str(tmp_path / "news"),
                "--store",
                str(tmp_path / "no_store"),
            ],
            fetch_pages=FakeNewsFetcher({}),
            sleep=lambda _s: None,
        )
        assert rc == 1
        assert "ERROR:" in capsys.readouterr().err

    def test_default_start_is_canonical(self) -> None:
        assert NEWS_BACKFILL_START == date(2025, 1, 2)

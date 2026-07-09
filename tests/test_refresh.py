"""Tests for data/refresh.py: incremental FMP refresh of an existing store (US-036)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from data.build_store import BuildError, TickerBundle, build_store
from data.fmp import Dividend, EodBar, Split
from data.refresh import (
    RefreshError,
    main,
    read_raw_bars,
    read_universes,
    refresh_store,
)
from tests.test_build_store import DAYS, FakeFmp, make_bars, read_bin

# Two more consecutive weekdays after DAYS (Tue 2024-01-09, Wed 2024-01-10).
NEW_DAYS = [date(2024, 1, 9), date(2024, 1, 10)]
LAST_OLD = DAYS[-1]  # 2024-01-08


class WindowedFakeFmp(FakeFmp):
    """FakeFmp that honors the [start, end] window and records each request."""

    def __init__(
        self,
        bars: dict[str, tuple[EodBar, ...]],
        splits: dict[str, tuple[Split, ...]] | None = None,
        dividends: dict[str, tuple[Dividend, ...]] | None = None,
    ) -> None:
        super().__init__(bars=bars, splits=splits, dividends=dividends)
        self.windows: list[tuple[str, str, str]] = []

    def get_eod_bars(self, symbol: str, start: object, end: object) -> list[EodBar]:
        self.windows.append((symbol, str(start), str(end)))
        all_bars = super().get_eod_bars(symbol, start, end)
        return [b for b in all_bars if str(start) <= b.date.isoformat() <= str(end)]


def build_fixture_store(tmp_path: Path, symbols: dict[str, tuple[EodBar, ...]]) -> Path:
    store = tmp_path / "us_data"
    build_store(
        [TickerBundle(sym, bars, (), ()) for sym, bars in symbols.items()],
        store,
    )
    return store


def store_snapshot(store: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) per file — enough to prove nothing was rewritten."""
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(store.rglob("*")):
        if path.is_file():
            stat = os.stat(path)
            out[str(path.relative_to(store))] = (stat.st_size, stat.st_mtime_ns)
    return out


# ---------------------------------------------------------------------------
# Reading the store back


def test_read_raw_bars_roundtrip(tmp_path: Path) -> None:
    """Raw bars survive the store round-trip (adjusted bins -> raw), split included."""
    bars = make_bars("AAPL")
    split = Split("AAPL", DAYS[2], 2.0, 1.0)
    dividend = Dividend("AAPL", DAYS[3], 0.5)
    store = tmp_path / "us_data"
    build_store([TickerBundle("AAPL", bars, (split,), (dividend,))], store)

    calendar_text = (store / "calendars" / "day.txt").read_text()
    calendar = [date.fromisoformat(x) for x in calendar_text.split()]
    recovered = read_raw_bars(store, "AAPL", calendar)
    assert len(recovered) == len(bars)
    for orig, back in zip(bars, recovered, strict=True):
        assert back.date == orig.date
        assert back.close == pytest.approx(orig.close, rel=1e-5)
        assert back.open == pytest.approx(orig.open, rel=1e-5)
        assert back.volume == pytest.approx(orig.volume, rel=1e-5)


def test_read_raw_bars_missing_ticker_raises(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    with pytest.raises(RefreshError, match="missing feature file"):
        read_raw_bars(store, "MSFT", DAYS)


# ---------------------------------------------------------------------------
# Refresh: appending new bars


def two_ticker_store(tmp_path: Path) -> Path:
    return build_fixture_store(
        tmp_path, {"AAPL": make_bars("AAPL"), "MSFT": make_bars("MSFT", close=200.0)}
    )


def test_refresh_appends_new_bars(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    client = WindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    result = refresh_store(store, client, end=NEW_DAYS[-1])
    assert result.updated is True
    assert result.last_date_before == LAST_OLD
    assert result.last_date_after == NEW_DAYS[-1]
    assert result.new_bars == {"AAPL": 2, "MSFT": 2}

    calendar = (store / "calendars" / "day.txt").read_text().split()
    assert calendar == [d.isoformat() for d in DAYS + NEW_DAYS]
    start, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    assert start == 0
    assert len(closes) == 7
    # make_bars: close = 100 + i; the two new bars continue the ramp (105, 106).
    assert closes[-2:] == pytest.approx([105.0, 106.0])
    # instruments spans extended
    all_lines = (store / "instruments" / "all.txt").read_text().splitlines()
    assert f"AAPL\t{DAYS[0].isoformat()}\t{NEW_DAYS[-1].isoformat()}" in all_lines


def test_refresh_fetches_only_since_each_tickers_last_date(tmp_path: Path) -> None:
    """Incremental windows: start = per-ticker last stored date + 1 day."""
    lagging = make_bars("MSFT", DAYS[:3], close=200.0)  # ends 2024-01-04
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL"), "MSFT": lagging})
    client = WindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    result = refresh_store(store, client, end=NEW_DAYS[-1])
    assert result.updated is True
    windows = dict((sym, (start, end)) for sym, start, end in client.windows)
    assert windows["AAPL"] == ("2024-01-09", "2024-01-10")
    assert windows["MSFT"] == ("2024-01-05", "2024-01-10")
    # MSFT caught up: 01-05, 01-08 + the two new days
    assert result.new_bars["MSFT"] == 4


def test_refresh_recomputes_factors_for_new_split(tmp_path: Path) -> None:
    """A split landing between refreshes re-scales the WHOLE stored history."""
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    _, closes_before = read_bin(store / "features" / "aapl" / "close.day.bin")
    new_split = Split("AAPL", NEW_DAYS[0], 2.0, 1.0)  # 2:1 on 2024-01-09
    client = WindowedFakeFmp(
        bars={"AAPL": make_bars("AAPL", DAYS + NEW_DAYS)},
        splits={"AAPL": (new_split,)},
    )
    result = refresh_store(store, client, end=NEW_DAYS[-1])
    assert result.updated is True
    _, factors = read_bin(store / "features" / "aapl" / "factor.day.bin")
    assert factors[:5] == pytest.approx([0.5] * 5)  # pre-split history re-adjusted
    assert factors[-1] == pytest.approx(1.0)  # backward adjustment anchor
    _, closes_after = read_bin(store / "features" / "aapl" / "close.day.bin")
    assert closes_after[:5] == pytest.approx(np.asarray(closes_before) * 0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# Idempotency


def test_refresh_noop_when_end_not_past_store(tmp_path: Path) -> None:
    """end <= last stored date: no fetch at all, store byte-for-byte untouched."""
    store = two_ticker_store(tmp_path)
    before = store_snapshot(store)
    client = WindowedFakeFmp(bars={})
    result = refresh_store(store, client, end=LAST_OLD)
    assert result.updated is False
    assert result.last_date_before == result.last_date_after == LAST_OLD
    assert client.windows == []
    assert store_snapshot(store) == before


def test_refresh_noop_when_fmp_has_nothing_new(tmp_path: Path) -> None:
    """Weekend/holiday window: FMP returns no bars, store untouched."""
    store = two_ticker_store(tmp_path)
    before = store_snapshot(store)
    client = WindowedFakeFmp(
        bars={"AAPL": make_bars("AAPL"), "MSFT": make_bars("MSFT", close=200.0)}
    )
    result = refresh_store(store, client, end=date(2024, 1, 9))
    assert result.updated is False
    assert len(client.windows) == 2  # it did look
    assert store_snapshot(store) == before


def test_refresh_twice_second_run_is_noop(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    bars = {
        "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
        "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
    }
    assert refresh_store(store, WindowedFakeFmp(bars=bars), end=NEW_DAYS[-1]).updated is True
    after_first = store_snapshot(store)
    second = refresh_store(store, WindowedFakeFmp(bars=bars), end=NEW_DAYS[-1])
    assert second.updated is False
    assert store_snapshot(store) == after_first


# ---------------------------------------------------------------------------
# Universe preservation


def test_refresh_preserves_universes_with_refreshed_spans(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    old_span = f"AAPL\t{DAYS[0].isoformat()}\t{LAST_OLD.isoformat()}\n"
    (store / "instruments" / "tech.txt").write_text(old_span)
    assert read_universes(store) == {"tech": ["AAPL"]}

    client = WindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    assert refresh_store(store, client, end=NEW_DAYS[-1]).updated is True
    refreshed = (store / "instruments" / "tech.txt").read_text()
    assert refreshed == f"AAPL\t{DAYS[0].isoformat()}\t{NEW_DAYS[-1].isoformat()}\n"


def test_build_store_rejects_bad_extra_instruments(tmp_path: Path) -> None:
    bundles = [TickerBundle("AAPL", make_bars("AAPL"), (), ())]
    with pytest.raises(BuildError, match="reserved"):
        build_store(bundles, tmp_path / "s1", extra_instruments={"all": ["AAPL"]})
    with pytest.raises(BuildError, match="not in the store"):
        build_store(bundles, tmp_path / "s2", extra_instruments={"tech": ["MSFT"]})
    # failed builds leave no partial store behind
    assert not (tmp_path / "s1").exists()
    assert not (tmp_path / "s2").exists()


# ---------------------------------------------------------------------------
# Guards + CLI


def test_refresh_refuses_missing_store(tmp_path: Path) -> None:
    with pytest.raises(RefreshError, match="no store at"):
        refresh_store(tmp_path / "nope", WindowedFakeFmp(bars={}))


def test_main_refresh_and_already_current(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    store = two_ticker_store(tmp_path)
    bars = {
        "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
        "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
    }
    argv = ["--store", str(store), "--end", NEW_DAYS[-1].isoformat()]
    assert main(argv, client=WindowedFakeFmp(bars=bars)) == 0
    out = capsys.readouterr().out
    assert "store refreshed" in out
    assert "+4 bars across 2 tickers" in out
    assert "2024-01-08 -> 2024-01-10" in out

    assert main(argv, client=WindowedFakeFmp(bars=bars)) == 0
    assert "already current (last date 2024-01-10)" in capsys.readouterr().out


def test_main_missing_store_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    argv = ["--store", str(tmp_path / "absent"), "--end", "2024-01-10"]
    assert main(argv, client=WindowedFakeFmp(bars={})) == 1
    assert "no store at" in capsys.readouterr().err

"""Tests for data/refresh.py: incremental FMP refresh of an existing store (US-036)."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.build_store import (
    COMMODITY_SYMBOLS,
    MARKET_FIELDS,
    BuildError,
    TickerBundle,
    build_store,
)
from data.fmp import CommodityEod, Dividend, EodBar, FmpError, Split, TreasuryCurve
from data.refresh import (
    RefreshError,
    extend_store,
    main,
    read_market_fields,
    read_market_series,
    read_raw_bars,
    read_universes,
    refresh_store,
)
from tests.test_build_store import (
    DAYS,
    FakeFmp,
    commodity_price,
    make_bars,
    make_curve,
    read_bin,
    weekdays,
)

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
    assert read_universes(store) == {
        "tech": [("AAPL", DAYS[0].isoformat(), LAST_OLD.isoformat())]
    }

    client = WindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    assert refresh_store(store, client, end=NEW_DAYS[-1]).updated is True
    refreshed = (store / "instruments" / "tech.txt").read_text()
    assert refreshed == f"AAPL\t{DAYS[0].isoformat()}\t{NEW_DAYS[-1].isoformat()}\n"


def test_refresh_preserves_multi_span_pit_universe(tmp_path: Path) -> None:
    """A PIT span file survives a refresh: closed spans verbatim, open span extended."""
    store = two_ticker_store(tmp_path)
    closed = ("AAPL", DAYS[0].isoformat(), DAYS[2].isoformat())  # PIT exit: history
    reopened = ("AAPL", DAYS[4].isoformat(), LAST_OLD.isoformat())  # member at store end
    lagging = ("MSFT", DAYS[0].isoformat(), DAYS[3].isoformat())  # exited before store end
    (store / "instruments" / "pit.txt").write_text(
        "".join(f"{s}\t{a}\t{b}\n" for s, a, b in (closed, reopened, lagging))
    )

    client = WindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    assert refresh_store(store, client, end=NEW_DAYS[-1]).updated is True
    refreshed = (store / "instruments" / "pit.txt").read_text().splitlines()
    assert refreshed == [
        f"AAPL\t{DAYS[0].isoformat()}\t{DAYS[2].isoformat()}",
        f"AAPL\t{DAYS[4].isoformat()}\t{NEW_DAYS[-1].isoformat()}",
        f"MSFT\t{DAYS[0].isoformat()}\t{DAYS[3].isoformat()}",
    ]


def test_read_universes_rejects_malformed_line(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    (store / "instruments" / "bad.txt").write_text("AAPL 2024-01-02 2024-01-08\n")
    with pytest.raises(RefreshError, match="malformed instruments line"):
        read_universes(store)


def test_build_store_rejects_bad_extra_instruments(tmp_path: Path) -> None:
    bundles = [TickerBundle("AAPL", make_bars("AAPL"), (), ())]
    row = ("AAPL", DAYS[0].isoformat(), LAST_OLD.isoformat())
    with pytest.raises(BuildError, match="reserved"):
        build_store(bundles, tmp_path / "s1", extra_instruments={"all": [row]})
    with pytest.raises(BuildError, match="not in the store"):
        build_store(
            bundles,
            tmp_path / "s2",
            extra_instruments={"tech": [("MSFT", row[1], row[2])]},
        )
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


# ---------------------------------------------------------------------------
# extend_store(): on-demand new-ticker backfill (custom-universe support)


def test_extend_store_adds_new_ticker_and_preserves_universes(tmp_path: Path) -> None:
    store = build_fixture_store(
        tmp_path, {"AAPL": make_bars("AAPL"), "MSFT": make_bars("MSFT")}
    )
    (store / "instruments" / "pair.txt").write_text("AAPL\t2024-01-02\t2024-01-08\n")
    fmp = WindowedFakeFmp(bars={"GOOG": make_bars("GOOG", close=50.0)})
    result = extend_store(store, fmp, ["goog"])
    assert result.added == {"GOOG": 5}
    assert result.missing == ()
    all_symbols = {
        line.split("\t")[0]
        for line in (store / "instruments" / "all.txt").read_text().splitlines()
        if line.strip()
    }
    assert all_symbols == {"AAPL", "MSFT", "GOOG"}
    assert read_universes(store)["pair"] == [("AAPL", "2024-01-02", "2024-01-08")]
    # Fetch window is aligned to the store's own calendar, never "today".
    assert fmp.windows == [("GOOG", "2024-01-02", "2024-01-08")]
    # Existing tickers survive the rebuild with their bars intact.
    calendar = [
        date.fromisoformat(line)
        for line in (store / "calendars" / "day.txt").read_text().splitlines()
        if line.strip()
    ]
    assert [bar.date for bar in read_raw_bars(store, "AAPL", calendar)] == DAYS


def test_extend_store_preserves_multi_span_universe_verbatim(tmp_path: Path) -> None:
    """Extending never touches existing tickers, so PIT spans pass through unchanged."""
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    spans = "AAPL\t2024-01-02\t2024-01-04\nAAPL\t2024-01-08\t2024-01-08\n"
    (store / "instruments" / "pit.txt").write_text(spans)
    fmp = WindowedFakeFmp(bars={"GOOG": make_bars("GOOG", close=50.0)})
    assert extend_store(store, fmp, ["GOOG"]).added == {"GOOG": 5}
    assert (store / "instruments" / "pit.txt").read_text() == spans


def test_extend_store_reports_symbols_fmp_lacks_and_leaves_store_untouched(
    tmp_path: Path,
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    snapshot = store_snapshot(store)
    fmp = WindowedFakeFmp(bars={"FAKE": ()})
    result = extend_store(store, fmp, ["FAKE"])
    assert result.added == {}
    assert result.missing == ("FAKE",)
    assert store_snapshot(store) == snapshot


def test_extend_store_skips_symbols_already_present(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    fmp = WindowedFakeFmp(bars={})
    result = extend_store(store, fmp, ["AAPL", "aapl"])
    assert result.added == {}
    assert result.missing == ()
    assert fmp.windows == []  # nothing fetched at all


def test_extend_store_partial_missing_still_adds_the_rest(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    fmp = WindowedFakeFmp(bars={"GOOG": make_bars("GOOG"), "FAKE": ()})
    result = extend_store(store, fmp, ["GOOG", "FAKE"])
    assert result.added == {"GOOG": 5}
    assert result.missing == ("FAKE",)


def test_main_add_tickers_extends_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    fmp = WindowedFakeFmp(bars={"GOOG": make_bars("GOOG")})
    assert main(["--store", str(store), "--add-tickers", "GOOG"], client=fmp) == 0
    assert "added GOOG: 5 bars" in capsys.readouterr().out


def test_main_add_tickers_missing_symbol_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    fmp = WindowedFakeFmp(bars={"FAKE": ()})
    assert main(["--store", str(store), "--add-tickers", "FAKE"], client=fmp) == 1
    assert "FMP has no data for: FAKE" in capsys.readouterr().err


def test_extend_store_excludes_gapped_symbols_and_reports_them(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    snapshot = store_snapshot(store)
    # A hole on 2024-01-04 inside GLXY's own span: unusable, must not build.
    holed = tuple(b for b in make_bars("GLXY") if b.date != DAYS[2])
    fmp = WindowedFakeFmp(bars={"GLXY": holed})
    result = extend_store(store, fmp, ["GLXY"])
    assert result.added == {}
    assert result.gapped == ("GLXY",)
    assert result.missing == ()
    assert store_snapshot(store) == snapshot


def test_extend_store_gapped_symbol_does_not_poison_the_batch(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    holed = tuple(b for b in make_bars("GLXY") if b.date != DAYS[2])
    fmp = WindowedFakeFmp(bars={"GLXY": holed, "GOOG": make_bars("GOOG")})
    result = extend_store(store, fmp, ["GLXY", "GOOG"])
    assert result.added == {"GOOG": 5}
    assert result.gapped == ("GLXY",)


# ---------------------------------------------------------------------------
# Market broadcast fields ($mkt_*, US-067)


class MarketWindowedFakeFmp(WindowedFakeFmp):
    """WindowedFakeFmp + windowed market endpoints with per-series failure."""

    def __init__(
        self,
        bars: dict[str, tuple[EodBar, ...]],
        commodities: dict[str, list[tuple[date, float]]] | None = None,
        y10: list[tuple[date, float | None]] | None = None,
        fail_commodities: set[str] | None = None,
        fail_treasury: bool = False,
    ) -> None:
        super().__init__(bars=bars)
        self.commodities = commodities or {}
        self.y10 = y10 or []
        self.fail_commodities = fail_commodities or set()
        self.fail_treasury = fail_treasury
        self.market_windows: list[tuple[str, str, str]] = []

    def get_commodity_eod(self, symbol: str, start: object, end: object) -> list[CommodityEod]:
        self.market_windows.append((symbol, str(start), str(end)))
        if symbol in self.fail_commodities:
            raise FmpError(f"simulated outage fetching {symbol}")
        return [
            CommodityEod(symbol, d, p, 0.0)
            for d, p in self.commodities.get(symbol, [])
            if str(start) <= d.isoformat() <= str(end)
        ]

    def get_treasury_rates(self, start: object, end: object) -> list[TreasuryCurve]:
        self.market_windows.append(("treasury", str(start), str(end)))
        if self.fail_treasury:
            raise FmpError("simulated treasury outage")
        return [
            make_curve(d, y) for d, y in self.y10 if str(start) <= d.isoformat() <= str(end)
        ]


def market_store(
    tmp_path: Path,
    days: list[date] | None = None,
    extra_series: dict[str, list[tuple[date, float]]] | None = None,
) -> Path:
    """Two-ticker store carrying mkt_gold + mkt_y10 broadcast bins."""
    days = DAYS if days is None else days
    series: dict[str, list[tuple[date, float]]] = {
        "mkt_gold": [(d, 2000.0 + i) for i, d in enumerate(days)],
        "mkt_y10": [(d, 4.0 + 0.25 * i) for i, d in enumerate(days)],
    }
    if extra_series:
        series.update(extra_series)
    store = tmp_path / "us_data"
    build_store(
        [
            TickerBundle("AAPL", make_bars("AAPL", days), (), ()),
            TickerBundle("MSFT", make_bars("MSFT", days, close=200.0), (), ()),
        ],
        store,
        market_series=series,
    )
    return store


def market_client(
    all_days: list[date],
    fail_commodities: set[str] | None = None,
    fail_treasury: bool = False,
) -> MarketWindowedFakeFmp:
    """Serves equity bars, GCUSD prices, and the 10y curve over all_days."""
    return MarketWindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", all_days),
            "MSFT": make_bars("MSFT", all_days, close=200.0),
        },
        commodities={"GCUSD": [(d, 2000.0 + i) for i, d in enumerate(all_days)]},
        y10=[(d, 4.0 + 0.25 * i) for i, d in enumerate(all_days)],
        fail_commodities=fail_commodities,
        fail_treasury=fail_treasury,
    )


def introduction_client(
    fail_commodities: set[str] | None = None, fail_treasury: bool = False
) -> MarketWindowedFakeFmp:
    """Serves every canonical market series over DAYS (no new equity bars)."""
    return MarketWindowedFakeFmp(
        bars={"AAPL": make_bars("AAPL"), "MSFT": make_bars("MSFT", close=200.0)},
        commodities={
            sym: [(d, commodity_price(sym, i)) for i, d in enumerate(DAYS)]
            for sym in COMMODITY_SYMBOLS.values()
        },
        y10=[(d, 4.0 + 0.25 * i) for i, d in enumerate(DAYS)],
        fail_commodities=fail_commodities,
        fail_treasury=fail_treasury,
    )


def test_refresh_appends_market_fields_alongside_equity_bars(tmp_path: Path) -> None:
    store = market_store(tmp_path)
    client = market_client(DAYS + NEW_DAYS)
    result = refresh_store(store, client, end=NEW_DAYS[-1])
    assert result.updated is True
    assert result.warnings == ()
    assert result.market_introduced == ()
    # Each series is pulled since the store's last date, same --end as equities.
    assert ("GCUSD", "2024-01-09", "2024-01-10") in client.market_windows
    assert ("treasury", "2024-01-09", "2024-01-10") in client.market_windows
    for sym in ("aapl", "msft"):  # identical broadcast across instruments
        start, gold = read_bin(store / "features" / sym / "mkt_gold.day.bin")
        assert start == 0
        assert len(gold) == 7
        assert gold[-2:] == pytest.approx([2005.0, 2006.0])
    _, y10 = read_bin(store / "features" / "aapl" / "mkt_y10.day.bin")
    assert y10[-2:] == pytest.approx([4.0 + 0.25 * 5, 4.0 + 0.25 * 6])


def test_refresh_noop_with_market_fields_store_untouched(tmp_path: Path) -> None:
    """Idempotency holds for a store carrying market bins: no market fetch at all."""
    store = market_store(tmp_path)
    before = store_snapshot(store)
    client = market_client(DAYS)
    result = refresh_store(store, client, end=LAST_OLD)
    assert result.updated is False
    assert client.market_windows == []
    assert store_snapshot(store) == before
    # Weekend variant: equities looked but found nothing -> still no market pull.
    client2 = market_client(DAYS)
    result2 = refresh_store(store, client2, end=date(2024, 1, 9))
    assert result2.updated is False
    assert len(client2.windows) == 2
    assert client2.market_windows == []
    assert store_snapshot(store) == before


def test_refresh_twice_with_market_fields_second_run_noop(tmp_path: Path) -> None:
    store = market_store(tmp_path)
    first = refresh_store(store, market_client(DAYS + NEW_DAYS), end=NEW_DAYS[-1])
    assert first.updated is True
    after_first = store_snapshot(store)
    second = refresh_store(store, market_client(DAYS + NEW_DAYS), end=NEW_DAYS[-1])
    assert second.updated is False
    assert store_snapshot(store) == after_first


def test_refresh_market_outage_forward_fills_warns_and_continues(tmp_path: Path) -> None:
    days = weekdays(date(2024, 1, 1), 250)
    new_days = weekdays(days[-1] + timedelta(days=1), 2)
    mystery = {"mkt_mystery": [(d, 7.0) for d in days]}
    store = market_store(tmp_path, days=days, extra_series=mystery)
    client = market_client(days + new_days, fail_commodities={"GCUSD"})
    result = refresh_store(store, client, end=new_days[-1])
    assert result.updated is True
    assert len(result.warnings) == 2
    outage = next(w for w in result.warnings if "mkt_gold" in w)
    assert "forward-filled" in outage
    unknown = next(w for w in result.warnings if "mkt_mystery" in w)
    assert "unknown market series" in unknown
    # The failed series forward-fills the new days from its last stored value...
    _, gold = read_bin(store / "features" / "aapl" / "mkt_gold.day.bin")
    assert len(gold) == 252
    assert gold[-2:] == pytest.approx([2000.0 + 249] * 2)
    _, myst = read_bin(store / "features" / "aapl" / "mkt_mystery.day.bin")
    assert myst[-2:] == pytest.approx([7.0, 7.0])
    # ...while the healthy series still advanced.
    _, y10 = read_bin(store / "features" / "aapl" / "mkt_y10.day.bin")
    assert y10[-2:] == pytest.approx([4.0 + 0.25 * 250, 4.0 + 0.25 * 251])


def test_main_market_outage_exits_zero_and_posts_slack_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    days = weekdays(date(2024, 1, 1), 250)
    new_days = weekdays(days[-1] + timedelta(days=1), 2)
    store = market_store(tmp_path, days=days)
    posted: list[str] = []
    client = market_client(days + new_days, fail_commodities={"GCUSD"})
    argv = ["--store", str(store), "--end", new_days[-1].isoformat()]
    assert main(argv, client=client, notify=posted.append) == 0
    captured = capsys.readouterr()
    assert "store refreshed" in captured.out
    assert "WARNING: market series mkt_gold" in captured.err
    assert posted and posted[0].startswith(":warning: store refresh: ")
    assert "mkt_gold" in posted[0]


def test_main_market_outage_slack_failure_still_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    days = weekdays(date(2024, 1, 1), 250)
    new_days = weekdays(days[-1] + timedelta(days=1), 2)
    store = market_store(tmp_path, days=days)

    def broken_notify(message: str) -> None:
        raise RuntimeError("slack down")

    client = market_client(days + new_days, fail_commodities={"GCUSD"})
    argv = ["--store", str(store), "--end", new_days[-1].isoformat()]
    assert main(argv, client=client, notify=broken_notify) == 0
    assert "slack notice failed (slack down)" in capsys.readouterr().err


def test_main_no_slack_skips_the_notice(tmp_path: Path) -> None:
    days = weekdays(date(2024, 1, 1), 250)
    new_days = weekdays(days[-1] + timedelta(days=1), 2)
    store = market_store(tmp_path, days=days)
    posted: list[str] = []
    client = market_client(days + new_days, fail_commodities={"GCUSD"})
    argv = ["--store", str(store), "--end", new_days[-1].isoformat(), "--no-slack"]
    assert main(argv, client=client, notify=posted.append) == 0
    assert posted == []


def test_refresh_market_start_introduces_missing_series(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)  # no mkt bins yet
    # No new equity bars: the introduction alone forces the rebuild.
    result = refresh_store(store, introduction_client(), end=LAST_OLD, market_start=DAYS[0])
    assert result.updated is True
    assert result.new_bars == {}
    assert result.market_introduced == MARKET_FIELDS
    assert result.warnings == ()
    for name in MARKET_FIELDS:
        for sym in ("aapl", "msft"):
            assert (store / "features" / sym / f"{name}.day.bin").exists()
    _, brent = read_bin(store / "features" / "aapl" / "mkt_brent.day.bin")
    assert brent == pytest.approx([commodity_price("BZUSD", i) for i in range(len(DAYS))])
    # Equity bins are value-identical through the market-only rebuild.
    _, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    assert closes == pytest.approx([100.0, 101.0, 102.0, 103.0, 104.0])


def test_refresh_market_start_partial_outage_skips_that_series(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    client = introduction_client(fail_commodities={"BZUSD"})
    result = refresh_store(store, client, end=LAST_OLD, market_start=DAYS[0])
    assert result.updated is True
    assert set(result.market_introduced) == set(MARKET_FIELDS) - {"mkt_brent"}
    warning = next(w for w in result.warnings if "mkt_brent" in w)
    assert "not introduced" in warning
    assert not (store / "features" / "aapl" / "mkt_brent.day.bin").exists()
    assert (store / "features" / "aapl" / "mkt_gold.day.bin").exists()


def test_refresh_market_start_total_outage_leaves_store_untouched(tmp_path: Path) -> None:
    store = two_ticker_store(tmp_path)
    before = store_snapshot(store)
    client = introduction_client(
        fail_commodities=set(COMMODITY_SYMBOLS.values()), fail_treasury=True
    )
    result = refresh_store(store, client, end=LAST_OLD, market_start=DAYS[0])
    assert result.updated is False
    assert len(result.warnings) == len(MARKET_FIELDS)
    assert store_snapshot(store) == before


def test_refresh_without_market_start_never_touches_market_endpoints(
    tmp_path: Path,
) -> None:
    store = two_ticker_store(tmp_path)
    client = MarketWindowedFakeFmp(
        bars={
            "AAPL": make_bars("AAPL", DAYS + NEW_DAYS),
            "MSFT": make_bars("MSFT", DAYS + NEW_DAYS, close=200.0),
        }
    )
    result = refresh_store(store, client, end=NEW_DAYS[-1])
    assert result.updated is True
    assert client.market_windows == []
    assert not (store / "features" / "aapl" / "mkt_gold.day.bin").exists()


def test_main_market_start_flag_introduces_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = two_ticker_store(tmp_path)
    argv = [
        "--store",
        str(store),
        "--end",
        LAST_OLD.isoformat(),
        "--market-start",
        DAYS[0].isoformat(),
    ]
    assert main(argv, client=introduction_client()) == 0
    out = capsys.readouterr().out
    assert "market series introduced:" in out
    assert "mkt_brent" in out


def test_extend_store_carries_market_fields_to_new_tickers(tmp_path: Path) -> None:
    store = market_store(tmp_path)
    fmp = WindowedFakeFmp(bars={"GOOG": make_bars("GOOG", close=50.0)})
    assert extend_store(store, fmp, ["GOOG"]).added == {"GOOG": 5}
    _, goog_gold = read_bin(store / "features" / "goog" / "mkt_gold.day.bin")
    assert goog_gold == pytest.approx([2000.0, 2001.0, 2002.0, 2003.0, 2004.0])
    _, aapl_gold = read_bin(store / "features" / "aapl" / "mkt_gold.day.bin")
    assert aapl_gold == pytest.approx(goog_gold)


def test_read_market_series_overlays_partial_spans_and_drops_nan_head(
    tmp_path: Path,
) -> None:
    """No single ticker spans the calendar; the overlay still recovers the series."""
    store = tmp_path / "us_data"
    build_store(
        [
            TickerBundle("AAPL", make_bars("AAPL", DAYS[:3]), (), ()),
            TickerBundle("LATE", make_bars("LATE", DAYS[2:], close=50.0), (), ()),
        ],
        store,
        market_series={
            "mkt_gold": [(d, 2000.0 + i) for i, d in enumerate(DAYS[1:], start=1)]
        },
    )
    fields = read_market_fields(store, ["AAPL", "LATE"])
    assert fields == ("mkt_gold",)
    series = read_market_series(store, ["AAPL", "LATE"], DAYS, fields)
    obs = series["mkt_gold"]
    assert [d for d, _ in obs] == DAYS[1:]  # the NaN head day is dropped
    assert [v for _, v in obs] == pytest.approx([2000.0 + i for i in range(1, 5)])
    # A ticker missing a market bin is corruption, reported loud.
    (store / "features" / "aapl" / "mkt_gold.day.bin").unlink()
    with pytest.raises(RefreshError, match="missing market bin"):
        read_market_series(store, ["AAPL", "LATE"], DAYS, fields)


def test_extend_store_drops_off_calendar_bars_keeping_calendar_invariant(
    tmp_path: Path,
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": make_bars("AAPL")})
    # A foreign-venue bar on a day the store has never seen (Sat 2024-01-06):
    # dropped, not merged — the calendar must not grow.
    foreign_day = date(2024, 1, 6)
    bars = tuple(
        sorted(make_bars("GDS") + make_bars("GDS", days=[foreign_day]), key=lambda b: b.date)
    )
    fmp = WindowedFakeFmp(bars={"GDS": bars})
    result = extend_store(store, fmp, ["GDS"])
    assert result.added == {"GDS": 5}  # the foreign bar did not count
    calendar = [
        date.fromisoformat(line)
        for line in (store / "calendars" / "day.txt").read_text().splitlines()
        if line.strip()
    ]
    assert calendar == DAYS

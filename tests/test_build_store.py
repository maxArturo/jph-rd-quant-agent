"""Tests for data/build_store.py: checkpointed backfill, bin layout, swap, qlib read."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.build_store import (
    COMMODITY_SYMBOLS,
    MARKET_FIELDS,
    NEWS_FIELD,
    BuildError,
    StoreValidationError,
    TickerBundle,
    backfill,
    build_from_fmp,
    build_store,
    fetch_bundle,
    fetch_market_series,
    main,
    resolve_tickers,
    validate_store,
)
from data.fmp import (
    _TREASURY_TENORS,
    CommodityEod,
    Dividend,
    EodBar,
    FmpClient,
    FmpError,
    Split,
    TreasuryCurve,
)

# Five consecutive US weekdays (Tue 2024-01-02 .. Mon 2024-01-08).
DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]


def make_bars(
    symbol: str, days: list[date] | None = None, close: float = 100.0
) -> tuple[EodBar, ...]:
    days = DAYS if days is None else days
    return tuple(
        EodBar(
            symbol=symbol,
            date=day,
            open=close - 1.0 + i,
            high=close + 2.0 + i,
            low=close - 2.0 + i,
            close=close + i,
            volume=1_000.0 + 10 * i,
        )
        for i, day in enumerate(days)
    )


class FakeFmp(FmpClient):
    """FmpClient stand-in serving canned per-symbol data; can fail on one symbol."""

    def __init__(
        self,
        bars: dict[str, tuple[EodBar, ...]],
        splits: dict[str, tuple[Split, ...]] | None = None,
        dividends: dict[str, tuple[Dividend, ...]] | None = None,
        fail_on: str | None = None,
    ) -> None:
        super().__init__(session=object())
        self.bars = bars
        self.splits = splits or {}
        self.dividends = dividends or {}
        self.fail_on = fail_on
        self.fetched: list[str] = []

    def get_eod_bars(self, symbol: str, start: object, end: object) -> list[EodBar]:
        self.fetched.append(symbol)
        if symbol == self.fail_on:
            raise FmpError(f"simulated crash fetching {symbol}")
        return list(self.bars[symbol])

    def get_splits(self, symbol: str) -> list[Split]:
        return list(self.splits.get(symbol, ()))

    def get_dividends(self, symbol: str) -> list[Dividend]:
        return list(self.dividends.get(symbol, ()))


def five_ticker_client(fail_on: str | None = None) -> FakeFmp:
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    return FakeFmp(
        bars={sym: make_bars(sym, close=100.0 + 50 * i) for i, sym in enumerate(symbols)},
        fail_on=fail_on,
    )


def read_bin(path: Path) -> tuple[int, np.ndarray]:
    data = np.fromfile(path, dtype="<f")
    return int(data[0]), data[1:]


def weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def commodity_price(symbol: str, i: int) -> float:
    """Deterministic, per-symbol-distinct canned commodity price."""
    return 10.0 * (1 + list(COMMODITY_SYMBOLS.values()).index(symbol)) + i


def make_curve(day: date, year10: float | None) -> TreasuryCurve:
    values: dict[str, float | None] = {tenor: 1.0 for tenor in _TREASURY_TENORS}
    values["year10"] = year10
    return TreasuryCurve(date=day, **values)


class MarketFakeFmp(FakeFmp):
    """FakeFmp that also serves canned commodity EOD prices and treasury curves."""

    def __init__(
        self,
        bars: dict[str, tuple[EodBar, ...]],
        market_days: list[date] | None = None,
        y10_none: set[date] | None = None,
    ) -> None:
        super().__init__(bars)
        self.market_days = market_days if market_days is not None else DAYS
        self.y10_none = y10_none or set()
        self.commodity_calls: list[str] = []

    def get_commodity_eod(self, symbol: str, start: object, end: object) -> list[CommodityEod]:
        self.commodity_calls.append(symbol)
        return [
            CommodityEod(symbol, day, commodity_price(symbol, i), 100.0)
            for i, day in enumerate(self.market_days)
        ]

    def get_treasury_rates(self, start: object, end: object) -> list[TreasuryCurve]:
        return [
            make_curve(day, None if day in self.y10_none else 4.0 + 0.1 * i)
            for i, day in enumerate(self.market_days)
        ]


def market_five_ticker_client() -> MarketFakeFmp:
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    return MarketFakeFmp(
        bars={sym: make_bars(sym, close=100.0 + 50 * i) for i, sym in enumerate(symbols)}
    )


# ---------------------------------------------------------------------------
# Ticker list resolution


def test_resolve_tickers_dedupes_and_uppercases() -> None:
    assert resolve_tickers("aapl, MSFT,aapl ,nvda", None) == ["AAPL", "MSFT", "NVDA"]


def test_resolve_tickers_from_file(tmp_path: Path) -> None:
    listing = tmp_path / "tickers.txt"
    listing.write_text("aapl\nMSFT\n\nnvda\n")
    assert resolve_tickers(None, str(listing)) == ["AAPL", "MSFT", "NVDA"]


def test_resolve_tickers_rejects_bad_arg_combos() -> None:
    with pytest.raises(BuildError):
        resolve_tickers(None, None)
    with pytest.raises(BuildError):
        resolve_tickers("AAPL", "somefile")
    with pytest.raises(BuildError):
        resolve_tickers(" , ", None)


# ---------------------------------------------------------------------------
# Checkpointed, resumable backfill


def test_crash_midlist_then_resume_without_duplicates(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    ckpt = tmp_path / "us_data.checkpoint"
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]

    crashing = five_ticker_client(fail_on="GOOG")
    with pytest.raises(FmpError):
        build_from_fmp(symbols, DAYS[0], DAYS[-1], store, ckpt, crashing)
    assert crashing.fetched == ["AAPL", "MSFT", "GOOG"]
    assert sorted(p.name for p in ckpt.glob("*.json")) == ["AAPL.json", "MSFT.json"]
    assert not store.exists()  # crash before build: no store, no partials

    healthy = five_ticker_client()
    build_from_fmp(symbols, DAYS[0], DAYS[-1], store, ckpt, healthy)
    # Resume: only the unfinished tail is refetched.
    assert healthy.fetched == ["GOOG", "AMZN", "NVDA"]
    lines = (store / "instruments" / "all.txt").read_text().splitlines()
    assert sorted(line.split("\t")[0] for line in lines) == sorted(symbols)
    assert len(lines) == len(symbols)  # no duplicates


def test_checkpoint_for_other_window_is_refetched(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    client = five_ticker_client()

    def fetch(symbol: str) -> TickerBundle:
        return fetch_bundle(client, symbol, DAYS[0], DAYS[-1])

    backfill(["AAPL"], fetch, ckpt, DAYS[0], DAYS[-1])
    backfill(["AAPL"], fetch, ckpt, DAYS[0], DAYS[-1])
    assert client.fetched == ["AAPL"]  # same window: served from checkpoint
    backfill(["AAPL"], fetch, ckpt, DAYS[0], DAYS[-1] + timedelta(days=1))
    assert client.fetched == ["AAPL", "AAPL"]  # window changed: refetched


# ---------------------------------------------------------------------------
# Store layout and adjustment wiring


def test_store_layout_calendar_instruments_and_bin_format(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    late = make_bars("LATE", days=DAYS[2:], close=50.0)
    bundles = [
        TickerBundle("AAPL", make_bars("AAPL"), (), ()),
        TickerBundle("LATE", late, (), ()),
    ]
    build_store(bundles, store)

    calendar = (store / "calendars" / "day.txt").read_text().splitlines()
    assert calendar == [d.isoformat() for d in DAYS]
    lines = sorted((store / "instruments" / "all.txt").read_text().splitlines())
    assert lines[0] == f"AAPL\t{DAYS[0]}\t{DAYS[-1]}"
    assert lines[1] == f"LATE\t{DAYS[2]}\t{DAYS[-1]}"

    start_index, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    assert start_index == 0
    np.testing.assert_allclose(closes, [100.0, 101.0, 102.0, 103.0, 104.0], rtol=1e-6)
    # LATE begins two calendar days in: its bin index reflects that.
    start_index, closes = read_bin(store / "features" / "late" / "close.day.bin")
    assert start_index == 2
    assert len(closes) == 3
    for field in ("open", "high", "low", "volume", "factor"):
        assert (store / "features" / "aapl" / f"{field}.day.bin").exists()
    _, factors = read_bin(store / "features" / "aapl" / "factor.day.bin")
    np.testing.assert_allclose(factors, np.ones(5), rtol=1e-6)


def test_split_adjusts_prices_volume_and_factor(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    split = Split("AAPL", DAYS[3], 2.0, 1.0)  # 2:1 split effective on day 4
    build_store([TickerBundle("AAPL", make_bars("AAPL"), (split,), ())], store)

    _, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    _, volumes = read_bin(store / "features" / "aapl" / "volume.day.bin")
    _, factors = read_bin(store / "features" / "aapl" / "factor.day.bin")
    np.testing.assert_allclose(factors, [0.5, 0.5, 0.5, 1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(closes, [50.0, 50.5, 51.0, 103.0, 104.0], rtol=1e-6)
    np.testing.assert_allclose(volumes, [2000.0, 2020.0, 2040.0, 1030.0, 1040.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# Temp-dir write, validation, atomic swap


def leftover_dirs(parent: Path, store_name: str) -> list[str]:
    return [p.name for p in parent.iterdir() if p.name.startswith(store_name) and p.is_dir()]


def test_failed_build_leaves_existing_store_untouched(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_store([TickerBundle("AAPL", make_bars("AAPL"), (), ())], store)
    before = (store / "instruments" / "all.txt").read_text()

    with pytest.raises(BuildError, match="no bars"):
        build_store(
            [
                TickerBundle("MSFT", make_bars("MSFT"), (), ()),
                TickerBundle("EMPTY", (), (), ()),
            ],
            store,
        )
    assert (store / "instruments" / "all.txt").read_text() == before
    assert leftover_dirs(tmp_path, "us_data") == ["us_data"]  # no .tmp/.old partials


def test_validation_failure_cleans_temp_and_never_swaps(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    # GAPPY trades on days 1 and 3 but not day 2: a NaN close inside its span.
    gappy = make_bars("GAPPY", days=[DAYS[0], DAYS[2]])
    bundles = [
        TickerBundle("AAPL", make_bars("AAPL"), (), ()),
        TickerBundle("GAPPY", gappy, (), ()),
    ]
    with pytest.raises(StoreValidationError, match="NaN close"):
        build_store(bundles, store)
    assert not store.exists()
    assert leftover_dirs(tmp_path, "us_data") == []


def test_rebuild_swaps_old_store_for_new(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_store([TickerBundle("AAPL", make_bars("AAPL", close=100.0), (), ())], store)
    build_store([TickerBundle("AAPL", make_bars("AAPL", close=500.0), (), ())], store)
    _, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    np.testing.assert_allclose(closes, [500.0, 501.0, 502.0, 503.0, 504.0], rtol=1e-6)
    assert leftover_dirs(tmp_path, "us_data") == ["us_data"]


def test_validate_store_catches_missing_feature_file(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_store([TickerBundle("AAPL", make_bars("AAPL"), (), ())], store)
    (store / "features" / "aapl" / "close.day.bin").unlink()
    with pytest.raises(StoreValidationError, match="missing feature file"):
        validate_store(store, ["AAPL"])


# ---------------------------------------------------------------------------
# Market broadcast fields ($mkt_*, US-066)


def test_market_fields_broadcast_identical_and_never_adjusted(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    split = Split("AAPL", DAYS[3], 2.0, 1.0)  # 2:1 split effective on day 4
    gold = [(day, 2000.0 + i) for i, day in enumerate(DAYS)]
    y10 = [(day, 4.0 + 0.1 * i) for i, day in enumerate(DAYS)]
    bundles = [
        TickerBundle("AAPL", make_bars("AAPL"), (split,), ()),
        TickerBundle("LATE", make_bars("LATE", days=DAYS[2:], close=50.0), (), ()),
    ]
    build_store(bundles, store, market_series={"mkt_gold": gold, "mkt_y10": y10})

    # RAW despite AAPL's split: the market series is never factor-adjusted...
    start_index, aapl_gold = read_bin(store / "features" / "aapl" / "mkt_gold.day.bin")
    assert start_index == 0
    np.testing.assert_allclose(aapl_gold, [2000.0, 2001.0, 2002.0, 2003.0, 2004.0], rtol=1e-6)
    _, aapl_y10 = read_bin(store / "features" / "aapl" / "mkt_y10.day.bin")
    np.testing.assert_allclose(aapl_y10, [4.0, 4.1, 4.2, 4.3, 4.4], rtol=1e-6)
    # ...and identical across instruments per date (LATE starts two days in).
    start_index, late_gold = read_bin(store / "features" / "late" / "mkt_gold.day.bin")
    assert start_index == 2
    np.testing.assert_allclose(late_gold, [2002.0, 2003.0, 2004.0], rtol=1e-6)
    # The equity adjustment math is untouched: the split still adjusts $close.
    _, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    np.testing.assert_allclose(closes, [50.0, 50.5, 51.0, 103.0, 104.0], rtol=1e-6)


def test_market_forward_fill_and_nan_before_first_observation(tmp_path: Path) -> None:
    days = weekdays(date(2024, 1, 1), 120)
    store = tmp_path / "us_data"
    monday = next(i for i, day in enumerate(days) if i > 12 and day.weekday() == 0)
    saturday = days[monday] - timedelta(days=2)
    assert saturday.weekday() == 5
    # mkt_wti starts 10 trading days in, skips the Monday, but printed Saturday.
    wti = [(day, 70.0 + i) for i, day in enumerate(days[10:]) if day != days[monday]]
    wti.append((saturday, 555.0))
    # mkt_gold spans the whole calendar and skips the Monday (commodity holiday).
    gold = [(day, 2000.0 + i) for i, day in enumerate(days) if day != days[monday]]
    build_store(
        [TickerBundle("AAPL", make_bars("AAPL", days=days), (), ())],
        store,
        market_series={"mkt_wti": wti, "mkt_gold": gold},
    )

    _, wti_values = read_bin(store / "features" / "aapl" / "mkt_wti.day.bin")
    assert np.isnan(wti_values[:10]).all()  # before the series' first observation
    assert not np.isnan(wti_values[10:]).any()  # forward-fill leaves no holes after it
    assert wti_values[monday] == pytest.approx(555.0)  # last observation was Saturday's
    _, gold_values = read_bin(store / "features" / "aapl" / "mkt_gold.day.bin")
    assert gold_values[monday] == gold_values[monday - 1]  # holiday: previous trading day


def test_market_low_coverage_fails_loud_with_series_named(tmp_path: Path) -> None:
    days = weekdays(date(2024, 1, 1), 120)
    store = tmp_path / "us_data"
    dxy = [(day, 100.0) for day in days[:60]]  # feed died halfway: 50% coverage
    with pytest.raises(StoreValidationError, match="mkt_dxy"):
        build_store(
            [TickerBundle("AAPL", make_bars("AAPL", days=days), (), ())],
            store,
            market_series={"mkt_dxy": dxy},
        )
    assert not store.exists()
    assert leftover_dirs(tmp_path, "us_data") == []


def test_market_series_bad_names_rejected(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    bundle = TickerBundle("AAPL", make_bars("AAPL"), (), ())
    series = [(day, 1.0) for day in DAYS]
    for bad in ("close", "$mkt_gold", "MKT_GOLD"):
        with pytest.raises(BuildError, match="invalid market series name"):
            build_store([bundle], store, market_series={bad: series})
    assert not store.exists()


def test_validate_store_catches_missing_market_bin(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_store(
        [TickerBundle("AAPL", make_bars("AAPL"), (), ())],
        store,
        market_series={"mkt_gold": [(day, 2000.0 + i) for i, day in enumerate(DAYS)]},
    )
    validate_store(store, ["AAPL"], market_fields=("mkt_gold",))  # intact store passes
    (store / "features" / "aapl" / "mkt_gold.day.bin").unlink()
    with pytest.raises(StoreValidationError, match="missing feature file"):
        validate_store(store, ["AAPL"], market_fields=("mkt_gold",))


def test_fetch_market_series_covers_every_field_and_skips_none_y10() -> None:
    client = MarketFakeFmp(bars={}, y10_none={DAYS[1]})
    series = fetch_market_series(client, DAYS[0], DAYS[-1])
    assert set(series) == set(MARKET_FIELDS)
    assert client.commodity_calls == list(COMMODITY_SYMBOLS.values())
    assert series["mkt_brent"][0] == (DAYS[0], commodity_price("BZUSD", 0))
    for field in COMMODITY_SYMBOLS:
        assert len(series[field]) == len(DAYS)
    # the day FMP reported no 10y value for is skipped, not emitted as None/NaN
    assert len(series["mkt_y10"]) == len(DAYS) - 1
    assert DAYS[1] not in {day for day, _ in series["mkt_y10"]}


def test_main_with_market_start_writes_all_mkt_bins(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    code = main(
        [
            "--tickers",
            "AAPL,MSFT,GOOG,AMZN,NVDA",
            "--start",
            DAYS[0].isoformat(),
            "--end",
            DAYS[-1].isoformat(),
            "--output",
            str(store),
            "--market-start",
            DAYS[0].isoformat(),
        ],
        client=market_five_ticker_client(),
    )
    assert code == 0
    for name in MARKET_FIELDS:
        _, aapl = read_bin(store / "features" / "aapl" / f"{name}.day.bin")
        _, nvda = read_bin(store / "features" / "nvda" / f"{name}.day.bin")
        np.testing.assert_allclose(aapl, nvda, rtol=1e-6)


# ---------------------------------------------------------------------------
# Per-ticker raw series ($news_ct_1d, US-073)


def test_ticker_series_written_raw_per_instrument_and_span_clipped(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    split = Split("AAPL", DAYS[3], 2.0, 1.0)  # 2:1 split effective on day 4
    counts = {
        "AAPL": [(day, float(i)) for i, day in enumerate(DAYS)],
        # LATE's series covers the whole calendar, but its bars start on day 3:
        # the observations before its span must be dropped, not written.
        "LATE": [(day, 7.0 + i) for i, day in enumerate(DAYS)],
    }
    bundles = [
        TickerBundle("AAPL", make_bars("AAPL"), (split,), ()),
        TickerBundle("LATE", make_bars("LATE", days=DAYS[2:], close=50.0), (), ()),
    ]
    build_store(bundles, store, ticker_series={NEWS_FIELD: counts})

    # RAW despite AAPL's split: the count is never factor-adjusted...
    start_index, aapl = read_bin(store / "features" / "aapl" / f"{NEWS_FIELD}.day.bin")
    assert start_index == 0
    np.testing.assert_allclose(aapl, [0.0, 1.0, 2.0, 3.0, 4.0], rtol=1e-6)
    # ...and per-ticker (NOT broadcast): LATE carries its own values, span-clipped.
    start_index, late = read_bin(store / "features" / "late" / f"{NEWS_FIELD}.day.bin")
    assert start_index == 2
    np.testing.assert_allclose(late, [9.0, 10.0, 11.0], rtol=1e-6)
    # The equity adjustment math is untouched: the split still adjusts $close.
    _, closes = read_bin(store / "features" / "aapl" / "close.day.bin")
    np.testing.assert_allclose(closes, [50.0, 50.5, 51.0, 103.0, 104.0], rtol=1e-6)


def test_ticker_series_nan_outside_coverage_zero_inside(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    # Coverage starts on day 3; days 3 and 5 are covered no-news days (0).
    counts = [(DAYS[2], 0.0), (DAYS[3], 5.0), (DAYS[4], 0.0)]
    build_store(
        [TickerBundle("AAPL", make_bars("AAPL"), (), ())],
        store,
        ticker_series={NEWS_FIELD: {"AAPL": counts}},
    )
    _, values = read_bin(store / "features" / "aapl" / f"{NEWS_FIELD}.day.bin")
    assert np.isnan(values[:2]).all()  # before coverage: NaN, never 0
    np.testing.assert_allclose(values[2:], [0.0, 5.0, 0.0], rtol=1e-6)


def test_ticker_series_symbol_without_observations_gets_all_nan_bin(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    bundles = [
        TickerBundle("AAPL", make_bars("AAPL"), (), ()),
        TickerBundle("MSFT", make_bars("MSFT", close=200.0), (), ()),
    ]
    build_store(
        bundles, store, ticker_series={NEWS_FIELD: {"AAPL": [(day, 1.0) for day in DAYS]}}
    )
    # Every instrument gets a bin (validation demands it); no data = all NaN.
    _, msft = read_bin(store / "features" / "msft" / f"{NEWS_FIELD}.day.bin")
    assert np.isnan(msft).all()


def test_ticker_series_bad_inputs_rejected(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    bundle = TickerBundle("AAPL", make_bars("AAPL"), (), ())
    good = [(day, 1.0) for day in DAYS]
    for bad in ("close", "$news_ct_1d", "NEWS_CT_1D"):
        with pytest.raises(BuildError, match="invalid ticker series name"):
            build_store([bundle], store, ticker_series={bad: {"AAPL": good}})
    with pytest.raises(BuildError, match="not in the store"):
        build_store([bundle], store, ticker_series={NEWS_FIELD: {"MSFT": good}})
    saturday = date(2024, 1, 6)
    with pytest.raises(BuildError, match="off-calendar"):
        build_store(
            [bundle], store, ticker_series={NEWS_FIELD: {"AAPL": [(saturday, 1.0)]}}
        )
    with pytest.raises(BuildError, match="non-finite"):
        build_store(
            [bundle], store, ticker_series={NEWS_FIELD: {"AAPL": [(DAYS[0], float("nan"))]}}
        )
    with pytest.raises(BuildError, match="both market_series and ticker_series"):
        build_store(
            [bundle], store, market_series={"mkt_gold": good},
            ticker_series={"mkt_gold": {"AAPL": good}},
        )
    assert not store.exists()
    assert leftover_dirs(tmp_path, "us_data") == []


def test_validate_store_catches_missing_ticker_bin(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_store(
        [TickerBundle("AAPL", make_bars("AAPL"), (), ())],
        store,
        ticker_series={NEWS_FIELD: {"AAPL": [(day, 1.0) for day in DAYS]}},
    )
    validate_store(store, ["AAPL"], ticker_fields=(NEWS_FIELD,))  # intact store passes
    (store / "features" / "aapl" / f"{NEWS_FIELD}.day.bin").unlink()
    with pytest.raises(StoreValidationError, match="missing feature file"):
        validate_store(store, ["AAPL"], ticker_fields=(NEWS_FIELD,))


# ---------------------------------------------------------------------------
# CLI


def test_main_builds_store_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    store = tmp_path / "us_data"
    code = main(
        [
            "--tickers",
            "AAPL,MSFT,GOOG,AMZN,NVDA",
            "--start",
            DAYS[0].isoformat(),
            "--end",
            DAYS[-1].isoformat(),
            "--output",
            str(store),
        ],
        client=five_ticker_client(),
    )
    assert code == 0
    assert "store built" in capsys.readouterr().out
    assert (store / "calendars" / "day.txt").exists()


def test_main_reports_errors_and_exits_nonzero(capsys: pytest.CaptureFixture) -> None:
    code = main(["--start", "2024-01-02", "--end", "2024-01-08"], client=five_ticker_client())
    assert code == 1
    assert "exactly one of" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Acceptance smoke: qlib reads the store back


def test_qlib_reads_aapl_ohlcv_and_market_fields(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    client = market_five_ticker_client()
    bundles = backfill(
        symbols,
        lambda s: fetch_bundle(client, s, DAYS[0], DAYS[-1]),
        tmp_path / "ckpt",
        DAYS[0],
        DAYS[-1],
    )
    build_store(
        bundles,
        store,
        market_series=fetch_market_series(client, DAYS[0], DAYS[-1]),
        ticker_series={
            NEWS_FIELD: {
                sym: [(day, float(10 * i + j)) for j, day in enumerate(DAYS)]
                for i, sym in enumerate(symbols)
            }
        },
    )

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(store), region="us")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(["AAPL"], fields, freq="day")
    assert len(df) == len(DAYS)
    assert not df.isna().any().any()
    np.testing.assert_allclose(
        df["$close"].to_numpy(), [100.0, 101.0, 102.0, 103.0, 104.0], rtol=1e-5
    )

    # Market broadcast fields read back per instrument/date, identical across tickers.
    mkt = D.features(["AAPL", "NVDA"], ["$mkt_gold", "$mkt_y10"], freq="day")
    assert not mkt.isna().any().any()
    gold = mkt["$mkt_gold"].unstack(level=0)
    assert (gold["AAPL"] == gold["NVDA"]).all()
    np.testing.assert_allclose(
        gold["AAPL"].to_numpy(),
        [commodity_price("GCUSD", i) for i in range(len(DAYS))],
        rtol=1e-5,
    )
    y10 = mkt["$mkt_y10"].unstack(level=0)
    np.testing.assert_allclose(y10["AAPL"].to_numpy(), [4.0, 4.1, 4.2, 4.3, 4.4], rtol=1e-5)

    # Per-ticker news counts read back per instrument/date, DISTINCT per ticker.
    news = D.features(["AAPL", "NVDA"], [f"${NEWS_FIELD}"], freq="day")
    counts = news[f"${NEWS_FIELD}"].unstack(level=0)
    np.testing.assert_allclose(counts["AAPL"].to_numpy(), [0.0, 1.0, 2.0, 3.0, 4.0], rtol=1e-5)
    np.testing.assert_allclose(counts["NVDA"].to_numpy(), [40.0, 41.0, 42.0, 43.0, 44.0], rtol=1e-5)

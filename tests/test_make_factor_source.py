"""Tests for data/make_factor_source.py: h5 shape, debug subset, refusal, qlib parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.build_store import MARKET_FIELDS, NEWS_FIELD, build_store, fetch_bundle
from data.make_factor_source import (
    COLUMNS,
    MARKET_H5,
    NEWS_H5,
    FactorSourceError,
    debug_subset,
    load_market_frame,
    load_news_frame,
    load_universe_frame,
    main,
    make_factor_source,
)
from data.make_universe import make_universe
from tests.test_build_store import (
    DAYS,
    MarketFakeFmp,
    build_from_fmp,
    commodity_price,
    fetch_market_series,
    five_ticker_client,
    make_bars,
    market_five_ticker_client,
)

FIVE = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]


def news_count(symbol: str, i: int) -> float:
    """Deterministic, per-symbol-distinct canned daily news count."""
    return float(10 * FIVE.index(symbol) + i)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """Five-ticker fixture store with a us_liquid-style custom universe inside."""
    store = tmp_path / "us_data"
    build_from_fmp(FIVE, DAYS[0], DAYS[-1], store, tmp_path / "ckpt", five_ticker_client())
    make_universe("fixture_univ", store, tickers=",".join(FIVE))
    make_universe("pair", store, tickers="AAPL,NVDA")
    return store


@pytest.fixture
def market_store(tmp_path: Path) -> Path:
    """Five-ticker fixture store that also carries all 8 $mkt_* broadcast fields."""
    store = tmp_path / "us_data_mkt"
    build_from_fmp(
        FIVE,
        DAYS[0],
        DAYS[-1],
        store,
        tmp_path / "ckpt_mkt",
        market_five_ticker_client(),
        market_start=DAYS[0],
    )
    make_universe("fixture_univ", store, tickers=",".join(FIVE))
    return store


@pytest.fixture
def news_store(tmp_path: Path) -> Path:
    """Five-ticker store carrying $mkt_* AND $news_ct_1d (news starts on day 2,
    so day 1 is a NaN head; NVDA's last-day value is an explicit 0)."""
    store = tmp_path / "us_data_news"
    client = market_five_ticker_client()
    bundles = [fetch_bundle(client, sym, DAYS[0], DAYS[-1]) for sym in FIVE]
    counts = {
        sym: [
            (day, 0.0 if (sym == "NVDA" and j == 3) else news_count(sym, j))
            for j, day in enumerate(DAYS[1:])
        ]
        for sym in FIVE
    }
    build_store(
        bundles,
        store,
        market_series=fetch_market_series(client, DAYS[0], DAYS[-1]),
        ticker_series={NEWS_FIELD: counts},
    )
    make_universe("fixture_univ", store, tickers=",".join(FIVE))
    return store


# ---------------------------------------------------------------------------
# Output h5 contract


def test_all_h5_has_multiindex_and_ohlcv_columns(store: Path, tmp_path: Path) -> None:
    all_path, debug_path = make_factor_source("fixture_univ", store, tmp_path / "src")
    for path in (all_path, debug_path):
        assert path.exists()
        frame = pd.read_hdf(path, key="data")
        assert isinstance(frame, pd.DataFrame)
        assert isinstance(frame.index, pd.MultiIndex)
        assert frame.index.names == ["datetime", "instrument"]
        assert list(frame.columns) == list(COLUMNS)

    frame = pd.read_hdf(all_path, key="data")
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(DAYS) * len(FIVE)
    assert not frame.isna().to_numpy().any()
    # AAPL fixture closes are 100..104 with factor 1.0 (no split/dividend events).
    aapl = frame.xs("AAPL", level="instrument")
    np.testing.assert_allclose(aapl["$close"].to_numpy(), [100, 101, 102, 103, 104], rtol=1e-6)
    np.testing.assert_allclose(aapl["$factor"].to_numpy(), np.ones(5), rtol=1e-6)
    assert frame.index.get_level_values("datetime")[0] == pd.Timestamp(DAYS[0])


def test_frame_is_sorted_and_scoped_to_the_universe(store: Path) -> None:
    frame = load_universe_frame(store, "pair")
    assert frame.index.is_monotonic_increasing
    assert set(frame.index.get_level_values("instrument")) == {"AAPL", "NVDA"}


def test_debug_subset_trims_days_and_instruments(store: Path, tmp_path: Path) -> None:
    _, debug_path = make_factor_source(
        "fixture_univ", store, tmp_path / "src", debug_days=3, debug_instruments=2
    )
    frame = pd.read_hdf(debug_path, key="data")
    assert isinstance(frame, pd.DataFrame)
    dates = frame.index.get_level_values("datetime").unique()
    assert list(dates) == [pd.Timestamp(d) for d in DAYS[-3:]]
    assert sorted(set(frame.index.get_level_values("instrument"))) == ["AAPL", "AMZN"]


def test_debug_subset_rejects_nonpositive_limits(store: Path) -> None:
    frame = load_universe_frame(store, "pair")
    with pytest.raises(FactorSourceError):
        debug_subset(frame, debug_days=0)
    with pytest.raises(FactorSourceError):
        debug_subset(frame, debug_instruments=0)


def test_consumable_folders_mirror_the_h5_files(store: Path, tmp_path: Path) -> None:
    output = tmp_path / "src"
    all_path, debug_path = make_factor_source("fixture_univ", store, output)
    for folder, source in (("data_folder", all_path), ("data_folder_debug", debug_path)):
        consumable = output / folder / "daily_pv.h5"
        assert consumable.exists()
        pd.testing.assert_frame_equal(
            pd.read_hdf(consumable, key="data"),  # type: ignore[arg-type]
            pd.read_hdf(source, key="data"),  # type: ignore[arg-type]
        )
        readme = (output / folder / "README.md").read_text()
        assert 'key="data"' in readme and "daily_pv.h5" in readme


def test_regeneration_overwrites_stale_output(store: Path, tmp_path: Path) -> None:
    output = tmp_path / "src"
    make_factor_source("fixture_univ", store, output)
    before = (output / "daily_pv_all.h5").stat().st_size
    make_factor_source("pair", store, output)
    frame = pd.read_hdf(output / "daily_pv_all.h5", key="data")
    assert isinstance(frame, pd.DataFrame)
    assert set(frame.index.get_level_values("instrument")) == {"AAPL", "NVDA"}
    assert (output / "daily_pv_all.h5").stat().st_size <= before


# ---------------------------------------------------------------------------
# market_series.h5 companion (US-068)


def test_market_h5_written_to_both_folders_with_identical_schema(
    market_store: Path, tmp_path: Path
) -> None:
    output = tmp_path / "src"
    make_factor_source("fixture_univ", market_store, output, debug_days=3)
    frames: dict[str, pd.DataFrame] = {}
    for folder in ("data_folder", "data_folder_debug"):
        path = output / folder / MARKET_H5
        assert path.exists()
        frame = pd.read_hdf(path, key="data")
        assert isinstance(frame, pd.DataFrame)
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert not isinstance(frame.index, pd.MultiIndex)
        assert list(frame.columns) == sorted(f"${f}" for f in MARKET_FIELDS)
        frames[folder] = frame

    full = frames["data_folder"]
    assert list(full.index) == [pd.Timestamp(d) for d in DAYS]
    np.testing.assert_allclose(
        full["$mkt_brent"].to_numpy(),
        [commodity_price("BZUSD", i) for i in range(len(DAYS))],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        full["$mkt_y10"].to_numpy(), [4.0 + 0.1 * i for i in range(len(DAYS))], rtol=1e-6
    )


def test_debug_market_h5_windowed_like_daily_pv_debug(
    market_store: Path, tmp_path: Path
) -> None:
    output = tmp_path / "src"
    make_factor_source("fixture_univ", market_store, output, debug_days=3)
    debug_pv = pd.read_hdf(output / "data_folder_debug" / "daily_pv.h5", key="data")
    debug_market = pd.read_hdf(output / "data_folder_debug" / MARKET_H5, key="data")
    assert isinstance(debug_pv, pd.DataFrame) and isinstance(debug_market, pd.DataFrame)
    pv_dates = debug_pv.index.get_level_values("datetime").unique()
    assert list(debug_market.index) == list(pv_dates)
    full_market = pd.read_hdf(output / "data_folder" / MARKET_H5, key="data")
    assert isinstance(full_market, pd.DataFrame)
    pd.testing.assert_frame_equal(debug_market, full_market.loc[pv_dates])


def test_market_readme_lines_in_both_folders(market_store: Path, tmp_path: Path) -> None:
    output = tmp_path / "src"
    make_factor_source("fixture_univ", market_store, output)
    for folder in ("data_folder", "data_folder_debug"):
        readme = (output / folder / "README.md").read_text()
        assert 'pd.read_hdf("market_series.h5", key="data")' in readme
        assert "ONE value per date shared by ALL instruments" in readme
        assert "betas, spreads" in readme
        assert "NEVER use a market series as a cross-sectional signal on its own" in readme


def test_market_nan_before_first_observation(tmp_path: Path) -> None:
    store = tmp_path / "us_data_late_mkt"
    client = MarketFakeFmp(
        bars={sym: make_bars(sym) for sym in ("AAPL", "NVDA")}, market_days=DAYS[1:]
    )
    build_from_fmp(
        ["AAPL", "NVDA"], DAYS[0], DAYS[-1], store, tmp_path / "ckpt", client,
        market_start=DAYS[0],
    )
    frame = load_market_frame(store, ["AAPL", "NVDA"])
    assert frame is not None
    assert np.isnan(frame["$mkt_gold"].iloc[0])
    assert not frame["$mkt_gold"].iloc[1:].isna().any()


def test_store_without_market_bins_writes_no_market_h5(
    store: Path, market_store: Path, tmp_path: Path
) -> None:
    output = tmp_path / "src"
    # First generate from the market-carrying store, then regenerate from the
    # plain store into the SAME output: stale market files must disappear.
    make_factor_source("fixture_univ", market_store, output)
    make_factor_source("fixture_univ", store, output)
    for folder in ("data_folder", "data_folder_debug"):
        assert not (output / folder / MARKET_H5).exists()
        readme = (output / folder / "README.md").read_text()
        assert "market_series.h5" not in readme
    assert load_market_frame(store, FIVE) is None


# ---------------------------------------------------------------------------
# daily_news.h5 companion (US-073)


def test_news_h5_round_trips_counts_into_both_folders(
    news_store: Path, tmp_path: Path
) -> None:
    """The US-014 round-trip: counts -> store bins -> factor-source read-back."""
    output = tmp_path / "src"
    make_factor_source("fixture_univ", news_store, output, debug_days=3)
    frames: dict[str, pd.DataFrame] = {}
    for folder in ("data_folder", "data_folder_debug"):
        path = output / folder / NEWS_H5
        assert path.exists()
        frame = pd.read_hdf(path, key="data")
        assert isinstance(frame, pd.DataFrame)
        assert isinstance(frame.index, pd.MultiIndex)
        assert frame.index.names == ["datetime", "instrument"]
        assert list(frame.columns) == [f"${NEWS_FIELD}"]
        frames[folder] = frame

    full = frames["data_folder"]
    assert len(full) == len(DAYS) * len(FIVE)
    aapl = full.xs("AAPL", level="instrument")[f"${NEWS_FIELD}"]
    assert np.isnan(aapl.iloc[0])  # day 1 predates news coverage: NaN, not 0
    np.testing.assert_allclose(
        aapl.iloc[1:].to_numpy(), [news_count("AAPL", j) for j in range(4)], rtol=1e-6
    )
    nvda = full.xs("NVDA", level="instrument")[f"${NEWS_FIELD}"]
    assert nvda.iloc[-1] == 0.0  # covered no-news day round-trips as explicit 0
    np.testing.assert_allclose(
        nvda.iloc[1:-1].to_numpy(), [news_count("NVDA", j) for j in range(3)], rtol=1e-6
    )


def test_debug_news_h5_windowed_like_daily_pv_debug(news_store: Path, tmp_path: Path) -> None:
    output = tmp_path / "src"
    make_factor_source(
        "fixture_univ", news_store, output, debug_days=3, debug_instruments=2
    )
    debug_pv = pd.read_hdf(output / "data_folder_debug" / "daily_pv.h5", key="data")
    debug_news = pd.read_hdf(output / "data_folder_debug" / NEWS_H5, key="data")
    assert isinstance(debug_pv, pd.DataFrame) and isinstance(debug_news, pd.DataFrame)
    # Same trading days AND same instrument subset as daily_pv_debug.
    assert list(debug_news.index.get_level_values("datetime").unique()) == list(
        debug_pv.index.get_level_values("datetime").unique()
    )
    assert sorted(set(debug_news.index.get_level_values("instrument"))) == sorted(
        set(debug_pv.index.get_level_values("instrument"))
    )
    full_news = pd.read_hdf(output / "data_folder" / NEWS_H5, key="data")
    assert isinstance(full_news, pd.DataFrame)
    pd.testing.assert_frame_equal(
        debug_news, full_news.loc[debug_news.index], check_freq=False
    )


def test_news_readme_lines_in_both_folders(news_store: Path, tmp_path: Path) -> None:
    output = tmp_path / "src"
    make_factor_source("fixture_univ", news_store, output)
    for folder in ("data_folder", "data_folder_debug"):
        readme = (output / folder / "README.md").read_text()
        assert 'pd.read_hdf("daily_news.h5", key="data")' in readme
        assert "published after 16:00 US/Eastern" in readme
        assert "counts toward the NEXT trading day" in readme
        assert "0 means the day is covered by news data" in readme
        assert "NaN means the day\nis OUTSIDE news coverage" in readme
        # market section still present too — both companions coexist
        assert 'pd.read_hdf("market_series.h5", key="data")' in readme


def test_store_without_news_bins_writes_no_news_h5(
    market_store: Path, news_store: Path, tmp_path: Path
) -> None:
    output = tmp_path / "src"
    # Generate from the news-carrying store, then regenerate from the
    # market-only store into the SAME output: stale news files must disappear.
    make_factor_source("fixture_univ", news_store, output)
    make_factor_source("fixture_univ", market_store, output)
    for folder in ("data_folder", "data_folder_debug"):
        assert not (output / folder / NEWS_H5).exists()
        readme = (output / folder / "README.md").read_text()
        assert "daily_news.h5" not in readme
        assert (output / folder / MARKET_H5).exists()  # market companion survives
    assert load_news_frame(market_store, FIVE) is None


def test_partial_news_bins_fail_loud(news_store: Path) -> None:
    (news_store / "features" / "nvda" / f"{NEWS_FIELD}.day.bin").unlink()
    with pytest.raises(FactorSourceError, match="NVDA"):
        load_news_frame(news_store, FIVE)


# ---------------------------------------------------------------------------
# Refusal paths


def test_refuses_missing_universe(store: Path, tmp_path: Path) -> None:
    with pytest.raises(FactorSourceError, match="make_universe"):
        make_factor_source("nonexistent", store, tmp_path / "src")
    assert not (tmp_path / "src").exists()  # nothing written on refusal


def test_cli_missing_universe_exits_nonzero(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = main(
        ["--universe", "nonexistent", "--store", str(store), "--output", str(tmp_path / "src")]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "nonexistent" in err and "make_universe" in err


def test_refuses_store_without_calendar(tmp_path: Path) -> None:
    bogus = tmp_path / "empty_store"
    (bogus / "instruments").mkdir(parents=True)
    (bogus / "instruments" / "u.txt").write_text("AAPL\t2024-01-02\t2024-01-08\n")
    with pytest.raises(FactorSourceError, match="calendar"):
        make_factor_source("u", bogus, tmp_path / "src")


def test_refuses_universe_symbol_with_missing_features(store: Path, tmp_path: Path) -> None:
    (store / "instruments" / "ghost.txt").write_text("ZZZZ\t2024-01-02\t2024-01-08\n")
    with pytest.raises(FactorSourceError, match="missing feature file"):
        make_factor_source("ghost", store, tmp_path / "src")


# ---------------------------------------------------------------------------
# CLI happy path


def test_cli_generates_files_and_exits_zero(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "src"
    rc = main(
        [
            "--universe",
            "fixture_univ",
            "--store",
            str(store),
            "--output",
            str(output),
            "--debug-days",
            "2",
            "--debug-instruments",
            "3",
        ]
    )
    assert rc == 0
    assert "fixture_univ" in capsys.readouterr().out
    assert (output / "daily_pv_all.h5").exists()
    assert (output / "daily_pv_debug.h5").exists()


# ---------------------------------------------------------------------------
# Parity with qlib D.features (what upstream generate.py produces)


def test_frame_matches_qlib_features_output(store: Path) -> None:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(store), region="us")
    expected = (
        D.features(D.instruments(market="fixture_univ"), list(COLUMNS), freq="day")
        .swaplevel()
        .sort_index()
    )
    ours = load_universe_frame(store, "fixture_univ")
    pd.testing.assert_frame_equal(
        ours, expected, check_names=False, check_dtype=False, rtol=1e-6
    )

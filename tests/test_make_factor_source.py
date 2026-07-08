"""Tests for data/make_factor_source.py: h5 shape, debug subset, refusal, qlib parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.make_factor_source import (
    COLUMNS,
    FactorSourceError,
    debug_subset,
    load_universe_frame,
    main,
    make_factor_source,
)
from data.make_universe import make_universe
from tests.test_build_store import DAYS, build_from_fmp, five_ticker_client

FIVE = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """Five-ticker fixture store with a us_liquid-style custom universe inside."""
    store = tmp_path / "us_data"
    build_from_fmp(FIVE, DAYS[0], DAYS[-1], store, tmp_path / "ckpt", five_ticker_client())
    make_universe("fixture_univ", store, tickers=",".join(FIVE))
    make_universe("pair", store, tickers="AAPL,NVDA")
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

"""Tests for data/build_store.py: checkpointed backfill, bin layout, swap, qlib read."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.build_store import (
    BuildError,
    StoreValidationError,
    TickerBundle,
    backfill,
    build_from_fmp,
    build_store,
    fetch_bundle,
    main,
    resolve_tickers,
    validate_store,
)
from data.fmp import Dividend, EodBar, FmpClient, FmpError, Split

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


def test_qlib_reads_aapl_ohlcv_with_no_nan_closes(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    build_from_fmp(
        ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
        DAYS[0],
        DAYS[-1],
        store,
        tmp_path / "ckpt",
        five_ticker_client(),
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

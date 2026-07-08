"""Tests for data/make_universe.py: filters, gap rejection, instruments format."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data.build_store import TickerBundle, build_store
from data.fmp import EodBar, Split
from data.make_universe import (
    DEFAULT_CONFIG_PATH,
    UniverseError,
    apply_filters,
    liquidity_stats,
    main,
    make_universe,
    read_instrument_spans,
    resolve_config,
    write_instruments_file,
)

DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]


def make_bars(
    symbol: str, close: float = 100.0, volume: float = 1_000.0, days: list[date] | None = None
) -> tuple[EodBar, ...]:
    """Flat-price bars so dollar volume is exactly close * volume every day."""
    return tuple(
        EodBar(
            symbol=symbol,
            date=day,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume,
        )
        for day in (DAYS if days is None else days)
    )


def build_fixture_store(
    tmp_path: Path,
    specs: dict[str, tuple[float, float]],
    splits: dict[str, tuple[Split, ...]] | None = None,
) -> Path:
    """Store with one ticker per spec entry: symbol -> (close, volume)."""
    bundles = [
        TickerBundle(
            symbol=sym,
            bars=make_bars(sym, close=c, volume=v),
            splits=(splits or {}).get(sym, ()),
            dividends=(),
        )
        for sym, (c, v) in specs.items()
    ]
    store = tmp_path / "us_data"
    build_store(bundles, store)
    return store


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


FILTER_CONFIG = """
universes:
  test_liquid:
    min_adv_usd: 100000
    min_price: 10.0
    adv_window: 20
"""


# ---------------------------------------------------------------------------
# Built-in config contract (the real data/config.yaml)


def test_us_liquid_builtin_has_adv_and_price_thresholds() -> None:
    config = resolve_config("us_liquid", DEFAULT_CONFIG_PATH)
    assert config.builtin
    assert config.min_adv_usd is not None and config.min_adv_usd > 0
    assert config.min_price is not None and config.min_price > 0
    assert config.adv_window > 0


def test_sp500_builtin_points_at_committed_snapshot() -> None:
    config = resolve_config("sp500", DEFAULT_CONFIG_PATH)
    assert config.builtin
    assert not config.has_filters
    assert config.tickers_file is not None and config.tickers_file.exists()
    tickers = config.tickers_file.read_text().split()
    assert len(tickers) > 400
    assert "AAPL" in tickers
    assert all(t == t.upper() for t in tickers)


def test_unknown_name_resolves_to_bare_custom_config() -> None:
    config = resolve_config("semis", DEFAULT_CONFIG_PATH)
    assert not config.builtin
    assert not config.has_filters
    assert config.tickers_file is None


# ---------------------------------------------------------------------------
# Liquidity filters on a fixture store


def test_adv_threshold_pass_at_limit_fail_below(tmp_path: Path) -> None:
    # close 10 * volume 10_000 = ADV exactly 100_000 (the min); RUNT is 10 short.
    store = build_fixture_store(
        tmp_path, {"LIQD": (10.0, 10_000.0), "RUNT": (10.0, 9_999.0)}
    )
    config = resolve_config("test_liquid", write_config(tmp_path, FILTER_CONFIG))
    kept, rejected = apply_filters(store, ["LIQD", "RUNT"], config)
    assert kept == ["LIQD"]
    assert [r.symbol for r in rejected] == ["RUNT"]
    assert "ADV" in rejected[0].reason and "min" in rejected[0].reason


def test_price_threshold_pass_at_limit_fail_below(tmp_path: Path) -> None:
    # Both clear the ADV bar; CHEP fails only on price (9.99 < 10).
    store = build_fixture_store(
        tmp_path, {"PRCY": (10.0, 100_000.0), "CHEP": (9.99, 100_000.0)}
    )
    config = resolve_config("test_liquid", write_config(tmp_path, FILTER_CONFIG))
    kept, rejected = apply_filters(store, ["PRCY", "CHEP"], config)
    assert kept == ["PRCY"]
    assert [r.symbol for r in rejected] == ["CHEP"]
    assert "price" in rejected[0].reason


def test_adv_uses_raw_dollar_volume_across_a_split(tmp_path: Path) -> None:
    # 2:1 split on the middle day: stored close/volume are adjusted, but their
    # product must recover the raw $100 * 1000 = $100k daily dollar volume.
    split = Split(symbol="SPLT", date=DAYS[2], numerator=2.0, denominator=1.0)
    store = build_fixture_store(
        tmp_path, {"SPLT": (100.0, 1_000.0)}, splits={"SPLT": (split,)}
    )
    adv, last_raw_price = liquidity_stats(store, "SPLT", adv_window=20)
    assert adv == pytest.approx(100_000.0, rel=1e-6)
    assert last_raw_price == pytest.approx(100.0, rel=1e-6)


def test_universe_without_filters_keeps_everything(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAA": (1.0, 1.0), "BBB": (2.0, 2.0)})
    kept, rejected = apply_filters(
        store, ["AAA", "BBB"], resolve_config("custom_thing", DEFAULT_CONFIG_PATH)
    )
    assert kept == ["AAA", "BBB"]
    assert rejected == []


# ---------------------------------------------------------------------------
# Gap rejection


def test_absent_tickers_rejected_with_gap_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    rc = main(["--name", "semis", "--tickers", "AAPL,NVDA,AVGO", "--store", str(store)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "NVDA" in err and "AVGO" in err and "build_store" in err
    assert not (store / "instruments" / "semis.txt").exists()


def test_gap_rejection_via_api_names_only_missing(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    with pytest.raises(UniverseError) as excinfo:
        make_universe("semis", store, tickers="AAPL,NVDA")
    assert "NVDA" in str(excinfo.value)
    assert "AAPL" not in str(excinfo.value).split(":")[-1]


# ---------------------------------------------------------------------------
# Instruments file format


def test_instruments_file_rows_match_master_spans(tmp_path: Path) -> None:
    store = build_fixture_store(
        tmp_path, {"MSFT": (300.0, 1_000.0), "AAPL": (100.0, 1_000.0)}
    )
    path = make_universe("pair", store, tickers="MSFT,AAPL")
    assert path == store / "instruments" / "pair.txt"
    spans = read_instrument_spans(store)
    expected = "".join(f"{s}\t{spans[s][0]}\t{spans[s][1]}\n" for s in ["AAPL", "MSFT"])
    assert path.read_text() == expected
    for line in path.read_text().splitlines():
        symbol, start, end = line.split("\t")
        assert symbol == symbol.upper()
        assert start == DAYS[0].isoformat() and end == DAYS[-1].isoformat()


def test_reserved_and_invalid_names_rejected(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    spans = read_instrument_spans(store)
    with pytest.raises(UniverseError, match="reserved"):
        write_instruments_file(store, "all", ["AAPL"], spans)
    for bad in ("My Universe", "UPPER", "../evil", "9lives"):
        with pytest.raises(UniverseError, match="invalid universe name"):
            write_instruments_file(store, bad, ["AAPL"], spans)


def test_empty_universe_after_filters_is_an_error(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"RUNT": (1.0, 1.0)})
    config_path = write_config(tmp_path, FILTER_CONFIG)
    rc = main(
        ["--name", "test_liquid", "--store", str(store), "--config", str(config_path)]
    )
    assert rc == 1
    assert not (store / "instruments" / "test_liquid.txt").exists()


# ---------------------------------------------------------------------------
# CLI behaviors


def test_builtin_with_filters_defaults_to_all_store_tickers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_fixture_store(
        tmp_path,
        {"LIQD": (10.0, 10_000.0), "RUNT": (10.0, 9_999.0), "CHEP": (9.0, 100_000.0)},
    )
    config_path = write_config(tmp_path, FILTER_CONFIG)
    rc = main(["--name", "test_liquid", "--store", str(store), "--config", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "filtered RUNT" in out and "filtered CHEP" in out
    written = (store / "instruments" / "test_liquid.txt").read_text()
    assert written.splitlines() == [f"LIQD\t{DAYS[0].isoformat()}\t{DAYS[-1].isoformat()}"]


def test_custom_universe_requires_tickers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    rc = main(["--name", "semis", "--store", str(store)])
    assert rc == 1
    assert "--tickers" in capsys.readouterr().err


def test_config_tickers_file_used_when_no_cli_tickers(tmp_path: Path) -> None:
    store = build_fixture_store(
        tmp_path, {"AAPL": (100.0, 1_000.0), "MSFT": (300.0, 1_000.0)}
    )
    listing = tmp_path / "pair.txt"
    listing.write_text("AAPL\nMSFT\n")
    config_path = write_config(
        tmp_path,
        f"universes:\n  pairlist:\n    tickers_file: {listing.name}\n",
    )
    rc = main(["--name", "pairlist", "--store", str(store), "--config", str(config_path)])
    assert rc == 0
    assert (store / "instruments" / "pairlist.txt").read_text().startswith("AAPL\t")


def test_cli_tickers_override_config_tickers_file(tmp_path: Path) -> None:
    store = build_fixture_store(
        tmp_path, {"AAPL": (100.0, 1_000.0), "MSFT": (300.0, 1_000.0)}
    )
    listing = tmp_path / "pair.txt"
    listing.write_text("AAPL\nMSFT\n")
    config_path = write_config(
        tmp_path,
        f"universes:\n  pairlist:\n    tickers_file: {listing.name}\n",
    )
    rc = main(
        [
            "--name",
            "pairlist",
            "--tickers",
            "MSFT",
            "--store",
            str(store),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 0
    assert (store / "instruments" / "pairlist.txt").read_text() == (
        f"MSFT\t{DAYS[0].isoformat()}\t{DAYS[-1].isoformat()}\n"
    )


def test_missing_store_is_actionable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--name", "semis", "--tickers", "AAPL", "--store", str(tmp_path / "nope")])
    assert rc == 1
    assert "build" in capsys.readouterr().err.lower()

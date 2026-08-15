"""Tests for data/make_universe.py: filters, gap rejection, instruments format, PIT mode."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.build_store import TickerBundle, build_store
from data.fmp import EodBar, Split
from data.make_universe import (
    DEFAULT_CONFIG_PATH,
    MODE_LAST_WINDOW,
    MODE_PIT,
    UniverseError,
    apply_filters,
    liquidity_stats,
    main,
    make_universe,
    month_start_indices,
    read_instrument_spans,
    resolve_config,
    write_instrument_rows,
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


# ---------------------------------------------------------------------------
# Point-in-time (PIT) mode
#
# Synthetic 8-month store, adv_window=3: dollar volume is close(100) * volume,
# so volume 10_000 -> ADV $1M (passes the $100k bar) and volume 10 -> $1k
# (fails). Evaluations land on the first trading day of each month.

PIT_DAYS = [d.date() for d in pd.bdate_range("2024-01-02", "2024-08-30")]
HIGH_VOL, LOW_VOL = 10_000.0, 10.0

PIT_CONFIG = """
universes:
  pit_liquid:
    min_adv_usd: 100000
    min_price: 10.0
    adv_window: 3
    mode: pit
"""


def var_bars(
    symbol: str,
    volume_on: Callable[[date], float],
    days: Sequence[date] = PIT_DAYS,
    close: float = 100.0,
) -> tuple[EodBar, ...]:
    return tuple(
        EodBar(
            symbol=symbol,
            date=day,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume_on(day),
        )
        for day in days
    )


def build_pit_store(tmp_path: Path, bars: dict[str, tuple[EodBar, ...]]) -> Path:
    store = tmp_path / "us_data"
    build_store(
        [
            TickerBundle(symbol=sym, bars=b, splits=(), dividends=())
            for sym, b in bars.items()
        ],
        store,
    )
    return store


def first_trading_day(month: int) -> date:
    return next(d for d in PIT_DAYS if d.month == month)


def read_rows(store: Path, name: str) -> list[tuple[str, str, str]]:
    lines = (store / "instruments" / f"{name}.txt").read_text().splitlines()
    rows = [tuple(line.split("\t")) for line in lines]
    assert all(len(r) == 3 for r in rows)
    return rows  # type: ignore[return-value]


def test_month_start_indices_pick_first_trading_days() -> None:
    calendar = [d.isoformat() for d in PIT_DAYS]
    picked = [calendar[i] for i in month_start_indices(calendar)]
    assert picked == [first_trading_day(m).isoformat() for m in range(1, 9)]


def test_pit_spans_from_synthetic_price_volume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Always-liquid = full span; mid-history riser enters after confirmation;
    never-liquid is filtered out entirely."""
    rise = date(2024, 3, 25)
    store = build_pit_store(
        tmp_path,
        {
            "ALLG": var_bars("ALLG", lambda d: HIGH_VOL),
            "RISR": var_bars("RISR", lambda d: HIGH_VOL if d >= rise else LOW_VOL),
            "RUNT": var_bars("RUNT", lambda d: LOW_VOL),
        },
    )
    config_path = write_config(tmp_path, PIT_CONFIG)
    rc = main(["--name", "pit_liquid", "--store", str(store), "--config", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "filtered RUNT" in out and "point-in-time" in out
    # RISR first passes at the Apr 1 evaluation (trailing window fully high),
    # confirms at May 1 (one-period entry hysteresis) -> member from May 1.
    assert read_rows(store, "pit_liquid") == [
        ("ALLG", "2024-01-02", "2024-08-30"),
        ("RISR", first_trading_day(5).isoformat(), "2024-08-30"),
    ]


def test_pit_hysteresis_prevents_single_month_churn(tmp_path: Path) -> None:
    """A one-evaluation dip never exits a member; a one-evaluation spike never
    admits a non-member."""
    dip = (date(2024, 5, 24), date(2024, 6, 4))  # covers exactly the Jun 3 eval window

    def dip_vol(d: date) -> float:
        return LOW_VOL if dip[0] <= d <= dip[1] else HIGH_VOL

    def spike_vol(d: date) -> float:
        return HIGH_VOL if dip[0] <= d <= dip[1] else LOW_VOL

    store = build_pit_store(
        tmp_path,
        {"DIPR": var_bars("DIPR", dip_vol), "SPIK": var_bars("SPIK", spike_vol)},
    )
    config_path = write_config(tmp_path, PIT_CONFIG)
    make_universe("pit_liquid", store, config_path=config_path)
    # DIPR: single unbroken span despite failing the June evaluation.
    # SPIK: passed only the June evaluation -> never confirmed, no rows.
    assert read_rows(store, "pit_liquid") == [("DIPR", "2024-01-02", "2024-08-30")]


def test_pit_multiple_spans_per_symbol_in_span_format(tmp_path: Path) -> None:
    """Liquid -> illiquid -> liquid again yields two rows for the same symbol."""
    low_from, high_again = date(2024, 3, 27), date(2024, 6, 26)

    def wobble(d: date) -> float:
        return LOW_VOL if low_from <= d < high_again else HIGH_VOL

    store = build_pit_store(tmp_path, {"WOBL": var_bars("WOBL", wobble)})
    config_path = write_config(tmp_path, PIT_CONFIG)
    make_universe("pit_liquid", store, config_path=config_path)
    # Exit confirmed at the May 1 eval -> first span ends the prior trading
    # day; re-entry confirmed at the Aug 1 eval -> second span starts there.
    may1 = first_trading_day(5)
    day_before_may1 = PIT_DAYS[PIT_DAYS.index(may1) - 1]
    assert read_rows(store, "pit_liquid") == [
        ("WOBL", "2024-01-02", day_before_may1.isoformat()),
        ("WOBL", first_trading_day(8).isoformat(), "2024-08-30"),
    ]


def test_pit_delisting_clamps_span_end_and_late_listing_skips_evals(tmp_path: Path) -> None:
    gone_days = [d for d in PIT_DAYS if d <= date(2024, 6, 14)]
    late_days = [d for d in PIT_DAYS if d >= date(2024, 4, 15)]
    store = build_pit_store(
        tmp_path,
        {
            "ALLG": var_bars("ALLG", lambda d: HIGH_VOL),
            "GONE": var_bars("GONE", lambda d: HIGH_VOL, days=gone_days),
            "LATE": var_bars("LATE", lambda d: HIGH_VOL, days=late_days),
        },
    )
    config_path = write_config(tmp_path, PIT_CONFIG)
    make_universe("pit_liquid", store, config_path=config_path)
    # GONE delists 2024-06-14: the exit is only CONFIRMED at the Aug eval, but
    # the span end clamps to its own last bar. LATE lists 2024-04-15:
    # pre-listing evaluations are skipped, membership starts at the first
    # post-listing evaluation (May 1).
    assert read_rows(store, "pit_liquid") == [
        ("ALLG", "2024-01-02", "2024-08-30"),
        ("GONE", "2024-01-02", "2024-06-14"),
        ("LATE", first_trading_day(5).isoformat(), "2024-08-30"),
    ]


def test_pit_price_threshold_applies_per_evaluation(tmp_path: Path) -> None:
    """Volume is fine throughout; the raw price dropping under $10 exits the name."""
    cheap_from = date(2024, 5, 24)
    bars = tuple(
        EodBar(
            symbol="FALL",
            date=day,
            open=100.0,
            high=100.0,
            low=100.0,
            close=9.0 if day >= cheap_from else 100.0,
            volume=HIGH_VOL,
        )
        for day in PIT_DAYS
    )
    store = build_pit_store(tmp_path, {"FALL": bars})
    config_path = write_config(tmp_path, PIT_CONFIG)
    make_universe("pit_liquid", store, config_path=config_path)
    # Fails the Jun 3 eval (price $9), confirmed at Jul 1 -> span ends the
    # trading day before Jul 1.
    jul1 = first_trading_day(7)
    day_before_jul1 = PIT_DAYS[PIT_DAYS.index(jul1) - 1]
    assert read_rows(store, "pit_liquid") == [
        ("FALL", "2024-01-02", day_before_jul1.isoformat())
    ]


def test_legacy_mode_flag_overrides_pit_config(tmp_path: Path) -> None:
    """--mode last_window on a PIT universe reproduces the legacy full-span shape
    (the retroactive-admission bias PIT exists to remove)."""
    rise = date(2024, 3, 25)
    store = build_pit_store(
        tmp_path,
        {"RISR": var_bars("RISR", lambda d: HIGH_VOL if d >= rise else LOW_VOL)},
    )
    config_path = write_config(tmp_path, PIT_CONFIG)
    rc = main(
        [
            "--name",
            "pit_liquid",
            "--store",
            str(store),
            "--config",
            str(config_path),
            "--mode",
            "last_window",
        ]
    )
    assert rc == 0
    # Legacy mode: recent window is liquid -> full history admitted.
    assert read_rows(store, "pit_liquid") == [("RISR", "2024-01-02", "2024-08-30")]


def test_pit_mode_requires_filters(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    config_path = write_config(
        tmp_path, "universes:\n  bare:\n    mode: pit\n"
    )
    with pytest.raises(UniverseError, match="needs liquidity filters"):
        make_universe("bare", store, config_path=config_path)


def test_unknown_mode_rejected_in_config_and_arg(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path, "universes:\n  odd:\n    min_price: 1.0\n    mode: monthly\n"
    )
    with pytest.raises(UniverseError, match="unknown mode 'monthly'"):
        resolve_config("odd", config_path)
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    with pytest.raises(UniverseError, match="unknown universe mode"):
        make_universe("semis", store, tickers="AAPL", mode="bogus")


def test_us_liquid_defaults_to_pit_mode() -> None:
    assert resolve_config("us_liquid", DEFAULT_CONFIG_PATH).mode == MODE_PIT
    assert resolve_config("sp500", DEFAULT_CONFIG_PATH).mode == MODE_LAST_WINDOW
    assert resolve_config("custom", DEFAULT_CONFIG_PATH).mode == MODE_LAST_WINDOW


def test_write_instrument_rows_sorts_and_allows_repeats(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, {"AAPL": (100.0, 1_000.0)})
    path = write_instrument_rows(
        store,
        "spanny",
        [
            ("ZZZ", "2024-01-02", "2024-02-01"),
            ("AAA", "2024-03-01", "2024-04-01"),
            ("AAA", "2024-01-02", "2024-02-01"),
        ],
    )
    assert path.read_text() == (
        "AAA\t2024-01-02\t2024-02-01\n"
        "AAA\t2024-03-01\t2024-04-01\n"
        "ZZZ\t2024-01-02\t2024-02-01\n"
    )

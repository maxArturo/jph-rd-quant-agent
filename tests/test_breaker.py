"""Tests for execution/breaker.py (US-031)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.breaker import (
    CONFIG_PATH,
    Breaker,
    BreakerConfig,
    BreakerConfigError,
    BreakerReason,
    BreakerStateError,
    load_breaker_config,
)

CONFIG = BreakerConfig(max_daily_notional_usd=200_000.0, max_drawdown_pct=20.0)


@pytest.fixture()
def breaker(tmp_path: Path) -> Breaker:
    return Breaker(
        CONFIG,
        halt_file=tmp_path / "halt",
        high_water_mark_file=tmp_path / "hwm.json",
    )


def write_hwm(breaker: Breaker, value: float) -> None:
    breaker.high_water_mark_file.write_text(json.dumps({"high_water_mark": value}))


def read_hwm(breaker: Breaker) -> float:
    return json.loads(breaker.high_water_mark_file.read_text())["high_water_mark"]


# ---------------------------------------------------------------- config


class TestLoadBreakerConfig:
    def write(self, tmp_path: Path, payload: object) -> Path:
        path = tmp_path / "breaker.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    def test_committed_config_loads(self) -> None:
        config = load_breaker_config(CONFIG_PATH)
        assert config.max_daily_notional_usd == 200_000.0
        assert config.max_drawdown_pct == 20.0

    def test_values_coerced_to_float(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, {"max_daily_notional_usd": 50000, "max_drawdown_pct": 15})
        config = load_breaker_config(path)
        assert isinstance(config.max_daily_notional_usd, float)
        assert isinstance(config.max_drawdown_pct, float)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(BreakerConfigError, match="not found"):
            load_breaker_config(tmp_path / "absent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "{nope")
        with pytest.raises(BreakerConfigError, match="not valid JSON"):
            load_breaker_config(path)

    def test_non_object(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, [1, 2])
        with pytest.raises(BreakerConfigError, match="JSON object"):
            load_breaker_config(path)

    def test_unknown_key_refused(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            {"max_daily_notional_usd": 1, "max_drawdown_pct": 1, "max_bananas": 3},
        )
        with pytest.raises(BreakerConfigError, match="unknown keys: max_bananas"):
            load_breaker_config(path)

    def test_missing_key_refused(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, {"max_daily_notional_usd": 1})
        with pytest.raises(BreakerConfigError, match="missing keys: max_drawdown_pct"):
            load_breaker_config(path)

    @pytest.mark.parametrize("bad", [0, -5, "10", None, True])
    def test_non_positive_or_non_numeric_refused(self, tmp_path: Path, bad: object) -> None:
        path = self.write(tmp_path, {"max_daily_notional_usd": bad, "max_drawdown_pct": 10})
        with pytest.raises(BreakerConfigError, match="max_daily_notional_usd"):
            load_breaker_config(path)

    @pytest.mark.parametrize("bad", [100, 150])
    def test_drawdown_at_or_over_100_refused(self, tmp_path: Path, bad: float) -> None:
        path = self.write(tmp_path, {"max_daily_notional_usd": 1, "max_drawdown_pct": bad})
        with pytest.raises(BreakerConfigError, match="below 100"):
            load_breaker_config(path)


# ------------------------------------------------------------- halt file


class TestHaltFile:
    def test_halt_file_trips(self, breaker: Breaker) -> None:
        breaker.halt_file.write_text("")
        trip = breaker.check(equity=100_000.0, day_notional_usd=0.0)
        assert trip is not None
        assert trip.reason is BreakerReason.HALT_FILE
        assert trip.message.startswith("halt_file:")
        assert str(breaker.halt_file) in trip.message

    def test_halt_note_surfaced_in_message(self, breaker: Breaker) -> None:
        breaker.halt_file.write_text("operator: bad fills\n")
        trip = breaker.check(equity=100_000.0, day_notional_usd=0.0)
        assert trip is not None
        assert "operator: bad fills" in trip.message

    def test_halt_checked_before_other_trips(self, breaker: Breaker) -> None:
        write_hwm(breaker, 200_000.0)  # 50% drawdown would also trip
        breaker.halt_file.write_text("")
        trip = breaker.check(equity=100_000.0, day_notional_usd=999_999.0)
        assert trip is not None
        assert trip.reason is BreakerReason.HALT_FILE

    def test_halt_and_clear_halt_round_trip(self, breaker: Breaker) -> None:
        assert not breaker.halted
        breaker.halt("manual stop")
        assert breaker.halted
        assert breaker.halt_file.read_text() == "manual stop\n"
        breaker.clear_halt()
        assert not breaker.halted
        assert breaker.check(equity=100_000.0, day_notional_usd=0.0) is None

    def test_halt_empty_note_writes_empty_file(self, breaker: Breaker) -> None:
        breaker.halt()
        assert breaker.halt_file.read_text() == ""

    def test_clear_halt_when_absent_is_noop(self, breaker: Breaker) -> None:
        breaker.clear_halt()  # must not raise
        assert not breaker.halted

    def test_halt_creates_parent_dirs(self, tmp_path: Path) -> None:
        breaker = Breaker(
            CONFIG,
            halt_file=tmp_path / "deep" / "dir" / "halt",
            high_water_mark_file=tmp_path / "hwm.json",
        )
        breaker.halt("stop")
        assert breaker.halted


# -------------------------------------------------------- daily notional


class TestDailyNotional:
    def test_at_limit_passes(self, breaker: Breaker) -> None:
        assert breaker.check(equity=100_000.0, day_notional_usd=200_000.0) is None

    def test_just_over_trips(self, breaker: Breaker) -> None:
        trip = breaker.check(equity=100_000.0, day_notional_usd=200_000.01)
        assert trip is not None
        assert trip.reason is BreakerReason.DAILY_NOTIONAL
        assert trip.message.startswith("max_daily_notional_usd:")
        assert "$200,000.01" in trip.message
        assert "$200,000.00" in trip.message

    def test_notional_trip_does_not_touch_high_water_mark(self, breaker: Breaker) -> None:
        write_hwm(breaker, 90_000.0)
        trip = breaker.check(equity=100_000.0, day_notional_usd=300_000.0)
        assert trip is not None
        assert read_hwm(breaker) == 90_000.0  # new-high equity NOT recorded on a trip

    def test_negative_notional_rejected(self, breaker: Breaker) -> None:
        with pytest.raises(ValueError, match="day_notional_usd"):
            breaker.check(equity=100_000.0, day_notional_usd=-1.0)

    def test_nan_notional_rejected(self, breaker: Breaker) -> None:
        with pytest.raises(ValueError, match="day_notional_usd"):
            breaker.check(equity=100_000.0, day_notional_usd=float("nan"))


# ------------------------------------------------------------- drawdown


class TestDrawdown:
    def test_at_limit_passes(self, breaker: Breaker) -> None:
        write_hwm(breaker, 100_000.0)
        assert breaker.check(equity=80_000.0, day_notional_usd=0.0) is None

    def test_just_over_trips(self, breaker: Breaker) -> None:
        write_hwm(breaker, 100_000.0)
        trip = breaker.check(equity=79_999.0, day_notional_usd=0.0)
        assert trip is not None
        assert trip.reason is BreakerReason.DRAWDOWN
        assert trip.message.startswith("max_drawdown_pct:")
        assert "$79,999.00" in trip.message
        assert "$100,000.00" in trip.message
        assert "20%" in trip.message

    def test_drawdown_trip_does_not_lower_high_water_mark(self, breaker: Breaker) -> None:
        write_hwm(breaker, 100_000.0)
        breaker.check(equity=50_000.0, day_notional_usd=0.0)
        assert read_hwm(breaker) == 100_000.0

    def test_no_high_water_mark_file_passes_and_seeds(self, breaker: Breaker) -> None:
        assert not breaker.high_water_mark_file.exists()
        assert breaker.check(equity=100_000.0, day_notional_usd=0.0) is None
        assert read_hwm(breaker) == 100_000.0

    @pytest.mark.parametrize(
        "payload",
        [
            "{corrupt",
            json.dumps([1, 2]),
            json.dumps({"wrong_key": 1.0}),
            json.dumps({"high_water_mark": "100000"}),
            json.dumps({"high_water_mark": -5}),
            json.dumps({"high_water_mark": 0}),
            json.dumps({"high_water_mark": True}),
        ],
    )
    def test_corrupt_high_water_mark_refuses_to_trade(
        self, breaker: Breaker, payload: str
    ) -> None:
        breaker.high_water_mark_file.write_text(payload)
        with pytest.raises(BreakerStateError, match="refusing to trade"):
            breaker.check(equity=100_000.0, day_notional_usd=0.0)

    @pytest.mark.parametrize("bad", [0.0, -100.0, float("nan"), float("inf")])
    def test_bad_equity_rejected(self, breaker: Breaker, bad: float) -> None:
        with pytest.raises(ValueError, match="equity"):
            breaker.check(equity=bad, day_notional_usd=0.0)


# ------------------------------------------------- clean pass + persistence


class TestHighWaterMarkPersistence:
    def test_clean_pass_raises_high_water_mark(self, breaker: Breaker) -> None:
        write_hwm(breaker, 100_000.0)
        assert breaker.check(equity=110_000.0, day_notional_usd=1_000.0) is None
        assert read_hwm(breaker) == 110_000.0

    def test_clean_pass_below_peak_keeps_high_water_mark(self, breaker: Breaker) -> None:
        write_hwm(breaker, 100_000.0)
        assert breaker.check(equity=95_000.0, day_notional_usd=0.0) is None
        assert read_hwm(breaker) == 100_000.0

    def test_high_water_mark_survives_restart(self, breaker: Breaker) -> None:
        assert breaker.check(equity=120_000.0, day_notional_usd=0.0) is None
        # New Breaker instance over the same files = process restart.
        reborn = Breaker(
            CONFIG,
            halt_file=breaker.halt_file,
            high_water_mark_file=breaker.high_water_mark_file,
        )
        trip = reborn.check(equity=90_000.0, day_notional_usd=0.0)  # 25% below 120k
        assert trip is not None
        assert trip.reason is BreakerReason.DRAWDOWN

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        breaker = Breaker(
            CONFIG,
            halt_file=tmp_path / "halt",
            high_water_mark_file=tmp_path / "deep" / "dir" / "hwm.json",
        )
        assert breaker.check(equity=100_000.0, day_notional_usd=0.0) is None
        assert read_hwm(breaker) == 100_000.0

    def test_no_stray_temp_file_left(self, breaker: Breaker) -> None:
        breaker.check(equity=100_000.0, day_notional_usd=0.0)
        leftovers = [p.name for p in breaker.high_water_mark_file.parent.iterdir()]
        assert breaker.high_water_mark_file.name in leftovers
        assert not any(name.endswith(".tmp") for name in leftovers)

    def test_default_paths_point_at_rdq_data(self) -> None:
        breaker = Breaker(CONFIG)
        assert breaker.halt_file == Path.home() / "rdq-data" / "breaker" / "halt"
        assert (
            breaker.high_water_mark_file
            == Path.home() / "rdq-data" / "breaker" / "high_water_mark.json"
        )

"""Tests for execution/divergence.py (US-016).

The pure half (evaluate_divergence, config loader, backtest stats) is tested
with hand-built inputs so the z math is asserted exactly; the runner is
driven through a fake PortfolioReader against a real promoted-strategy state
DB and fixture workspace. Slack is a list-append notify; the halt file is
verified through the real Breaker so "same file the breaker checks" is a
tested fact, not a comment.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pandas as pd
import pytest

from execution.alpaca_client import AlpacaError, PortfolioEntry, PortfolioHistory
from execution.breaker import Breaker, BreakerConfig, BreakerReason
from execution.divergence import (
    BacktestStats,
    DivergenceConfig,
    DivergenceConfigError,
    DivergenceError,
    choose_period,
    equity_since_promotion,
    evaluate_divergence,
    load_backtest_stats,
    load_divergence_config,
    max_drawdown,
    realized_daily_returns,
    run_divergence,
    write_halt_file,
)
from orchestrator.state import StateStore

MDD_CSV_KEY = "1day.excess_return_with_cost.max_drawdown"


# --- fixtures -----------------------------------------------------------------


def write_ret(path: Path, returns: list[float], cost: float = 0.0) -> None:
    index = pd.bdate_range("2025-01-02", periods=len(returns))
    pd.DataFrame(
        {"return": returns, "cost": [cost] * len(returns)}, index=index
    ).to_pickle(path)


def make_history(
    promoted_date: dt.date,
    returns: list[float],
    start_equity: float = 100_000.0,
    equities: list[float] | None = None,
) -> PortfolioHistory:
    """History: promotion-day baseline point, then one point per return."""
    entries = [
        PortfolioEntry(
            date=promoted_date, equity=start_equity, profit_loss=None, profit_loss_pct=None
        )
    ]
    equity = start_equity
    for i, daily in enumerate(returns):
        equity = equities[i] if equities is not None else equity * (1 + daily)
        entries.append(
            PortfolioEntry(
                date=promoted_date + dt.timedelta(days=i + 1),
                equity=equity,
                profit_loss=equity * daily,
                profit_loss_pct=daily,
            )
        )
    return PortfolioHistory(timeframe="1D", base_value=start_equity, entries=entries)


class FakeAlpaca:
    def __init__(
        self, history: PortfolioHistory | None = None, error: Exception | None = None
    ) -> None:
        self.history = history
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> PortfolioHistory:
        self.calls.append((period, timeframe))
        if self.error is not None:
            raise self.error
        assert self.history is not None
        return self.history


def promoted_env(
    tmp_path: Path,
    backtest_returns: list[float] | None = None,
    mdd: float = -0.20,
    write_csv: bool = True,
) -> tuple[Path, Path, dt.date]:
    """Real state DB + fixture workspace; returns (db, workspace, promoted_date)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Alternating ±1% -> mean exactly 0 (so mu_adj = expected = 0), sigma known.
    write_ret(workspace / "ret.pkl", backtest_returns or [0.01, -0.01] * 30)
    if write_csv:
        pd.Series({MDD_CSV_KEY: mdd}).to_csv(workspace / "qlib_res.csv")
    db_path = tmp_path / "state.sqlite"
    store = StateStore(db_path)
    store.set_promoted_strategy(str(workspace), {"universe": "us_liquid"}, source="cli")
    promoted = store.get_promoted_strategy()
    assert promoted is not None
    promoted_date = dt.date.fromisoformat(promoted.promoted_at[:10])
    return db_path, workspace, promoted_date


def fixture_sigma(returns: list[float] | None = None) -> float:
    series = pd.Series(returns or [0.01, -0.01] * 30, dtype=float)
    return float(series.std())


# --- config loader ------------------------------------------------------------


def test_missing_config_file_yields_defaults(tmp_path: Path) -> None:
    config = load_divergence_config(tmp_path / "absent.yaml")
    assert config == DivergenceConfig()
    assert (config.haircut, config.warn_z, config.halt_z) == (0.5, -2.0, -3.0)
    assert (config.mdd_tolerance, config.window_days) == (1.25, 20)


def test_config_section_parsed(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "divergence:\n  haircut: 0.7\n  warn_z: -1.5\n  halt_z: -2.5\n"
        "  mdd_tolerance: 1.1\n  window_days: 10\n"
    )
    config = load_divergence_config(path)
    assert config == DivergenceConfig(
        haircut=0.7, warn_z=-1.5, halt_z=-2.5, mdd_tolerance=1.1, window_days=10
    )


def test_repo_config_section_loads() -> None:
    # The shipped orchestrator/config.yaml carries the section with defaults.
    assert load_divergence_config() == DivergenceConfig()


@pytest.mark.parametrize(
    "body",
    [
        "divergence:\n  haircut: fast\n",
        "divergence:\n  window_days: 2.5\n",
        "divergence: [1, 2]\n",
        "divergence:\n  warn_z: -3.0\n  halt_z: -2.0\n",  # halt milder than warn
        "divergence:\n  haircut: -0.5\n",
    ],
)
def test_bad_config_raises(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(DivergenceConfigError):
        load_divergence_config(path)


# --- backtest stats -----------------------------------------------------------


def test_load_backtest_stats_reads_ret_and_csv_mdd(tmp_path: Path) -> None:
    _, workspace, _ = promoted_env(tmp_path, mdd=-0.16)
    stats = load_backtest_stats(workspace)
    assert stats.mean == pytest.approx(0.0)
    assert stats.sigma == pytest.approx(fixture_sigma())
    assert stats.mdd == pytest.approx(0.16)  # csv value, magnitude
    assert stats.days == 60


def test_load_backtest_stats_mdd_falls_back_to_ret_curve(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # One -10% day then flat: compounded max drawdown is exactly 10%.
    write_ret(workspace / "ret.pkl", [0.02, -0.10] + [0.0] * 20)
    stats = load_backtest_stats(workspace)
    assert stats.mdd == pytest.approx(0.10)


def test_load_backtest_stats_net_of_cost(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    write_ret(workspace / "ret.pkl", [0.011, -0.009] * 10, cost=0.001)
    stats = load_backtest_stats(workspace)
    assert stats.mean == pytest.approx(0.0)  # net = return - cost


def test_load_backtest_stats_missing_ret_pkl_raises(tmp_path: Path) -> None:
    with pytest.raises(DivergenceError, match="no ret.pkl"):
        load_backtest_stats(tmp_path)


def test_load_backtest_stats_zero_variance_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    write_ret(workspace / "ret.pkl", [0.001] * 30)
    with pytest.raises(DivergenceError, match="zero variance"):
        load_backtest_stats(workspace)


# --- realized-side helpers ----------------------------------------------------


def test_realized_returns_start_strictly_after_promotion() -> None:
    promoted = dt.date(2026, 7, 1)
    history = make_history(promoted, [0.01, 0.02, 0.03])
    as_of = promoted + dt.timedelta(days=2)
    # The promotion-day point (pct None) is excluded; only days 1..2 count.
    assert realized_daily_returns(history, promoted, as_of) == [0.01, 0.02]
    equities = equity_since_promotion(history, promoted, as_of)
    assert len(equities) == 3 and equities[0] == 100_000.0  # baseline included


def test_max_drawdown() -> None:
    assert max_drawdown([]) == 0.0
    assert max_drawdown([100.0, 110.0, 121.0]) == 0.0
    assert max_drawdown([100.0, 80.0, 90.0]) == pytest.approx(0.20)
    assert max_drawdown([100.0, 120.0, 90.0, 130.0]) == pytest.approx(0.25)


def test_choose_period() -> None:
    assert choose_period(0) == "3M"
    assert choose_period(80) == "3M"
    assert choose_period(81) == "1A"
    assert choose_period(350) == "1A"
    assert choose_period(351) == "all"


# --- evaluate_divergence (pure) -----------------------------------------------


STATS = BacktestStats(mean=0.001, sigma=0.01, mdd=0.20, days=60)
CONFIG = DivergenceConfig()  # haircut 0.5, warn -2, halt -3, tolerance 1.25, 20d


def test_warmup_below_window_days() -> None:
    assert evaluate_divergence([0.001] * 19, [100.0] * 19, STATS, CONFIG) is None


def test_z_math_with_haircut() -> None:
    # Realized exactly the haircut expectation: mu_adj = 0.5 * 0.001 = 0.0005,
    # 20 days of +0.0005 sum to the expected 0.01 -> z = 0.
    result = evaluate_divergence([0.0005] * 20, [100.0] * 20, STATS, CONFIG)
    assert result is not None
    assert result.mu_adj == pytest.approx(0.0005)
    assert result.expected == pytest.approx(0.01)
    assert result.z == pytest.approx(0.0, abs=1e-12)
    assert result.status == "ok"
    # Without the haircut the same realized path would sit BELOW expectation:
    # raw expectation is 0.02, z_raw = (0.01 - 0.02) / (0.01 * sqrt(20)).
    no_haircut = DivergenceConfig(haircut=1.0)
    raw = evaluate_divergence([0.0005] * 20, [100.0] * 20, STATS, no_haircut)
    assert raw is not None
    assert raw.z == pytest.approx(-0.01 / (0.01 * math.sqrt(20)))


def test_only_trailing_window_counts() -> None:
    # A catastrophic day 40 days ago must not move the trailing-20 z.
    returns = [-0.5] + [0.0005] * 20
    result = evaluate_divergence(returns, [100.0] * 21, STATS, CONFIG)
    assert result is not None
    assert result.realized == pytest.approx(0.01)
    assert result.z == pytest.approx(0.0, abs=1e-12)


def test_alert_threshold() -> None:
    # 20 days of -0.005: realized -0.10, expected 0.01,
    # z = -0.11 / (0.01 * sqrt(20)) = -2.46 -> warn, not halt.
    result = evaluate_divergence([-0.005] * 20, [100.0, 95.0], STATS, CONFIG)
    assert result is not None
    assert result.z == pytest.approx(-0.11 / (0.01 * math.sqrt(20)))
    assert result.status == "warn"
    assert result.triggers == ()


def test_halt_threshold_via_z() -> None:
    # 20 days of -0.008: z = -0.17 / (0.01 * sqrt(20)) = -3.80 -> halt.
    result = evaluate_divergence([-0.008] * 20, [100.0, 90.0], STATS, CONFIG)
    assert result is not None
    assert result.status == "halt"
    assert len(result.triggers) == 1 and "z-score" in result.triggers[0]


def test_halt_threshold_via_drawdown() -> None:
    # Mild trailing window (z fine) but a 30% dip since promotion:
    # limit = 0.20 * 1.25 = 0.25 -> halt via drawdown.
    result = evaluate_divergence(
        [0.0005] * 20, [100.0, 70.0, 95.0], STATS, CONFIG
    )
    assert result is not None
    assert result.z == pytest.approx(0.0, abs=1e-12)
    assert result.status == "halt"
    assert len(result.triggers) == 1 and "drawdown" in result.triggers[0]
    assert result.drawdown == pytest.approx(0.30)
    assert result.drawdown_limit == pytest.approx(0.25)


def test_drawdown_at_limit_passes() -> None:
    # Exactly AT the limit passes (strictly-over trips) — breaker convention.
    result = evaluate_divergence([0.0005] * 20, [100.0, 75.0], STATS, CONFIG)
    assert result is not None
    assert result.drawdown == pytest.approx(result.drawdown_limit)
    assert result.status == "ok"


# --- halt file ----------------------------------------------------------------


def test_write_halt_file_and_breaker_sees_it(tmp_path: Path) -> None:
    halt_file = tmp_path / "breaker" / "halt"
    assert write_halt_file(halt_file, "divergence auto-halt test")
    breaker = Breaker(
        BreakerConfig(max_daily_notional_usd=1.0, max_drawdown_pct=10.0),
        halt_file=halt_file,
        high_water_mark_file=tmp_path / "breaker" / "hwm.json",
    )
    assert breaker.halted
    assert breaker.halt_note == "divergence auto-halt test"
    trip = breaker.check(equity=100_000.0, day_notional_usd=0.0)
    assert trip is not None and trip.reason is BreakerReason.HALT_FILE


def test_write_halt_file_never_overwrites(tmp_path: Path) -> None:
    halt_file = tmp_path / "halt"
    halt_file.write_text("operator note\n")
    assert not write_halt_file(halt_file, "divergence auto-halt")
    assert halt_file.read_text() == "operator note\n"


# --- run_divergence (runner) --------------------------------------------------


def run_env(
    tmp_path: Path, returns: list[float], equities: list[float] | None = None
) -> tuple[FakeAlpaca, list[str], dict[str, Path], dt.date]:
    db_path, _, promoted_date = promoted_env(tmp_path)
    client = FakeAlpaca(history=make_history(promoted_date, returns, equities=equities))
    paths = {
        "db_path": db_path,
        "config_path": tmp_path / "absent.yaml",  # defaults
        "halt_file": tmp_path / "breaker" / "halt",
    }
    as_of = promoted_date + dt.timedelta(days=len(returns) + 1)
    return client, [], paths, as_of


def test_run_warmup_posts_nothing(tmp_path: Path) -> None:
    client, posts, paths, as_of = run_env(tmp_path, [0.001] * 19)
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert posts == []
    assert not paths["halt_file"].exists()
    assert client.calls == [("3M", "1D")]


def test_run_ok_posts_nothing(tmp_path: Path) -> None:
    client, posts, paths, as_of = run_env(tmp_path, [0.0] * 20)
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert posts == []
    assert not paths["halt_file"].exists()


def test_run_warn_posts_numbers_no_halt(tmp_path: Path) -> None:
    # Backtest mean 0 -> expected 0; sigma ~0.010084; 20 days of -0.005 ->
    # z = -0.10 / (sigma * sqrt(20)) ~ -2.22: warn but not halt.
    client, posts, paths, as_of = run_env(tmp_path, [-0.005] * 20)
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert len(posts) == 1
    assert ":warning:" in posts[0]
    assert "realized -10.00%" in posts[0]
    assert "expected +0.00%" in posts[0]
    assert "drawdown since promotion" in posts[0]
    assert not paths["halt_file"].exists()


def test_run_halt_via_z_writes_halt_file(tmp_path: Path) -> None:
    # 20 days of -0.008 -> z ~ -3.55 < -3: auto-halt.
    client, posts, paths, as_of = run_env(tmp_path, [-0.008] * 20)
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert len(posts) == 1
    assert ":rotating_light:" in posts[0]
    assert "z-score" in posts[0]
    assert "resume_trading" in posts[0]  # manual clear procedure
    assert paths["halt_file"].is_file()
    assert "divergence auto-halt" in paths["halt_file"].read_text()


def test_run_halt_via_drawdown(tmp_path: Path) -> None:
    # Trailing window mild, but a 30% dip since promotion breaches the
    # 0.20 * 1.25 = 25% drawdown limit.
    returns = [-0.30] + [0.0005] * 20
    equities = [70_000.0] + [70_000.0 * (1.0005**i) for i in range(1, 21)]
    client, posts, paths, as_of = run_env(tmp_path, returns, equities=equities)
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert len(posts) == 1
    assert ":rotating_light:" in posts[0]
    assert "drawdown" in posts[0]
    assert paths["halt_file"].is_file()


def test_run_halt_keeps_existing_halt_file(tmp_path: Path) -> None:
    client, posts, paths, as_of = run_env(tmp_path, [-0.008] * 20)
    paths["halt_file"].parent.mkdir(parents=True)
    paths["halt_file"].write_text("operator halt: investigating\n")
    rc = run_divergence(client, posts.append, as_of=as_of, **paths)
    assert rc == 0
    assert paths["halt_file"].read_text() == "operator halt: investigating\n"
    assert len(posts) == 1 and "already present" in posts[0]


def test_run_nothing_promoted_is_silent_skip(tmp_path: Path) -> None:
    posts: list[str] = []
    rc = run_divergence(
        FakeAlpaca(),
        posts.append,
        as_of=dt.date(2026, 8, 15),
        db_path=tmp_path / "absent.sqlite",
        config_path=tmp_path / "absent.yaml",
        halt_file=tmp_path / "halt",
    )
    assert rc == 0
    assert posts == []


def test_run_broker_failure_posts_error(tmp_path: Path) -> None:
    db_path, _, promoted_date = promoted_env(tmp_path)
    posts: list[str] = []
    rc = run_divergence(
        FakeAlpaca(error=AlpacaError("boom")),
        posts.append,
        as_of=promoted_date + dt.timedelta(days=30),
        db_path=db_path,
        config_path=tmp_path / "absent.yaml",
        halt_file=tmp_path / "halt",
    )
    assert rc == 1
    assert len(posts) == 1 and "FAILED" in posts[0] and "boom" in posts[0]
    assert not (tmp_path / "halt").exists()


def test_run_unusable_workspace_posts_error(tmp_path: Path) -> None:
    db_path, workspace, promoted_date = promoted_env(tmp_path)
    (workspace / "ret.pkl").unlink()
    posts: list[str] = []
    rc = run_divergence(
        FakeAlpaca(),
        posts.append,
        as_of=promoted_date + dt.timedelta(days=30),
        db_path=db_path,
        config_path=tmp_path / "absent.yaml",
        halt_file=tmp_path / "halt",
    )
    assert rc == 1
    assert len(posts) == 1 and "FAILED" in posts[0] and "ret.pkl" in posts[0]

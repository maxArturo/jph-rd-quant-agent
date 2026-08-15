"""Live-vs-backtest divergence tracker (US-016).

Runs after the close on trading days (US-017 wires the timer): compares the
paper account's realized returns since promotion against the promoted
strategy's own backtest distribution, and escalates when live performance
drifts below what the backtest — honestly haircut — promised.

The honest expectation: backtests flatter live trading (survivorship gap,
fill assumptions, refresh drift), so the expected daily return is
``mu_adj = haircut * backtest mean`` (haircut default 0.5) rather than the
raw backtest mean. Sigma comes from the FULL test window's net daily
returns (ret.pkl, ``return - cost`` — the same series summary.compute_sharpe
reads). Over the trailing ``window_days`` (default 20) trading days:

    z = (realized_sum - window_days * mu_adj) / (sigma * sqrt(window_days))

Escalation ladder (z thresholds are strict ``<``; the drawdown limit passes
at-limit and trips strictly over, matching the gate/breaker convention):

* ``z < warn_z`` (default -2): Slack warning with realized vs expected and
  the drawdown since promotion.
* ``z < halt_z`` (default -3) OR drawdown since promotion strictly greater
  than ``backtest |MDD| * mdd_tolerance`` (default 1.25): writes the
  rebalancer's breaker halt file (the SAME file execution/breaker.py checks,
  so the next rebalance halts) and posts a :rotating_light: notice naming
  the trigger and the manual clear procedure. An existing halt file is never
  overwritten — the operator's note wins.

Fewer than ``window_days`` realized daily returns since promotion is warmup:
the tracker prints a line and exits 0 without posting anything.

This module never trades and modifies no state other than the halt file (it
does not touch the breaker's high-water mark). Failures post a Slack error
notice and exit 1, same pattern as rebalance/pred_refresh. Thresholds live
in the ``divergence:`` section of orchestrator/config.yaml
(``load_divergence_config``; missing section = defaults).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from execution.alpaca_client import AlpacaClient, AlpacaError, PortfolioHistory
from execution.breaker import DEFAULT_HALT_FILE
from execution.promoted import NoPromotedStrategyError, load_promoted_strategy
from execution.rebalance import MARKET_TZ, Notify, _safe_notify
from orchestrator.state import DEFAULT_DB_PATH

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "orchestrator" / "config.yaml"

# qlib_res.csv key for the backtest max drawdown (summary.METRIC_SPECS "MDD").
_MDD_CSV_KEY = "1day.excess_return_with_cost.max_drawdown"

# Operator clear procedure, quoted in every auto-halt notice.
CLEAR_PROCEDURE = (
    "clear with the resume_trading Slack tool (or delete the halt file) "
    "after reviewing the divergence"
)


class PortfolioReader(Protocol):
    """The one broker read this module needs (satisfied by AlpacaClient)."""

    def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> PortfolioHistory: ...


class DivergenceError(RuntimeError):
    """Any condition that must fail the divergence check (posted to Slack)."""


class DivergenceConfigError(DivergenceError):
    """The divergence: section of config.yaml is malformed."""


@dataclass(frozen=True)
class DivergenceConfig:
    """Thresholds — defaults mirror the divergence: section shipped in
    orchestrator/config.yaml (PRD US-009)."""

    haircut: float = 0.5
    warn_z: float = -2.0
    halt_z: float = -3.0
    mdd_tolerance: float = 1.25
    window_days: int = 20


def load_divergence_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DivergenceConfig:
    """Read the divergence: section; a missing file/section means defaults."""
    import yaml

    loaded: Any = None
    if config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text())
        except Exception as exc:  # noqa: BLE001 — one actionable error type for callers
            raise DivergenceConfigError(f"cannot parse {config_path}: {exc}") from exc
    section = loaded.get("divergence") if isinstance(loaded, dict) else None
    if section is None:
        return DivergenceConfig()
    if not isinstance(section, dict):
        raise DivergenceConfigError(f"divergence section in {config_path} must be a mapping")
    config = DivergenceConfig(
        haircut=_config_float(section, "haircut", DivergenceConfig.haircut),
        warn_z=_config_float(section, "warn_z", DivergenceConfig.warn_z),
        halt_z=_config_float(section, "halt_z", DivergenceConfig.halt_z),
        mdd_tolerance=_config_float(section, "mdd_tolerance", DivergenceConfig.mdd_tolerance),
        window_days=_config_int(section, "window_days", DivergenceConfig.window_days),
    )
    if config.halt_z > config.warn_z:
        raise DivergenceConfigError(
            f"divergence.halt_z ({config.halt_z:g}) must not be above warn_z "
            f"({config.warn_z:g}) — the halt threshold is the more extreme one"
        )
    if config.haircut <= 0 or config.mdd_tolerance <= 0 or config.window_days < 1:
        raise DivergenceConfigError(
            "divergence.haircut and mdd_tolerance must be positive and "
            f"window_days >= 1, got haircut={config.haircut!r} "
            f"mdd_tolerance={config.mdd_tolerance!r} window_days={config.window_days!r}"
        )
    return config


def _config_float(section: Mapping[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DivergenceConfigError(f"divergence.{key} must be a number, got {raw!r}")
    return float(raw)


def _config_int(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DivergenceConfigError(f"divergence.{key} must be an integer, got {raw!r}")
    return raw


# -- backtest distribution -----------------------------------------------------


@dataclass(frozen=True)
class BacktestStats:
    """The promoted strategy's backtest distribution over its full test window.

    ``mean``/``sigma`` are the net-of-cost daily return moments from ret.pkl;
    ``mdd`` is the backtest max drawdown as a POSITIVE magnitude fraction
    (qlib reports it negative).
    """

    mean: float
    sigma: float
    mdd: float
    days: int


def load_backtest_stats(workspace: Path) -> BacktestStats:
    """ret.pkl + qlib_res.csv -> BacktestStats; loud on anything unusable.

    Unlike the gate's degrade-per-field loader, every gap raises
    :class:`DivergenceError`: a tracker running on a partial distribution
    would compute a meaningless z and silently disarm the halt.
    """
    import pandas as pd  # lazy, keeps offline imports fast

    ret_pkl = workspace / "ret.pkl"
    if not ret_pkl.is_file():
        raise DivergenceError(f"promoted workspace has no ret.pkl: {ret_pkl}")
    try:
        frame = pd.read_pickle(ret_pkl)
    except Exception as exc:  # noqa: BLE001 — one actionable error type for callers
        raise DivergenceError(f"cannot read {ret_pkl}: {exc}") from exc
    if not isinstance(frame, pd.DataFrame) or "return" not in frame.columns:
        raise DivergenceError(f"{ret_pkl} is not a report frame with a 'return' column")
    net = frame["return"].astype(float)
    if "cost" in frame.columns:
        net = net - frame["cost"].astype(float)
    net = net.dropna()
    if len(net) < 2:
        raise DivergenceError(f"{ret_pkl} has {len(net)} usable daily returns (need >= 2)")
    sigma = float(net.std())
    # A constant series' std can come back as float residue (~1e-19), not an
    # exact 0 — anything this small is still a degenerate distribution.
    if math.isnan(sigma) or sigma < 1e-12:
        raise DivergenceError(f"{ret_pkl} daily returns have zero variance — cannot compute z")
    return BacktestStats(
        mean=float(net.mean()),
        sigma=sigma,
        mdd=_backtest_mdd(workspace, net),
        days=len(net),
    )


def _backtest_mdd(workspace: Path, net: Any) -> float:
    """Backtest |MDD|: the qlib_res.csv metric, else derived from ret.pkl."""
    csv = workspace / "qlib_res.csv"
    if csv.is_file():
        from orchestrator.summary import SummaryError, load_metrics

        try:
            metrics = load_metrics(csv)
        except SummaryError:
            metrics = {}
        if _MDD_CSV_KEY in metrics:
            return abs(float(metrics[_MDD_CSV_KEY]))
    # Fallback: max drawdown of the compounded net-return curve.
    curve = (1.0 + net).cumprod()
    drawdown = float((curve / curve.cummax() - 1.0).min())
    return abs(drawdown)


# -- realized side -------------------------------------------------------------


def choose_period(days_since_promotion: int) -> str:
    """Smallest Alpaca portfolio-history period safely covering the span."""
    if days_since_promotion <= 80:
        return "3M"
    if days_since_promotion <= 350:
        return "1A"
    return "all"


def realized_daily_returns(
    history: PortfolioHistory, promoted_date: dt.date, as_of: dt.date
) -> list[float]:
    """Daily return fractions attributable to the promoted strategy.

    Each portfolio-history point's profit_loss_pct is the move vs the
    PREVIOUS point, so the first attributable day is the first point strictly
    after the promotion date.
    """
    return [
        float(entry.profit_loss_pct)
        for entry in history.entries
        if promoted_date < entry.date <= as_of and entry.profit_loss_pct is not None
    ]


def equity_since_promotion(
    history: PortfolioHistory, promoted_date: dt.date, as_of: dt.date
) -> list[float]:
    """Equity points since promotion (promotion-day close = drawdown baseline)."""
    return [
        float(entry.equity)
        for entry in history.entries
        if promoted_date <= entry.date <= as_of and entry.equity is not None
    ]


def max_drawdown(equities: Sequence[float]) -> float:
    """Peak-to-trough drawdown magnitude of an equity series (0.0 when < 2 points)."""
    peak = 0.0
    worst = 0.0
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


# -- evaluation (pure) ---------------------------------------------------------


@dataclass(frozen=True)
class DivergenceResult:
    """One day's realized-vs-backtest comparison.

    ``status`` is ``ok`` / ``warn`` / ``halt``; ``triggers`` names every halt
    condition that fired (z, drawdown, or both).
    """

    status: str
    z: float
    realized: float  # trailing window sum of realized daily returns
    expected: float  # window_days * mu_adj
    mu_adj: float
    sigma: float
    window_days: int
    drawdown: float  # magnitude fraction since promotion
    drawdown_limit: float  # backtest |MDD| * mdd_tolerance
    triggers: tuple[str, ...] = ()


def evaluate_divergence(
    daily_returns: Sequence[float],
    equities: Sequence[float],
    stats: BacktestStats,
    config: DivergenceConfig,
) -> DivergenceResult | None:
    """PURE comparison; None = warmup (fewer than window_days realized days)."""
    if len(daily_returns) < config.window_days:
        return None
    window = daily_returns[-config.window_days :]
    realized = float(sum(window))
    mu_adj = config.haircut * stats.mean
    expected = config.window_days * mu_adj
    z = (realized - expected) / (stats.sigma * math.sqrt(config.window_days))
    drawdown = max_drawdown(equities)
    drawdown_limit = stats.mdd * config.mdd_tolerance

    triggers: list[str] = []
    if z < config.halt_z:
        triggers.append(
            f"z-score {z:.2f} below the {config.halt_z:g} halt threshold"
        )
    if drawdown > drawdown_limit:
        triggers.append(
            f"drawdown since promotion {drawdown:.2%} over the "
            f"{drawdown_limit:.2%} limit (backtest MDD {stats.mdd:.2%} × "
            f"{config.mdd_tolerance:g})"
        )
    if triggers:
        status = "halt"
    elif z < config.warn_z:
        status = "warn"
    else:
        status = "ok"
    return DivergenceResult(
        status=status,
        z=z,
        realized=realized,
        expected=expected,
        mu_adj=mu_adj,
        sigma=stats.sigma,
        window_days=config.window_days,
        drawdown=drawdown,
        drawdown_limit=drawdown_limit,
        triggers=tuple(triggers),
    )


def _numbers_block(result: DivergenceResult, promoted_at: str) -> str:
    return (
        f"• trailing {result.window_days} trading days realized "
        f"{result.realized:+.2%} vs expected {result.expected:+.2%} "
        f"(haircut μ_adj {result.mu_adj:+.4%}/day) — z = {result.z:.2f}"
        f"\n• drawdown since promotion ({promoted_at[:10]}): "
        f"{result.drawdown:.2%} (halt limit {result.drawdown_limit:.2%})"
    )


# -- halt file (the breaker's kill switch) -------------------------------------


def write_halt_file(halt_file: Path, note: str) -> bool:
    """Write the breaker halt file; False when one already exists.

    Never overwrites: an existing halt means trading is already stopped and
    the file's note (possibly an operator's) must not be clobbered.
    """
    if halt_file.exists():
        return False
    halt_file.parent.mkdir(parents=True, exist_ok=True)
    halt_file.write_text(note.strip() + "\n")
    return True


# -- runner --------------------------------------------------------------------


def run_divergence(
    client: PortfolioReader,
    notify: Notify,
    as_of: dt.date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    halt_file: Path = DEFAULT_HALT_FILE,
) -> int:
    """Run the daily divergence check; returns the process exit code.

    0 = checked (ok/warn/halt posted as appropriate), warmup, or nothing
    promoted (both silent skips); 1 = the check itself failed (posted to
    Slack — a broken tracker must not be a silent one).
    """
    if as_of is None:
        as_of = dt.datetime.now(MARKET_TZ).date()
    try:
        try:
            promoted = load_promoted_strategy(db_path)
        except NoPromotedStrategyError as exc:
            print(f"divergence check skipped ({as_of}): {exc}")
            return 0
        workspace = Path(promoted.workspace_path).expanduser()
        try:
            promoted_date = dt.date.fromisoformat(promoted.promoted_at[:10])
        except ValueError as exc:
            raise DivergenceError(
                f"cannot parse promoted_at {promoted.promoted_at!r} as a date"
            ) from exc

        config = load_divergence_config(config_path)
        stats = load_backtest_stats(workspace)
        history = client.get_portfolio_history(
            period=choose_period((as_of - promoted_date).days), timeframe="1D"
        )
        daily_returns = realized_daily_returns(history, promoted_date, as_of)
        equities = equity_since_promotion(history, promoted_date, as_of)
        result = evaluate_divergence(daily_returns, equities, stats, config)
        if result is None:
            print(
                f"divergence check ({as_of}): warmup — {len(daily_returns)} realized "
                f"trading day(s) since promotion ({promoted_date}), need "
                f"{config.window_days}"
            )
            return 0

        tag = workspace.name[:8]
        if result.status == "halt":
            trigger_text = "; ".join(result.triggers)
            note = f"divergence auto-halt {as_of} ({tag}): {trigger_text}"
            written = write_halt_file(halt_file, note)
            halt_line = (
                f"halt file written: {halt_file}"
                if written
                else f"halt file already present at {halt_file} — existing note kept"
            )
            message = (
                f":rotating_light: divergence AUTO-HALT ({as_of}): live performance of "
                f"`{tag}` breached the kill threshold — {trigger_text}"
                f"\n{_numbers_block(result, promoted.promoted_at)}"
                f"\n• {halt_line}; the next rebalance will refuse to trade"
                f"\n• manual clear: {CLEAR_PROCEDURE}"
            )
            _safe_notify(notify, message)
            print(message)
            return 0
        if result.status == "warn":
            message = (
                f":warning: divergence warning ({as_of}): live performance of `{tag}` "
                f"is drifting below the haircut backtest expectation "
                f"(z = {result.z:.2f} < {config.warn_z:g})"
                f"\n{_numbers_block(result, promoted.promoted_at)}"
            )
            _safe_notify(notify, message)
            print(message)
            return 0
        print(
            f"divergence check ({as_of}): ok — z = {result.z:.2f}, "
            f"drawdown {result.drawdown:.2%} (limit {result.drawdown_limit:.2%})"
        )
        return 0
    except (DivergenceError, AlpacaError) as exc:
        message = f"divergence check FAILED ({as_of}): {exc}"
        _safe_notify(notify, message)
        print(message, file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected bug: tell the operator, then crash loudly
        _safe_notify(notify, f"divergence check CRASHED ({as_of}): {exc!r}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Daily live-vs-backtest divergence check for the promoted strategy "
            "(US-016): warns on drift, writes the breaker halt file on severe breach"
        )
    )
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="YYYY-MM-DD (default: today in America/New_York)",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--halt-file", type=Path, default=DEFAULT_HALT_FILE)
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="print notices to stderr instead of Slack (supervised local runs)",
    )
    args = parser.parse_args(argv)

    from execution.rebalance import slack_notifier, stderr_notifier
    from orchestrator.config import ConfigError

    if args.no_slack:
        notify = stderr_notifier()
    else:
        try:
            notify = slack_notifier()
        except ConfigError as exc:
            print(
                f"ERROR: {exc}\nRefusing to run unattended without a Slack channel for "
                "divergence notices; pass --no-slack for a supervised local run.",
                file=sys.stderr,
            )
            return 1

    return run_divergence(
        AlpacaClient(),
        notify,
        as_of=args.as_of,
        db_path=args.db_path,
        config_path=args.config,
        halt_file=args.halt_file,
    )


if __name__ == "__main__":
    sys.exit(main())

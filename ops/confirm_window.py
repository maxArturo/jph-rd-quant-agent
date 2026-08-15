"""Confirmation-window re-prediction and portfolio daily returns (US-009).

The promotion gate's confirmation criterion (US-010) needs each strategy's
portfolio daily returns over the reserved confirmation window — the slice
(TEST_END, store end] the hypothesis search never saw (US-008). This module
produces them for ANY workspace that carries pred-refresh snapshot files
(execution/pred_refresh.py): the fetched candidate and the promoted incumbent
alike.

Flow (``confirmation_returns``):

1. Require the workspace's pred-refresh snapshot (conf_pred_refresh.yaml,
   pred_refresh.env, pred_refresh_params.pkl). Missing files raise
   ``ConfirmWindowError`` — the caller decides whether to snapshot first
   (``execution.pred_refresh.snapshot_pred_refresh``); this module never
   writes snapshot files itself.
2. Re-predict from the snapshotted EXACT weights (the US-049 docker
   machinery, ``test_end`` overridden to the window end). Skipped when the
   workspace's newest pred.pkl already covers every needed signal day:
   inference from fixed weights is deterministic, so a covering pred is the
   same answer — the promoted incumbent's daily refresh usually makes its
   docker run unnecessary, while a fresh candidate (pred ends at TEST_END)
   always needs one.
3. Simulate the shared TopkDropoutStrategy over the window's trading days
   with the workspace's own topk/n_drop and cost params (execution.signal
   loaders — the gate's parity layer guarantees both sides carry equal
   values).

Simulation semantics (both sides of a comparison run through the SAME code,
which is what the confirmation criterion needs):

* Day d's return is earned by the book selected from predictions dated the
  previous trading day (pred made FROM d-1 FOR d — the live rebalancer's
  timing; qlib's own backtest lags one extra day by executing at d's close).
* Returns are close-to-close on the store's ADJUSTED closes (split/dividend
  correct — no factor division here; that identity recovers RAW prices, see
  execution.rebalance.latest_store_price).
* Books are equal-weight; costs charge open_cost on the bought fraction of
  the new book and close_cost on the sold fraction of the old book.
  min_cost is an absolute-dollar knob and is ignored (there is no notional
  here); it is identical on both sides, so the comparison is unaffected.

Failure policy: every gap raises ``ConfirmWindowError`` (missing snapshot,
window outside the store calendar, docker failure, pred not covering the
window, missing prices). Never a silent None — US-010 maps the error to
``confirmation_unavailable`` and blocks promotion.
"""

from __future__ import annotations

import datetime as dt
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from execution import pred_refresh, signal
from execution.pred_refresh import PredRefreshError, Runner, _docker_runner
from execution.rebalance import DEFAULT_STORE_PATH

SNAPSHOT_FILES = (
    pred_refresh.SNAPSHOT_CONF_NAME,
    pred_refresh.SNAPSHOT_ENV_NAME,
    pred_refresh.SNAPSHOT_PARAMS_NAME,
)


class ConfirmWindowError(RuntimeError):
    """Any condition that must fail confirmation evaluation (never silent)."""


@dataclass(frozen=True)
class WindowReturns:
    """Portfolio daily returns over the trading days actually evaluated.

    ``daily_returns`` is net of costs (the confirmation criterion's input);
    ``gross_returns`` is kept alongside for operator-facing breakdowns.
    """

    workspace: str
    window: tuple[str, str]  # first/last trading day evaluated, ISO, inclusive
    daily_returns: tuple[float, ...]
    gross_returns: tuple[float, ...]
    repredicted: bool  # whether a docker re-predict was needed


def annualized_ir(returns: Sequence[float]) -> float | None:
    """mean/std(ddof=1) × sqrt(252) of daily returns, or None when degenerate.

    Benchmark-free, matching orchestrator.summary.compute_sharpe — the
    confirmation comparison only needs both sides computed identically.
    """
    from orchestrator.summary import TRADING_DAYS_PER_YEAR

    values = [float(value) for value in returns]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0 or math.isnan(std):
        return None
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def confirmation_returns(
    workspace: Path,
    window_start: dt.date,
    window_end: dt.date,
    *,
    params: signal.StrategyParams | None = None,
    cost_params: Mapping[str, float] | None = None,
    store_path: Path = DEFAULT_STORE_PATH,
    qlib_dir: Path = pred_refresh.DEFAULT_QLIB_DIR,
    image: str = pred_refresh.DEFAULT_IMAGE,
    timeout_minutes: float = pred_refresh.DEFAULT_TIMEOUT_MINUTES,
    runner: Runner = _docker_runner,
    force_repredict: bool = False,
) -> WindowReturns:
    """Exact-weights portfolio daily returns for [window_start, window_end].

    The window bounds are calendar dates (inclusive); evaluation covers the
    store trading days inside them. params/cost_params default to the
    workspace's own conf values.
    """
    ws = Path(workspace).expanduser()
    missing = [name for name in SNAPSHOT_FILES if not (ws / name).is_file()]
    if missing:
        raise ConfirmWindowError(
            f"workspace {ws} lacks pred-refresh snapshot file(s) {', '.join(missing)} — "
            "run execution.pred_refresh.snapshot_pred_refresh(workspace) first"
        )
    if window_start > window_end:
        raise ConfirmWindowError(f"window start {window_start} is after window end {window_end}")

    calendar = _calendar(store_path)
    if window_start < calendar[0] or window_end > calendar[-1]:
        raise ConfirmWindowError(
            f"window {window_start} → {window_end} outside the store calendar range "
            f"{calendar[0]} → {calendar[-1]} — refresh the store or shrink the window"
        )
    window_days = [day for day in calendar if window_start <= day <= window_end]
    if not window_days:
        raise ConfirmWindowError(f"no trading days in window {window_start} → {window_end}")
    first_idx = calendar.index(window_days[0])
    if first_idx == 0:
        raise ConfirmWindowError(
            f"window starts at the first store calendar day {window_days[0]} — no prior "
            "trading day to take signals from"
        )
    # window_days are consecutive calendar entries, so each day's signal day
    # (the previous trading day) is the calendar entry one slot earlier.
    signal_days = calendar[first_idx - 1 : first_idx - 1 + len(window_days)]

    try:
        if params is None:
            params = signal.load_strategy_params(ws)
        if cost_params is None:
            cost_params = signal.load_cost_params(ws)
    except signal.SignalError as exc:
        raise ConfirmWindowError(f"cannot load strategy/cost params from {ws}: {exc}") from exc

    needed = set(signal_days)
    pred = _try_load_pred(ws)
    repredicted = False
    if force_repredict or pred is None or not needed <= pred.dates:
        _run_repredict(
            ws,
            window_end,
            qlib_dir=qlib_dir,
            image=image,
            timeout_minutes=timeout_minutes,
            runner=runner,
        )
        repredicted = True
        pred = _try_load_pred(ws)
        if pred is None:
            raise ConfirmWindowError(
                f"re-predict finished but no readable pred.pkl under {ws}/mlruns"
            )
        still_missing = sorted(needed - pred.dates)
        if still_missing:
            raise ConfirmWindowError(
                "re-predict finished but pred.pkl still lacks cross-sections for "
                f"{', '.join(day.isoformat() for day in still_missing)} — the snapshot "
                "conf's test segment may not reach the confirmation window"
            )

    open_cost = float(cost_params.get("open_cost", 0.0))
    close_cost = float(cost_params.get("close_cost", 0.0))
    closes: dict[str, dict[dt.date, float]] = {}

    def close_for(symbol: str, day: dt.date) -> float:
        if symbol not in closes:
            closes[symbol] = _close_series(store_path, symbol, calendar)
        price = closes[symbol].get(day)
        if price is None:
            raise ConfirmWindowError(
                f"no store close for {symbol} on {day} "
                f"(features/{symbol.lower()}/close.day.bin) — cannot price the "
                "confirmation book"
            )
        return price

    book: list[str] = []
    gross: list[float] = []
    net: list[float] = []
    for day, sig_day in zip(window_days, signal_days, strict=True):
        cross = _cross_section(pred.series, pred.level, sig_day)
        if cross is None:
            raise ConfirmWindowError(
                f"pred.pkl has no usable cross-section for signal day {sig_day}"
            )
        try:
            new_book = signal.topk_dropout_holdings(cross, book, params)
        except signal.SignalError as exc:
            raise ConfirmWindowError(
                f"selection failed for {day} (signals {sig_day}): {exc}"
            ) from exc
        if not new_book:
            raise ConfirmWindowError(f"selection produced an empty book for {day}")
        day_return = sum(
            close_for(symbol, day) / close_for(symbol, sig_day) - 1.0 for symbol in new_book
        ) / len(new_book)
        bought = len(set(new_book) - set(book))
        sold = len(set(book) - set(new_book))
        cost = open_cost * bought / len(new_book)
        if book:
            cost += close_cost * sold / len(book)
        gross.append(day_return)
        net.append(day_return - cost)
        book = new_book

    return WindowReturns(
        workspace=str(ws),
        window=(window_days[0].isoformat(), window_days[-1].isoformat()),
        daily_returns=tuple(net),
        gross_returns=tuple(gross),
        repredicted=repredicted,
    )


# -- internals -----------------------------------------------------------------


def _calendar(store_path: Path) -> list[dt.date]:
    calendar_path = store_path.expanduser() / "calendars" / "day.txt"
    try:
        return sorted(signal._read_calendar(calendar_path))
    except signal.SignalError as exc:
        raise ConfirmWindowError(str(exc)) from exc


@dataclass(frozen=True)
class _PredSeries:
    """The newest pred.pkl, loaded whole (every cross-section, not just the
    latest — signal.load_latest_cross_section only serves the rebalancer)."""

    series: Any  # pd.Series, MultiIndex (datetime, instrument)
    level: int  # datetime level position
    dates: frozenset[dt.date]


def _try_load_pred(ws: Path) -> _PredSeries | None:
    """Newest pred.pkl as a _PredSeries; None on ANY problem.

    Pre-repredict a None just means "run the re-predict"; post-repredict the
    caller turns it into a ConfirmWindowError.
    """
    import pandas as pd

    try:
        obj = pd.read_pickle(signal.locate_pred(ws))
    except Exception:  # noqa: BLE001 — unusable pred means "re-predict", not crash
        return None
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return None
        series = obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        return None
    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2 or len(series) == 0:
        return None
    names = list(index.names)
    level = names.index("datetime") if "datetime" in names else 0
    # cast: pandas stubs type Timestamp(...) as possibly-NaT; the isna guard
    # already excludes that.
    dates = frozenset(
        cast(dt.date, pd.Timestamp(value).date())
        for value in index.get_level_values(level).unique()
        if not pd.isna(value)
    )
    return _PredSeries(series=series, level=level, dates=dates)


def _cross_section(series: Any, level: int, day: dt.date) -> Any | None:
    """The day's cross-section with NaN scores dropped; None when absent/empty."""
    import pandas as pd

    try:
        cross = series.xs(pd.Timestamp(day), level=level)
    except KeyError:
        return None
    cross = cross.dropna()
    if len(cross) == 0:
        return None
    cross.index = cross.index.map(str)
    return cross


def _run_repredict(
    ws: Path,
    test_end: dt.date,
    *,
    qlib_dir: Path,
    image: str,
    timeout_minutes: float,
    runner: Runner,
) -> None:
    """One exact-weights docker re-predict with test_end = the window end.

    Byte-for-byte the daily refresh's mechanism (env snapshot + container env
    + fresh script copy + docker_command) — only the test_end value and the
    log/container names differ.
    """
    try:
        env = pred_refresh.load_env_file(ws / pred_refresh.SNAPSHOT_ENV_NAME)
    except PredRefreshError as exc:
        raise ConfirmWindowError(str(exc)) from exc
    env.update(pred_refresh._CONTAINER_ENV)
    env["test_end"] = test_end.isoformat()
    shutil.copyfile(
        Path(pred_refresh.__file__).with_name(pred_refresh.PREDICT_SCRIPT_NAME),
        ws / pred_refresh.PREDICT_SCRIPT_NAME,
    )
    log_path = ws / "logs" / f"confirm_repredict_{test_end:%Y%m%d}.log"
    log_path.parent.mkdir(exist_ok=True)
    command = pred_refresh.docker_command(
        ws, env, image=image, qlib_dir=qlib_dir, name=f"rdq-confirm-{ws.name[:12]}-{test_end}"
    )
    try:
        returncode = runner(command, log_path, timeout_minutes * 60)
    except PredRefreshError as exc:
        raise ConfirmWindowError(str(exc)) from exc
    if returncode != 0:
        raise ConfirmWindowError(
            f"confirmation re-predict for {ws.name} exited {returncode} — see {log_path}"
        )


def _close_series(
    store_path: Path, symbol: str, calendar: Sequence[dt.date]
) -> dict[dt.date, float]:
    """Adjusted close per trading day from the store bins.

    Element 0 of a .bin is the calendar-index header (offset of the first
    value). ADJUSTED closes on purpose: daily returns must be split/dividend
    correct, so no division by factor here.
    """
    import numpy as np

    path = store_path.expanduser() / "features" / symbol.lower() / "close.day.bin"
    if not path.is_file():
        return {}
    values = np.fromfile(path, dtype="<f")
    if len(values) < 2:
        return {}
    offset = int(values[0])
    series: dict[dt.date, float] = {}
    for i, raw in enumerate(values[1:]):
        index = offset + i
        value = float(raw)
        if 0 <= index < len(calendar) and math.isfinite(value) and value > 0:
            series[calendar[index]] = value
    return series

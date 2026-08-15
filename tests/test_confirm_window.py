"""Tests for ops/confirm_window.py (US-009): confirmation-window returns."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from execution.pred_refresh import PredRefreshError
from ops.confirm_window import (
    ConfirmWindowError,
    annualized_ir,
    confirmation_returns,
)
from tests.test_rebalance import write_bins
from tests.test_signal import write_calendar, write_conf, write_pred

CAL = ["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08"]
D = [dt.date.fromisoformat(day) for day in CAL]

COSTS = {"open_cost": 0.001, "close_cost": 0.002, "min_cost": 5}

# topk=2/n_drop=1 walk over the window D[2]..D[4]:
#   D[2]: signals D[1] -> book [AAPL, MSFT] from empty (full-book buy cost)
#   D[3]: signals D[2] -> unchanged (no trades)
#   D[4]: signals D[3] -> NVDA displaces MSFT (one buy, one sell)
PRED = {
    "2026-07-02": {"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.1},
    "2026-07-06": {"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.1},
    "2026-07-07": {"NVDA": 0.95, "AAPL": 0.9, "MSFT": 0.1},
}

# Hand-computed from the closes in make_store():
GROSS = (
    0.02,  # mean(AAPL 100->102, MSFT 50->51)
    (104.0 / 102.0 - 1.0) / 2,  # mean(AAPL 102->104, MSFT flat)
    0.05,  # mean(AAPL flat, NVDA 10->11)
)
NET = (
    GROSS[0] - 0.001,  # open_cost on the whole starting book
    GROSS[1],  # no trades
    GROSS[2] - (0.001 * 0.5 + 0.002 * 0.5),  # buy NVDA, sell MSFT (half book each)
)


def make_store(tmp_path: Path) -> Path:
    store = tmp_path / "us_data"
    write_calendar(store / "calendars" / "day.txt", CAL)
    write_bins(store, "AAPL", [100.0, 100.0, 102.0, 104.0, 104.0], [1.0] * 5)
    write_bins(store, "MSFT", [50.0, 50.0, 51.0, 51.0, 51.0], [1.0] * 5)
    write_bins(store, "NVDA", [10.0, 10.0, 10.0, 10.0, 11.0], [1.0] * 5)
    return store


def make_workspace(
    tmp_path: Path,
    pred: dict[str, dict[str, float]] | None = PRED,
    snapshot: bool = True,
) -> Path:
    ws = tmp_path / "workspace"
    write_conf(ws, "conf.yaml", topk=2, n_drop=1, costs=COSTS)
    if snapshot:
        (ws / "conf_pred_refresh.yaml").write_text("record:\n    - class: SignalRecord\n")
        (ws / "pred_refresh.env").write_text("train_start=2020-01-01\ntest_end=2026-07-02\n")
        (ws / "pred_refresh_params.pkl").write_bytes(b"fake-weights")
    if pred is not None:
        write_pred(ws, pred, mtime=1.0)
    return ws


def refuse_runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
    raise AssertionError("re-predict must not run when pred.pkl already covers the window")


# --- successful window returns --------------------------------------------------


def test_window_returns_without_repredict(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    result = confirmation_returns(ws, D[2], D[4], store_path=store, runner=refuse_runner)
    assert result.workspace == str(ws)
    assert result.window == ("2026-07-06", "2026-07-08")
    assert result.repredicted is False
    assert result.gross_returns == pytest.approx(GROSS)
    assert result.daily_returns == pytest.approx(NET)


def test_window_bounds_trim_to_trading_days(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    # 07-04/07-05 is a weekend; bounds need not be trading days themselves.
    result = confirmation_returns(
        ws, dt.date(2026, 7, 5), D[4], store_path=store, runner=refuse_runner
    )
    assert result.window == ("2026-07-06", "2026-07-08")
    assert result.daily_returns == pytest.approx(NET)


def test_cost_params_override(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    zero = {"open_cost": 0.0, "close_cost": 0.0, "min_cost": 0.0}
    result = confirmation_returns(
        ws, D[2], D[4], cost_params=zero, store_path=store, runner=refuse_runner
    )
    assert result.daily_returns == pytest.approx(result.gross_returns)


# --- the re-predict path ---------------------------------------------------------


def test_repredict_runs_when_pred_stops_at_test_end(tmp_path: Path) -> None:
    """A candidate's pred ends at TEST_END; the helper re-predicts to cover."""
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path, pred={"2026-07-02": PRED["2026-07-02"]})
    commands: list[list[str]] = []

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        commands.append(list(command))
        log_path.write_text("re-predict ok\n")
        write_pred(ws, PRED, run="confirm")
        return 0

    result = confirmation_returns(ws, D[2], D[4], store_path=store, runner=runner)
    assert result.repredicted is True
    assert result.daily_returns == pytest.approx(NET)
    (command,) = commands
    # test_end overridden to the window end; snapshot env still aboard.
    assert "test_end=2026-07-08" in command
    assert "train_start=2020-01-01" in command
    assert f"{ws}:/workspace/qlib_workspace" in command
    # The predict script deploys fresh from the repo copy.
    assert (ws / "pred_refresh_predict.py").is_file()
    assert (ws / "logs" / "confirm_repredict_20260708.log").is_file()


def test_force_repredict_runs_even_when_covered(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    calls: list[int] = []

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        calls.append(1)
        write_pred(ws, PRED, run="confirm")
        return 0

    result = confirmation_returns(
        ws, D[2], D[4], store_path=store, runner=runner, force_repredict=True
    )
    assert result.repredicted is True
    assert len(calls) == 1
    assert result.daily_returns == pytest.approx(NET)


def test_repredict_nonzero_exit_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path, pred=None)

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        return 3

    with pytest.raises(ConfirmWindowError, match="exited 3"):
        confirmation_returns(ws, D[2], D[4], store_path=store, runner=runner)


def test_repredict_timeout_wraps_to_typed_error(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path, pred=None)

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        raise PredRefreshError("docker re-predict exceeded 40 min")

    with pytest.raises(ConfirmWindowError, match="exceeded 40 min"):
        confirmation_returns(ws, D[2], D[4], store_path=store, runner=runner)


def test_repredict_that_still_misses_days_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path, pred=None)

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        write_pred(ws, {"2026-07-02": PRED["2026-07-02"]}, run="confirm")
        return 0

    with pytest.raises(ConfirmWindowError, match="still lacks cross-sections for 2026-07-06"):
        confirmation_returns(ws, D[2], D[4], store_path=store, runner=runner)


# --- typed errors ----------------------------------------------------------------


def test_missing_snapshot_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path, snapshot=False)
    with pytest.raises(ConfirmWindowError, match="snapshot"):
        confirmation_returns(ws, D[2], D[4], store_path=store, runner=refuse_runner)


def test_window_beyond_store_end_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    with pytest.raises(ConfirmWindowError, match="outside the store calendar range"):
        confirmation_returns(
            ws, D[2], dt.date(2026, 8, 1), store_path=store, runner=refuse_runner
        )


def test_window_before_store_start_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    with pytest.raises(ConfirmWindowError, match="outside the store calendar range"):
        confirmation_returns(
            ws, dt.date(2026, 6, 1), D[4], store_path=store, runner=refuse_runner
        )


def test_inverted_window_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    with pytest.raises(ConfirmWindowError, match="after window end"):
        confirmation_returns(ws, D[4], D[2], store_path=store, runner=refuse_runner)


def test_window_with_no_trading_days_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    weekend = dt.date(2026, 7, 4)
    with pytest.raises(ConfirmWindowError, match="no trading days"):
        confirmation_returns(ws, weekend, weekend, store_path=store, runner=refuse_runner)


def test_window_starting_at_calendar_start_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ws = make_workspace(tmp_path)
    with pytest.raises(ConfirmWindowError, match="no prior"):
        confirmation_returns(ws, D[0], D[2], store_path=store, runner=refuse_runner)


def test_missing_price_raises(tmp_path: Path) -> None:
    store = tmp_path / "us_data"
    write_calendar(store / "calendars" / "day.txt", CAL)
    write_bins(store, "AAPL", [100.0, 100.0, 102.0, 104.0, 104.0], [1.0] * 5)
    write_bins(store, "MSFT", [50.0, 50.0, 51.0, 51.0, 51.0], [1.0] * 5)
    write_bins(store, "NVDA", [10.0, 10.0, 10.0, 10.0], [1.0] * 4)  # no close on D[4]
    ws = make_workspace(tmp_path)
    with pytest.raises(ConfirmWindowError, match="no store close for NVDA on 2026-07-08"):
        confirmation_returns(ws, D[2], D[4], store_path=store, runner=refuse_runner)


# --- annualized_ir ---------------------------------------------------------------


def test_annualized_ir_matches_hand_math() -> None:
    import statistics

    returns = [0.01, -0.02, 0.03, 0.005]
    expected = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
    assert annualized_ir(returns) == pytest.approx(expected)


def test_annualized_ir_degenerate_inputs() -> None:
    assert annualized_ir([]) is None
    assert annualized_ir([0.01]) is None
    assert annualized_ir([0.01, 0.01, 0.01]) is None  # zero variance

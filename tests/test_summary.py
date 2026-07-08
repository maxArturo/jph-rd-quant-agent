"""US-022: metrics summary formatter + equity-curve chart rendering.

Fixtures mirror the real artifact shapes: qlib_res.csv is a pandas Series
written by read_exp_res.py (metric name -> value, qlib's real recorder keys),
ret.pkl is qlib's report_normal_1day.pkl portfolio DataFrame. The fixture
builders are reused by tests/test_poller.py's completion tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orchestrator.summary import (
    TRADING_DAYS_PER_YEAR,
    SummaryError,
    compute_sharpe,
    format_summary,
    load_metrics,
    render_equity_curve,
)

# qlib's real metric keys (SigAnaRecord + risk_analysis of excess returns).
FIXTURE_METRICS: dict[str, float] = {
    "IC": 0.0432,
    "ICIR": 0.3121,
    "Rank IC": 0.0401,
    "Rank ICIR": 0.2933,
    "1day.excess_return_with_cost.mean": 0.0005,
    "1day.excess_return_with_cost.std": 0.0080,
    "1day.excess_return_with_cost.annualized_return": 0.1234,
    "1day.excess_return_with_cost.information_ratio": 1.0200,
    "1day.excess_return_with_cost.max_drawdown": -0.0840,
    "1day.excess_return_without_cost.annualized_return": 0.1500,
}


def write_qlib_res_csv(path: Path, metrics: dict[str, float] | None = None) -> Path:
    """Write a fixture qlib_res.csv exactly like read_exp_res.py does."""
    pd.Series(metrics if metrics is not None else FIXTURE_METRICS).to_csv(path)
    return path


def write_ret_pkl(path: Path, *, days: int = 60, with_bench: bool = True) -> Path:
    """Write a fixture ret.pkl shaped like qlib's report_normal_1day.pkl."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2025-01-02", periods=days)
    returns = rng.normal(0.001, 0.01, days)
    frame = pd.DataFrame(
        {
            "account": 1_000_000 * (1 + pd.Series(returns, index=index)).cumprod(),
            "return": returns,
            "total_turnover": rng.uniform(0, 5e5, days),
            "turnover": rng.uniform(0, 0.3, days),
            "cost": np.full(days, 0.0002),
            "total_cost": rng.uniform(0, 200, days),
            "value": rng.uniform(9e5, 1.1e6, days),
            "cash": rng.uniform(0, 1e5, days),
        },
        index=index,
    )
    if with_bench:
        frame["bench"] = rng.normal(0.0005, 0.008, days)
    frame.to_pickle(path)
    return path


# --- load_metrics -------------------------------------------------------------


def test_load_metrics_parses_fixture_csv(tmp_path: Path) -> None:
    csv = write_qlib_res_csv(tmp_path / "qlib_res.csv")
    metrics = load_metrics(csv)
    assert metrics["IC"] == pytest.approx(0.0432)
    assert metrics["1day.excess_return_with_cost.max_drawdown"] == pytest.approx(-0.084)
    assert len(metrics) == len(FIXTURE_METRICS)


def test_load_metrics_drops_nan_and_non_numeric(tmp_path: Path) -> None:
    csv = tmp_path / "qlib_res.csv"
    csv.write_text(",0\nIC,0.05\nbroken,not-a-number\nICIR,\n")
    metrics = load_metrics(csv)
    assert metrics == {"IC": pytest.approx(0.05)}


def test_load_metrics_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SummaryError, match="cannot read metrics csv"):
        load_metrics(tmp_path / "absent.csv")


# --- format_summary -----------------------------------------------------------


def test_summary_contains_all_seven_metrics_formatted(tmp_path: Path) -> None:
    metrics = load_metrics(write_qlib_res_csv(tmp_path / "qlib_res.csv"))
    text = format_summary(metrics, sharpe=1.15, workspace_path="/tmp/ws")
    for label in ("IC", "ICIR", "Rank IC", "ARR", "IR", "MDD", "Sharpe"):
        assert f"*{label}:*" in text
    assert "*IC:* 0.0432" in text
    assert "*ICIR:* 0.3121" in text
    assert "*Rank IC:* 0.0401" in text
    assert "*ARR:* +12.34%" in text
    assert "*IR:* 1.0200" in text
    assert "*MDD:* -8.40%" in text
    assert "*Sharpe:* 1.1500" in text
    assert "`/tmp/ws`" in text


def test_summary_reports_missing_metrics_as_na() -> None:
    text = format_summary({"IC": 0.01}, sharpe=None)
    assert "*IC:* 0.0100" in text
    for label in ("ICIR", "Rank IC", "ARR", "IR", "MDD", "Sharpe"):
        assert f"*{label}:* n/a" in text


def test_summary_csv_sharpe_key_wins_over_fallback() -> None:
    text = format_summary({"1day.excess_return_with_cost.sharpe": 2.5}, sharpe=1.0)
    assert "*Sharpe:* 2.5000" in text


# --- compute_sharpe -----------------------------------------------------------


def test_compute_sharpe_matches_hand_calculation(tmp_path: Path) -> None:
    pkl = write_ret_pkl(tmp_path / "ret.pkl")
    frame = pd.read_pickle(pkl)
    net = frame["return"] - frame["cost"]
    expected = net.mean() / net.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert compute_sharpe(pkl) == pytest.approx(expected)


def test_compute_sharpe_treats_missing_cost_as_zero(tmp_path: Path) -> None:
    index = pd.bdate_range("2025-01-02", periods=4)
    frame = pd.DataFrame({"return": [0.01, 0.02, -0.01, 0.005]}, index=index)
    frame.to_pickle(tmp_path / "ret.pkl")
    ret = frame["return"]
    expected = ret.mean() / ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert compute_sharpe(tmp_path / "ret.pkl") == pytest.approx(expected)


def test_compute_sharpe_unusable_frames_return_none(tmp_path: Path) -> None:
    index = pd.bdate_range("2025-01-02", periods=3)
    cases = {
        "no_return.pkl": pd.DataFrame({"bench": [0.1, 0.2, 0.3]}, index=index),
        "one_row.pkl": pd.DataFrame({"return": [0.01]}, index=index[:1]),
        "zero_var.pkl": pd.DataFrame({"return": [0.01, 0.01, 0.01]}, index=index),
    }
    for name, frame in cases.items():
        frame.to_pickle(tmp_path / name)
        assert compute_sharpe(tmp_path / name) is None, name


def test_compute_sharpe_corrupt_pickle_raises(tmp_path: Path) -> None:
    bad = tmp_path / "ret.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(SummaryError, match="cannot read ret.pkl"):
        compute_sharpe(bad)


# --- render_equity_curve --------------------------------------------------------


def test_render_produces_non_empty_png_file(tmp_path: Path) -> None:
    png = render_equity_curve(write_ret_pkl(tmp_path / "ret.pkl"))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "equity_curve.png"
    out.write_bytes(png)
    assert out.stat().st_size > 1000  # a real chart, not a stub image


def test_render_without_bench_column_still_renders(tmp_path: Path) -> None:
    pkl = write_ret_pkl(tmp_path / "ret.pkl", with_bench=False)
    png = render_equity_curve(pkl, title="no bench")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_rejects_non_portfolio_pickles(tmp_path: Path) -> None:
    pd.Series([1, 2, 3]).to_pickle(tmp_path / "series.pkl")
    with pytest.raises(SummaryError, match="'return' column"):
        render_equity_curve(tmp_path / "series.pkl")

    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"garbage")
    with pytest.raises(SummaryError, match="cannot read ret.pkl"):
        render_equity_curve(corrupt)

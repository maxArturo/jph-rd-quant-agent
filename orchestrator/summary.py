"""Loop-completion summary: backtest metrics + equity-curve chart (US-022).

Parses a finished workspace's ``qlib_res.csv`` (a pandas Series written by the
template's read_exp_res.py: metric name -> value) into the operator-facing
metric set and renders ``ret.pkl`` (qlib's ``report_normal_1day.pkl``
portfolio DataFrame: columns account/return/turnover/cost/bench/... indexed by
trading day) into a PNG equity-curve chart.

Metric-name mapping (qlib's real recorder keys, from SigAnaRecord and
risk_analysis at the pinned versions):

- IC / ICIR / Rank IC        -> logged verbatim by SigAnaRecord
- ARR / IR / MDD             -> ``1day.excess_return_with_cost.{annualized_return,
                                information_ratio,max_drawdown}``
- Sharpe                     -> NOT in qlib's metric set. Read from the csv if a
                                sharpe-named key ever appears; otherwise derived
                                from ret.pkl's daily net return
                                (mean/std * sqrt(252)). This is the absolute-
                                return Sharpe, complementing IR (excess-return
                                based); ``n/a`` when neither source is available.

Missing metrics render as ``n/a`` — honest reporting over pretty numbers.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from pathlib import Path

# Annualization factor for daily US equity returns (trading days per year).
TRADING_DAYS_PER_YEAR = 252

# Display label -> (csv key candidates, format style). Order = display order.
_PCT = "pct"
_NUM = "num"
METRIC_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("IC", ("IC",), _NUM),
    ("ICIR", ("ICIR",), _NUM),
    ("Rank IC", ("Rank IC",), _NUM),
    ("ARR", ("1day.excess_return_with_cost.annualized_return",), _PCT),
    ("IR", ("1day.excess_return_with_cost.information_ratio",), _NUM),
    ("MDD", ("1day.excess_return_with_cost.max_drawdown",), _PCT),
]
SHARPE_CSV_KEYS: tuple[str, ...] = ("1day.excess_return_with_cost.sharpe", "Sharpe", "sharpe")


class SummaryError(RuntimeError):
    """A completion artifact could not be parsed or rendered."""


def load_metrics(qlib_res_csv: str | Path) -> dict[str, float]:
    """Read qlib_res.csv (metric-name index, one value column) into a dict.

    Non-numeric or NaN values are dropped rather than surfaced — the formatter
    then reports the affected metric as ``n/a``.
    """
    import pandas as pd

    path = Path(qlib_res_csv)
    try:
        frame = pd.read_csv(path, index_col=0)
    except Exception as exc:  # noqa: BLE001 - one actionable error type for callers
        raise SummaryError(f"cannot read metrics csv {path}: {exc}") from exc
    if frame.shape[1] < 1:
        raise SummaryError(f"metrics csv {path} has no value column")
    out: dict[str, float] = {}
    for key, raw in frame.iloc[:, 0].items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            out[str(key)] = value
    return out


def compute_sharpe(ret_pkl: str | Path) -> float | None:
    """Annualized Sharpe of the strategy's daily net return from ret.pkl.

    net return = ``return`` - ``cost`` (cost treated as 0 when absent);
    Sharpe = mean/std * sqrt(252). None when the frame is unusable (missing
    ``return`` column, <2 rows, or zero variance).
    """
    import pandas as pd

    try:
        frame = pd.read_pickle(Path(ret_pkl))
    except Exception as exc:  # noqa: BLE001 - one actionable error type for callers
        raise SummaryError(f"cannot read ret.pkl {ret_pkl}: {exc}") from exc
    if not isinstance(frame, pd.DataFrame) or "return" not in frame.columns:
        return None
    net = frame["return"].astype(float)
    if "cost" in frame.columns:
        net = net - frame["cost"].astype(float)
    net = net.dropna()
    if len(net) < 2:
        return None
    std = float(net.std())
    if std == 0 or math.isnan(std):
        return None
    return float(net.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _format_value(value: float | None, style: str) -> str:
    if value is None:
        return "n/a"
    if style == _PCT:
        return f"{value:+.2%}"
    return f"{value:.4f}"


def format_summary(
    metrics: Mapping[str, float],
    sharpe: float | None = None,
    *,
    workspace_path: str | Path | None = None,
) -> str:
    """Slack mrkdwn block of the headline backtest metrics.

    ``sharpe`` is the ret.pkl-derived fallback; a sharpe-named key in the csv
    wins when present.
    """
    lines = ["*Backtest metrics*"]
    if workspace_path is not None:
        lines[0] += f" (workspace `{workspace_path}`)"
    for label, keys, style in METRIC_SPECS:
        value = next((metrics[k] for k in keys if k in metrics), None)
        lines.append(f"• *{label}:* {_format_value(value, style)}")
    csv_sharpe = next((metrics[k] for k in SHARPE_CSV_KEYS if k in metrics), None)
    effective_sharpe = csv_sharpe if csv_sharpe is not None else sharpe
    lines.append(f"• *Sharpe:* {_format_value(effective_sharpe, _NUM)}")
    return "\n".join(lines)


def render_equity_curve(ret_pkl: str | Path, *, title: str = "Equity curve") -> bytes:
    """Render ret.pkl to a PNG equity-curve chart; returns the PNG bytes.

    Plots the cumulative strategy return net of cost (qlib accumulates by
    summation — see qlib risk_analysis) and, when present, the benchmark's
    cumulative return.
    """
    import matplotlib

    matplotlib.use("Agg")  # server-side rendering: no display, thread-safe enough here
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        frame = pd.read_pickle(Path(ret_pkl))
    except Exception as exc:  # noqa: BLE001 - one actionable error type for callers
        raise SummaryError(f"cannot read ret.pkl {ret_pkl}: {exc}") from exc
    if not isinstance(frame, pd.DataFrame) or "return" not in frame.columns:
        raise SummaryError(f"ret.pkl {ret_pkl} is not a portfolio DataFrame with a 'return' column")

    net = frame["return"].astype(float)
    if "cost" in frame.columns:
        net = net - frame["cost"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    try:
        ax.plot(net.index, net.cumsum(), label="strategy (net of cost)", linewidth=1.6)
        if "bench" in frame.columns:
            bench = frame["bench"].astype(float)
            ax.plot(bench.index, bench.cumsum(), label="benchmark", linewidth=1.2, alpha=0.8)
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("cumulative return")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
    finally:
        plt.close(fig)
    png = buffer.getvalue()
    if not png:
        raise SummaryError(f"rendered an empty PNG from {ret_pkl}")
    return png

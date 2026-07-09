"""Signal extraction: promoted workspace pred.pkl -> equal-weight target holdings.

Replicates qlib's ``TopkDropoutStrategy`` stock selection (the strategy every
backtest in research/us_templates runs: method_buy="top", method_sell="bottom")
so the paper book tracks what the backtest actually simulated. The reference
implementation is qlib/contrib/strategy/signal_strategy.py
``TopkDropoutStrategy.generate_trade_decision``; ``topk_dropout_holdings``
mirrors its selection lines one-for-one, with two deliberate deviations:

* Ties are deterministic here: equal scores rank alphabetically by symbol
  (upstream uses an unstable sort, so tie order is sort-algorithm luck — not
  acceptable for a money-touching rebalancer).
* Degenerate slices are clamped: upstream's ``comb[-n_drop:]`` with n_drop=0
  selects the WHOLE book for sale, and ``today[:negative]`` (over-held book)
  buys a nonsense slice. Here n_drop=0 sells nothing and a negative buy
  budget buys nothing.

Failure policy per US-029: stale or missing predictions raise SignalError /
exit nonzero. No partial book is ever returned — every failure raises before
any targets object exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_CALENDAR_PATH = Path("~/.qlib/qlib_data/us_data/calendars/day.txt")


class SignalError(Exception):
    """Any condition that must abort signal extraction (no partial output)."""


@dataclass(frozen=True)
class StrategyParams:
    """TopkDropoutStrategy knobs, sourced from the workspace qlib config."""

    topk: int
    n_drop: int

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise SignalError(f"topk must be >= 1, got {self.topk}")
        if self.n_drop < 0:
            raise SignalError(f"n_drop must be >= 0, got {self.n_drop}")


@dataclass(frozen=True)
class TargetBook:
    """The extracted target portfolio: equal-weight holdings for pred_date."""

    pred_date: dt.date
    weights: dict[str, float]
    params: StrategyParams
    pred_path: Path


def load_strategy_params(workspace: Path, config_name: str | None = None) -> StrategyParams:
    """Read topk/n_drop from the workspace's qlib conf yaml(s).

    Workspace confs keep their jinja placeholders (qrun renders them at run
    time), so render with tolerant undefineds before yaml-parsing. With no
    explicit config_name every conf*.yaml is scanned; they must agree —
    disagreement means we cannot know which strategy the backtest ran.
    """
    import yaml
    from jinja2 import Environment, Undefined

    workspace = workspace.expanduser()
    if config_name is not None:
        conf_files = [workspace / config_name]
        if not conf_files[0].is_file():
            raise SignalError(f"config not found: {conf_files[0]}")
    else:
        conf_files = sorted(workspace.glob("conf*.yaml"))
        if not conf_files:
            raise SignalError(f"no conf*.yaml found in workspace {workspace}")

    env = Environment(undefined=Undefined, autoescape=False)
    found: dict[str, tuple[int, int]] = {}
    for conf in conf_files:
        try:
            rendered = env.from_string(conf.read_text()).render()
            data = yaml.safe_load(rendered)
        except Exception as exc:  # jinja/yaml errors both mean "unreadable conf"
            raise SignalError(f"cannot parse {conf}: {exc}") from exc
        params = _find_topk_dropout_kwargs(data)
        if params is not None:
            found[conf.name] = params

    if not found:
        raise SignalError(
            f"no TopkDropoutStrategy config found in {[c.name for c in conf_files]} "
            f"under {workspace}"
        )
    if len(set(found.values())) > 1:
        raise SignalError(f"conflicting TopkDropoutStrategy params across configs: {found}")
    topk, n_drop = next(iter(found.values()))
    return StrategyParams(topk=topk, n_drop=n_drop)


def _find_topk_dropout_kwargs(node: Any) -> tuple[int, int] | None:
    """Recursively find a {class: TopkDropoutStrategy, kwargs: {...}} block."""
    if isinstance(node, dict):
        if node.get("class") == "TopkDropoutStrategy":
            kwargs = node.get("kwargs")
            if not isinstance(kwargs, dict) or "topk" not in kwargs or "n_drop" not in kwargs:
                raise SignalError(f"TopkDropoutStrategy block missing topk/n_drop kwargs: {node}")
            return int(kwargs["topk"]), int(kwargs["n_drop"])
        for value in node.values():
            hit = _find_topk_dropout_kwargs(value)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for item in node:
            hit = _find_topk_dropout_kwargs(item)
            if hit is not None:
                return hit
    return None


def locate_pred(workspace: Path) -> Path:
    """Find the newest pred.pkl under the workspace's mlruns tree.

    qlib's SignalRecord logs pred.pkl as an mlflow artifact
    (mlruns/<exp>/<run>/artifacts/pred.pkl); one workspace can hold several
    runs, newest mtime wins (same "latest recorder" intent as the workspace's
    read_exp_res.py without importing mlflow).
    """
    workspace = workspace.expanduser()
    candidates = sorted(
        workspace.glob("mlruns/**/pred.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise SignalError(f"predictions absent: no mlruns/**/pred.pkl under {workspace}")
    return candidates[0]


def load_latest_cross_section(pred_path: Path) -> tuple[dt.date, Any]:
    """Load pred.pkl and return (pred_date, scores) for the latest datetime.

    pred.pkl is a DataFrame (or Series) with MultiIndex (datetime, instrument);
    like upstream, only the first column is used. NaN scores are dropped
    (no signal for that name).
    """
    import pandas as pd

    try:
        obj = pd.read_pickle(pred_path)
    except Exception as exc:
        raise SignalError(f"cannot unpickle predictions {pred_path}: {exc}") from exc

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            raise SignalError(f"predictions {pred_path} have no columns")
        series = obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        raise SignalError(f"predictions {pred_path} are {type(obj).__name__}, not a DataFrame")

    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise SignalError(f"predictions {pred_path} lack a (datetime, instrument) MultiIndex")
    names = list(index.names)
    dt_level = names.index("datetime") if "datetime" in names else 0

    if len(series) == 0:
        raise SignalError(f"predictions {pred_path} are empty")
    latest_raw: Any = index.get_level_values(dt_level).max()
    if pd.isna(latest_raw):
        raise SignalError(f"predictions {pred_path} have no valid datetime index")
    latest = pd.Timestamp(latest_raw)
    pred_date = cast(dt.datetime, latest.to_pydatetime()).date()
    cross = series.xs(latest, level=dt_level).dropna()
    if len(cross) == 0:
        raise SignalError(f"latest cross-section {pred_date} in {pred_path} is all-NaN")
    cross.index = cross.index.map(str)
    return pred_date, cross


def last_trading_day(as_of: dt.date, calendar_path: Path = DEFAULT_CALENDAR_PATH) -> dt.date:
    """Most recent trading day on/before as_of, per the store calendar."""
    calendar_path = calendar_path.expanduser()
    if not calendar_path.is_file():
        raise SignalError(f"trading calendar not found: {calendar_path}")
    days: list[dt.date] = []
    for line in calendar_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            days.append(dt.date.fromisoformat(line[:10]))
        except ValueError as exc:
            raise SignalError(f"bad calendar line in {calendar_path}: {line!r}") from exc
    past = [d for d in days if d <= as_of]
    if not past:
        raise SignalError(f"no trading day on/before {as_of} in {calendar_path}")
    return max(past)


def assert_fresh(pred_date: dt.date, as_of: dt.date, calendar_path: Path) -> None:
    """Abort if the latest prediction is older than the last trading day."""
    last_td = last_trading_day(as_of, calendar_path)
    if pred_date < last_td:
        raise SignalError(
            f"predictions stale: latest cross-section is {pred_date} but the last "
            f"trading day on/before {as_of} is {last_td} — refresh the store and "
            f"regenerate predictions before trading"
        )


def _rank_desc(scores: Any) -> list[str]:
    """Symbols by descending score; ties alphabetical; NaN scores last."""
    ordered = scores.sort_index(kind="stable").sort_values(
        ascending=False, kind="stable", na_position="last"
    )
    return [str(s) for s in ordered.index]


def topk_dropout_holdings(
    scores: Any, current: Sequence[str], params: StrategyParams
) -> list[str]:
    """One TopkDropoutStrategy step: next holdings from scores + current book.

    Mirrors TopkDropoutStrategy.generate_trade_decision (method_buy="top",
    method_sell="bottom"): rank current holdings, admit at most
    n_drop + topk - len(current) new candidates, drop holdings ranking in the
    bottom n_drop of the combined list, buy up to the topk budget. Holdings
    missing from scores rank last (NaN), so they are dropped first.
    """
    import pandas as pd

    if len(set(current)) != len(current):
        raise SignalError(f"current holdings contain duplicates: {sorted(current)}")
    scores = scores.dropna()
    if len(scores) == 0:
        raise SignalError("prediction cross-section is empty after dropping NaN scores")
    if scores.index.has_duplicates:
        dupes = sorted(set(scores.index[scores.index.duplicated()]))
        raise SignalError(f"prediction cross-section has duplicate instruments: {dupes}")

    # last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
    last = _rank_desc(scores.reindex(list(current)))
    # today = top (n_drop + topk - len(last)) of the not-currently-held names
    n_new = max(params.n_drop + params.topk - len(last), 0)
    today = _rank_desc(scores[~scores.index.isin(last)])[:n_new]
    # comb = combined ranking of held + admitted names (drop from this list,
    # so a new name scoring below the worst holding drops itself, not the book)
    comb = _rank_desc(scores.reindex(pd.Index(last).union(pd.Index(today, dtype=object))))
    # sell = holdings in the bottom n_drop of comb
    bottom = set(comb[-params.n_drop :]) if params.n_drop > 0 else set()
    sell = [s for s in last if s in bottom]
    # buy = today[: len(sell) + topk - len(last)], clamped (see module docstring)
    buy = today[: max(len(sell) + params.topk - len(last), 0)]

    holdings = [s for s in last if s not in bottom] + buy
    return sorted(holdings)


def equal_weight_targets(holdings: Sequence[str]) -> dict[str, float]:
    """Equal-weight the holdings; weights sum to 1.0 (fully invested)."""
    if not holdings:
        raise SignalError("no holdings selected; refusing to emit an empty target book")
    weight = 1.0 / len(holdings)
    return {symbol: weight for symbol in sorted(holdings)}


def extract_targets(
    workspace: Path,
    current_holdings: Sequence[str],
    params: StrategyParams | None = None,
    as_of: dt.date | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
) -> TargetBook:
    """workspace pred.pkl -> fresh, equal-weight TargetBook (or SignalError)."""
    if params is None:
        params = load_strategy_params(workspace)
    if as_of is None:
        as_of = dt.date.today()
    pred_path = locate_pred(workspace)
    pred_date, scores = load_latest_cross_section(pred_path)
    assert_fresh(pred_date, as_of, calendar_path)
    holdings = topk_dropout_holdings(scores, current_holdings, params)
    weights = equal_weight_targets(holdings)
    return TargetBook(pred_date=pred_date, weights=weights, params=params, pred_path=pred_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract equal-weight target holdings from a workspace's pred.pkl"
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--current", default="", help="comma-separated currently held symbols (default: none)"
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument(
        "--as-of", type=dt.date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_PATH)
    args = parser.parse_args(argv)

    if (args.topk is None) != (args.n_drop is None):
        parser.error("--topk and --n-drop must be given together")

    current = [s.strip().upper() for s in args.current.split(",") if s.strip()]
    try:
        params = (
            StrategyParams(topk=args.topk, n_drop=args.n_drop) if args.topk is not None else None
        )
        book = extract_targets(
            args.workspace,
            current,
            params=params,
            as_of=args.as_of,
            calendar_path=args.calendar,
        )
    except SignalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "pred_date": book.pred_date.isoformat(),
                "topk": book.params.topk,
                "n_drop": book.params.n_drop,
                "weights": book.weights,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""US-029: signal extraction — pred.pkl -> equal-weight target holdings."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from execution.signal import (
    SignalError,
    StrategyParams,
    assert_fresh,
    equal_weight_targets,
    extract_targets,
    last_trading_day,
    load_latest_cross_section,
    load_strategy_params,
    locate_pred,
    main,
    topk_dropout_holdings,
)

US_TEMPLATES = Path(__file__).resolve().parents[1] / "research" / "us_templates"


# ---------------------------------------------------------------- fixtures


def make_scores(mapping: dict[str, float]) -> pd.Series:
    return pd.Series(mapping, dtype=float)


def write_pred(
    workspace: Path,
    rows: dict[str, dict[str, float]],
    run: str = "run1",
    mtime: float | None = None,
) -> Path:
    """Write an mlflow-artifact-shaped pred.pkl: MultiIndex (datetime, instrument)."""
    tuples = []
    values = []
    for day, scores in rows.items():
        for symbol, score in scores.items():
            tuples.append((pd.Timestamp(day), symbol))
            values.append(score)
    index = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    df = pd.DataFrame({"score": values}, index=index)
    path = workspace / "mlruns" / "1" / run / "artifacts" / "pred.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def write_calendar(path: Path, days: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(days) + "\n")
    return path


def write_conf(workspace: Path, name: str, topk: int, n_drop: int) -> Path:
    """A minimal workspace conf with the jinja placeholders real confs keep."""
    workspace.mkdir(parents=True, exist_ok=True)
    text = f"""\
qlib_init:
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us
data_handler_config: &data_handler_config
    start_time: {{{{ train_start | default("2008-01-01", true) }}}}
    end_time: {{{{ test_end | default("null", true) }}}}
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: {topk}
            n_drop: {n_drop}
"""
    path = workspace / name
    path.write_text(text)
    return path


def fixture_workspace(
    tmp_path: Path,
    rows: dict[str, dict[str, float]],
    topk: int = 3,
    n_drop: int = 1,
) -> tuple[Path, Path]:
    """Workspace with conf + pred, and a calendar ending on the pred's last day."""
    workspace = tmp_path / "workspace"
    write_conf(workspace, "conf_combined_factors.yaml", topk=topk, n_drop=n_drop)
    write_pred(workspace, rows)
    calendar = write_calendar(tmp_path / "calendars" / "day.txt", sorted(rows))
    return workspace, calendar


# ------------------------------------------------- topk_dropout_holdings


def test_cold_start_selects_top_topk() -> None:
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    held = topk_dropout_holdings(scores, [], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "C"]


def test_drop_rule_replaces_worst_holding_with_better_candidate() -> None:
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 10, "E": 1})
    held = topk_dropout_holdings(scores, ["A", "B", "C"], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "D"]


def test_at_most_n_drop_replaced_per_step() -> None:
    # Three candidates beat every holding, but n_drop=1 admits/replaces one.
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 10, "E": 9, "F": 8})
    held = topk_dropout_holdings(scores, ["A", "B", "C"], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "D"]


def test_no_churn_when_candidates_score_below_holdings() -> None:
    # The admitted candidate ranks bottom of the combined list, so it drops
    # itself and the book is untouched (upstream's sell-high-buy-low guard).
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 1})
    held = topk_dropout_holdings(scores, ["A", "B", "C"], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "C"]


def test_holding_missing_from_predictions_dropped_first() -> None:
    # X has no score: it ranks last (NaN) and is the first name dropped.
    scores = make_scores({"A": 5, "B": 4, "D": 2})
    held = topk_dropout_holdings(scores, ["A", "B", "X"], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "D"]


def test_tie_at_topk_boundary_resolves_alphabetically() -> None:
    scores = make_scores({"C": 2.0, "B": 2.0, "A": 1.0})
    held = topk_dropout_holdings(scores, [], StrategyParams(topk=2, n_drop=1))
    assert held == ["B", "C"]

    # Tie at the drop boundary: held B and C tie at the bottom of comb;
    # alphabetical rank puts C last, so C is the one dropped.
    scores = make_scores({"A": 5.0, "B": 1.0, "C": 1.0, "D": 3.0})
    held = topk_dropout_holdings(scores, ["A", "B", "C"], StrategyParams(topk=3, n_drop=1))
    assert held == ["A", "B", "D"]


def test_n_drop_zero_never_sells() -> None:
    # Deviation guard: upstream's comb[-0:] slice would sell the whole book.
    scores = make_scores({"A": 1, "B": 2, "C": 99})
    held = topk_dropout_holdings(scores, ["A", "B"], StrategyParams(topk=2, n_drop=0))
    assert held == ["A", "B"]


def test_underfilled_book_tops_up_to_topk() -> None:
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 2})
    held = topk_dropout_holdings(scores, ["D"], StrategyParams(topk=3, n_drop=1))
    # n_new = 1 + 3 - 1 = 3 -> today=[A,B,C]; comb=[A,B,C,D] -> sell=[D];
    # buy budget = 1 + 3 - 1 = 3 -> [A,B,C]
    assert held == ["A", "B", "C"]


def test_overheld_book_shrinks_without_nonsense_buys() -> None:
    # More holdings than topk (e.g. topk shrank between promotions): the
    # negative buy budget is clamped to zero and the book shrinks by n_drop.
    scores = make_scores({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
    held = topk_dropout_holdings(
        scores, ["A", "B", "C", "D", "E"], StrategyParams(topk=3, n_drop=1)
    )
    assert held == ["A", "B", "C", "D"]


def test_duplicate_current_holdings_rejected() -> None:
    scores = make_scores({"A": 1})
    with pytest.raises(SignalError, match="duplicates"):
        topk_dropout_holdings(scores, ["A", "A"], StrategyParams(topk=2, n_drop=1))


def test_all_nan_scores_rejected() -> None:
    scores = pd.Series({"A": float("nan")})
    with pytest.raises(SignalError, match="empty"):
        topk_dropout_holdings(scores, [], StrategyParams(topk=2, n_drop=1))


def test_duplicate_instruments_in_cross_section_rejected() -> None:
    scores = pd.Series([1.0, 2.0], index=["A", "A"])
    with pytest.raises(SignalError, match="duplicate instruments"):
        topk_dropout_holdings(scores, [], StrategyParams(topk=2, n_drop=1))


def test_parity_with_upstream_selection_lines() -> None:
    """Differential check against the verbatim upstream selection code.

    These lines are transcribed from qlib/contrib/strategy/signal_strategy.py
    TopkDropoutStrategy.generate_trade_decision (method_buy="top",
    method_sell="bottom", only_tradable=False). With distinct scores (ties are
    where we deliberately deviate) both implementations must agree.
    """
    import random

    def upstream(pred_score: Any, current: list[str], topk: int, n_drop: int) -> list[str]:
        last: Any = pred_score.reindex(current).sort_values(ascending=False).index
        today = list(
            pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index
        )[: n_drop + topk - len(last)]
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        sell: Any = last[last.isin(list(comb)[-n_drop:])]
        buy = today[: len(sell) + topk - len(last)]
        return sorted([str(c) for c in last if c not in sell] + [str(c) for c in buy])

    rng = random.Random(29)
    symbols = [f"S{i:02d}" for i in range(20)]
    for topk, n_drop in [(3, 1), (5, 2), (10, 3), (4, 4)]:
        for _ in range(25):
            universe = rng.sample(symbols, rng.randint(max(topk, 2), len(symbols)))
            scores = pd.Series(rng.sample(range(1000), len(universe)), index=universe, dtype=float)
            n_held = rng.randint(0, min(topk, len(universe)))
            current = rng.sample(universe, n_held)
            ours = topk_dropout_holdings(
                scores, current, StrategyParams(topk=topk, n_drop=n_drop)
            )
            assert ours == upstream(scores, current, topk, n_drop), (
                f"divergence: topk={topk} n_drop={n_drop} current={current} "
                f"scores={scores.to_dict()}"
            )


# ----------------------------------------------------- equal_weight_targets


def test_equal_weights_sum_to_one() -> None:
    weights = equal_weight_targets(["B", "A", "C"])
    third = pytest.approx(1 / 3)
    assert weights == {"A": third, "B": third, "C": third}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_empty_holdings_refused() -> None:
    with pytest.raises(SignalError, match="empty target book"):
        equal_weight_targets([])


# ------------------------------------------------------------ params


def test_strategy_params_validation() -> None:
    with pytest.raises(SignalError, match="topk"):
        StrategyParams(topk=0, n_drop=1)
    with pytest.raises(SignalError, match="n_drop"):
        StrategyParams(topk=1, n_drop=-1)


def test_load_strategy_params_renders_jinja_conf(tmp_path: Path) -> None:
    write_conf(tmp_path, "conf_baseline.yaml", topk=50, n_drop=5)
    assert load_strategy_params(tmp_path) == StrategyParams(topk=50, n_drop=5)


def test_load_strategy_params_from_real_us_templates() -> None:
    params = load_strategy_params(US_TEMPLATES / "factor_template")
    assert params == StrategyParams(topk=50, n_drop=5)


def test_load_strategy_params_conflicting_confs_rejected(tmp_path: Path) -> None:
    write_conf(tmp_path, "conf_a.yaml", topk=50, n_drop=5)
    write_conf(tmp_path, "conf_b.yaml", topk=30, n_drop=3)
    with pytest.raises(SignalError, match="conflicting"):
        load_strategy_params(tmp_path)


def test_load_strategy_params_missing_conf_rejected(tmp_path: Path) -> None:
    with pytest.raises(SignalError, match="no conf"):
        load_strategy_params(tmp_path)


def test_load_strategy_params_explicit_config_name(tmp_path: Path) -> None:
    write_conf(tmp_path, "conf_a.yaml", topk=50, n_drop=5)
    write_conf(tmp_path, "conf_b.yaml", topk=30, n_drop=3)
    assert load_strategy_params(tmp_path, "conf_b.yaml") == StrategyParams(topk=30, n_drop=3)


# ------------------------------------------------------- pred loading


def test_locate_pred_newest_mtime_wins(tmp_path: Path) -> None:
    write_pred(tmp_path, {"2026-07-01": {"A": 1.0}}, run="old", mtime=1_000)
    newest = write_pred(tmp_path, {"2026-07-02": {"A": 1.0}}, run="new", mtime=2_000)
    assert locate_pred(tmp_path) == newest


def test_locate_pred_absent_aborts(tmp_path: Path) -> None:
    with pytest.raises(SignalError, match="predictions absent"):
        locate_pred(tmp_path)


def test_load_latest_cross_section_picks_last_date_and_drops_nan(tmp_path: Path) -> None:
    path = write_pred(
        tmp_path,
        {
            "2026-07-01": {"A": 9.0, "B": 8.0},
            "2026-07-02": {"A": 1.0, "B": float("nan"), "C": 2.0},
        },
    )
    pred_date, scores = load_latest_cross_section(path)
    assert pred_date == dt.date(2026, 7, 2)
    assert dict(scores) == {"A": 1.0, "C": 2.0}


def test_load_latest_cross_section_rejects_flat_index(tmp_path: Path) -> None:
    path = tmp_path / "pred.pkl"
    pd.Series({"A": 1.0}).to_pickle(path)
    with pytest.raises(SignalError, match="MultiIndex"):
        load_latest_cross_section(path)


def test_load_latest_cross_section_rejects_non_frame(tmp_path: Path) -> None:
    path = tmp_path / "pred.pkl"
    pd.to_pickle({"not": "a frame"}, path)
    with pytest.raises(SignalError, match="not a DataFrame"):
        load_latest_cross_section(path)


# --------------------------------------------------------- freshness


def test_last_trading_day(tmp_path: Path) -> None:
    calendar = write_calendar(tmp_path / "day.txt", ["2026-07-06", "2026-07-07", "2026-07-08"])
    # As-of a non-trading date: the last entry on/before it.
    assert last_trading_day(dt.date(2026, 7, 9), calendar) == dt.date(2026, 7, 8)
    assert last_trading_day(dt.date(2026, 7, 7), calendar) == dt.date(2026, 7, 7)
    with pytest.raises(SignalError, match="no trading day"):
        last_trading_day(dt.date(2026, 7, 5), calendar)


def test_last_trading_day_missing_calendar(tmp_path: Path) -> None:
    with pytest.raises(SignalError, match="calendar not found"):
        last_trading_day(dt.date(2026, 7, 9), tmp_path / "nope.txt")


def test_assert_fresh_accepts_current_and_rejects_stale(tmp_path: Path) -> None:
    calendar = write_calendar(tmp_path / "day.txt", ["2026-07-07", "2026-07-08"])
    assert_fresh(dt.date(2026, 7, 8), dt.date(2026, 7, 9), calendar)
    with pytest.raises(SignalError, match="stale"):
        assert_fresh(dt.date(2026, 7, 7), dt.date(2026, 7, 9), calendar)


# ------------------------------------------------------ extract_targets


def test_extract_targets_end_to_end(tmp_path: Path) -> None:
    workspace, calendar = fixture_workspace(
        tmp_path,
        {
            "2026-07-07": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 9.0},
            "2026-07-08": {"A": 5.0, "B": 4.0, "C": 3.0, "D": 10.0, "E": 1.0},
        },
    )
    book = extract_targets(
        workspace, ["A", "B", "C"], as_of=dt.date(2026, 7, 8), calendar_path=calendar
    )
    assert book.pred_date == dt.date(2026, 7, 8)
    assert book.params == StrategyParams(topk=3, n_drop=1)
    assert set(book.weights) == {"A", "B", "D"}
    assert all(w == pytest.approx(1 / 3) for w in book.weights.values())


def test_extract_targets_stale_predictions_abort(tmp_path: Path) -> None:
    workspace, _ = fixture_workspace(tmp_path, {"2026-07-07": {"A": 1.0, "B": 2.0}})
    calendar = write_calendar(
        tmp_path / "calendars" / "day.txt", ["2026-07-07", "2026-07-08"]
    )
    with pytest.raises(SignalError, match="stale"):
        extract_targets(workspace, [], as_of=dt.date(2026, 7, 9), calendar_path=calendar)


def test_extract_targets_missing_pred_aborts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write_conf(workspace, "conf_baseline.yaml", topk=3, n_drop=1)
    calendar = write_calendar(tmp_path / "day.txt", ["2026-07-08"])
    with pytest.raises(SignalError, match="predictions absent"):
        extract_targets(workspace, [], as_of=dt.date(2026, 7, 8), calendar_path=calendar)


# --------------------------------------------------------------- CLI


def test_cli_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace, calendar = fixture_workspace(
        tmp_path, {"2026-07-08": {"A": 5.0, "B": 4.0, "C": 3.0, "D": 10.0}}
    )
    rc = main(
        [
            "--workspace", str(workspace),
            "--current", "a,b,c",
            "--as-of", "2026-07-08",
            "--calendar", str(calendar),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pred_date"] == "2026-07-08"
    assert payload["topk"] == 3 and payload["n_drop"] == 1
    assert set(payload["weights"]) == {"A", "B", "D"}


def test_cli_stale_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace, _ = fixture_workspace(tmp_path, {"2026-07-07": {"A": 1.0}})
    calendar = write_calendar(tmp_path / "cal2" / "day.txt", ["2026-07-07", "2026-07-08"])
    rc = main(
        [
            "--workspace", str(workspace),
            "--as-of", "2026-07-09",
            "--calendar", str(calendar),
        ]
    )
    assert rc == 1
    assert "stale" in capsys.readouterr().err


def test_cli_missing_pred_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    write_conf(workspace, "conf_baseline.yaml", topk=3, n_drop=1)
    calendar = write_calendar(tmp_path / "day.txt", ["2026-07-08"])
    rc = main(
        [
            "--workspace", str(workspace),
            "--as-of", "2026-07-08",
            "--calendar", str(calendar),
        ]
    )
    assert rc == 1
    assert "predictions absent" in capsys.readouterr().err


def test_cli_explicit_params_override_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, calendar = fixture_workspace(
        tmp_path, {"2026-07-08": {"A": 3.0, "B": 2.0, "C": 1.0}}
    )
    rc = main(
        [
            "--workspace", str(workspace),
            "--topk", "2",
            "--n-drop", "1",
            "--as-of", "2026-07-08",
            "--calendar", str(calendar),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topk"] == 2
    assert set(payload["weights"]) == {"A", "B"}

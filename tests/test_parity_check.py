"""Tests for ops/parity_check.py (US-069): refresh-parity harness for new fields.

Hermetic: docker never runs — a fake runner plays both the training qrun (it
writes the mlflow artifact set a real qrun leaves behind, including the task
artifact and the render-context log line) and the exact-weights re-predict, so
the full pipeline (conf render -> snapshot -> re-predict -> spearman) is
exercised end-to-end against a fixture store.
"""

from __future__ import annotations

import datetime as dt
import os
import pickle
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import yaml
from jinja2 import Environment, Undefined

from execution import pred_refresh
from ops import parity_check
from ops.parity_check import (
    ParityDates,
    ParityError,
    ParityResult,
    derive_dates,
    feature_spec,
    measured_reproduction,
    render_conf,
    resolve_field,
    result_line,
    run_field,
    train_command,
)
from tests.test_build_store import weekdays
from tests.test_menu import fixture_store

# The fixture store's calendar (tests/test_build_store.DAYS).
CAL = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
D = [dt.date.fromisoformat(day) for day in CAL]
DATES = ParityDates(D[0], D[1], D[2], D[2], D[3], D[3])

# Backtested pred the fake training run writes: two test-window days, four
# names, distinct ranks.
BASE_PRED = {
    "2024-01-05": {"AAPL": 0.9, "MSFT": 0.5, "GOOG": 0.1, "NVDA": -0.4},
    "2024-01-08": {"AAPL": 0.2, "MSFT": 0.8, "GOOG": -0.3, "NVDA": 0.6},
}


def rendered(conf_text: str, context: dict[str, str] | None = None) -> dict:
    """yaml-load a conf after jinja-rendering it the way qrun does."""
    env = Environment(undefined=Undefined, autoescape=False)
    return yaml.safe_load(env.from_string(conf_text).render(**(context or {})))


def write_run(
    workspace: Path,
    run: str,
    pred: dict[str, dict[str, float]],
    mtime: float,
    backtested: bool,
) -> None:
    """An mlruns run shaped like what qrun / the re-predict script leaves."""
    artifacts = workspace / "mlruns" / "1" / run / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    tuples = [
        (pd.Timestamp(day), symbol) for day, scores in pred.items() for symbol in scores
    ]
    values = [score for scores in pred.values() for score in scores.values()]
    index = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    pd.DataFrame({"score": values}, index=index).to_pickle(artifacts / "pred.pkl")
    (artifacts / "params.pkl").write_bytes(pickle.dumps({"weights": run}))
    if backtested:
        # The recorded task must signature-match the workspace conf, exactly
        # like a real qrun records the task it executed.
        task = rendered((workspace / parity_check.CONF_NAME).read_text())["task"]
        (artifacts / "task").write_bytes(pickle.dumps(task))
        (artifacts / "portfolio_analysis").mkdir(exist_ok=True)
        (artifacts / "portfolio_analysis" / "report.pkl").write_bytes(b"x")
    for path in artifacts.rglob("*"):
        os.utime(path, (mtime, mtime))
    os.utime(artifacts / "pred.pkl", (mtime, mtime))


def make_fake_runner(repredict_pred: dict[str, dict[str, float]]):
    """Plays qrun (training) and the exact-weights re-predict container."""

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        workspace = log_path.parent.parent
        script = command[-1]
        if "qrun" in script:
            # qrun logs the render context; recover_context reads it back.
            env_pairs = dict(
                arg.split("=", 1) for arg in command if isinstance(arg, str) and "=" in arg
            )
            context = {
                key: env_pairs[key]
                for key in ("train_start", "train_end", "test_start", "test_end")
                if key in env_pairs
            }
            log_path.write_text(
                f"Render the template with the context: {context!r}\nqrun ok\n"
            )
            write_run(workspace, "train", BASE_PRED, mtime=100.0, backtested=True)
            return 0
        assert pred_refresh.PREDICT_SCRIPT_NAME in script
        assert (workspace / pred_refresh.PREDICT_SCRIPT_NAME).is_file()
        write_run(workspace, "refresh", repredict_pred, mtime=200.0, backtested=False)
        return 0

    return runner


# --- date derivation -------------------------------------------------------------


def test_derive_dates_splits_and_holds_out_the_lag() -> None:
    calendar = weekdays(dt.date(2025, 1, 2), 400)
    dates = derive_dates(calendar, dt.date(2025, 1, 2), lag=5)
    usable = calendar[:-5]
    assert dates.train_start == usable[0]
    assert dates.test_end == usable[-1]
    # Segments are contiguous and ordered.
    assert dates.train_end < dates.valid_start <= dates.valid_end < dates.test_start
    assert calendar.index(dates.valid_start) == calendar.index(dates.train_end) + 1
    assert calendar.index(dates.test_start) == calendar.index(dates.valid_end) + 1
    # ~50/25/25.
    train_days = calendar.index(dates.train_end) + 1
    assert train_days == len(usable) // 2


def test_derive_dates_respects_start() -> None:
    calendar = weekdays(dt.date(2024, 1, 1), 500)
    start = calendar[100]
    dates = derive_dates(calendar, start, lag=5)
    assert dates.train_start == start


def test_derive_dates_too_few_days_raises() -> None:
    calendar = weekdays(dt.date(2025, 1, 2), 8)
    with pytest.raises(ParityError, match="not enough"):
        derive_dates(calendar, dt.date(2025, 1, 2), lag=5)


# --- feature expressions and conf ------------------------------------------------


def test_feature_spec_references_the_field() -> None:
    for field, market_level in (("$mkt_brent", True), ("$close", False)):
        expressions, names = feature_spec(field, market_level)
        assert len(expressions) == len(names) == 3
        assert all(field in expr for expr in expressions)
        assert len(set(names)) == 3


def test_market_level_expressions_have_a_cross_section() -> None:
    # A broadcast series alone is cross-sectionally constant — every market
    # expression must mix in a per-ticker field.
    expressions, _ = feature_spec("$mkt_gold", True)
    assert all("$close" in expr for expr in expressions)


def test_render_conf_is_valid_rendered_yaml() -> None:
    conf_text = render_conf("$mkt_wti", True)
    data = rendered(conf_text, DATES.context())
    assert data["task"]["model"]["class"] == "LGBModel"
    assert data["task"]["dataset"]["kwargs"]["segments"]["train"] == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
    ]
    feature = data["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"][
        "kwargs"
    ]["dataloader_l"][0]["kwargs"]["config"]["feature"]
    assert all("$mkt_wti" in expr for expr in feature[0])
    records = [entry["class"] for entry in data["task"]["record"]]
    assert records == ["SignalRecord", "SigAnaRecord", "PortAnaRecord"]


def test_conf_name_is_a_snapshot_candidate() -> None:
    # snapshot_pred_refresh only considers known conf filenames.
    assert parity_check.CONF_NAME in pred_refresh._SOURCE_CONF_CANDIDATES


def test_conf_survives_reduce_records_and_signature_match() -> None:
    conf_text = render_conf("$mkt_y10", True)
    reduced = pred_refresh.reduce_records(conf_text)
    data = rendered(reduced, DATES.context())
    records = [entry["class"] for entry in data["task"]["record"]]
    assert records == ["SignalRecord"]
    signature = pred_refresh._conf_task_signature(conf_text)
    assert signature is not None
    assert signature[0] == "LGBModel"


def test_train_command_mount_and_qrun() -> None:
    command = train_command(Path("/ws/mkt_gold"), {"train_start": "2025-01-02"})
    assert command[:3] == ["docker", "run", "--rm"]
    assert "-e" in command and "train_start=2025-01-02" in command
    assert f"qrun {parity_check.CONF_NAME}" in command[-1]
    assert "/ws/mkt_gold:/workspace/qlib_workspace" in command


# --- field resolution ------------------------------------------------------------


def test_resolve_field_kinds(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    assert resolve_field(store, "$mkt_brent") == ("$mkt_brent", True)
    assert resolve_field(store, "mkt_gold") == ("$mkt_gold", True)
    assert resolve_field(store, "$close") == ("$close", False)


def test_resolve_field_unknown_raises_naming_the_field(tmp_path: Path) -> None:
    # $news_sent stays absent from the fixture store until US-074 (sentiment).
    store = fixture_store(tmp_path)
    with pytest.raises(ParityError, match=r"\$news_sent is not in the store"):
        resolve_field(store, "$news_sent")


# --- end-to-end with a fake runner ------------------------------------------------


def test_run_field_pass_end_to_end(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    result = run_field(
        "$mkt_gold",
        store=store,
        root=tmp_path / "root",
        market="tiny_pit",
        dates=DATES,
        qlib_dir=tmp_path,
        runner=make_fake_runner(BASE_PRED),
    )
    assert result == ParityResult("$mkt_gold", 1.0, 2, True)
    workspace = tmp_path / "root" / "mkt_gold"
    # The real snapshot path ran: all three pred-refresh files exist and the
    # recovered context round-tripped through the env file.
    for name in (
        pred_refresh.SNAPSHOT_CONF_NAME,
        pred_refresh.SNAPSHOT_ENV_NAME,
        pred_refresh.SNAPSHOT_PARAMS_NAME,
    ):
        assert (workspace / name).is_file()
    env = pred_refresh.load_env_file(workspace / pred_refresh.SNAPSHOT_ENV_NAME)
    assert env["train_start"] == "2024-01-02"
    assert "SignalRecord" in (workspace / pred_refresh.SNAPSHOT_CONF_NAME).read_text()


def test_run_field_fail_measures_the_spearman(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    inverted = {
        day: {symbol: -score for symbol, score in scores.items()}
        for day, scores in BASE_PRED.items()
    }
    result = run_field(
        "$mkt_gold",
        store=store,
        root=tmp_path / "root",
        market="tiny_pit",
        dates=DATES,
        qlib_dir=tmp_path,
        runner=make_fake_runner(inverted),
    )
    assert result.passed is False
    assert result.spearman == pytest.approx(-1.0)
    line = result_line(result, 0.98)
    assert "PARITY FAIL $mkt_gold" in line
    assert "-1.0000" in line
    assert "0.98" in line


def test_run_field_recreates_the_workspace(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    workspace = tmp_path / "root" / "mkt_gold"
    stale = workspace / "mlruns" / "1" / "old" / "artifacts"
    stale.mkdir(parents=True)
    (stale / "pred.pkl").write_bytes(b"stale")
    result = run_field(
        "$mkt_gold",
        store=store,
        root=tmp_path / "root",
        market="tiny_pit",
        dates=DATES,
        qlib_dir=tmp_path,
        runner=make_fake_runner(BASE_PRED),
    )
    assert result.passed is True
    assert not (workspace / "mlruns" / "1" / "old").exists()


def test_run_field_training_failure_names_the_field(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)

    def failing_runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        return 7

    with pytest.raises(ParityError, match=r"\$mkt_brent: training qrun exited 7"):
        run_field(
            "$mkt_brent",
            store=store,
            root=tmp_path / "root",
            market="tiny_pit",
            dates=DATES,
            qlib_dir=tmp_path,
            runner=failing_runner,
        )


def test_measured_reproduction_needs_two_preds(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / parity_check.CONF_NAME).parent.mkdir(parents=True)
    (workspace / parity_check.CONF_NAME).write_text(render_conf("$close", False))
    write_run(workspace, "only", BASE_PRED, mtime=100.0, backtested=True)
    with pytest.raises(ParityError, match="found 1 pred.pkl"):
        measured_reproduction(workspace)


def test_result_line_pass_names_field_and_value() -> None:
    line = result_line(ParityResult("$mkt_dxy", 0.9993, 96, True), 0.98)
    assert "PARITY PASS $mkt_dxy" in line
    assert "0.9993" in line
    assert "96" in line


def test_remove_workspace_reclaims_unwritable_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A 0o555 directory blocks unlink for the owner too — same failure shape
    # as an interrupted run's root-owned docker leftovers.
    workspace = tmp_path / "ws"
    locked = workspace / "mlruns" / "locked"
    locked.mkdir(parents=True)
    (locked / "filelock").write_bytes(b"x")
    locked.chmod(0o555)
    reclaimed: list[Path] = []

    def fake_force_writable(ws: Path, image: str) -> None:
        reclaimed.append(ws)
        locked.chmod(0o755)

    monkeypatch.setattr(parity_check, "_force_writable", fake_force_writable)
    parity_check._remove_workspace(workspace, "img")
    assert reclaimed == [workspace]
    assert not workspace.exists()


def test_train_command_chmods_the_whole_workspace() -> None:
    # Not just mlruns: qrun leaves a root-owned nested workspace/ tree too.
    command = train_command(Path("/ws/mkt_gold"), {})
    assert "chmod -R 777 /workspace/qlib_workspace;" in command[-1]


# --- CLI ---------------------------------------------------------------------------


def test_main_all_mkt_sweeps_every_market_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from data.build_store import MARKET_FIELDS

    checked: list[str] = []

    def fake_run_field(field: str, **kwargs: object) -> ParityResult:
        checked.append(field)
        return ParityResult(field, 0.999, 10, True)

    monkeypatch.setattr(parity_check, "run_field", fake_run_field)
    assert parity_check.main(["--all-mkt"]) == 0
    assert checked == [f"${name}" for name in MARKET_FIELDS]
    out = capsys.readouterr().out
    assert f"parity: {len(MARKET_FIELDS)}/{len(MARKET_FIELDS)} field(s) passed" in out


def test_main_single_field_failure_exits_nonzero_naming_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_field(field: str, **kwargs: object) -> ParityResult:
        return ParityResult(field, 0.42, 10, False)

    monkeypatch.setattr(parity_check, "run_field", fake_run_field)
    assert parity_check.main(["--field", "$mkt_brent"]) == 1
    captured = capsys.readouterr()
    assert "PARITY FAIL $mkt_brent" in captured.err
    assert "0.4200" in captured.err
    assert "parity: 0/1 field(s) passed" in captured.out


def test_main_parity_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_field(field: str, **kwargs: object) -> ParityResult:
        raise ParityError("store menu unreadable")

    monkeypatch.setattr(parity_check, "run_field", fake_run_field)
    assert parity_check.main(["--field", "$close"]) == 1
    assert "PARITY ERROR $close: store menu unreadable" in capsys.readouterr().err

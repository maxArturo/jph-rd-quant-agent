"""US-048: automated daily prediction refresh for the promoted strategy.

Covers the promote-time snapshot (source-conf choice, record reduction to
SignalRecord-only with jinja preserved, context recovery from docker logs)
and the morning refresh pipeline (docker invocation shape, test_end override,
freshness self-check against a tmp-dir store, clean skips, failure exits,
and refresh-run pruning). The docker runner is injected — no containers run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, Undefined

from execution.pred_refresh import (
    PredRefreshError,
    choose_source_conf,
    docker_command,
    load_env_file,
    prune_refresh_runs,
    recover_context,
    reduce_records,
    run_pred_refresh,
    snapshot_pred_refresh,
)
from orchestrator.state import StateStore
from tests.test_signal import write_calendar, write_pred

AS_OF = dt.date(2026, 7, 17)  # Friday
CALENDAR = ["2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16"]
FRESH_DAY = "2026-07-16"  # last trading day on/before AS_OF
STALE_DAY = "2026-07-15"

# Shaped like the real us_templates confs: jinja placeholders, three records.
BASELINE_CONF = """\
qlib_init:
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us
data_handler_config: &data_handler_config
    start_time: {{ train_start | default("2008-01-01", true) }}
    end_time: {{ test_end | default("null", true) }}
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
task:
    model:
        class: GeneralPTNN
        kwargs:
            n_epochs: {{ n_epochs }}
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""

SOTA_CONF = BASELINE_CONF.replace(
    "data_loader_placeholder", "unused"
) + """\
# sota variant marker
sota_extra:
    data_loader:
        - class: qlib.data.dataset.loader.StaticDataLoader
          kwargs:
            config: "combined_factors_df.parquet"
"""

CONTEXT_LINE = (
    "[8:MainThread](2026-07-14 15:18:15,501) INFO - qlib.qrun - [run.py:78] - "
    "Render the template with the context: {'n_epochs': '100', 'test_end': '2026-07-10', "
    "'train_start': '2016-01-01', 'lr': '5e-4', "
    "'feature_names': \"['RESI5', 'WVMA5']\"}"
)
TRAIN_LINE = "GeneralPTNN parameters setting: {'num_features': 20, 'lr': 0.0005}"


def make_workspace(tmp_path: Path, sota: bool = False) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "conf_baseline_factors_model.yaml").write_text(BASELINE_CONF)
    if sota:
        (workspace / "conf_sota_factors_model.yaml").write_text(SOTA_CONF)
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "docker_execution_20260714_151810.log").write_text(
        f"boilerplate\n{CONTEXT_LINE}\nmore lines\n{TRAIN_LINE}\n"
    )
    return workspace


def rendered(conf_text: str) -> dict:
    env = Environment(undefined=Undefined, autoescape=False)
    return yaml.safe_load(env.from_string(conf_text).render())


# --- snapshot: record reduction ----------------------------------------------


def test_snapshot_reduces_records_to_signal_only(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    conf_path, _ = snapshot_pred_refresh(workspace)
    data = rendered(conf_path.read_text())
    records = data["task"]["record"]
    assert [r["class"] for r in records] == ["SignalRecord"]
    # The structural property that sidesteps qlib's end-of-calendar
    # IndexError: nothing in the reduced conf runs a backtest.
    assert "PortAnaRecord" not in conf_path.read_text()
    assert "SigAnaRecord" not in conf_path.read_text()


def test_snapshot_keeps_jinja_placeholders(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    conf_path, _ = snapshot_pred_refresh(workspace)
    text = conf_path.read_text()
    assert '{{ test_end | default("null", true) }}' in text
    assert "{{ n_epochs }}" in text


def test_reduce_records_requires_a_record_block() -> None:
    with pytest.raises(PredRefreshError, match="no record: block"):
        reduce_records("task:\n    model:\n        class: X\n")


def test_reduce_records_requires_signal_record() -> None:
    conf = "task:\n    record:\n        - class: PortAnaRecord\n          kwargs: {}\n"
    with pytest.raises(PredRefreshError, match="no SignalRecord"):
        reduce_records(conf)


# --- snapshot: source conf choice ---------------------------------------------


def test_snapshot_prefers_sota_conf_when_its_parquet_exists(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path, sota=True)
    (workspace / "combined_factors_df.parquet").write_bytes(b"parquet")
    assert choose_source_conf(workspace).name == "conf_sota_factors_model.yaml"


def test_snapshot_falls_back_to_baseline_when_parquet_cleaned(tmp_path: Path) -> None:
    """RD-Agent cleans combined_factors_df.parquet after runs without SOTA
    factor experiments — the sota conf is then unrunnable."""
    workspace = make_workspace(tmp_path, sota=True)
    assert choose_source_conf(workspace).name == "conf_baseline_factors_model.yaml"


def test_snapshot_without_any_conf_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(PredRefreshError, match="no usable source conf"):
        choose_source_conf(tmp_path / "empty")


# --- snapshot: context recovery -----------------------------------------------


def test_snapshot_env_recovers_context_and_container_env(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    _, env_path = snapshot_pred_refresh(workspace)
    env = load_env_file(env_path)
    assert env["test_end"] == "2026-07-10"
    assert env["n_epochs"] == "100"
    assert env["feature_names"] == "['RESI5', 'WVMA5']"
    # num_features is not a context var in the baseline conf — recovered from
    # the training log instead.
    assert env["num_features"] == "20"
    # QTDockerEnv parity, persisted so refresh time needs nothing else.
    assert env["PYTHONPATH"] == "/workspace/qlib_workspace"
    assert env["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_snapshot_without_context_line_raises(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    for log in (workspace / "logs").glob("*.log"):
        log.write_text("no context here\n")
    with pytest.raises(PredRefreshError, match="cannot recover the jinja context"):
        recover_context(workspace)


def test_recover_context_uses_newest_log(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    newer = workspace / "logs" / "docker_execution_20260715_000000.log"
    newer.write_text(CONTEXT_LINE.replace("'2026-07-10'", "'2026-07-15'") + "\n")
    import os

    old = workspace / "logs" / "docker_execution_20260714_151810.log"
    os.utime(old, (1000.0, 1000.0))
    assert recover_context(workspace)["test_end"] == "2026-07-15"


def test_load_env_file_parses_the_backfilled_format(tmp_path: Path) -> None:
    """The manually backfilled pred_refresh.env (2026-07-14) must keep parsing."""
    path = tmp_path / "pred_refresh.env"
    path.write_text(
        "feature_names=['RESI5', 'WVMA5']\n"
        "lr=5e-4\n"
        "test_end=2026-07-16\n"
        "PYTHONPATH=/workspace/qlib_workspace\n"
        "MLFLOW_ALLOW_FILE_STORE=true\n"
    )
    env = load_env_file(path)
    assert env["lr"] == "5e-4"
    assert env["feature_names"] == "['RESI5', 'WVMA5']"


# --- refresh pipeline -----------------------------------------------------------


def promoted_env(tmp_path: Path, snapshot: bool = True) -> tuple[Path, Path, Path]:
    """(workspace with a promoted row, db_path, store_path) in a tmp dir."""
    workspace = make_workspace(tmp_path)
    if snapshot:
        snapshot_pred_refresh(workspace)
    db_path = tmp_path / "state.sqlite"
    StateStore(db_path).set_promoted_strategy(str(workspace), {"topk": 50, "n_drop": 5})
    store_path = tmp_path / "us_data"
    write_calendar(store_path / "calendars" / "day.txt", CALENDAR)
    return workspace, db_path, store_path


def fresh_writer(workspace: Path, day: str = FRESH_DAY):
    """A fake runner that 'trains': drops a pred.pkl cross-section for day."""

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        log_path.write_text("qrun ok\n")
        write_pred(workspace, {day: {"AAPL": 1.0, "MSFT": 0.5}}, run="refresh1")
        return 0

    return runner


def test_refresh_happy_path_posts_success(tmp_path: Path) -> None:
    workspace, db_path, store_path = promoted_env(tmp_path)
    notices: list[str] = []
    commands: list[list[str]] = []

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        commands.append(list(command))
        return fresh_writer(workspace)(command, log_path, timeout_seconds)

    rc = run_pred_refresh(
        notices.append, as_of=AS_OF, db_path=db_path, store_path=store_path, runner=runner
    )
    assert rc == 0
    (notice,) = notices
    assert "pred refresh (2026-07-17)" in notice
    assert FRESH_DAY in notice
    # qrun output went to a real log file inside the workspace.
    assert (workspace / "logs" / "pred_refresh_20260717.log").is_file()

    (command,) = commands
    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert f"{workspace}:/workspace/qlib_workspace" in command
    assert "-w /workspace/qlib_workspace" in joined
    assert "local_qlib:latest" in command
    assert "--name rdq-pred-refresh-2026-07-17" in joined
    # The stored context rides along; test_end is overridden to as_of.
    assert "-e test_end=2026-07-17" in joined
    assert "test_end=2026-07-16" not in joined
    assert "-e n_epochs=100" in joined
    assert "-e MLFLOW_ALLOW_FILE_STORE=true" in joined
    # qrun + in-container chmod (new mlruns files are root-owned).
    assert command[-3:-1] == ["sh", "-c"]
    assert "qrun conf_pred_refresh.yaml" in command[-1]
    assert "chmod -R 777 /workspace/qlib_workspace/mlruns" in command[-1]


def test_refresh_self_check_runs_the_rebalancer_gate(tmp_path: Path) -> None:
    """AC: the post-run check is assert_fresh against the tmp store calendar —
    a refresh that produces a stale cross-section fails loudly."""
    workspace, db_path, store_path = promoted_env(tmp_path)
    notices: list[str] = []
    rc = run_pred_refresh(
        notices.append,
        as_of=AS_OF,
        db_path=db_path,
        store_path=store_path,
        runner=fresh_writer(workspace, day=STALE_DAY),
    )
    assert rc == 1
    (notice,) = notices
    assert "pred refresh FAILED" in notice
    assert "predictions stale" in notice


def test_refresh_short_circuits_when_already_fresh(tmp_path: Path) -> None:
    workspace, db_path, store_path = promoted_env(tmp_path)
    write_pred(workspace, {FRESH_DAY: {"AAPL": 1.0}})

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        raise AssertionError("docker must not run when predictions are fresh")

    notices: list[str] = []
    rc = run_pred_refresh(
        notices.append, as_of=AS_OF, db_path=db_path, store_path=store_path, runner=runner
    )
    assert rc == 0
    assert "already fresh" in notices[0]


def test_refresh_skips_cleanly_without_promotion(tmp_path: Path) -> None:
    """AC: exit 0 + notice when nothing is promoted (also covers a missing DB)."""
    notices: list[str] = []
    rc = run_pred_refresh(
        notices.append,
        as_of=AS_OF,
        db_path=tmp_path / "absent.sqlite",
        store_path=tmp_path / "us_data",
        runner=fresh_writer(tmp_path),
    )
    assert rc == 0
    assert "pred refresh skipped" in notices[0]


def test_refresh_fails_when_snapshot_missing(tmp_path: Path) -> None:
    workspace, db_path, store_path = promoted_env(tmp_path, snapshot=False)
    notices: list[str] = []
    rc = run_pred_refresh(
        notices.append,
        as_of=AS_OF,
        db_path=db_path,
        store_path=store_path,
        runner=fresh_writer(workspace),
    )
    assert rc == 1
    assert "snapshot missing" in notices[0]
    assert "re-promote" in notices[0]


def test_refresh_fails_on_docker_nonzero_exit(tmp_path: Path) -> None:
    workspace, db_path, store_path = promoted_env(tmp_path)
    notices: list[str] = []
    rc = run_pred_refresh(
        notices.append,
        as_of=AS_OF,
        db_path=db_path,
        store_path=store_path,
        runner=lambda command, log_path, timeout: 137,
    )
    assert rc == 1
    assert "docker qrun exited 137" in notices[0]


def test_refresh_crash_notifies_then_raises(tmp_path: Path) -> None:
    workspace, db_path, store_path = promoted_env(tmp_path)
    notices: list[str] = []

    def runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
        raise OSError("docker daemon unreachable")

    with pytest.raises(OSError, match="docker daemon unreachable"):
        run_pred_refresh(
            notices.append, as_of=AS_OF, db_path=db_path, store_path=store_path, runner=runner
        )
    assert "pred refresh CRASHED" in notices[0]


# --- pruning --------------------------------------------------------------------


def signal_only_run(workspace: Path, run: str, mtime: float) -> Path:
    write_pred(workspace, {FRESH_DAY: {"AAPL": 1.0}}, run=run, mtime=mtime)
    artifacts = workspace / "mlruns" / "1" / run / "artifacts"
    (artifacts / "label.pkl").write_bytes(b"x")
    (artifacts / "params.pkl").write_bytes(b"x")
    return workspace / "mlruns" / "1" / run


def test_prune_keeps_newest_and_protects_research_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs = [signal_only_run(workspace, f"refresh{i}", mtime=1000.0 + i) for i in range(7)]
    # The original promoted backtest run: pred.pkl plus portfolio artifacts.
    research = signal_only_run(workspace, "research", mtime=1.0)
    (research / "artifacts" / "portfolio_analysis").mkdir()

    removed = prune_refresh_runs(workspace, keep=5)
    assert removed == 2
    assert not runs[0].exists() and not runs[1].exists()
    assert all(run.exists() for run in runs[2:])
    # Oldest of all, but never a pruning candidate.
    assert research.exists()


def test_prune_refuses_nonpositive_keep(tmp_path: Path) -> None:
    with pytest.raises(PredRefreshError, match="keep must be >= 1"):
        prune_refresh_runs(tmp_path, keep=0)


# --- docker command shape ---------------------------------------------------------


def test_docker_command_mounts_qlib_dir(tmp_path: Path) -> None:
    command = docker_command(
        tmp_path, {"a": "1"}, image="img:tag", qlib_dir=tmp_path / "qlib"
    )
    assert f"{tmp_path / 'qlib'}:/root/.qlib" in command
    assert "img:tag" in command


def test_env_snapshot_refuses_unstorable_keys(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    log = workspace / "logs" / "docker_execution_20260714_151810.log"
    log.write_text(
        "Render the template with the context: {'bad=key': '1', 'test_end': '2026-07-10'}\n"
    )
    with pytest.raises(PredRefreshError, match="cannot be stored"):
        snapshot_pred_refresh(workspace)

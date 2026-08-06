"""Offline tests for promoting fetched GPU-run workspaces (ops/promote_fetched.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

import ops.promote_fetched as promote_fetched
from ops.promote_fetched import PromoteFetchedError, read_tickers, validate_workspace
from orchestrator.state import StateStore

CONF = """\
market: &market us_liquid
benchmark: &benchmark SPY
qlib_init:
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 20
            n_drop: 3
task:
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
"""

CONTEXT_LINE = (
    "[8:MainThread](2026-08-06 15:18:15,501) INFO - qlib.qrun - [run.py:78] - "
    "Render the template with the context: {'test_end': '2026-07-10'}"
)


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws" / "abc123def4567890"
    workspace.mkdir(parents=True)
    (workspace / "conf_baseline.yaml").write_text(CONF)
    (workspace / "qlib_res.csv").write_text(
        ",0\nIC,0.0186\n1day.excess_return_with_cost.annualized_return,0.71\n"
        "1day.excess_return_with_cost.max_drawdown,-0.14\n"
    )
    pred = workspace / "mlruns" / "1" / "run1" / "artifacts" / "pred.pkl"
    pred.parent.mkdir(parents=True)
    pred.write_bytes(b"\x00")
    logs = workspace / "logs"
    logs.mkdir()
    (logs / "docker_execution_20260806_150000.log").write_text(f"x\n{CONTEXT_LINE}\n")
    return workspace


def make_store(tmp_path: Path) -> Path:
    store = tmp_path / "store" / "instruments"
    store.mkdir(parents=True)
    (store / "us_liquid.txt").write_text(
        "AAPL\t2016-01-04\t2026-08-05\nSPY\t2016-01-04\t2026-08-05\n"
    )
    return store.parent


class TestValidateWorkspace:
    def test_happy_path(self, tmp_path: Path) -> None:
        candidate = validate_workspace(make_workspace(tmp_path))
        assert candidate["market"] == "us_liquid"
        assert candidate["topk"] == 20
        assert candidate["n_drop"] == 3
        assert candidate["metrics"]["IC"] == 0.0186

    def test_refuses_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(PromoteFetchedError, match="not a directory"):
            validate_workspace(tmp_path / "nope")

    def test_refuses_without_backtest(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        (workspace / "qlib_res.csv").unlink()
        with pytest.raises(PromoteFetchedError, match="qlib_res.csv"):
            validate_workspace(workspace)

    def test_refuses_without_predictions(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        next(workspace.glob("mlruns/*/*/artifacts/pred.pkl")).unlink()
        with pytest.raises(PromoteFetchedError, match="pred.pkl"):
            validate_workspace(workspace)

    def test_refuses_without_docker_log(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        next((workspace / "logs").glob("docker_execution_*.log")).unlink()
        with pytest.raises(PromoteFetchedError, match="docker_execution"):
            validate_workspace(workspace)

    def test_refuses_conf_without_market(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        stripped = CONF.replace("market: &market us_liquid\n", "")
        (workspace / "conf_baseline.yaml").write_text(stripped)
        with pytest.raises(PromoteFetchedError, match="conf is unusable"):
            validate_workspace(workspace)


class TestReadTickers:
    def test_reads_sorted_unique_symbols(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        assert read_tickers(store / "instruments", "us_liquid") == ["AAPL", "SPY"]

    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        assert read_tickers(tmp_path, "nope") is None


class TestMain:
    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys) -> None:
        workspace = make_workspace(tmp_path)
        store = make_store(tmp_path)
        db = tmp_path / "state.sqlite"
        StateStore(db)  # create schema so --db exists
        rc = promote_fetched.main(
            ["--workspace", str(workspace), "--db", str(db), "--store", str(store), "--no-slack"]
        )
        assert rc == 0
        assert "dry-run" in capsys.readouterr().out
        assert not (workspace / "conf_pred_refresh.yaml").exists()
        assert StateStore(db).get_promoted_strategy() is None

    def test_refuses_missing_db(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        workspace = make_workspace(tmp_path)
        args = ["--workspace", str(workspace), "--db", str(tmp_path / "absent.sqlite")]
        rc = promote_fetched.main([*args, "--yes", "--no-slack"])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_yes_writes_snapshot_and_row(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        store = make_store(tmp_path)
        db = tmp_path / "state.sqlite"
        StateStore(db)
        rc = promote_fetched.main(
            [
                "--workspace",
                str(workspace),
                "--db",
                str(db),
                "--store",
                str(store),
                "--yes",
                "--no-slack",
            ]
        )
        assert rc == 0
        assert (workspace / "conf_pred_refresh.yaml").is_file()
        assert (workspace / "pred_refresh.env").is_file()
        promoted = StateStore(db).get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(workspace)
        assert promoted.config["universe"] == "us_liquid"
        assert promoted.config["universe_tickers"] == ["AAPL", "SPY"]
        assert promoted.config["topk"] == 20
        assert promoted.config["n_drop"] == 3

    def test_refused_workspace_exits_one(self, tmp_path: Path, capsys) -> None:
        rc = promote_fetched.main(["--workspace", str(tmp_path / "missing"), "--no-slack"])
        assert rc == 1
        assert "REFUSED" in capsys.readouterr().err

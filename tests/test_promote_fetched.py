"""Offline tests for promoting fetched GPU-run workspaces (ops/promote_fetched.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

import ops.promote_fetched as promote_fetched
from ops.promote_fetched import (
    PromoteFetchedError,
    evaluate_advisory_gate,
    read_tickers,
    validate_workspace,
)
from ops.promotion_gate import CriterionResult, GateConfig, GateVerdict
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
    # The backtested run's shape: US-049 snapshots its params.pkl.
    (pred.parent / "params.pkl").write_bytes(b"gpu-weights")
    (pred.parent / "portfolio_analysis").mkdir()
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


def make_verdict(passed: bool) -> GateVerdict:
    """A minimal stub verdict for driving main()'s advisory-gate branches."""
    return GateVerdict(
        parity_ok=True,
        passed=passed,
        parity_mismatches=(),
        criteria=(CriterionResult("IR", passed, "stub reason"),),
        candidate_workspace="/ws/candidate",
        incumbent_workspace="/ws/incumbent",
        config=GateConfig(),
    )


def stub_gate(
    monkeypatch: pytest.MonkeyPatch, verdict: GateVerdict | None, error: str | None = None
) -> None:
    monkeypatch.setattr(
        promote_fetched, "evaluate_advisory_gate", lambda *a, **kw: (verdict, error)
    )


@pytest.fixture(autouse=True)
def capture_decisions(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Never let a test reach the real onecli subprocess (it writes Notion)."""
    import ops.promotion_decision as promotion_decision

    calls: list[dict] = []

    def fake_record(payload, **_kwargs):
        calls.append(dict(payload))
        return True

    monkeypatch.setattr(promotion_decision, "record_via_onecli", fake_record)
    return calls


class TestReadTickers:
    def test_reads_sorted_unique_symbols(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        assert read_tickers(store / "instruments", "us_liquid") == ["AAPL", "SPY"]

    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        assert read_tickers(tmp_path, "nope") is None


class TestMain:
    def base_args(self, tmp_path: Path) -> tuple[Path, Path, list[str]]:
        workspace = make_workspace(tmp_path)
        store = make_store(tmp_path)
        db = tmp_path / "state.sqlite"
        StateStore(db)  # create schema so --db exists
        args = [
            "--workspace",
            str(workspace),
            "--db",
            str(db),
            "--store",
            str(store),
            "--no-slack",
        ]
        return workspace, db, args

    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys) -> None:
        workspace, db, args = self.base_args(tmp_path)
        rc = promote_fetched.main(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "dry-run" in out
        # The real advisory gate ran: empty DB = no incumbent, allow_first off.
        assert "*Promotion gate:*" in out and "FAIL" in out
        assert "--force" in out
        assert not (workspace / "conf_pred_refresh.yaml").exists()
        assert StateStore(db).get_promoted_strategy() is None

    def test_refuses_missing_db(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        workspace = make_workspace(tmp_path)
        args = ["--workspace", str(workspace), "--db", str(tmp_path / "absent.sqlite")]
        rc = promote_fetched.main([*args, "--yes", "--no-slack"])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_yes_writes_snapshot_and_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, db, args = self.base_args(tmp_path)
        stub_gate(monkeypatch, make_verdict(passed=True))
        rc = promote_fetched.main([*args, "--yes", "--no-notion"])
        assert rc == 0
        assert (workspace / "conf_pred_refresh.yaml").is_file()
        assert (workspace / "pred_refresh.env").is_file()
        assert (workspace / "pred_refresh_params.pkl").read_bytes() == b"gpu-weights"
        promoted = StateStore(db).get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(workspace)
        assert promoted.config["universe"] == "us_liquid"
        assert promoted.config["universe_tickers"] == ["AAPL", "SPY"]
        assert promoted.config["topk"] == 20
        assert promoted.config["n_drop"] == 3
        # US-012: the CLI path leaves the same history record as the others.
        (history,) = StateStore(db).list_promotion_history()
        assert history.source == "cli"
        assert history.gate_verdict is not None
        assert history.gate_verdict["pass"] is True
        assert history.gate_verdict["forced"] is False

    def test_failing_gate_blocks_yes_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        workspace, db, args = self.base_args(tmp_path)
        stub_gate(monkeypatch, make_verdict(passed=False))
        rc = promote_fetched.main([*args, "--yes", "--no-notion"])
        assert rc == 1
        assert "--force" in capsys.readouterr().err
        assert not (workspace / "conf_pred_refresh.yaml").exists()
        assert StateStore(db).get_promoted_strategy() is None
        assert StateStore(db).list_promotion_history() == []

    def test_force_promotes_and_records_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, db, args = self.base_args(tmp_path)
        stub_gate(monkeypatch, make_verdict(passed=False))
        rc = promote_fetched.main([*args, "--yes", "--force", "--no-notion"])
        assert rc == 0
        promoted = StateStore(db).get_promoted_strategy()
        assert promoted is not None and promoted.workspace_path == str(workspace)
        (history,) = StateStore(db).list_promotion_history()
        assert history.source == "cli"
        assert history.gate_verdict is not None
        assert history.gate_verdict["pass"] is False
        assert history.gate_verdict["forced"] is True

    def test_force_covers_gate_error_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, db, args = self.base_args(tmp_path)
        stub_gate(monkeypatch, None, "store exploded")
        assert promote_fetched.main([*args, "--yes", "--no-notion"]) == 1
        rc = promote_fetched.main([*args, "--yes", "--force", "--no-notion"])
        assert rc == 0
        (history,) = StateStore(db).list_promotion_history()
        assert history.gate_verdict == {"error": "store exploded", "forced": True}

    def test_decision_log_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_decisions: list[dict]
    ) -> None:
        """US-012: the CLI writes the Decision Log row it used to skip."""
        workspace, _db, args = self.base_args(tmp_path)
        stub_gate(monkeypatch, make_verdict(passed=False))
        rc = promote_fetched.main([*args, "--yes", "--force"])
        assert rc == 0
        (payload,) = capture_decisions
        assert payload["source"] == "cli"
        assert payload["workspace"] == str(workspace)
        assert payload["market"] == "us_liquid"
        assert payload["metrics"]["IC"] == 0.0186
        assert payload["replaced_workspace"] is None
        assert payload["forced"] is True
        assert payload["gate_verdict"]["pass"] is False

    def test_refused_workspace_exits_one(self, tmp_path: Path, capsys) -> None:
        rc = promote_fetched.main(["--workspace", str(tmp_path / "missing"), "--no-slack"])
        assert rc == 1
        assert "REFUSED" in capsys.readouterr().err


class TestAdvisoryGate:
    def test_no_incumbent_runs_pure_gate(self, tmp_path: Path) -> None:
        workspace = make_workspace(tmp_path)
        db = StateStore(tmp_path / "state.sqlite")
        verdict, error = evaluate_advisory_gate(workspace, ["AAPL", "SPY"], db)
        assert error is None
        assert verdict is not None
        assert verdict.passed is False  # allow_first defaults off
        assert verdict.incumbent_workspace is None

    def test_gate_assembly_error_degrades_never_raises(self, tmp_path: Path) -> None:
        """With an incumbent the confirmation leg needs the store calendar —
        the fixture has none, and the contract is (None, error), no raise."""
        workspace = make_workspace(tmp_path)
        db = StateStore(tmp_path / "state.sqlite")
        db.set_promoted_strategy(str(tmp_path / "old"), {"universe": "us_liquid"})
        verdict, error = evaluate_advisory_gate(
            workspace, ["AAPL", "SPY"], db, store_path=tmp_path / "store"
        )
        assert verdict is None
        assert error is not None

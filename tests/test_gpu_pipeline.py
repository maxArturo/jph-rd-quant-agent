"""Offline tests for the GPU pipeline driver (ops/gpu_pipeline.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops.gpu_pipeline import (
    PipelineOptions,
    RunDates,
    SlackThread,
    StatusFile,
    build_notion_context,
    build_run_args,
    compute_run_dates,
    format_final_summary,
    format_loop_digest,
    format_run_start,
    gate_and_promote,
    incumbent_report,
    parse_size_plan,
    reportable,
    resolve_instrument_hash,
    run_status,
    worker_sh,
)
from ops.promotion_gate import hash_instruments
from orchestrator.state import StateStore
from tests.test_promote_fetched import make_store as make_instrument_store
from tests.test_promote_fetched import make_workspace


class TestSizePlan:
    def test_parses_ordered_pairs(self) -> None:
        plan = parse_size_plan("gpu-4000adax1-20gb:tor1, gpu-6000adax1-48gb:nyc2")
        assert plan == [("gpu-4000adax1-20gb", "tor1"), ("gpu-6000adax1-48gb", "nyc2")]

    def test_rejects_missing_region(self) -> None:
        with pytest.raises(ValueError, match="SIZE:REGION"):
            parse_size_plan("gpu-4000adax1-20gb")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_size_plan(" , ")


def make_store(tmp_path: Path, *, periods: int = 60) -> tuple[Path, list[str]]:
    """Fake qlib store: business-day calendar + a span-format universe file."""
    import pandas as pd

    store = tmp_path / "us_data"
    (store / "calendars").mkdir(parents=True)
    days = [d.date().isoformat() for d in pd.bdate_range("2026-01-01", periods=periods)]
    (store / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    (store / "instruments").mkdir()
    (store / "instruments" / "us_liquid.txt").write_text(
        # Multiple spans per symbol (PIT format, US-023) — must dedup to one.
        "MSFT\t2016-01-04\t2020-06-30\n"
        "AAPL\t2016-01-04\t2026-06-30\n"
        "MSFT\t2021-01-04\t2026-06-30\n"
    )
    return store, days


class TestRunDates:
    def test_test_end_rolls_back_confirm_days_trading_days(self, tmp_path: Path) -> None:
        store, days = make_store(tmp_path)
        dates = compute_run_dates(store)
        assert dates.confirm_days == 42
        assert dates.test_end == days[-43]
        assert dates.confirm_start == days[-42]
        assert dates.store_end == days[-1]

    def test_confirm_days_configurable(self, tmp_path: Path) -> None:
        store, days = make_store(tmp_path)
        dates = compute_run_dates(store, confirm_days=5)
        assert dates.test_end == days[-6]
        assert dates.confirm_start == days[-5]
        assert dates.store_end == days[-1]

    def test_calendar_too_short_fails_loud(self, tmp_path: Path) -> None:
        store, _ = make_store(tmp_path, periods=42)
        with pytest.raises(RuntimeError, match="cannot reserve"):
            compute_run_dates(store)

    def test_missing_calendar_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="calendar"):
            compute_run_dates(tmp_path / "nowhere")

    def test_rejects_nonpositive_confirm_days(self, tmp_path: Path) -> None:
        store, _ = make_store(tmp_path)
        with pytest.raises(ValueError, match="confirm_days"):
            compute_run_dates(store, confirm_days=0)


class TestInstrumentHash:
    def test_hashes_sorted_deduped_symbols(self, tmp_path: Path) -> None:
        from ops.promotion_gate import hash_instruments

        store, _ = make_store(tmp_path)
        # Span rows collapse to the symbol set — same hash as the plain list.
        assert resolve_instrument_hash("us_liquid", store) == hash_instruments(["AAPL", "MSFT"])

    def test_missing_universe_file_fails_loud(self, tmp_path: Path) -> None:
        store, _ = make_store(tmp_path)
        with pytest.raises(RuntimeError, match="make_universe"):
            resolve_instrument_hash("nonexistent", store)


class TestLaunchComposition:
    def test_run_args_always_carry_test_end(self, tmp_path: Path) -> None:
        dates = compute_run_dates(make_store(tmp_path)[0], confirm_days=5)
        args = build_run_args(PipelineOptions(loop_n=7), dates)
        assert args[:3] == ["run", "--loop_n", "7"]
        assert args[3:5] == ["--test-end", dates.test_end]

    def test_run_args_optional_flags(self, tmp_path: Path) -> None:
        dates = compute_run_dates(make_store(tmp_path)[0], confirm_days=5)
        options = PipelineOptions(
            loop_n=3, all_duration="12h", universe="my_universe", instruction="try momentum"
        )
        args = build_run_args(options, dates)
        assert args[-6:] == [
            "--all_duration", "12h", "--universe", "my_universe", "--instruction", "try momentum"
        ]

    def test_run_start_message_states_window_and_hash(self, tmp_path: Path) -> None:
        dates = compute_run_dates(make_store(tmp_path)[0], confirm_days=5)
        text = format_run_start(PipelineOptions(loop_n=10), dates, "abcd1234abcd1234")
        assert "budget 10 hypotheses" in text
        assert f"Test window ends {dates.test_end}" in text
        assert f"{dates.confirm_start} → {dates.store_end}" in text
        assert "5 trading days" in text
        assert "`abcd1234abcd1234`" in text


class TestFormatting:
    def test_sota_loop_digest(self) -> None:
        digest = format_loop_digest(
            {
                "loop": 5,
                "action": "factor",
                "decision": True,
                "hypothesis": "vol-normalized momentum",
                "workspace": "/x/ws/0bf074144a98499a8ddb31fc3df65fa8",
                "metrics": {"IC": 0.0186, "ARR": 0.7128, "MDD": -0.14},
            }
        )
        assert "Loop 5" in digest
        assert "new SOTA" in digest
        assert "IC 0.0186" in digest
        assert "`0bf07414`" in digest

    def test_pending_loop_digest_degrades(self) -> None:
        digest = format_loop_digest({"loop": 1, "decision": None})
        assert "no verdict" in digest
        assert "no backtest artifacts" in digest
        assert "n/a" in digest

    def test_reportable_requires_verdict(self) -> None:
        assert reportable({"decision": True})
        assert reportable({"decision": False})
        assert not reportable({"decision": None})
        assert not reportable({})

    def test_final_summary_with_candidate(self) -> None:
        status = {
            "loops": [
                {"loop": 0, "decision": True, "workspace": "/x/aa", "metrics": {"IC": 0.01}},
                {"loop": 1, "decision": False, "workspace": "/x/bb", "metrics": {"IC": 0.02}},
            ],
            "candidate_loop": 0,
        }
        text = format_final_summary(status, 0, 1.5, "gpu-6000adax1-48gb")
        assert "exit 0" in text
        assert "2 loops, 1 SOTA" in text
        assert "ops.promote_fetched" in text
        assert "$2.3" in text  # 1.5h * $1.57/hr ≈ $2.35
        # No incumbent in the status → the summary must say so, not go silent.
        assert "No promoted strategy on record" in text

    def test_final_summary_without_candidate(self) -> None:
        text = format_final_summary({"loops": [], "candidate_loop": None}, 1, 0.5, "unknown-size")
        assert "nothing to promote" in text

    def _candidate_status(self, **extra) -> dict:
        return {
            "loops": [
                {
                    "loop": 5,
                    "decision": True,
                    "workspace": "/x/fefa27ea8aa4",
                    "metrics": {"IC": 0.0214, "ARR": 0.6435, "MDD": -0.2822},
                }
            ],
            "candidate_loop": 5,
            **extra,
        }

    def test_final_summary_incumbent_same_window(self) -> None:
        status = self._candidate_status(
            candidate_window=["2025-01-02", "2026-07-10"],
            candidate_factors=["extension_penalty", "downside_share_60"],
            incumbent={
                "workspace": "/y/e05ad9b46f4d",
                "metrics": {"IC": 0.0217, "ARR": 0.5936, "MDD": -0.2665},
                "window": ["2025-01-02", "2026-07-10"],
            },
        )
        text = format_final_summary(status, 0, 1.0, "gpu-4000adax1-20gb")
        assert "backtest 2025-01-02 → 2026-07-10" in text
        assert "Candidate factors: extension_penalty, downside_share_60" in text
        assert "Incumbent (promoted `e05ad9b4`): IC 0.0217 · ARR 0.5936 · MDD -0.2665" in text
        assert "same backtest window" in text
        assert "not directly comparable" not in text

    def test_final_summary_incumbent_window_mismatch_warns(self) -> None:
        status = self._candidate_status(
            candidate_window=["2025-01-02", "2026-08-11"],
            incumbent={
                "workspace": "/y/e05ad9b46f4d",
                "metrics": {"IC": 0.0217},
                "window": ["2025-01-02", "2026-07-10"],
            },
        )
        text = format_final_summary(status, 0, 1.0, "gpu-4000adax1-20gb")
        assert "windows differ" in text
        assert "not directly comparable" in text

    def test_final_summary_incumbent_metrics_unreadable(self) -> None:
        status = self._candidate_status(
            incumbent={"workspace": "/y/e05ad9b46f4d", "metrics": None, "window": None}
        )
        text = format_final_summary(status, 0, 1.0, "gpu-4000adax1-20gb")
        assert "incumbent baseline unavailable" in text


class TestIncumbentReport:
    def test_none_when_nothing_promoted(self, tmp_path: Path) -> None:
        assert incumbent_report(tmp_path / "state.sqlite") is None

    def test_reads_promoted_metrics_and_window(self, tmp_path: Path) -> None:
        import pandas as pd

        from orchestrator.state import StateStore

        workspace = tmp_path / "e05ad9b46f4d"
        workspace.mkdir()
        (workspace / "qlib_res.csv").write_text(
            ",0\n"
            "IC,0.0217\n"
            "1day.excess_return_with_cost.annualized_return,0.5936\n"
            "1day.excess_return_with_cost.max_drawdown,-0.2665\n"
        )
        pd.DataFrame(
            {"return": [0.01, 0.02]}, index=pd.to_datetime(["2025-01-02", "2026-07-10"])
        ).to_pickle(workspace / "ret.pkl")
        db_path = tmp_path / "state.sqlite"
        StateStore(db_path).set_promoted_strategy(str(workspace), {"universe": "us_liquid"})

        report = incumbent_report(db_path)
        assert report is not None
        assert report["workspace"] == str(workspace)
        assert report["metrics"] == pytest.approx({"IC": 0.0217, "ARR": 0.5936, "MDD": -0.2665})
        assert report["window"] == ["2025-01-02", "2026-07-10"]


class TestNotionContext:
    """The context file is all ops.notion_summary's subprocess ever sees — the
    run_summary fields (US-013) must ride it."""

    DATES = RunDates(
        test_end="2026-06-12",
        confirm_start="2026-06-15",
        store_end="2026-08-13",
        confirm_days=42,
    )

    def test_carries_run_summary_inputs(self) -> None:
        options = PipelineOptions(universe=None, instruction="chase divergence")
        final_status = {
            "loops": [
                {"loop": 0, "decision": False},
                {"loop": 1, "decision": True, "hypothesis": "h1"},
                {"loop": 2, "decision": None},
            ],
            "candidate_window": ["2025-01-02", "2026-06-12"],
            "candidate_factors": ["f1"],
            "incumbent": {"workspace": "/y/inc"},
        }
        candidate = {"loop": 1, "hypothesis": "h1", "metrics": {"IC": 0.02}}
        context = build_notion_context(
            options,
            final_status,
            candidate,
            dates=self.DATES,
            instrument_hash="6fbafedc13ed9a52",
            exit_code=0,
        )
        assert context["loops"] == final_status["loops"]
        assert context["run_status"] == "completed"
        assert context["instrument_hash"] == "6fbafedc13ed9a52"
        assert context["test_end"] == "2026-06-12"
        assert context["confirmation_window"] == ["2026-06-15", "2026-08-13"]
        assert context["directive"] == "chase divergence"
        assert context["loops_total"] == 2  # verdicts only
        assert context["sota_count"] == 1
        assert context["candidate"]["window"] == ["2025-01-02", "2026-06-12"]
        # The whole context must survive the JSON hop to the subprocess.
        assert json.loads(json.dumps(context)) == context

    def test_directive_strips_run_memory_digest(self) -> None:
        """US-015: the instruction may carry the run-history digest — durable
        records must keep the BARE directive or digests would compound."""
        from orchestrator.run_memory import compose_instruction

        composed = compose_instruction("chase divergence", "Run-history digest:\n\nold stuff")
        options = PipelineOptions(instruction=composed)
        context = build_notion_context(options, {}, {}, exit_code=0)
        assert context["directive"] == "chase divergence"

    def test_degrades_without_launch_facts(self) -> None:
        context = build_notion_context(PipelineOptions(), {}, {}, exit_code=None)
        assert context["run_status"] == "stopped"
        assert context["instrument_hash"] is None
        assert context["test_end"] is None
        assert context["confirmation_window"] is None
        assert context["loops"] == []

    def test_run_status_mapping(self) -> None:
        assert run_status(0) == "completed"
        assert run_status(None) == "stopped"
        assert run_status(1) == "failed"


class TestSlackFallback:
    def test_disabled_thread_prints_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        thread = SlackThread(enabled=False)
        thread.post("hello world")
        assert "hello world" in capsys.readouterr().err


class TestWorkerSh:
    def test_failure_raises_with_stderr_tail(self) -> None:
        with pytest.raises(RuntimeError, match="unknown subcommand"):
            worker_sh("frobnicate")


# ------------------------------------------------------------------ gate + auto-promotion
# The candidate workspace fixture (conf + qlib_res + pred/params + docker log)
# is the promote_fetched one — auto-promotion runs through that exact path.

TEST_WINDOW = ("2025-01-02", "2026-06-12")
GATE_DATES = RunDates(
    test_end="2026-06-12", confirm_start="2026-06-15", store_end="2026-08-13", confirm_days=42
)
LAUNCH_HASH = hash_instruments(["AAPL", "SPY"])  # matches make_instrument_store's list


class RecordingSlack(SlackThread):
    def __init__(self) -> None:
        super().__init__(enabled=False)
        self.posts: list[str] = []

    def post(self, text: str) -> None:
        self.posts.append(text)


def gate_config_yaml(tmp_path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "ir_margin": 1.05,
        "mdd_tolerance": 1.25,
        "min_ic": 0.0,
        "allow_first": False,
        "confirm_ir_margin": 1.0,
        "auto_promote": True,
    }
    values.update(overrides)
    path = tmp_path / "gate_config.yaml"
    path.write_text(
        "promotion_gate:\n"
        + "".join(f"  {key}: {json.dumps(value)}\n" for key, value in values.items())
    )
    return path


def install_gate_fakes(
    monkeypatch: pytest.MonkeyPatch,
    candidate: Path,
    incumbent: Path,
    *,
    cand_ir: float = 2.0,
    inc_ir: float = 1.0,
    conf_cand_ir: float = 2.0,
    conf_inc_ir: float = 1.0,
) -> None:
    """Stub the gate's IO half; evaluate_gate itself runs for real."""
    from ops import promotion_gate

    metrics = {
        str(candidate): {"IR": cand_ir, "MDD": -0.10, "IC": 0.02},
        str(incumbent): {"IR": inc_ir, "MDD": -0.10, "IC": 0.02},
    }

    def fake_bundle(workspace, *, instrument_hash=None, config_name=None):
        ws = str(Path(workspace).expanduser())
        return promotion_gate.MetricBundle(
            workspace=ws,
            metrics=metrics[ws],
            window=TEST_WINDOW,
            market="us_liquid",
            instrument_hash=instrument_hash,
            topk=20,
            n_drop=3,
            cost_params={"open_cost": 0.0005, "close_cost": 0.0005, "min_cost": 5.0},
        )

    def fake_evidence(cand, inc, start, end, **kwargs):
        window = (start.isoformat(), end.isoformat())

        def side(workspace, ir):
            return promotion_gate.ConfirmationSide(
                workspace=str(workspace),
                ir=ir,
                window=window,
                days=42,
                repredicted=True,
                reproduction=0.999,
            )

        return promotion_gate.ConfirmationEvidence(
            window=window, candidate=side(cand, conf_cand_ir), incumbent=side(inc, conf_inc_ir)
        )

    monkeypatch.setattr(promotion_gate, "load_metric_bundle", fake_bundle)
    monkeypatch.setattr(promotion_gate, "load_confirmation_evidence", fake_evidence)


class TestGateAndPromote:
    @pytest.fixture(autouse=True)
    def capture_decisions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """US-012 Decision Log rows are captured — the real record_via_onecli
        would shell out to onecli and write actual Notion rows from tests."""
        import ops.promotion_decision as promotion_decision

        self.decisions: list[dict] = []

        def fake_record(payload, **_kwargs):
            self.decisions.append(dict(payload))
            return True

        monkeypatch.setattr(promotion_decision, "record_via_onecli", fake_record)

    def setup_box(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """Candidate workspace, incumbent workspace, store, and a promoted DB."""
        candidate = make_workspace(tmp_path)
        incumbent = tmp_path / "inc" / "e05ad9b46f4d"
        incumbent.mkdir(parents=True)
        store = make_instrument_store(tmp_path)
        db = tmp_path / "state.sqlite"
        StateStore(db).set_promoted_strategy(
            str(incumbent),
            {"universe": "us_liquid", "universe_tickers": ["AAPL", "SPY"], "topk": 20, "n_drop": 3},
        )
        return candidate, incumbent, store, db

    def run_gate(
        self, tmp_path: Path, candidate: Path, store: Path, db: Path, config: Path
    ) -> tuple[bool, RecordingSlack, Path]:
        slack = RecordingSlack()
        status_path = tmp_path / "pipeline_status.json"
        promoted = gate_and_promote(
            str(candidate),
            GATE_DATES,
            LAUNCH_HASH,
            slack,
            StatusFile(status_path, "111.222"),
            db_path=db,
            store_path=store,
            config_path=config,
        )
        return promoted, slack, status_path

    def test_promotes_on_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)
        install_gate_fakes(monkeypatch, candidate, incumbent)
        promoted, slack, status_path = self.run_gate(
            tmp_path, candidate, store, db, gate_config_yaml(tmp_path)
        )
        assert promoted is True
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(candidate)
        # Snapshot ran (the promote_fetched write path, not a bare pointer flip).
        assert (candidate / "conf_pred_refresh.yaml").is_file()
        latest = StateStore(db).list_promotion_history()[0]
        assert latest.source == "auto_gate"
        assert latest.gate_verdict is not None and latest.gate_verdict["pass"] is True
        assert latest.replaced_workspace == str(incumbent)
        text = "\n".join(slack.posts)
        assert "*Promotion gate:*" in text and "PASS" in text
        assert "replacing `e05ad9b4`" in text
        assert "ops.rollback_promotion --yes" in text
        status = json.loads(status_path.read_text())
        assert status["auto_promoted"] is True
        assert status["gate"]["pass"] is True
        # US-012: the auto-promotion left a Decision Log record too.
        (decision,) = self.decisions
        assert decision["source"] == "auto_gate"
        assert decision["workspace"] == str(candidate)
        assert decision["replaced_workspace"] == str(incumbent)
        assert decision["gate_verdict"]["pass"] is True
        assert decision["forced"] is False

    def test_no_promote_on_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)
        # IR leg fails: 1.0 is not > 1.0 × 1.05.
        install_gate_fakes(monkeypatch, candidate, incumbent, cand_ir=1.0)
        promoted, slack, status_path = self.run_gate(
            tmp_path, candidate, store, db, gate_config_yaml(tmp_path)
        )
        assert promoted is False
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(incumbent)
        assert len(StateStore(db).list_promotion_history()) == 1  # only the original
        assert not (candidate / "conf_pred_refresh.yaml").exists()
        text = "\n".join(slack.posts)
        assert "FAIL" in text
        assert "failing criteria: IR" in text
        assert json.loads(status_path.read_text())["auto_promoted"] is False
        assert self.decisions == []  # no promotion, no Decision Log row

    def test_no_promote_on_parity_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)
        install_gate_fakes(monkeypatch, candidate, incumbent)  # metrics would pass
        slack = RecordingSlack()
        promoted = gate_and_promote(
            str(candidate),
            GATE_DATES,
            "0" * 16,  # launch hash disagrees with the incumbent's recorded universe
            slack,
            StatusFile(None, None),
            db_path=db,
            store_path=store,
            config_path=gate_config_yaml(tmp_path),
        )
        assert promoted is False
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(incumbent)
        text = "\n".join(slack.posts)
        assert "instrument list mismatch" in text
        assert "failing criteria: parity" in text

    def test_no_promote_on_snapshot_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)
        install_gate_fakes(monkeypatch, candidate, incumbent)

        def broken_snapshot(workspace):
            raise RuntimeError("jinja context unrecoverable")

        monkeypatch.setattr("ops.promote_fetched.snapshot_pred_refresh", broken_snapshot)
        promoted, slack, status_path = self.run_gate(
            tmp_path, candidate, store, db, gate_config_yaml(tmp_path)
        )
        assert promoted is False
        # The failure happened BEFORE the pointer flip: incumbent untouched.
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(incumbent)
        assert len(StateStore(db).list_promotion_history()) == 1
        text = "\n".join(slack.posts)
        assert ":rotating_light:" in text and "UNPROMOTED" in text
        assert "jinja context unrecoverable" in text
        assert "ops.promote_fetched" in text  # manual recovery command
        assert json.loads(status_path.read_text())["auto_promoted"] is False

    def test_kill_switch_reverts_to_report_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)
        install_gate_fakes(monkeypatch, candidate, incumbent)
        promoted, slack, _ = self.run_gate(
            tmp_path, candidate, store, db, gate_config_yaml(tmp_path, auto_promote=False)
        )
        assert promoted is False
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(incumbent)
        assert not (candidate / "conf_pred_refresh.yaml").exists()
        text = "\n".join(slack.posts)
        # The verdict block still posts in full — only the promotion is withheld.
        assert "*Promotion gate:*" in text and "PASS" in text
        assert "report-only" in text
        assert "ops.promote_fetched" in text

    def test_gate_error_never_fails_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidate, incumbent, store, db = self.setup_box(tmp_path)

        def exploding_bundle(workspace, *, instrument_hash=None, config_name=None):
            raise RuntimeError("qlib_res parse exploded")

        monkeypatch.setattr("ops.promotion_gate.load_metric_bundle", exploding_bundle)
        promoted, slack, status_path = self.run_gate(
            tmp_path, candidate, store, db, gate_config_yaml(tmp_path)
        )
        assert promoted is False
        row = StateStore(db).get_promoted_strategy()
        assert row is not None and row.workspace_path == str(incumbent)
        text = "\n".join(slack.posts)
        assert "promotion gate errored" in text and "qlib_res parse exploded" in text
        assert json.loads(status_path.read_text())["gate"] == {
            "error": "qlib_res parse exploded"
        }

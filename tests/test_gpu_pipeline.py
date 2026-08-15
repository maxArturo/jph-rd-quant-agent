"""Offline tests for the GPU pipeline driver (ops/gpu_pipeline.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.gpu_pipeline import (
    PipelineOptions,
    SlackThread,
    build_run_args,
    compute_run_dates,
    format_final_summary,
    format_loop_digest,
    format_run_start,
    incumbent_report,
    parse_size_plan,
    reportable,
    resolve_instrument_hash,
    worker_sh,
)


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


class TestSlackFallback:
    def test_disabled_thread_prints_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        thread = SlackThread(enabled=False)
        thread.post("hello world")
        assert "hello world" in capsys.readouterr().err


class TestWorkerSh:
    def test_failure_raises_with_stderr_tail(self) -> None:
        with pytest.raises(RuntimeError, match="unknown subcommand"):
            worker_sh("frobnicate")

"""Offline tests for the GPU pipeline driver (ops/gpu_pipeline.py)."""

from __future__ import annotations

import pytest

from ops.gpu_pipeline import (
    SlackThread,
    format_final_summary,
    format_loop_digest,
    parse_size_plan,
    reportable,
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

    def test_final_summary_without_candidate(self) -> None:
        text = format_final_summary({"loops": [], "candidate_loop": None}, 1, 0.5, "unknown-size")
        assert "nothing to promote" in text


class TestSlackFallback:
    def test_disabled_thread_prints_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        thread = SlackThread(enabled=False)
        thread.post("hello world")
        assert "hello world" in capsys.readouterr().err


class TestWorkerSh:
    def test_failure_raises_with_stderr_tail(self) -> None:
        with pytest.raises(RuntimeError, match="unknown subcommand"):
            worker_sh("frobnicate")

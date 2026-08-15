"""US-012: Notion Decision Log rows for CLI / auto-gate promotions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import ops.promotion_decision as promotion_decision
from ops.promote_fetched import PromotionResult
from ops.promotion_decision import (
    PromotionDecisionError,
    build_payload,
    decision_details,
    decision_title,
    record_via_onecli,
    write_decision,
)
from ops.promotion_gate import GATE_NOT_EVALUATED, gate_summary_line
from orchestrator.notion_client import NotionClient
from orchestrator.state import StateStore
from tests.test_notion_client import FakeSession
from tests.test_notion_recorder import page_response

PASSING_VERDICT = {
    "pass": True,
    "parity_ok": True,
    "criteria": [{"name": "IR", "passed": True}],
}
FAILING_VERDICT = {
    "pass": False,
    "parity_ok": False,
    "criteria": [
        {"name": "IR", "passed": False},
        {"name": "IC", "passed": True},
        {"name": "confirmation", "passed": False},
    ],
}


def make_result(workspace: str = "/runs/ws/abc123def4567890") -> PromotionResult:
    return PromotionResult(
        workspace=workspace,
        market="us_liquid",
        topk=20,
        n_drop=3,
        metrics={"IC": 0.0186, "ARR": 0.71, "MDD": -0.14},
        tickers=["AAPL", "SPY"],
        replaced_workspace="/runs/ws/oldworkspace1234",
        promoted_at="2026-08-15T12:00:00+00:00",
        snapshot_files=("conf_pred_refresh.yaml", "pred_refresh.env", "pred_refresh_params.pkl"),
    )


class TestGateSummaryLine:
    def test_pass(self) -> None:
        assert gate_summary_line(PASSING_VERDICT) == "Gate: PASS — parity ok, all criteria met"

    def test_fail_names_parity_and_failing_criteria(self) -> None:
        assert gate_summary_line(FAILING_VERDICT) == "Gate: FAIL — parity, IR, confirmation"

    def test_forced_suffix(self) -> None:
        line = gate_summary_line(FAILING_VERDICT, forced=True)
        assert line.startswith("Gate: FAIL — parity, IR, confirmation")
        assert "promoted anyway (operator --force)" in line

    def test_error(self) -> None:
        line = gate_summary_line(None, error="store exploded", forced=True)
        assert "Gate: ERROR — store exploded" in line
        assert "promoted anyway" in line

    def test_none_means_not_evaluated(self) -> None:
        assert gate_summary_line(None) == "Gate: not evaluated"

    def test_conversational_constant_shape(self) -> None:
        assert GATE_NOT_EVALUATED.startswith("Gate: not evaluated")


class TestPayloadAndDetails:
    def test_build_payload_is_json_safe(self) -> None:
        payload = build_payload(
            make_result(),
            source="auto_gate",
            gate_verdict=PASSING_VERDICT,
            thread_ts="123.456",
        )
        assert json.loads(json.dumps(payload)) == payload
        assert payload["source"] == "auto_gate"
        assert payload["thread_ts"] == "123.456"
        assert payload["forced"] is False

    def test_title_uses_workspace_tag(self) -> None:
        payload = build_payload(make_result(), source="cli")
        assert decision_title(payload) == "Promote 'abc123de' to paper trading"

    def test_details_carry_everything_the_prd_asks(self) -> None:
        payload = build_payload(
            make_result(), source="cli", gate_verdict=FAILING_VERDICT, forced=True
        )
        details = decision_details(payload)
        assert "Source: cli (forced)" in details
        assert "Workspace: /runs/ws/abc123def4567890" in details
        assert "Universe: us_liquid" in details
        assert "TopkDropoutStrategy: topk=20, n_drop=3" in details
        assert "IC: 0.0186" in details
        assert "ARR: 0.7100" in details
        assert "MDD: -0.1400" in details
        assert "Replaced: /runs/ws/oldworkspace1234" in details
        assert "Gate: FAIL — parity, IR, confirmation — promoted anyway" in details

    def test_details_first_promotion(self) -> None:
        payload = build_payload(
            make_result(),
            source="auto_gate",
            gate_verdict=PASSING_VERDICT,
        )
        payload["replaced_workspace"] = None
        details = decision_details(payload)
        assert "Replaced: none (first promotion)" in details
        assert "Gate: PASS" in details


def notion_config(tmp_path: Path) -> Path:
    names = (
        "research_ideas",
        "hypothesis_log",
        "backtest_results",
        "decision_log",
        "trade_ledger",
        "account_snapshots",
        "strategy_notes",
    )
    lines = ["notion:", "  databases:"]
    lines += [f"    {name}: db-{name}" for name in names]
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


class TestWriteDecision:
    def test_writes_through_notion_recorder(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        StateStore(db)
        session = FakeSession([page_response("page-dec")])
        payload = build_payload(make_result(), source="cli", gate_verdict=PASSING_VERDICT)
        page_id = write_decision(
            payload,
            db_path=db,
            config_path=notion_config(tmp_path),
            notion=NotionClient(session=session, sleep=lambda _s: None, max_retries=0),
        )
        assert page_id == "page-dec"
        (create,) = session.calls
        assert create["method"] == "POST"
        body = create["json"]
        assert body["parent"] == {"type": "database_id", "database_id": "db-decision_log"}
        props = body["properties"]
        assert (
            props["Decision"]["title"][0]["text"]["content"]
            == "Promote 'abc123de' to paper trading"
        )
        assert props["Type"] == {"select": {"name": "promotion"}}
        details = props["Details"]["rich_text"][0]["text"]["content"]
        assert "Source: cli" in details and "Gate: PASS" in details
        assert "Idea" not in props  # no thread, no idea relation

    def test_links_idea_page_via_thread(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        store = StateStore(db)
        store.set_notion_page("idea", "123.456", "page-idea")
        session = FakeSession([page_response("page-dec")])
        payload = build_payload(
            make_result(), source="auto_gate", gate_verdict=PASSING_VERDICT, thread_ts="123.456"
        )
        write_decision(
            payload,
            db_path=db,
            config_path=notion_config(tmp_path),
            notion=NotionClient(session=session, sleep=lambda _s: None, max_retries=0),
        )
        props = session.calls[0]["json"]["properties"]
        assert props["Idea"] == {"relation": [{"id": "page-idea"}]}

    def test_refuses_to_create_the_state_db(self, tmp_path: Path) -> None:
        payload = build_payload(make_result(), source="cli")
        with pytest.raises(PromotionDecisionError, match="does not exist"):
            write_decision(
                payload,
                db_path=tmp_path / "absent.sqlite",
                config_path=notion_config(tmp_path),
            )
        assert not (tmp_path / "absent.sqlite").exists()


class FakeCompleted:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


class TestRecordViaOnecli:
    def test_runs_under_the_orchestrator_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            # The payload file must exist DURING the call (deleted after).
            seen["payload"] = json.loads(Path(cmd[-1]).read_text())
            return FakeCompleted(0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        payload = build_payload(make_result(), source="cli", gate_verdict=PASSING_VERDICT)
        assert record_via_onecli(payload) is True
        cmd = seen["cmd"]
        assert cmd[:4] == ["onecli", "run", "--agent", "rdq-orchestrator"]
        assert "ops.promotion_decision" in cmd
        assert seen["payload"] == payload
        assert not Path(cmd[-1]).exists()  # temp payload cleaned up

    def test_failure_reports_false(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **kw: FakeCompleted(1, stderr="401 unauthorized")
        )
        payload = build_payload(make_result(), source="cli")
        assert record_via_onecli(payload) is False
        assert "401 unauthorized" in capsys.readouterr().err

    def test_missing_onecli_reports_false(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def no_binary(*_a, **_kw):
            raise FileNotFoundError("onecli")

        monkeypatch.setattr(subprocess, "run", no_binary)
        assert record_via_onecli(build_payload(make_result(), source="cli")) is False
        assert "failed to launch" in capsys.readouterr().err


class TestMain:
    def test_prints_page_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        payload_path = tmp_path / "payload.json"
        payload_path.write_text(json.dumps(build_payload(make_result(), source="cli")))
        monkeypatch.setattr(promotion_decision, "write_decision", lambda *a, **kw: "page-77")
        assert promotion_decision.main(["--payload", str(payload_path)]) == 0
        assert "page-77" in capsys.readouterr().out

    def test_swallowed_recorder_failure_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        payload_path = tmp_path / "payload.json"
        payload_path.write_text(json.dumps(build_payload(make_result(), source="cli")))
        monkeypatch.setattr(promotion_decision, "write_decision", lambda *a, **kw: None)
        assert promotion_decision.main(["--payload", str(payload_path)]) == 1
        assert "failed" in capsys.readouterr().err

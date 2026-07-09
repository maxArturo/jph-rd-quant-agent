"""US-038: operator halt/resume trading tools + breaker state in the summary.

Tool tests drive the REAL ConversationCore with FakeClient scripts (mocked
Anthropic, recording say) over a real Breaker on tmp paths; Decision Log
writes are asserted against a mocked Notion session; and the tool-written
halt file is proven to gate the real rebalance pipeline (FakeBroker from
tests/test_rebalance.py). No network anywhere.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from execution.breaker import Breaker, BreakerConfig
from execution.order_gate import Limits
from execution.rebalance import breaker_state_line, format_daily_summary
from orchestrator.conversation import (
    ConversationCore,
    format_trading_halted,
    format_trading_resumed,
)
from orchestrator.llm import ModelRouter
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.state import StateStore
from tests.test_conversation import THREAD, RecordingSay, StubLauncher
from tests.test_llm import FakeClient, message, text_block, tool_use_block
from tests.test_notion_client import FakeSession
from tests.test_notion_recorder import DBS, page_response, plain_text
from tests.test_rebalance import (
    STORE_DAYS,
    FakeBroker,
    run,
    write_bins,
)
from tests.test_signal import write_calendar, write_conf, write_pred


def make_breaker(tmp_path: Path) -> Breaker:
    return Breaker(
        BreakerConfig(max_daily_notional_usd=200_000.0, max_drawdown_pct=20.0),
        halt_file=tmp_path / "breaker" / "halt",
        high_water_mark_file=tmp_path / "breaker" / "hwm.json",
    )


@pytest.fixture
def rebalance_env(tmp_path: Path) -> SimpleNamespace:
    """Mirror of tests/test_rebalance.py's env fixture (importing a pytest
    fixture into another module trips ruff F811, so it is rebuilt here from
    the same shared helpers): topk=2/n_drop=1 selects AAPL+MSFT at 0.5 each,
    i.e. buy 250 AAPL @ 201.00 and buy 125 MSFT @ 402.00 on the $100k account.
    """
    store = tmp_path / "us_data"
    write_calendar(store / "calendars" / "day.txt", STORE_DAYS)
    write_bins(store, "AAPL", [199.0, 200.0], [1.0, 1.0])
    write_bins(store, "MSFT", [398.0, 400.0], [1.0, 1.0])
    write_bins(store, "NVDA", [99.0, 100.0], [1.0, 1.0])

    workspace = tmp_path / "workspace"
    write_conf(workspace, "conf.yaml", topk=2, n_drop=1)
    write_pred(workspace, {"2026-07-08": {"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.1}})

    db_path = tmp_path / "state.sqlite"
    StateStore(db_path).set_promoted_strategy(
        str(workspace), {"universe": "us_liquid", "topk": 2, "n_drop": 1}
    )
    limits = Limits(
        max_order_notional_usd=60_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    return SimpleNamespace(
        store=store,
        workspace=workspace,
        db_path=db_path,
        breaker=make_breaker(tmp_path),
        limits=limits,
    )


def make_core(
    tmp_path: Path,
    client: FakeClient,
    breaker: Breaker,
    recorder: NotionRecorder | None = None,
) -> ConversationCore:
    return ConversationCore(
        store=StateStore(db_path=tmp_path / "conv.sqlite"),
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        recorder=recorder,
        breaker=breaker,
    )


def make_recorder(tmp_path: Path, responses: list[Any]) -> tuple[NotionRecorder, FakeSession]:
    store = StateStore(db_path=tmp_path / "recorder.sqlite")
    session = FakeSession(responses)
    client = NotionClient(session=session, sleep=lambda _s: None, max_retries=0)
    recorder = NotionRecorder(client, DBS, store, permalink=lambda _ts: None)
    return recorder, session


def halt_script(reason: str | None, final_reply: str = "Trading is halted.") -> list[Any]:
    args: dict[str, Any] = {} if reason is None else {"reason": reason}
    return [
        message("tool_use", [tool_use_block("tu_halt", "halt_trading", args)]),
        message("end_turn", [text_block(final_reply)]),
    ]


def resume_script(final_reply: str = "Trading resumed.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_resume", "resume_trading", {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


# --- halt_trading ----------------------------------------------------------------


def test_halt_trading_writes_halt_file_and_confirms(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    client = FakeClient(judgment_messages=halt_script("flash crash, step away"))
    core = make_core(tmp_path, client, breaker)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "halt all trading", say)

    assert breaker.halted
    assert breaker.halt_note == "flash crash, step away"
    assert say.calls[0]["text"] == format_trading_halted(
        "flash crash, step away", breaker.halt_file
    )
    assert say.calls[0]["thread_ts"] == THREAD
    assert reply == "Trading is halted."


def test_halt_trading_without_reason_notes_the_thread(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    client = FakeClient(judgment_messages=halt_script(None))
    core = make_core(tmp_path, client, breaker)

    core.handle_message(THREAD, "halt trading", RecordingSay())

    assert breaker.halted
    assert THREAD in breaker.halt_note


def test_halt_trading_when_already_halted_errors_and_keeps_note(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.halt("original note")
    client = FakeClient(judgment_messages=halt_script("new note", "Already halted."))
    core = make_core(tmp_path, client, breaker)
    say = RecordingSay()

    core.handle_message(THREAD, "halt trading", say)

    assert breaker.halt_note == "original note"  # not overwritten
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "already halted" in tool_result["content"]
    assert "original note" in tool_result["content"]
    # no halt confirmation was posted — only the final reply
    assert [c["text"] for c in say.calls] == ["Already halted."]


# --- resume_trading --------------------------------------------------------------


def test_resume_trading_removes_halt_file_and_confirms(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.halt("maintenance")
    client = FakeClient(judgment_messages=resume_script())
    core = make_core(tmp_path, client, breaker)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "resume trading", say)

    assert not breaker.halted
    assert not breaker.halt_file.exists()
    assert say.calls[0]["text"] == format_trading_resumed(breaker.halt_file)
    assert reply == "Trading resumed."


def test_resume_trading_when_not_halted_errors(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    client = FakeClient(judgment_messages=resume_script("Nothing was halted."))
    core = make_core(tmp_path, client, breaker)
    say = RecordingSay()

    core.handle_message(THREAD, "resume trading", say)

    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "not halted" in tool_result["content"]
    assert [c["text"] for c in say.calls] == ["Nothing was halted."]


# --- Decision Log rows -----------------------------------------------------------


def test_halt_trading_writes_decision_log_row(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    recorder, session = make_recorder(tmp_path, [page_response("page-dec")])
    client = FakeClient(judgment_messages=halt_script("flash crash"))
    core = make_core(tmp_path, client, breaker, recorder=recorder)

    core.handle_message(THREAD, "halt trading", RecordingSay())

    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/pages")
    body = call["json"]
    assert body["parent"] == {"type": "database_id", "database_id": "db-dec"}
    props = body["properties"]
    assert plain_text(props["Decision"], "title") == "Trading halted"
    assert props["Type"] == {"select": {"name": "halt"}}
    details = plain_text(props["Details"], "rich_text")
    assert "flash crash" in details
    assert str(breaker.halt_file) in details


def test_resume_trading_writes_decision_log_row(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.halt("flash crash")
    recorder, session = make_recorder(tmp_path, [page_response("page-dec")])
    client = FakeClient(judgment_messages=resume_script())
    core = make_core(tmp_path, client, breaker, recorder=recorder)

    core.handle_message(THREAD, "resume trading", RecordingSay())

    (call,) = session.calls
    props = call["json"]["properties"]
    assert plain_text(props["Decision"], "title") == "Trading resumed"
    assert props["Type"] == {"select": {"name": "resume"}}
    assert "flash crash" in plain_text(props["Details"], "rich_text")


# --- the tool-written file gates the rebalancer (AC 2) ---------------------------


def test_tool_driven_halt_and_resume_gate_the_rebalancer(
    rebalance_env: Any, tmp_path: Path
) -> None:
    """halt_trading's file makes rebalance exit 0 with no orders; resume restores it."""
    client = FakeClient(
        judgment_messages=[
            *halt_script("flash crash"),
            *resume_script(),
        ]
    )
    core = make_core(tmp_path, client, rebalance_env.breaker)
    say = RecordingSay()

    core.handle_message(THREAD, "halt trading now", say)
    broker = FakeBroker()
    notes: list[str] = []
    assert run(rebalance_env, broker, notes) == 0
    assert "halted" in notes[0]
    assert "flash crash" in notes[0]
    assert broker.session.posts() == []

    core.handle_message(THREAD, "resume trading", say)
    broker2 = FakeBroker()
    notes2: list[str] = []
    assert run(rebalance_env, broker2, notes2) == 0
    assert len(broker2.session.posts()) == 2  # AAPL + MSFT buys submitted again
    assert "breaker: normal (high-water mark $100,000.00)" in notes2[0]


# --- breaker state in the daily summary (AC 3) -----------------------------------


def test_breaker_state_line_halted_carries_the_note(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.halt("weekend maintenance")
    line = breaker_state_line(breaker)
    assert line.startswith("breaker: HALTED")
    assert "weekend maintenance" in line


def test_breaker_state_line_normal_with_high_water_mark(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    assert breaker.check(112_500.0, 0.0) is None
    assert breaker_state_line(breaker) == "breaker: normal (high-water mark $112,500.00)"


def test_breaker_state_line_normal_before_first_clean_pass(tmp_path: Path) -> None:
    assert breaker_state_line(make_breaker(tmp_path)) == (
        "breaker: normal (no high-water mark recorded yet)"
    )


def test_breaker_state_line_reports_corrupt_state_without_raising(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.high_water_mark_file.parent.mkdir(parents=True, exist_ok=True)
    breaker.high_water_mark_file.write_text("not json")
    line = breaker_state_line(breaker)
    assert line.startswith("breaker: STATE ERROR")


def test_daily_summary_includes_breaker_state_line(tmp_path: Path) -> None:
    breaker = make_breaker(tmp_path)
    breaker.halt("weekend maintenance")
    text = format_daily_summary(
        dt.date(2026, 7, 9),
        100_000.0,
        [],
        [],
        no_trade_note="no orders — book already on target",
        breaker_state=breaker_state_line(breaker),
    )
    lines = text.splitlines()
    assert "breaker: HALTED — weekend maintenance (resume trading to lift it)" in lines
    assert lines.index("gate/breaker rejections: none") < lines.index(
        "breaker: HALTED — weekend maintenance (resume trading to lift it)"
    )

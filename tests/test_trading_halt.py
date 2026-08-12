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
    format_live_trading_halted,
    format_live_trading_resumed,
    format_trading_halted,
    format_trading_resumed,
)
from orchestrator.llm import ModelRouter
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.state import StateStore
from tests.test_conversation import THREAD, ChannelSay, RecordingSay, StubLauncher
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
from tests.test_slack_app import CHANNEL, LIVE_CHANNEL


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


# --- US-008: halt_live_trading / resume_live_trading ------------------------------


def make_live_breaker(tmp_path: Path) -> Breaker:
    return Breaker(
        BreakerConfig(max_daily_notional_usd=5_000.0, max_drawdown_pct=5.0),
        halt_file=tmp_path / "breaker-live" / "halt",
        high_water_mark_file=tmp_path / "breaker-live" / "hwm.json",
    )


def make_live_core(
    tmp_path: Path,
    client: FakeClient,
    paper_breaker: Breaker,
    live_breaker: Breaker,
    recorder: NotionRecorder | None = None,
) -> ConversationCore:
    """Core with the live channel armed: paper + live breakers on tmp paths."""
    return ConversationCore(
        store=StateStore(db_path=tmp_path / "conv.sqlite"),
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        recorder=recorder,
        breaker=paper_breaker,
        channel_id=CHANNEL,
        live_channel_id=LIVE_CHANNEL,
        live_breaker=live_breaker,
    )


def live_halt_script(reason: str | None, final_reply: str = "Live halted.") -> list[Any]:
    args: dict[str, Any] = {} if reason is None else {"reason": reason}
    return [
        message("tool_use", [tool_use_block("tu_lhalt", "halt_live_trading", args)]),
        message("end_turn", [text_block(final_reply)]),
    ]


def live_resume_script(final_reply: str = "Live resumed.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_lresume", "resume_live_trading", {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def tool_names(client: FakeClient) -> list[str]:
    return [tool["name"] for tool in client.stream_calls[0]["tools"]]


def tool_error(client: FakeClient) -> dict[str, Any]:
    """The is_error tool_result fed back to the model on the second call."""
    result = client.stream_calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    return result


def test_live_tools_not_registered_without_live_channel(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core = make_core(tmp_path, client, make_breaker(tmp_path))

    core.handle_message(THREAD, "hello", RecordingSay())

    names = tool_names(client)
    assert "halt_live_trading" not in names
    assert "resume_live_trading" not in names


def test_live_tools_registered_when_live_channel_armed(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core = make_live_core(
        tmp_path, client, make_breaker(tmp_path), make_live_breaker(tmp_path)
    )

    core.handle_message(THREAD, "hello", ChannelSay(LIVE_CHANNEL))

    names = tool_names(client)
    assert "halt_live_trading" in names
    assert "resume_live_trading" in names


def test_halt_live_trading_halts_live_and_leaves_paper_file_absent(tmp_path: Path) -> None:
    paper = make_breaker(tmp_path)
    live = make_live_breaker(tmp_path)
    client = FakeClient(judgment_messages=live_halt_script("bad fill quality"))
    core = make_live_core(tmp_path, client, paper, live)
    say = ChannelSay(LIVE_CHANNEL)

    reply = core.handle_message(THREAD, "halt live trading", say)

    assert live.halted
    assert live.halt_note == "bad fill quality"
    assert not paper.halted
    assert not paper.halt_file.exists()  # live halt never touches paper's file
    assert say.calls[0]["text"] == format_live_trading_halted(
        "bad fill quality", live.halt_file
    )
    assert "LIVE" in say.calls[0]["text"]  # unmistakably live wording
    assert say.calls[0]["text"] != format_trading_halted("bad fill quality", live.halt_file)
    assert reply == "Live halted."


def test_paper_halt_leaves_live_file_absent(tmp_path: Path) -> None:
    paper = make_breaker(tmp_path)
    live = make_live_breaker(tmp_path)
    client = FakeClient(judgment_messages=halt_script("flash crash"))
    core = make_live_core(tmp_path, client, paper, live)

    core.handle_message(THREAD, "halt trading", ChannelSay(CHANNEL))

    assert paper.halted
    assert not live.halted
    assert not live.halt_file.exists()  # paper halt never touches live's file


def test_halt_live_trading_refused_from_paper_channel(tmp_path: Path) -> None:
    live = make_live_breaker(tmp_path)
    client = FakeClient(judgment_messages=live_halt_script("x", "Refused."))
    core = make_live_core(tmp_path, client, make_breaker(tmp_path), live)

    core.handle_message(THREAD, "halt live trading", ChannelSay(CHANNEL))

    error = tool_error(client)
    assert LIVE_CHANNEL in error["content"]  # pointer to the live channel
    assert "REAL-MONEY" in error["content"]
    assert not live.halted
    assert not live.halt_file.exists()


def test_halt_live_trading_refused_from_unknown_channel(tmp_path: Path) -> None:
    """Real-money control demands positive channel identification."""
    live = make_live_breaker(tmp_path)
    client = FakeClient(judgment_messages=live_halt_script("x", "Refused."))
    core = make_live_core(tmp_path, client, make_breaker(tmp_path), live)

    core.handle_message(THREAD, "halt live trading", RecordingSay())

    assert LIVE_CHANNEL in tool_error(client)["content"]
    assert not live.halted


def test_paper_halt_trading_refused_from_live_channel(tmp_path: Path) -> None:
    paper = make_breaker(tmp_path)
    client = FakeClient(judgment_messages=halt_script("x", "Refused."))
    core = make_live_core(tmp_path, client, paper, make_live_breaker(tmp_path))

    core.handle_message(THREAD, "halt trading", ChannelSay(LIVE_CHANNEL))

    error = tool_error(client)
    assert CHANNEL in error["content"]  # mirror pointer to the paper channel
    assert "halt_live_trading" in error["content"]
    assert not paper.halted
    assert not paper.halt_file.exists()


def test_paper_resume_trading_refused_from_live_channel(tmp_path: Path) -> None:
    paper = make_breaker(tmp_path)
    paper.halt("paper maintenance")
    client = FakeClient(judgment_messages=resume_script("Refused."))
    core = make_live_core(tmp_path, client, paper, make_live_breaker(tmp_path))

    core.handle_message(THREAD, "resume trading", ChannelSay(LIVE_CHANNEL))

    error = tool_error(client)
    assert "resume_live_trading" in error["content"]
    assert paper.halted  # the paper halt survives the refused call
    assert paper.halt_note == "paper maintenance"


def test_resume_live_trading_clears_live_only(tmp_path: Path) -> None:
    paper = make_breaker(tmp_path)
    paper.halt("paper note")
    live = make_live_breaker(tmp_path)
    live.halt("live note")
    client = FakeClient(judgment_messages=live_resume_script())
    core = make_live_core(tmp_path, client, paper, live)
    say = ChannelSay(LIVE_CHANNEL)

    reply = core.handle_message(THREAD, "resume live trading", say)

    assert not live.halted
    assert not live.halt_file.exists()
    assert paper.halted  # resuming live never lifts the paper halt
    assert say.calls[0]["text"] == format_live_trading_resumed(live.halt_file)
    assert "LIVE" in say.calls[0]["text"]
    assert reply == "Live resumed."


def test_resume_live_trading_refused_from_paper_channel(tmp_path: Path) -> None:
    live = make_live_breaker(tmp_path)
    live.halt("live note")
    client = FakeClient(judgment_messages=live_resume_script("Refused."))
    core = make_live_core(tmp_path, client, make_breaker(tmp_path), live)

    core.handle_message(THREAD, "resume live trading", ChannelSay(CHANNEL))

    assert LIVE_CHANNEL in tool_error(client)["content"]
    assert live.halted  # the live halt survives the refused call


def test_halt_live_trading_when_already_halted_keeps_note(tmp_path: Path) -> None:
    live = make_live_breaker(tmp_path)
    live.halt("original live note")
    client = FakeClient(judgment_messages=live_halt_script("new note", "Already halted."))
    core = make_live_core(tmp_path, client, make_breaker(tmp_path), live)

    core.handle_message(THREAD, "halt live trading", ChannelSay(LIVE_CHANNEL))

    assert live.halt_note == "original live note"  # not overwritten
    error = tool_error(client)
    assert "already halted" in error["content"]
    assert "resume_live_trading" in error["content"]


def test_resume_live_trading_when_not_halted_errors(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=live_resume_script("Nothing halted."))
    core = make_live_core(
        tmp_path, client, make_breaker(tmp_path), make_live_breaker(tmp_path)
    )

    core.handle_message(THREAD, "resume live trading", ChannelSay(LIVE_CHANNEL))

    assert "not halted" in tool_error(client)["content"]


def test_halt_live_trading_writes_decision_log_row(tmp_path: Path) -> None:
    live = make_live_breaker(tmp_path)
    recorder, session = make_recorder(tmp_path, [page_response("page-dec")])
    client = FakeClient(judgment_messages=live_halt_script("bad fill quality"))
    core = make_live_core(
        tmp_path, client, make_breaker(tmp_path), live, recorder=recorder
    )

    core.handle_message(THREAD, "halt live trading", ChannelSay(LIVE_CHANNEL))

    (call,) = session.calls
    props = call["json"]["properties"]
    assert plain_text(props["Decision"], "title") == "LIVE trading halted"
    assert props["Type"] == {"select": {"name": "halt_live"}}
    details = plain_text(props["Details"], "rich_text")
    assert "bad fill quality" in details
    assert str(live.halt_file) in details


def test_resume_live_trading_writes_decision_log_row(tmp_path: Path) -> None:
    live = make_live_breaker(tmp_path)
    live.halt("bad fill quality")
    recorder, session = make_recorder(tmp_path, [page_response("page-dec")])
    client = FakeClient(judgment_messages=live_resume_script())
    core = make_live_core(
        tmp_path, client, make_breaker(tmp_path), live, recorder=recorder
    )

    core.handle_message(THREAD, "resume live trading", ChannelSay(LIVE_CHANNEL))

    (call,) = session.calls
    props = call["json"]["properties"]
    assert plain_text(props["Decision"], "title") == "LIVE trading resumed"
    assert props["Type"] == {"select": {"name": "resume_live"}}
    assert "bad fill quality" in plain_text(props["Details"], "rich_text")

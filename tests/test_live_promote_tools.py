"""US-010: promote_to_live / demote_live Slack tools — one message, no confirmation.

The tools are thin wrappers over the US-009 LivePromotion backend (tested in
tests/test_live_promotion.py), so these tests drive the REAL ConversationCore
with FakeClient scripts over a FAKE backend and recording says: the arm path
posts the armed summary, every refusal path writes nothing, demote posts what
was demoted. No network anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from execution import breaker as breaker_module
from execution import order_gate
from execution.breaker import load_breaker_config
from execution.order_gate import load_limits
from orchestrator.conversation import (
    LIVE_REBALANCE_SCHEDULE,
    ConversationCore,
    format_live_armed,
    format_live_demoted,
)
from orchestrator.llm import ModelRouter
from orchestrator.promotion import LivePromotionError, LivePromotionResult
from orchestrator.state import PromotedStrategy, StateStore
from tests.test_conversation import THREAD, ChannelSay, RecordingSay, StubLauncher
from tests.test_llm import FakeClient, message, text_block, tool_use_block
from tests.test_slack_app import CHANNEL, LIVE_CHANNEL
from tests.test_trading_halt import (
    make_breaker,
    make_live_breaker,
    tool_error,
    tool_names,
)

LIVE_LIMITS = load_limits(order_gate.LIVE_LIMITS_PATH)
LIVE_BREAKER_CONFIG = load_breaker_config(breaker_module.LIVE_CONFIG_PATH)

WORKSPACE = "/home/x/rdq-runs/workspaces/e05ad9b4"


def make_promoted(**config_overrides: Any) -> PromotedStrategy:
    config: dict[str, Any] = {
        "universe": "us_liquid",
        "universe_tickers": ["AAPL", "MSFT", "NVDA"],
        "topk": 2,
        "n_drop": 1,
        "thread_ts": "1751000000.000200",
        "session_path": "/home/x/rdq-runs/traces/abc123",
        "live_equity_allocation_pct": 10.0,
    }
    config.update(config_overrides)
    return PromotedStrategy(
        workspace_path=WORKSPACE, config=config, promoted_at="2026-08-12T21:00:00"
    )


def make_result(**overrides: Any) -> LivePromotionResult:
    fields: dict[str, Any] = {
        "promoted": make_promoted(),
        "source": "run",
        "source_thread_ts": "1751000000.000200",
        "metrics": {"IC": 0.0512, "1day.excess_return_with_cost.annualized_return": 0.148},
        "sharpe": 1.31,
        "universe_label": None,
        "warnings": (),
        "replaced": None,
    }
    fields.update(overrides)
    return LivePromotionResult(**fields)


class FakeLivePromotions:
    """LivePromotionManager stub recording calls; scriptable success/refusal."""

    def __init__(
        self,
        result: LivePromotionResult | None = None,
        error: Exception | None = None,
        demoted: PromotedStrategy | None = None,
        demote_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.demoted = demoted
        self.demote_error = demote_error
        self.promote_calls: list[dict[str, Any]] = []
        self.demote_calls: list[str | None] = []

    def promote(
        self,
        reference: str | None = None,
        thread_ts: str | None = None,
        trigger_permalink: str | None = None,
    ) -> LivePromotionResult:
        self.promote_calls.append(
            {
                "reference": reference,
                "thread_ts": thread_ts,
                "trigger_permalink": trigger_permalink,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result

    def demote(self, trigger_permalink: str | None = None) -> PromotedStrategy:
        self.demote_calls.append(trigger_permalink)
        if self.demote_error is not None:
            raise self.demote_error
        assert self.demoted is not None
        return self.demoted


def make_core(
    tmp_path: Path,
    client: FakeClient,
    live_promotions: FakeLivePromotions | None,
    live_breaker: Any = None,
    permalink: Any = None,
    live_channel_id: str | None = LIVE_CHANNEL,
) -> ConversationCore:
    return ConversationCore(
        store=StateStore(db_path=tmp_path / "conv.sqlite"),
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        breaker=make_breaker(tmp_path),
        channel_id=CHANNEL,
        live_channel_id=live_channel_id,
        live_breaker=live_breaker
        if live_breaker is not None
        else make_live_breaker(tmp_path),
        live_promotions=live_promotions,
        permalink=permalink,
    )


def promote_script(
    args: dict[str, Any] | None = None, final_reply: str = "Armed."
) -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_plive", "promote_to_live", args or {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def demote_script(final_reply: str = "Demoted.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_dlive", "demote_live", {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


# --- registration -----------------------------------------------------------------


def test_tools_not_registered_without_live_channel(tmp_path: Path) -> None:
    """AC: the core refuses these tools entirely when live_channel_id is unset."""
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core = make_core(
        tmp_path,
        client,
        FakeLivePromotions(result=make_result()),
        live_channel_id=None,
    )

    core.handle_message(THREAD, "hello", RecordingSay())

    names = tool_names(client)
    assert "promote_to_live" not in names
    assert "demote_live" not in names


def test_tools_registered_when_live_channel_and_backend_wired(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core = make_core(tmp_path, client, FakeLivePromotions(result=make_result()))

    core.handle_message(THREAD, "hello", ChannelSay(LIVE_CHANNEL))

    names = tool_names(client)
    assert "promote_to_live" in names
    assert "demote_live" in names


def test_tools_not_registered_without_backend(tmp_path: Path) -> None:
    """Partial wiring (live channel armed, no backend) leaves the tools out."""
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core = make_core(tmp_path, client, live_promotions=None)

    core.handle_message(THREAD, "hello", ChannelSay(LIVE_CHANNEL))

    names = tool_names(client)
    assert "promote_to_live" not in names
    assert "demote_live" not in names
    assert "halt_live_trading" in names  # the breaker tools stay


# --- the arm path ------------------------------------------------------------------


def test_promote_to_live_arms_immediately_and_posts_summary(tmp_path: Path) -> None:
    result = make_result()
    backend = FakeLivePromotions(result=result)
    client = FakeClient(judgment_messages=promote_script())
    core = make_core(tmp_path, client, backend)
    say = ChannelSay(LIVE_CHANNEL)

    reply = core.handle_message(THREAD, "promote to live", say)

    # One tool call → one backend write; nothing blocked on a confirmation.
    assert len(backend.promote_calls) == 1
    assert backend.promote_calls[0]["reference"] is None
    assert backend.promote_calls[0]["thread_ts"] == THREAD
    assert say.calls[0]["text"] == format_live_armed(
        result, LIVE_LIMITS, LIVE_BREAKER_CONFIG
    )
    assert say.calls[0]["thread_ts"] == THREAD
    assert reply == "Armed."


def test_armed_summary_restates_everything_in_force() -> None:
    text = format_live_armed(make_result(), LIVE_LIMITS, LIVE_BREAKER_CONFIG)
    assert "LIVE trading is ARMED" in text
    assert "`e05ad9b4`" in text  # workspace id
    assert "direct run promotion (thread 1751000000.000200)" in text  # source
    assert "us_liquid (3 tickers pinned)" in text  # universe + ticker count
    assert "IC 0.0512" in text and "ARR 0.1480" in text  # headline metrics
    assert "Sharpe 1.3100" in text
    assert "10% of live equity" in text  # allocation pct
    # live limits and breaker values in force (committed live config numbers)
    assert "$500/order" in text
    assert "10% max position" in text
    assert "60 orders/day" in text
    assert "60 positions max" in text
    assert "$5,000 daily notional" in text
    assert "5% drawdown" in text
    assert LIVE_REBALANCE_SCHEDULE in text  # when the next live rebalance fires


def test_armed_summary_paper_source_mismatch_warnings_and_replaced() -> None:
    result = make_result(
        source="paper",
        source_thread_ts=None,
        metrics={},
        sharpe=None,
        universe_label="ai_semis",
        warnings=("an existing conf_pred_refresh.yaml was regenerated",),
        replaced=make_promoted(),
    )
    text = format_live_armed(result, LIVE_LIMITS, LIVE_BREAKER_CONFIG)
    assert "copy of the paper-promoted strategy" in text
    assert "metrics n/a" in text
    assert "labeled `ai_semis`" in text  # label-vs-conf mismatch call-out
    assert "trades live is `us_liquid`" in text
    assert ":warning: an existing conf_pred_refresh.yaml was regenerated" in text
    assert "*Replaced:* `e05ad9b4` (promoted 2026-08-12T21:00:00)" in text


def test_promote_to_live_passes_run_reference(tmp_path: Path) -> None:
    backend = FakeLivePromotions(result=make_result())
    client = FakeClient(
        judgment_messages=promote_script({"run_reference": "1751000000.000200"})
    )
    core = make_core(tmp_path, client, backend)

    core.handle_message(THREAD, "promote run 1751000000.000200 to live", ChannelSay(LIVE_CHANNEL))

    assert backend.promote_calls[0]["reference"] == "1751000000.000200"


def test_promote_to_live_resolves_live_channel_permalink(tmp_path: Path) -> None:
    backend = FakeLivePromotions(result=make_result())
    seen: list[tuple[str, str]] = []

    def permalink(channel: str, message_ts: str) -> str:
        seen.append((channel, message_ts))
        return "https://slack.example/p1"

    client = FakeClient(judgment_messages=promote_script())
    core = make_core(tmp_path, client, backend, permalink=permalink)

    core.handle_message(THREAD, "promote to live", ChannelSay(LIVE_CHANNEL))

    assert seen == [(LIVE_CHANNEL, THREAD)]
    assert backend.promote_calls[0]["trigger_permalink"] == "https://slack.example/p1"


def test_permalink_failure_never_blocks_arming(tmp_path: Path) -> None:
    backend = FakeLivePromotions(result=make_result())

    def permalink(channel: str, message_ts: str) -> str:
        raise RuntimeError("slack api down")

    client = FakeClient(judgment_messages=promote_script())
    core = make_core(tmp_path, client, backend, permalink=permalink)
    say = ChannelSay(LIVE_CHANNEL)

    core.handle_message(THREAD, "promote to live", say)

    assert backend.promote_calls[0]["trigger_permalink"] is None
    assert "LIVE trading is ARMED" in say.calls[0]["text"]


# --- refusal paths (each writes nothing) --------------------------------------------


def test_promote_refused_from_paper_channel(tmp_path: Path) -> None:
    backend = FakeLivePromotions(result=make_result())
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)
    say = ChannelSay(CHANNEL)

    core.handle_message(THREAD, "promote to live", say)

    error = tool_error(client)
    assert LIVE_CHANNEL in error["content"]  # pointer to the live channel
    assert backend.promote_calls == []  # nothing written
    assert [c["text"] for c in say.calls] == ["Refused."]


def test_promote_refused_from_unknown_channel(tmp_path: Path) -> None:
    """Real-money arming demands positive channel identification."""
    backend = FakeLivePromotions(result=make_result())
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)

    core.handle_message(THREAD, "promote to live", RecordingSay())

    assert LIVE_CHANNEL in tool_error(client)["content"]
    assert backend.promote_calls == []


def test_promote_refused_when_live_breaker_halted(tmp_path: Path) -> None:
    backend = FakeLivePromotions(result=make_result())
    live = make_live_breaker(tmp_path)
    live.halt("bad fill quality")
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend, live_breaker=live)
    say = ChannelSay(LIVE_CHANNEL)

    core.handle_message(THREAD, "promote to live", say)

    error = tool_error(client)
    assert "halted" in error["content"]
    assert "bad fill quality" in error["content"]
    assert "resume_live_trading" in error["content"]
    assert backend.promote_calls == []
    assert [c["text"] for c in say.calls] == ["Refused."]


def test_promote_refused_when_live_limits_config_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "limits.live.json"
    bad.write_text("{not json")
    monkeypatch.setattr(order_gate, "LIVE_LIMITS_PATH", bad)
    backend = FakeLivePromotions(result=make_result())
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)

    core.handle_message(THREAD, "promote to live", ChannelSay(LIVE_CHANNEL))

    error = tool_error(client)
    assert "guardrail config is missing or malformed" in error["content"]
    assert backend.promote_calls == []


def test_promote_refused_when_live_breaker_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        breaker_module, "LIVE_CONFIG_PATH", tmp_path / "missing" / "breaker.live.json"
    )
    backend = FakeLivePromotions(result=make_result())
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)

    core.handle_message(THREAD, "promote to live", ChannelSay(LIVE_CHANNEL))

    assert "guardrail config is missing or malformed" in tool_error(client)["content"]
    assert backend.promote_calls == []


def test_backend_refusal_is_relayed_and_nothing_posted(tmp_path: Path) -> None:
    """Non-promotable status / missing artifacts refuse inside the backend;
    the tool relays the reason and posts no armed summary."""
    backend = FakeLivePromotions(
        error=LivePromotionError(
            "run 1751000000.000200 is 'running' — only a completed or"
            " operator-stopped run can be promoted to live"
        )
    )
    client = FakeClient(judgment_messages=promote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)
    say = ChannelSay(LIVE_CHANNEL)

    core.handle_message(THREAD, "promote to live", say)

    error = tool_error(client)
    assert "is 'running'" in error["content"]
    assert [c["text"] for c in say.calls] == ["Refused."]  # no armed summary


# --- demote_live --------------------------------------------------------------------


def test_demote_live_clears_slot_and_posts_notice(tmp_path: Path) -> None:
    backend = FakeLivePromotions(demoted=make_promoted())
    client = FakeClient(judgment_messages=demote_script())
    core = make_core(tmp_path, client, backend)
    say = ChannelSay(LIVE_CHANNEL)

    reply = core.handle_message(THREAD, "demote live", say)

    assert backend.demote_calls == [None]
    text = say.calls[0]["text"]
    assert text == format_live_demoted(make_promoted())
    assert "`e05ad9b4`" in text  # what was demoted
    assert "abort with no promoted strategy" in text  # next-rebalance behavior
    assert reply == "Demoted."


def test_demote_live_refused_when_slot_empty(tmp_path: Path) -> None:
    backend = FakeLivePromotions(
        demote_error=LivePromotionError("the live slot is already empty — nothing to demote")
    )
    client = FakeClient(judgment_messages=demote_script(final_reply="Nothing to demote."))
    core = make_core(tmp_path, client, backend)
    say = ChannelSay(LIVE_CHANNEL)

    core.handle_message(THREAD, "demote live", say)

    assert "already empty" in tool_error(client)["content"]
    assert [c["text"] for c in say.calls] == ["Nothing to demote."]


def test_demote_live_refused_from_paper_channel(tmp_path: Path) -> None:
    backend = FakeLivePromotions(demoted=make_promoted())
    client = FakeClient(judgment_messages=demote_script(final_reply="Refused."))
    core = make_core(tmp_path, client, backend)

    core.handle_message(THREAD, "demote live", ChannelSay(CHANNEL))

    assert LIVE_CHANNEL in tool_error(client)["content"]
    assert backend.demote_calls == []


def test_demote_live_passes_permalink(tmp_path: Path) -> None:
    backend = FakeLivePromotions(demoted=make_promoted())
    client = FakeClient(judgment_messages=demote_script())
    core = make_core(
        tmp_path,
        client,
        backend,
        permalink=lambda _channel, _ts: "https://slack.example/p2",
    )

    core.handle_message(THREAD, "demote live", ChannelSay(LIVE_CHANNEL))

    assert backend.demote_calls == ["https://slack.example/p2"]

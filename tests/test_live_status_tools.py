"""US-012: live read-only status tools (check_live_account/orders/pnl).

Drives the REAL ConversationCore with FakeClient scripts over a
StubLiveStatus, mirroring tests/test_broker_tools.py. The live tools answer
from the Live Notion databases + orchestrator state (the US-053 read-path
decision, docs/decisions.md) — a source-grep test pins that orchestrator/
can never construct a live-host broker client.

LiveStatusReader itself is proven over the real NotionClient +
tests/test_notion_client.py's FakeSession, the established Notion testing
pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orchestrator
from execution.breaker import Breaker
from orchestrator.conversation import (
    ConversationCore,
    format_live_account_report,
    format_live_orders_report,
    format_live_pnl_report,
)
from orchestrator.live_status import LiveOrder, LiveSnapshot, LiveStatusReader
from orchestrator.llm import ModelRouter
from orchestrator.state import PromotedStrategy, StateStore
from tests.test_broker_tools import StubBroker, tool_result, tool_script
from tests.test_conversation import THREAD, ChannelSay, RecordingSay, StubLauncher
from tests.test_llm import FakeClient, message, text_block
from tests.test_notion_client import make_client, query_result
from tests.test_slack_app import CHANNEL, LIVE_CHANNEL
from tests.test_trading_halt import make_breaker, make_live_breaker, tool_error, tool_names

SNAPSHOT = LiveSnapshot(
    date="2026-08-14",
    equity=5_023.5,
    cash=1_523.5,
    positions=6.0,
    orders_placed=4.0,
    orders_filled=4.0,
    outcome="traded",
    day_pl=12.4,
    day_pl_pct=0.0025,
    pl_day="2026-08-13",
    breaker="normal (high-water mark $5,023.50)",
    notes="",
)

ORDERS = [
    LiveOrder(
        title="2026-08-14 BUY 2 MSFT",
        symbol="MSFT",
        side="buy",
        status="filled",
        qty=2.0,
        filled_qty=2.0,
        limit_price=402.0,
        filled_avg_price=401.5,
        submitted_at="2026-08-14T12:10:00.000+00:00",
        notes="",
    ),
    LiveOrder(
        title="2026-08-14 SELL 1 AAPL",
        symbol="AAPL",
        side="sell",
        status="submitted",
        qty=1.0,
        filled_qty=0.0,
        limit_price=201.0,
        filled_avg_price=None,
        submitted_at="2026-08-14T12:10:00.000+00:00",
        notes="",
    ),
]

# Newest first, like the reader returns them.
HISTORY = [
    SNAPSHOT,
    LiveSnapshot(
        date="2026-08-13",
        equity=5_011.1,
        cash=1_511.1,
        positions=6.0,
        orders_placed=2.0,
        orders_filled=2.0,
        outcome="traded",
        day_pl=11.1,
        day_pl_pct=0.0022,
        pl_day="2026-08-12",
        breaker="",
        notes="",
    ),
    LiveSnapshot(
        date="2026-08-12",
        equity=5_000.0,
        cash=1_500.0,
        positions=6.0,
        orders_placed=6.0,
        orders_filled=6.0,
        outcome="traded",
        day_pl=None,
        day_pl_pct=None,
        pl_day=None,
        breaker="",
        notes="first live day",
    ),
]


class StubLiveStatus:
    """Records read calls; serves canned Live-Notion-derived rows."""

    def __init__(
        self,
        snapshot: LiveSnapshot | None = None,
        history: list[LiveSnapshot] | None = None,
        orders: list[LiveOrder] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._snapshot = snapshot
        self._history = history or []
        self._orders = orders or []

    def latest_snapshot(self) -> LiveSnapshot | None:
        self.calls.append({"method": "latest_snapshot"})
        return self._snapshot

    def snapshot_history(self, limit: int = 30) -> list[LiveSnapshot]:
        self.calls.append({"method": "snapshot_history", "limit": limit})
        return self._history[:limit]

    def recent_orders(self, limit: int = 10) -> list[LiveOrder]:
        self.calls.append({"method": "recent_orders", "limit": limit})
        return self._orders[:limit]


def make_core(
    tmp_path: Path,
    client: FakeClient,
    live_status: StubLiveStatus | None,
    broker: StubBroker | None = None,
    live_breaker: Breaker | None = None,
    live_channel_id: str | None = LIVE_CHANNEL,
) -> tuple[ConversationCore, StateStore]:
    store = StateStore(db_path=tmp_path / "conv.sqlite")
    core = ConversationCore(
        store=store,
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        breaker=make_breaker(tmp_path),
        broker=broker or StubBroker(),
        channel_id=CHANNEL,
        live_channel_id=live_channel_id,
        live_breaker=live_breaker
        or (make_live_breaker(tmp_path) if live_channel_id else None),
        live_status=live_status,
    )
    return core, store


# --------------------------------------------------------------- registration


def test_live_check_tools_not_registered_without_live_channel(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core, _ = make_core(tmp_path, client, StubLiveStatus(), live_channel_id=None)

    core.handle_message(THREAD, "hello", RecordingSay())

    names = tool_names(client)
    assert "check_live_account" not in names
    assert "check_live_orders" not in names
    assert "check_live_pnl" not in names


def test_live_check_tools_not_registered_without_live_status(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core, _ = make_core(tmp_path, client, live_status=None)

    core.handle_message(THREAD, "hello", ChannelSay(LIVE_CHANNEL))

    names = tool_names(client)
    assert "halt_live_trading" in names  # live channel armed...
    assert "check_live_account" not in names  # ...but no status source wired


def test_live_check_tools_registered_alongside_paper_ones(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("hi")])])
    core, _ = make_core(tmp_path, client, StubLiveStatus())

    core.handle_message(THREAD, "hello", ChannelSay(LIVE_CHANNEL))

    names = tool_names(client)
    assert {"check_live_account", "check_live_orders", "check_live_pnl"} <= set(names)
    # Paper tools stay registered (their handler refuses in the live channel).
    assert {"check_account", "check_orders", "check_pnl"} <= set(names)


# ------------------------------------------------------------------ account


def test_check_live_account_reports_slot_snapshot_and_breaker(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_account", {}))
    status = StubLiveStatus(snapshot=SNAPSHOT)
    core, store = make_core(tmp_path, client, status)
    store.set_promoted_strategy("/ws/paper_abc123", {"universe": "us_liquid"})
    store.set_promoted_strategy_live(
        "/ws/live_def456",
        {"universe": "us_liquid_promoted_30", "live_equity_allocation_pct": 10.0},
    )

    core.handle_message(THREAD, "how is the live account?", ChannelSay(LIVE_CHANNEL))

    assert status.calls == [{"method": "latest_snapshot"}]
    result = tool_result(client)
    assert "is_error" not in result
    report = result["content"]
    assert "LIVE account (real money)" in report
    assert "live strategy: live_def456" in report
    assert "10% equity allocation" in report
    # Differing slots name BOTH workspaces.
    assert "paper trades paper_abc123, live trades live_def456" in report
    assert "live trading: active" in report
    assert "equity $5,023.50, cash $1,523.50, 6 positions" in report
    assert "day outcome: traded (4 orders placed, 4 filled)" in report
    assert "previous completed day (2026-08-13): +$12.40 (+0.25%)" in report
    assert "breaker at snapshot time: normal" in report


def test_check_live_account_same_workspace_notes_parity(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_account", {}))
    core, store = make_core(tmp_path, client, StubLiveStatus(snapshot=SNAPSHOT))
    store.set_promoted_strategy("/ws/shared_ws1", {"universe": "us_liquid"})
    store.set_promoted_strategy_live(
        "/ws/shared_ws1",
        {"universe": "us_liquid", "live_equity_allocation_pct": 10.0},
    )

    core.handle_message(THREAD, "live account?", ChannelSay(LIVE_CHANNEL))

    report = tool_result(client)["content"]
    assert "paper slot: same strategy as live" in report
    assert "differs" not in report


def test_check_live_account_reports_live_halt(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_account", {}))
    live_breaker = make_live_breaker(tmp_path)
    live_breaker.halt("fat finger")
    core, _ = make_core(
        tmp_path, client, StubLiveStatus(snapshot=SNAPSHOT), live_breaker=live_breaker
    )

    core.handle_message(THREAD, "live status?", ChannelSay(LIVE_CHANNEL))

    report = tool_result(client)["content"]
    assert "live trading: HALTED — fat finger (resume_live_trading lifts it)" in report


def test_check_live_account_before_any_live_data(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_account", {}))
    core, _ = make_core(tmp_path, client, StubLiveStatus(snapshot=None))

    core.handle_message(THREAD, "live account?", ChannelSay(LIVE_CHANNEL))

    result = tool_result(client)
    assert "is_error" not in result  # graceful, not an error
    report = result["content"]
    assert "no live data yet" in report
    assert "live strategy: none promoted" in report


# ------------------------------------------------------------------- orders


def test_check_live_orders_formats_ledger_rows(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_orders", {"limit": 5}))
    status = StubLiveStatus(orders=ORDERS)
    core, _ = make_core(tmp_path, client, status)

    core.handle_message(THREAD, "did live trade?", ChannelSay(LIVE_CHANNEL))

    assert status.calls == [{"method": "recent_orders", "limit": 5}]
    report = tool_result(client)["content"]
    assert "2 live order(s) from the Trade Ledger (Live)" in report
    assert "buy 2 MSFT @ limit $402.00 — filled 2 @ $401.50" in report
    assert "sell 1 AAPL @ limit $201.00 — submitted" in report


def test_check_live_orders_clamps_limit(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_orders", {"limit": 999}))
    status = StubLiveStatus(orders=ORDERS)
    core, _ = make_core(tmp_path, client, status)

    core.handle_message(THREAD, "orders?", ChannelSay(LIVE_CHANNEL))

    assert status.calls == [{"method": "recent_orders", "limit": 50}]


def test_check_live_orders_before_any_live_data(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_orders", {}))
    core, _ = make_core(tmp_path, client, StubLiveStatus())

    core.handle_message(THREAD, "live orders?", ChannelSay(LIVE_CHANNEL))

    result = tool_result(client)
    assert "is_error" not in result
    assert "no live data yet" in result["content"]


# --------------------------------------------------------------------- pnl


def test_check_live_pnl_reports_period_totals(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_pnl", {"days": 7}))
    status = StubLiveStatus(history=HISTORY)
    core, _ = make_core(tmp_path, client, status)

    core.handle_message(THREAD, "how is live doing?", ChannelSay(LIVE_CHANNEL))

    assert status.calls == [{"method": "snapshot_history", "limit": 7}]
    report = tool_result(client)["content"]
    assert "live P/L over the last 3 recorded rebalance day(s)" in report
    assert "period total: +$23.50 (+0.47%) — equity $5,000.00 -> $5,023.50" in report
    assert "2026-08-14: equity $5,023.50, prev-day +$12.40 (+0.25%)" in report
    assert "2026-08-12: equity $5,000.00, prev-day P/L n/a" in report


def test_check_live_pnl_before_any_live_data(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_pnl", {}))
    core, _ = make_core(tmp_path, client, StubLiveStatus())

    core.handle_message(THREAD, "live pnl?", ChannelSay(LIVE_CHANNEL))

    result = tool_result(client)
    assert "is_error" not in result
    assert "no live data yet" in result["content"]


# ------------------------------------------------------------ channel gates


def test_check_live_account_refused_from_paper_channel(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_live_account", {}))
    status = StubLiveStatus(snapshot=SNAPSHOT)
    core, _ = make_core(tmp_path, client, status)

    core.handle_message(THREAD, "live account?", ChannelSay(CHANNEL))

    error = tool_error(client)
    assert LIVE_CHANNEL in error["content"]  # pointer to the live channel
    assert "REAL-MONEY" in error["content"]
    assert "check_account" in error["content"]  # the paper reads hint
    assert status.calls == []


def test_check_live_tools_refused_from_unknown_channel(tmp_path: Path) -> None:
    """Even reads about the real-money account demand positive channel ID."""
    client = FakeClient(judgment_messages=tool_script("check_live_pnl", {}))
    status = StubLiveStatus(history=HISTORY)
    core, _ = make_core(tmp_path, client, status)

    core.handle_message(THREAD, "live pnl?", RecordingSay())

    assert LIVE_CHANNEL in tool_error(client)["content"]
    assert status.calls == []


def test_paper_check_account_refused_from_live_channel(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=tool_script("check_account", {}))
    broker = StubBroker()
    core, _ = make_core(tmp_path, client, StubLiveStatus(), broker=broker)

    core.handle_message(THREAD, "check the account", ChannelSay(LIVE_CHANNEL))

    error = tool_error(client)
    assert "check_live_account" in error["content"]  # pointer to the live twin
    assert CHANNEL in error["content"]
    assert broker.calls == []  # the paper broker was never touched


def test_paper_check_tools_unchanged_in_paper_and_unknown_channels(tmp_path: Path) -> None:
    """Paper freeze: with live armed, paper reads still work outside the live channel."""
    for say in (ChannelSay(CHANNEL), RecordingSay()):
        client = FakeClient(judgment_messages=tool_script("check_account", {}))
        broker = StubBroker()
        core, _ = make_core(tmp_path, client, StubLiveStatus(), broker=broker)

        core.handle_message(THREAD, "account?", say)

        assert [c["method"] for c in broker.calls] == ["get_account", "get_positions"]
        assert "equity: $101,250.50" in tool_result(client)["content"]


# ------------------------------------------------------------- formatters


def test_live_account_report_without_optional_snapshot_fields() -> None:
    sparse = LiveSnapshot(
        date=None, equity=None, cash=None, positions=None, orders_placed=None,
        orders_filled=None, outcome=None, day_pl=None, day_pl_pct=None,
        pl_day=None, breaker="", notes="",
    )
    report = format_live_account_report(sparse, None, None, "active")
    assert "latest live snapshot (undated): equity n/a, cash n/a, n/a positions" in report
    assert "day outcome: unknown (? orders placed, ? filled)" in report
    assert "previous completed day" not in report
    assert "breaker at snapshot time" not in report


def test_live_account_report_live_only_slot_names_no_paper() -> None:
    live = PromotedStrategy(
        workspace_path="/ws/live_only1",
        config={"universe": "us_liquid", "live_equity_allocation_pct": 10.0},
        promoted_at="2026-08-14T00:00:00Z",
    )
    report = format_live_account_report(None, live, None, "active")
    assert "live strategy: live_only1" in report
    assert "paper slot" not in report  # nothing to compare against


def test_live_orders_report_row_without_submitted_at() -> None:
    bare = LiveOrder(
        title="2026-08-14 BUY 3 NVDA", symbol="NVDA", side="buy", status="filled",
        qty=None, filled_qty=3.0, limit_price=None, filled_avg_price=100.0,
        submitted_at=None, notes="",
    )
    report = format_live_orders_report([bare])
    assert "unknown time buy 3 NVDA — filled 3 @ $100.00" in report


def test_live_pnl_report_without_equity_values() -> None:
    assert "no live data yet" in format_live_pnl_report([])


# --------------------------------------------------------- LiveStatusReader


def snapshot_page(date: str, equity: float) -> dict[str, Any]:
    """A realistic Account Snapshots (Live) query result page."""
    return {
        "object": "page",
        "id": f"page-{date}",
        "properties": {
            "Snapshot": {"title": [{"plain_text": f"{date} — equity ${equity:,.2f}"}]},
            "Date": {"date": {"start": date}},
            "Equity": {"number": equity},
            "Cash": {"number": 1_500.0},
            "Positions": {"number": 6},
            "Orders Placed": {"number": 4},
            "Orders Filled": {"number": 4},
            "Outcome": {"select": {"name": "traded"}},
            "Day P/L": {"number": 12.4},
            "Day P/L %": {"number": 0.0025},
            "P/L Day": {"date": {"start": "2026-08-13"}},
            "Breaker": {"rich_text": [{"plain_text": "normal"}]},
            "Notes": {"rich_text": []},
        },
    }


def ledger_page() -> dict[str, Any]:
    """A realistic Trade Ledger (Live) query result page (text.content shape)."""
    return {
        "object": "page",
        "id": "page-ord",
        "properties": {
            "Order": {"title": [{"text": {"content": "2026-08-14 BUY 2 MSFT"}}]},
            "Symbol": {"rich_text": [{"text": {"content": "MSFT"}}]},
            "Side": {"select": {"name": "buy"}},
            "Status": {"select": {"name": "filled"}},
            "Qty": {"number": 2},
            "Filled Qty": {"number": 2},
            "Limit Price": {"number": 402.0},
            "Filled Avg Price": {"number": 401.5},
            "Submitted At": {"date": {"start": "2026-08-14T12:10:00.000+00:00"}},
        },
    }


def test_reader_without_database_ids_reads_nothing() -> None:
    notion, session, _ = make_client([])  # any request would exhaust the queue
    reader = LiveStatusReader(notion, snapshots_db=None, ledger_db=None)

    assert reader.latest_snapshot() is None
    assert reader.snapshot_history() == []
    assert reader.recent_orders() == []
    assert session.calls == []


def test_reader_snapshot_history_queries_newest_first_and_parses() -> None:
    notion, session, _ = make_client(
        [query_result([snapshot_page("2026-08-14", 5_023.5)])]
    )
    reader = LiveStatusReader(notion, snapshots_db="db-snap", ledger_db=None)

    (snapshot,) = reader.snapshot_history(limit=5)

    (call,) = session.calls
    assert call["url"].endswith("/v1/databases/db-snap/query")
    assert call["json"]["sorts"] == [{"property": "Date", "direction": "descending"}]
    assert call["json"]["page_size"] == 5
    assert snapshot == LiveSnapshot(
        date="2026-08-14", equity=5_023.5, cash=1_500.0, positions=6.0,
        orders_placed=4.0, orders_filled=4.0, outcome="traded", day_pl=12.4,
        day_pl_pct=0.0025, pl_day="2026-08-13", breaker="normal", notes="",
    )


def test_reader_latest_snapshot_fetches_a_single_row() -> None:
    notion, session, _ = make_client(
        [query_result([snapshot_page("2026-08-14", 5_023.5)], has_more=True, cursor="c2")]
    )
    reader = LiveStatusReader(notion, snapshots_db="db-snap", ledger_db=None)

    snapshot = reader.latest_snapshot()

    assert snapshot is not None and snapshot.date == "2026-08-14"
    (call,) = session.calls  # max_results=1 stopped pagination despite has_more
    assert call["json"]["page_size"] == 1


def test_reader_recent_orders_sorts_by_created_time_and_parses() -> None:
    notion, session, _ = make_client([query_result([ledger_page()])])
    reader = LiveStatusReader(notion, snapshots_db=None, ledger_db="db-ledger")

    (order,) = reader.recent_orders(limit=10)

    (call,) = session.calls
    assert call["url"].endswith("/v1/databases/db-ledger/query")
    assert call["json"]["sorts"] == [
        {"timestamp": "created_time", "direction": "descending"}
    ]
    assert order == LiveOrder(
        title="2026-08-14 BUY 2 MSFT", symbol="MSFT", side="buy", status="filled",
        qty=2.0, filled_qty=2.0, limit_price=402.0, filled_avg_price=401.5,
        submitted_at="2026-08-14T12:10:00.000+00:00", notes="",
    )


# ------------------------------------------------------------- source grep


def test_orchestrator_source_never_touches_the_live_broker_host() -> None:
    """The US-053 read-path decision, pinned: no live-host client in orchestrator/.

    'paper-api.alpaca.markets' (the paper host) is stripped before the grep;
    what must never appear is the bare live host or the opt-in flag that
    unlocks it.
    """
    for path in sorted(Path(orchestrator.__file__).parent.glob("*.py")):
        source = path.read_text().replace("paper-api.alpaca.markets", "")
        assert "api.alpaca.markets" not in source, path
        assert "allow_live" not in source, path

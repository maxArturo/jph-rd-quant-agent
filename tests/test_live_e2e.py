"""US-021: end-to-end live-path dry run in fixture mode — the pre-go-live gate.

Drives the COMPLETE --live code path exactly as the 08:10 ET timer would run
it, with the operator's manual step (funding + live keys) replaced by fakes:
the REAL AlpacaClient against the live host over FakeBroker's realistic
/v2/account, /v2/positions, /v2/orders, /v2/calendar (+ portfolio-history)
payloads; a temp state DB with a LIVE promoted slot; TEMP live guardrail
config files resolved through the live path constants (limits.live.json /
breaker.live.json / allocation.live.json + the live breaker state dir); the
REAL TradeLedger/AccountSnapshotLog recorders over fake Notion sessions at
the (Live) database ids; and the REAL slack_notifier(live=True) over a fake
WebClient. No network, no real accounts.

ops/runbook.md §8 names this file as the pre-go-live gate:

    .venv/bin/python -m pytest tests/test_live_e2e.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from execution import allocation as allocation_mod
from execution import breaker as breaker_mod
from execution import order_gate as order_gate_mod
from execution.account_log import AccountSnapshotLog
from execution.alpaca_client import LIVE_BASE_URL, AlpacaClient
from execution.ledger import TradeLedger
from execution.rebalance import run_rebalance, slack_notifier
from orchestrator.config import SlackConfig
from orchestrator.notion_client import NotionClient
from orchestrator.state import StateStore
from tests.test_account_log import route_history
from tests.test_notion_client import FakeResponse as NotionResponse
from tests.test_notion_client import FakeSession as NotionSession
from tests.test_rebalance import AS_OF, STORE_DAYS, FakeBroker, write_bins
from tests.test_rebalance_live import FakeWebClient
from tests.test_signal import write_calendar, write_conf, write_pred

LIVE_LEDGER_DB = "db-trade-ledger-live"
LIVE_SNAPSHOTS_DB = "db-account-snapshots-live"
LIVE_CHANNEL = "C0LIVE"


def notion_page(page_id: str) -> NotionResponse:
    return NotionResponse(200, {"object": "page", "id": page_id})


@pytest.fixture
def gate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Everything the 08:10 live run touches, rebuilt from fixtures.

    Store + workspace + state DB carry the same numbers as the paper env in
    tests/test_rebalance.py, but the strategy is pinned in the LIVE slot:
    topk=2/n_drop=1 over {AAPL: 0.9, MSFT: 0.8, NVDA: 0.1} selects AAPL+MSFT
    at 0.5 weight each. At the 10% allocation on the $100k account the diff
    sizes against $10k: exactly buy 25 AAPL @ 201.00 and buy 12 MSFT @ 402.00.

    Guardrails are NOT injected — run_rebalance(live=True) must resolve them
    itself through the live path constants, which all point at temp files
    here. The temp limits permit the $5,025 AAPL buy; the committed files
    (paper limits.json AND limits.live.json, both $500/order) would gate-
    reject it — so the trade completing at all proves the live paths were
    the ones consulted.
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
    StateStore(db_path).set_promoted_strategy_live(
        str(workspace),
        {"universe": "us_liquid", "topk": 2, "n_drop": 1, "live_equity_allocation_pct": 10.0},
    )

    # Temp live guardrail config files, resolved via the live path constants.
    limits_path = tmp_path / "limits.live.json"
    limits_path.write_text(
        json.dumps(
            {
                "max_order_notional_usd": 6_000,
                "max_position_pct_equity": 60,
                "max_day_orders": 120,
                "max_total_positions": 60,
            }
        )
    )
    breaker_path = tmp_path / "breaker.live.json"
    breaker_path.write_text(
        json.dumps({"max_daily_notional_usd": 200_000, "max_drawdown_pct": 20})
    )
    allocation_path = tmp_path / "allocation.live.json"
    allocation_path.write_text(json.dumps({"live_equity_allocation_pct": 10}))
    halt_file = tmp_path / "breaker-live" / "halt"
    hwm_file = tmp_path / "breaker-live" / "high_water_mark.json"
    monkeypatch.setattr(order_gate_mod, "LIVE_LIMITS_PATH", limits_path)
    monkeypatch.setattr(breaker_mod, "LIVE_CONFIG_PATH", breaker_path)
    monkeypatch.setattr(breaker_mod, "LIVE_HALT_FILE", halt_file)
    monkeypatch.setattr(breaker_mod, "LIVE_HWM_FILE", hwm_file)
    monkeypatch.setattr(allocation_mod, "ALLOCATION_PATH", allocation_path)

    # Real slack_notifier(live=True) over the fake WebClient.
    import slack_sdk

    import orchestrator.config as config_mod

    FakeWebClient.instances = []
    monkeypatch.setattr(slack_sdk, "WebClient", FakeWebClient)
    monkeypatch.setattr(
        config_mod,
        "load_slack_config",
        lambda: SlackConfig(
            bot_token="xoxb-test",
            app_token="xapp-test",
            channel_id="C0PAPER",
            live_channel_id=LIVE_CHANNEL,
        ),
    )
    return SimpleNamespace(
        store=store,
        db_path=db_path,
        halt_file=halt_file,
        hwm_file=hwm_file,
    )


def live_recorders(
    ledger_responses: list[NotionResponse], snapshot_responses: list[NotionResponse]
) -> SimpleNamespace:
    """The REAL recorders over fake Notion sessions at the (Live) db ids."""
    ledger_session = NotionSession(ledger_responses)
    snap_session = NotionSession(snapshot_responses)
    return SimpleNamespace(
        ledger=TradeLedger(NotionClient(session=ledger_session), LIVE_LEDGER_DB),
        ledger_session=ledger_session,
        snapshots=AccountSnapshotLog(
            NotionClient(session=snap_session, sleep=lambda _s: None, max_retries=0),
            LIVE_SNAPSHOTS_DB,
        ),
        snapshots_session=snap_session,
    )


def run_gate(env: SimpleNamespace, broker: FakeBroker, recorders: SimpleNamespace) -> int:
    return run_rebalance(
        AlpacaClient(LIVE_BASE_URL, session=broker.session, allow_live=True),
        slack_notifier(live=True),
        as_of=AS_OF,
        db_path=env.db_path,
        store_path=env.store,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.1,
        sleep=lambda _s: None,
        ledger=recorders.ledger,
        snapshots=recorders.snapshots,
        live=True,
    )


def test_live_path_end_to_end_traded_day(gate_env: SimpleNamespace) -> None:
    broker = FakeBroker()
    route_history(broker)
    recorders = live_recorders(
        [notion_page("pg-1"), notion_page("pg-2"), notion_page("pg-1"), notion_page("pg-2")],
        [notion_page("pg-snap")],
    )

    assert run_gate(gate_env, broker, recorders) == 0

    # Targets scaled to the allocation pct: $100k equity * 10% = $10k for the
    # diff -> floor(5000/200)=25 AAPL, floor(5000/400)=12 MSFT (the unscaled
    # book would buy 250/125), every order stamped with the rdq-live- prefix.
    posts = [c["json"] for c in broker.session.posts()]
    assert [(p["symbol"], p["side"], p["qty"], p["limit_price"]) for p in posts] == [
        ("AAPL", "buy", "25", "201"),
        ("MSFT", "buy", "12", "402"),
    ]
    assert [p["client_order_id"] for p in posts] == [
        "rdq-live-2026-07-09-buy-AAPL",
        "rdq-live-2026-07-09-buy-MSFT",
    ]

    # Live limits consulted at the live path: the committed limits files
    # (paper AND live, both $500/order) would have rejected the $5,025 AAPL
    # buy — only the temp file at the patched LIVE_LIMITS_PATH permits it.
    # Live breaker consulted at the live state paths: the high-water mark
    # landed in the temp live state dir, recording the REAL (unscaled) equity.
    assert json.loads(gate_env.hwm_file.read_text()) == {"high_water_mark": 100_000.0}

    # 1:1 ledger reconciliation: every submitted order has a Trade Ledger
    # (Live) row, and every row names an order Alpaca actually accepted.
    creates = [c for c in recorders.ledger_session.calls if c["method"] == "POST"]
    assert len(creates) == len(broker.submitted) == 2
    assert all(
        c["json"]["parent"] == {"type": "database_id", "database_id": LIVE_LEDGER_DB}
        for c in creates
    )
    ledger_order_ids = {
        c["json"]["properties"]["Order ID"]["rich_text"][0]["text"]["content"]
        for c in creates
    }
    assert ledger_order_ids == {row["id"] for row in broker.submitted}
    finals = [c for c in recorders.ledger_session.calls if c["method"] == "PATCH"]
    assert len(finals) == 2
    assert all(
        c["json"]["properties"]["Status"] == {"select": {"name": "filled"}} for c in finals
    )
    assert recorders.ledger.failures == []

    # Daily snapshot row written to Account Snapshots (Live).
    (snap_call,) = recorders.snapshots_session.calls
    assert snap_call["json"]["parent"] == {
        "type": "database_id",
        "database_id": LIVE_SNAPSHOTS_DB,
    }
    row = snap_call["json"]["properties"]
    assert row["Outcome"] == {"select": {"name": "traded"}}
    assert row["Orders Placed"] == {"number": 2}
    assert row["Orders Filled"] == {"number": 2}
    assert row["Equity"] == {"number": 100_000.0}
    assert row["Day P/L"] == {"number": 1_250.5}
    assert recorders.snapshots.failures == []

    # Summary posted to the LIVE channel, saying LIVE and the allocation.
    (web_client,) = FakeWebClient.instances
    (message,) = web_client.messages
    assert message["channel"] == LIVE_CHANNEL
    assert "daily LIVE rebalance summary (2026-07-09)" in message["text"]
    assert "live allocation: 10% of equity (sizing against $10,000.00)" in message["text"]
    assert "2/2 orders filled" in message["text"]
    assert "breaker: normal (high-water mark $100,000.00)" in message["text"]


def test_live_halt_file_refusal_end_to_end(gate_env: SimpleNamespace) -> None:
    # The LIVE halt file is present: exit 0, no orders submitted, no ledger
    # rows (an empty response queue makes any ledger call fail loudly), a
    # "halted" snapshot row, and the halt line posted to the live channel.
    gate_env.halt_file.parent.mkdir(parents=True, exist_ok=True)
    gate_env.halt_file.write_text("operator halt before go-live\n")
    broker = FakeBroker()
    route_history(broker)
    recorders = live_recorders([], [notion_page("pg-snap")])

    assert run_gate(gate_env, broker, recorders) == 0

    assert broker.session.posts() == []
    assert broker.submitted == []
    assert recorders.ledger_session.calls == []
    (snap_call,) = recorders.snapshots_session.calls
    row = snap_call["json"]["properties"]
    assert row["Outcome"] == {"select": {"name": "halted"}}
    assert row["Orders Placed"] == {"number": 0}

    (web_client,) = FakeWebClient.instances
    (message,) = web_client.messages
    assert message["channel"] == LIVE_CHANNEL
    assert "rebalance halted (2026-07-09)" in message["text"]

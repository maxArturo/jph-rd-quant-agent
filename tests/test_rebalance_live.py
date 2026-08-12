"""US-013: rebalance --live — live slot, live guardrails, scaled equity, live ids.

Drives run_rebalance(live=True) end-to-end through the REAL AlpacaClient over
tests/test_rebalance.py's FakeBroker (fake HTTP session), plus the main()/
slack_notifier() wiring that picks the live host and the live Slack channel.
Paper behavior is regression-frozen: nothing here touches the paper tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import execution.rebalance as rebalance_mod
from execution import allocation as allocation_mod
from execution import breaker as breaker_mod
from execution.allocation import LiveAllocation
from execution.alpaca_client import BASE_URL, LIVE_BASE_URL, AlpacaClient
from execution.breaker import Breaker, BreakerConfig
from execution.order_gate import Limits
from execution.rebalance import run_rebalance, slack_notifier
from orchestrator.config import ConfigError, SlackConfig
from orchestrator.state import StateStore
from tests.test_rebalance import AS_OF, STORE_DAYS, FakeBroker, write_bins
from tests.test_signal import write_calendar, write_conf, write_pred

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def live_env(tmp_path: Path) -> SimpleNamespace:
    """Store + LIVE-promoted workspace + state DB + permissive tmp guardrails.

    Same numbers as the paper env in tests/test_rebalance.py, but the
    strategy is pinned in the LIVE slot: topk=2/n_drop=1 over {AAPL: 0.9,
    MSFT: 0.8, NVDA: 0.1} selects AAPL+MSFT at 0.5 weight each. At the 10%
    allocation on $100k equity the diff sizes against $10k: exactly
    buy 25 AAPL @ 201.00 and buy 12 MSFT @ 402.00.
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

    breaker = Breaker(
        BreakerConfig(max_daily_notional_usd=200_000.0, max_drawdown_pct=20.0),
        halt_file=tmp_path / "halt",
        high_water_mark_file=tmp_path / "hwm.json",
    )
    limits = Limits(
        max_order_notional_usd=60_000.0,
        max_position_pct_equity=60.0,
        max_day_orders=120,
        max_total_positions=60,
    )
    return SimpleNamespace(
        store=store, workspace=workspace, db_path=db_path, breaker=breaker, limits=limits
    )


def live_client(broker: FakeBroker) -> AlpacaClient:
    """The real client against the live host, driven by the fake session."""
    return AlpacaClient(LIVE_BASE_URL, session=broker.session, allow_live=True)


def run_live(
    env: SimpleNamespace, broker: FakeBroker, notes: list[str], **overrides: Any
) -> int:
    kwargs: dict[str, Any] = dict(
        dry_run=False,
        as_of=AS_OF,
        db_path=env.db_path,
        store_path=env.store,
        limits=env.limits,
        breaker=env.breaker,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.1,
        sleep=lambda _s: None,
        live=True,
        allocation=LiveAllocation(live_equity_allocation_pct=10.0),
    )
    kwargs.update(overrides)
    return run_rebalance(live_client(broker), notes.append, **kwargs)


# ------------------------------------------- end-to-end: scaled targets, live ids


def test_live_scales_targets_and_stamps_live_order_ids(live_env: SimpleNamespace) -> None:
    broker = FakeBroker()
    notes: list[str] = []
    assert run_live(live_env, broker, notes) == 0

    posts = [c["json"] for c in broker.session.posts()]
    # $100k equity * 10% allocation = $10k for the diff: floor(5000/200)=25
    # AAPL, floor(5000/400)=12 MSFT (the unscaled book would buy 250/125).
    assert [(p["symbol"], p["side"], p["qty"], p["limit_price"]) for p in posts] == [
        ("AAPL", "buy", "25", "201"),
        ("MSFT", "buy", "12", "402"),
    ]
    assert posts[0]["client_order_id"] == "rdq-live-2026-07-09-buy-AAPL"
    assert posts[1]["client_order_id"] == "rdq-live-2026-07-09-buy-MSFT"
    assert len(notes) == 1
    assert "2/2 orders filled" in notes[0]


def test_live_buying_power_cap_uses_real_buying_power(live_env: SimpleNamespace) -> None:
    # $6k REAL buying power funds the $5,025 AAPL buy (an allocation-scaled
    # figure of $600 would defer everything); the $4,824 MSFT buy no longer
    # fits and defers with a warning in the summary.
    broker = FakeBroker(buying_power=6_000.0)
    notes: list[str] = []
    assert run_live(live_env, broker, notes) == 0

    posts = [c["json"] for c in broker.session.posts()]
    assert [p["symbol"] for p in posts] == ["AAPL"]
    assert posts[0]["client_order_id"] == "rdq-live-2026-07-09-buy-AAPL"
    assert "MSFT" in notes[0]
    assert "deferred" in notes[0]


# ------------------------------------------------------------- slot independence


def test_live_reads_the_live_slot_never_the_paper_row(
    live_env: SimpleNamespace, tmp_path: Path
) -> None:
    # A paper-promoted strategy with an EMPTY live slot must refuse — never
    # fall back to the paper row.
    db_path = tmp_path / "paper-only.sqlite"
    StateStore(db_path).set_promoted_strategy(
        str(live_env.workspace), {"universe": "us_liquid", "topk": 2, "n_drop": 1}
    )
    broker = FakeBroker()
    notes: list[str] = []
    assert run_live(live_env, broker, notes, db_path=db_path) == 1
    assert "no live promoted strategy" in notes[0]
    assert broker.session.posts() == []


# ------------------------------------------- live guardrail + state-path defaults


def test_live_defaults_load_the_committed_live_guardrails(
    live_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No injected limits/breaker/allocation: the committed limits.live.json
    # ($500/order) rejects the $5,025 AAPL buy, proving live mode defaults to
    # the live files. Live breaker state paths point at tmp so the test never
    # touches ~/rdq-data/breaker-live/.
    monkeypatch.setattr(breaker_mod, "LIVE_HALT_FILE", tmp_path / "live-halt")
    monkeypatch.setattr(breaker_mod, "LIVE_HWM_FILE", tmp_path / "live-hwm.json")
    broker = FakeBroker()
    notes: list[str] = []
    assert run_live(live_env, broker, notes, limits=None, breaker=None, allocation=None) == 1
    assert "max_order_notional_usd" in notes[0]
    assert broker.session.posts() == []


def test_live_halt_file_halts_the_live_run_and_exits_zero(
    live_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Default live breaker + a touched LIVE halt file: exit 0, "halted"
    # notice posted, nothing submitted (permissive injected limits so the
    # gate passes first).
    halt = tmp_path / "live-halt"
    halt.write_text("operator halt\n")
    monkeypatch.setattr(breaker_mod, "LIVE_HALT_FILE", halt)
    monkeypatch.setattr(breaker_mod, "LIVE_HWM_FILE", tmp_path / "live-hwm.json")
    broker = FakeBroker()
    notes: list[str] = []
    assert run_live(live_env, broker, notes, breaker=None) == 0
    assert "rebalance halted" in notes[0]
    assert broker.session.posts() == []


def test_live_allocation_config_failure_aborts_without_trading(
    live_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(allocation_mod, "ALLOCATION_PATH", tmp_path / "missing.json")
    broker = FakeBroker()
    notes: list[str] = []
    assert run_live(live_env, broker, notes, allocation=None) == 1
    assert "allocation config not found" in notes[0]
    assert broker.session.posts() == []


# --------------------------------------------------- slack_notifier live channel


class FakeWebClient:
    instances: list[FakeWebClient] = []

    def __init__(self, token: str) -> None:
        self.token = token
        self.messages: list[dict[str, str]] = []
        FakeWebClient.instances.append(self)


    def chat_postMessage(self, channel: str, text: str) -> None:  # noqa: N802 - slack_sdk API
        self.messages.append({"channel": channel, "text": text})


def _slack_config(live_channel_id: str | None) -> SlackConfig:
    return SlackConfig(
        bot_token="xoxb-test",
        app_token="xapp-test",
        channel_id="C0PAPER",
        live_channel_id=live_channel_id,
    )


@pytest.fixture
def fake_slack(monkeypatch: pytest.MonkeyPatch) -> type[FakeWebClient]:
    import slack_sdk

    FakeWebClient.instances = []
    monkeypatch.setattr(slack_sdk, "WebClient", FakeWebClient)
    return FakeWebClient


def test_slack_notifier_live_posts_to_the_live_channel(
    fake_slack: type[FakeWebClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    import orchestrator.config as config_mod

    monkeypatch.setattr(config_mod, "load_slack_config", lambda: _slack_config("C0LIVE"))
    notify = slack_notifier(live=True)
    notify("live notice")
    assert fake_slack.instances[-1].messages == [{"channel": "C0LIVE", "text": "live notice"}]


def test_slack_notifier_paper_default_ignores_the_live_channel(
    fake_slack: type[FakeWebClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    import orchestrator.config as config_mod

    monkeypatch.setattr(config_mod, "load_slack_config", lambda: _slack_config("C0LIVE"))
    notify = slack_notifier()
    notify("paper notice")
    assert fake_slack.instances[-1].messages == [
        {"channel": "C0PAPER", "text": "paper notice"}
    ]


def test_slack_notifier_live_refuses_without_live_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orchestrator.config as config_mod

    monkeypatch.setattr(config_mod, "load_slack_config", lambda: _slack_config(None))
    with pytest.raises(ConfigError, match="SLACK_LIVE_CHANNEL_ID"):
        slack_notifier(live=True)


# ------------------------------------------------------------------ main() wiring


def test_main_live_wires_live_client_flag_and_no_paper_notion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(client: AlpacaClient, notify: Any, **kwargs: Any) -> int:
        captured["client"] = client
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(rebalance_mod, "run_rebalance", fake_run)
    assert rebalance_mod.main(["--live", "--no-slack"]) == 0
    assert captured["client"].base_url == LIVE_BASE_URL
    assert captured["live"] is True
    # Live Notion routing lands with US-014 — until then --live must not
    # write live fills into the PAPER ledger/snapshots.
    assert captured["ledger"] is None
    assert captured["snapshots"] is None


def test_main_paper_default_keeps_the_paper_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(client: AlpacaClient, notify: Any, **kwargs: Any) -> int:
        captured["client"] = client
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(rebalance_mod, "run_rebalance", fake_run)
    assert rebalance_mod.main(["--no-slack", "--no-notion"]) == 0
    assert captured["client"].base_url == BASE_URL
    assert captured["live"] is False

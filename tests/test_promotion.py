"""US-033: strategy promotion flow.

Covers the Promote button on completed-run summaries, the confirmation step
(restating universe, topk/n_drop, and headline metrics), the pinning of the
promoted strategy (with a replacement notice), the Notion Decision Log write
(mocked HTTP), Bolt routing of the three promotion actions, and the
rebalancer entrypoint check that refuses to run without a promotion.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from execution.promoted import NoPromotedStrategyError, load_promoted_strategy
from execution.signal import SignalError, StrategyParams, load_strategy_params
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.poller import HypothesisPoller
from orchestrator.promotion import (
    ACTION_PROMOTE,
    ACTION_PROMOTE_CANCEL,
    ACTION_PROMOTE_CONFIRM,
    PromotionFlow,
)
from orchestrator.rdagent_client import ArtifactNotFoundError, RunArtifacts, RunStatus
from orchestrator.state import StateStore
from tests.test_notion_client import FakeResponse, FakeSession
from tests.test_notion_recorder import DBS, page_response
from tests.test_poller import (
    FINISHED_OK,
    SESSION,
    THREAD,
    TRACE_ID,
    FakeSlack,
    RecordingSay,
    StubRdAgent,
    make_artifacts,
)
from tests.test_slack_app import CHANNEL, make_app

# A workspace conf shaped like the real us_templates ones (jinja placeholders
# survive in real workspaces; load_strategy_params renders them tolerantly).
WORKSPACE_CONF = """\
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy.signal_strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
"""

PARAMS = StrategyParams(topk=50, n_drop=5)


def promotable_artifacts(tmp_path: Path) -> RunArtifacts:
    """A fixture workspace with metrics AND a parseable strategy conf."""
    artifacts = make_artifacts(tmp_path)
    (Path(artifacts.workspace_path) / "conf.yaml").write_text(WORKSPACE_CONF)
    return artifacts


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    s = StateStore(tmp_path / "state.sqlite")
    s.create_run(THREAD, SESSION, universe="us_liquid", universe_tickers=("AAPL", "MSFT"))
    s.update_run_status(THREAD, "completed")
    return s


def make_flow(
    store: StateStore,
    artifacts: RunArtifacts,
    recorder: NotionRecorder | None = None,
) -> PromotionFlow:
    return PromotionFlow(store, recorder=recorder, locate=lambda _session: artifacts)


# --- Promote button on the completion summary (poller) -----------------------


def completion_slack(
    store: StateStore, artifacts_or_exc: Any, status: RunStatus = FINISHED_OK
) -> FakeSlack:
    """Run one completion poll and return the FakeSlack that captured it."""
    rd = StubRdAgent()
    rd.status_by_trace[TRACE_ID] = status
    slack = FakeSlack()

    def locate(_session: str | Path) -> RunArtifacts:
        if isinstance(artifacts_or_exc, Exception):
            raise artifacts_or_exc
        return artifacts_or_exc

    HypothesisPoller(store, rd, slack, CHANNEL, locate=locate).poll_once()
    return slack


@pytest.fixture
def running_store(tmp_path: Path) -> StateStore:
    s = StateStore(tmp_path / "state.sqlite")
    s.create_run(THREAD, SESSION, universe="us_liquid")
    return s


def test_completed_run_summary_offers_promote_button(
    running_store: StateStore, tmp_path: Path
) -> None:
    slack = completion_slack(running_store, promotable_artifacts(tmp_path))
    (post,) = slack.posts
    section, actions = post["blocks"]
    assert section["text"]["text"] == post["text"]
    (button,) = actions["elements"]
    assert button["action_id"] == ACTION_PROMOTE
    assert button["value"] == THREAD
    assert button["text"]["text"] == "Promote to paper trading"


def test_stopped_run_summary_offers_promote_button(
    running_store: StateStore, tmp_path: Path
) -> None:
    # end_code -1 = operator stop — the only way an unbounded orchestrator run
    # ever ends, so its summary must still offer promotion (US-044).
    stopped = RunStatus(finished=True, end_code=-1, error_msg=None)
    slack = completion_slack(running_store, promotable_artifacts(tmp_path), status=stopped)
    (post,) = slack.posts
    _section, actions = post["blocks"]
    (button,) = actions["elements"]
    assert button["action_id"] == ACTION_PROMOTE
    assert button["value"] == THREAD


def test_failed_run_summary_has_no_promote_button(
    running_store: StateStore, tmp_path: Path
) -> None:
    failed = RunStatus(finished=True, end_code=2, error_msg="subprocess died")
    slack = completion_slack(running_store, promotable_artifacts(tmp_path), status=failed)
    (post,) = slack.posts
    assert "blocks" not in post


def test_summary_without_artifacts_has_no_promote_button(running_store: StateStore) -> None:
    slack = completion_slack(running_store, ArtifactNotFoundError("no runner result"))
    (post,) = slack.posts
    assert "blocks" not in post


# --- confirmation step --------------------------------------------------------


def test_request_promotion_restates_universe_params_and_metrics(
    store: StateStore, tmp_path: Path
) -> None:
    flow = make_flow(store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)

    (call,) = say.calls
    assert call["thread_ts"] == THREAD
    text = call["text"]
    assert "`us_liquid`" in text
    assert "topk=50" in text and "n_drop=5" in text
    assert str(tmp_path / "workspace") in text
    assert "*IC:* 0.0432" in text  # headline metrics restated (FIXTURE_METRICS)
    assert "*MDD:* -8.40%" in text
    # Confirm/Cancel buttons carry the thread_ts.
    _section, actions = call["blocks"]
    by_id = {e["action_id"]: e for e in actions["elements"]}
    assert set(by_id) == {ACTION_PROMOTE_CONFIRM, ACTION_PROMOTE_CANCEL}
    assert all(e["value"] == THREAD for e in actions["elements"])
    # The confirmation alone promotes nothing.
    assert store.get_promoted_strategy() is None


def test_request_promotion_uses_the_real_workspace_conf(
    store: StateStore, tmp_path: Path
) -> None:
    artifacts = promotable_artifacts(tmp_path)
    # No injected load_params: the flow's default reads conf.yaml (50, 5).
    flow = PromotionFlow(store, locate=lambda _session: artifacts)
    assert load_strategy_params(Path(artifacts.workspace_path)) == PARAMS
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "topk=50" in say.calls[0]["text"]


def test_request_promotion_warns_when_replacing(store: StateStore, tmp_path: Path) -> None:
    store.set_promoted_strategy("/old/workspace", {"universe": "us_liquid"})
    flow = make_flow(store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "replaces the currently promoted strategy" in say.calls[0]["text"]
    assert "/old/workspace" in say.calls[0]["text"]


def test_request_promotion_refuses_thread_without_run(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite")  # no run rows at all
    flow = make_flow(store, make_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "no research run" in say.calls[0]["text"]
    assert store.get_promoted_strategy() is None


def test_request_promotion_refuses_running_run(
    running_store: StateStore, tmp_path: Path
) -> None:
    flow = make_flow(running_store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "'running'" in say.calls[0]["text"]
    assert "only a completed or operator-stopped run" in say.calls[0]["text"]


def test_request_promotion_refuses_failed_run(
    running_store: StateStore, tmp_path: Path
) -> None:
    running_store.update_run_status(THREAD, "failed")
    flow = make_flow(running_store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "'failed'" in say.calls[0]["text"]
    assert running_store.get_promoted_strategy() is None


def test_request_promotion_allows_operator_stopped_run(
    running_store: StateStore, tmp_path: Path
) -> None:
    # Orchestrator-started runs are unbounded; a deliberate stop at a SOTA
    # result is their normal successful ending and must be promotable (US-044).
    running_store.update_run_status(THREAD, "stopped")
    flow = make_flow(running_store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "topk=50" in say.calls[0]["text"]  # confirmation, not a refusal
    flow.confirm_promotion(THREAD, say)
    promoted = running_store.get_promoted_strategy()
    assert promoted is not None
    assert promoted.config["universe"] == "us_liquid"


def test_request_promotion_refuses_when_params_unreadable(
    store: StateStore, tmp_path: Path
) -> None:
    # make_artifacts writes no conf.yaml -> the real loader would refuse; the
    # injected loader refuses the same way.
    def no_params(_workspace: Path) -> StrategyParams:
        raise SignalError("no conf*.yaml found in workspace")

    flow = PromotionFlow(
        store, locate=lambda _s: make_artifacts(tmp_path), load_params=no_params
    )
    say = RecordingSay()
    flow.request_promotion(THREAD, say)
    assert "topk/n_drop" in say.calls[0]["text"]
    assert store.get_promoted_strategy() is None


# --- confirm flow ---------------------------------------------------------------


def test_confirm_promotion_pins_workspace_and_config(
    store: StateStore, tmp_path: Path
) -> None:
    flow = make_flow(store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.confirm_promotion(THREAD, say)

    promoted = store.get_promoted_strategy()
    assert promoted is not None
    assert promoted.workspace_path == str(tmp_path / "workspace")
    assert promoted.config["universe"] == "us_liquid"
    assert promoted.config["universe_tickers"] == ["AAPL", "MSFT"]
    assert promoted.config["topk"] == 50
    assert promoted.config["n_drop"] == 5
    assert promoted.config["thread_ts"] == THREAD
    assert promoted.config["session_path"] == SESSION

    (call,) = say.calls
    assert "promoted to paper trading" in call["text"]
    assert "Replaced" not in call["text"]  # nothing was promoted before


def test_confirm_promotion_replacement_notice_and_single_row(
    store: StateStore, tmp_path: Path
) -> None:
    store.set_promoted_strategy("/old/workspace", {"universe": "old_univ"})
    flow = make_flow(store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.confirm_promotion(THREAD, say)

    (call,) = say.calls
    assert "Replaced the previously promoted strategy" in call["text"]
    assert "/old/workspace" in call["text"]
    promoted = store.get_promoted_strategy()
    assert promoted is not None
    assert promoted.workspace_path == str(tmp_path / "workspace")
    # The schema allows exactly one promoted strategy — replaced, not added.
    with sqlite3.connect(store.db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM promoted_strategy").fetchone()
    assert count == 1


def test_confirm_promotion_refusal_promotes_nothing(
    running_store: StateStore, tmp_path: Path
) -> None:
    flow = make_flow(running_store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.confirm_promotion(THREAD, say)
    assert "Cannot promote" in say.calls[0]["text"]
    assert running_store.get_promoted_strategy() is None


def test_cancel_promotion_changes_nothing(store: StateStore, tmp_path: Path) -> None:
    flow = make_flow(store, promotable_artifacts(tmp_path))
    say = RecordingSay()
    flow.cancel_promotion(THREAD, say)
    assert "cancelled" in say.calls[0]["text"]
    assert "nothing was changed" in say.calls[0]["text"]
    assert store.get_promoted_strategy() is None


# --- Notion Decision Log (mocked HTTP) -------------------------------------------


def test_confirm_promotion_writes_decision_log_row(
    store: StateStore, tmp_path: Path
) -> None:
    store.create_directive(THREAD, "12-1 momentum in liquid US names")
    store.set_notion_page("idea", THREAD, "page-idea")
    session = FakeSession([page_response("page-dec"), FakeResponse(200, {"object": "page"})])
    recorder = NotionRecorder(
        NotionClient(session=session, sleep=lambda _s: None, max_retries=0), DBS, store
    )
    flow = make_flow(store, promotable_artifacts(tmp_path), recorder=recorder)
    flow.confirm_promotion(THREAD, RecordingSay())

    create, status_update = session.calls
    assert create["method"] == "POST"
    assert create["url"].endswith("/v1/pages")
    body = create["json"]
    assert body["parent"] == {"type": "database_id", "database_id": DBS.decision_log}
    props = body["properties"]
    title = props["Decision"]["title"][0]["text"]["content"]
    assert title == "Promote '12-1 momentum in liquid US names' to paper trading"
    assert props["Type"] == {"select": {"name": "promotion"}}
    assert "start" in props["Decided At"]["date"]
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}
    details = props["Details"]["rich_text"][0]["text"]["content"]
    assert "topk=50" in details and "n_drop=5" in details
    assert "us_liquid" in details
    assert "IC: 0.0432" in details

    # The idea page's Status moves to 'promoted'.
    assert status_update["method"] == "PATCH"
    assert status_update["url"].endswith("/v1/pages/page-idea")
    assert status_update["json"]["properties"]["Status"] == {"select": {"name": "promoted"}}


def test_decision_log_replacement_detail(store: StateStore, tmp_path: Path) -> None:
    store.set_promoted_strategy("/old/workspace", {"universe": "old"})
    session = FakeSession([page_response("page-dec")])
    recorder = NotionRecorder(
        NotionClient(session=session, sleep=lambda _s: None, max_retries=0), DBS, store
    )
    flow = make_flow(store, promotable_artifacts(tmp_path), recorder=recorder)
    flow.confirm_promotion(THREAD, RecordingSay())
    (create, *_rest) = session.calls
    details = create["json"]["properties"]["Details"]["rich_text"][0]["text"]["content"]
    assert "Replaced: /old/workspace" in details


# --- Bolt routing -----------------------------------------------------------------


class FakePromotions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def request_promotion(self, thread_ts: str, say: Any) -> None:
        self.calls.append(("request_promotion", thread_ts))

    def confirm_promotion(self, thread_ts: str, say: Any) -> None:
        self.calls.append(("confirm_promotion", thread_ts))

    def cancel_promotion(self, thread_ts: str, say: Any) -> None:
        self.calls.append(("cancel_promotion", thread_ts))


@pytest.mark.parametrize(
    ("action_id", "method"),
    [
        (ACTION_PROMOTE, "request_promotion"),
        (ACTION_PROMOTE_CONFIRM, "confirm_promotion"),
        (ACTION_PROMOTE_CANCEL, "cancel_promotion"),
    ],
)
def test_promotion_buttons_route_through_bolt(
    monkeypatch: pytest.MonkeyPatch, action_id: str, method: str
) -> None:
    from tests.test_poller import dispatch_action

    promotions = FakePromotions()
    app, _client, _conversation = make_app(monkeypatch, promotions=promotions)
    dispatch_action(app, action_id, value=THREAD)
    assert promotions.calls == [(method, THREAD)]


# --- rebalancer entrypoint check (execution/promoted.py) ---------------------------


def test_rebalancer_refuses_without_state_database(tmp_path: Path) -> None:
    db_path = tmp_path / "absent.sqlite"
    with pytest.raises(NoPromotedStrategyError, match="state database not found"):
        load_promoted_strategy(db_path)
    assert not db_path.exists()  # the check never creates orchestrator state


def test_rebalancer_refuses_without_promotion(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite"
    StateStore(db_path)  # schema exists, but nothing was ever promoted
    with pytest.raises(NoPromotedStrategyError, match="no promoted strategy exists"):
        load_promoted_strategy(db_path)


def test_rebalancer_refuses_when_workspace_gone(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite"
    StateStore(db_path).set_promoted_strategy(str(tmp_path / "vanished"), {"topk": 50})
    with pytest.raises(NoPromotedStrategyError, match="missing on disk"):
        load_promoted_strategy(db_path)


def test_rebalancer_loads_the_promoted_strategy(store: StateStore, tmp_path: Path) -> None:
    flow = make_flow(store, promotable_artifacts(tmp_path))
    flow.confirm_promotion(THREAD, RecordingSay())

    promoted = load_promoted_strategy(store.db_path)
    assert promoted.workspace_path == str(tmp_path / "workspace")
    assert promoted.config["topk"] == 50
    assert promoted.config["n_drop"] == 5
    assert promoted.config["universe"] == "us_liquid"

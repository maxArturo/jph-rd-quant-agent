"""US-027: research runs recorded in Notion end-to-end.

The recorder is unit-tested against a mocked Notion HTTP session (FakeSession
from tests/test_notion_client.py), then each lifecycle hook is driven through
the REAL ConversationCore (FakeClient tool scripts) and HypothesisPoller
(StubRdAgent/FakeSlack from tests/test_poller.py) to prove the writes fire at
the right points with the right payloads. No network anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.conversation import ConversationCore
from orchestrator.llm import ModelRouter
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import (
    NotionDatabases,
    NotionRecorder,
    RecorderConfigError,
    directive_details,
    load_notion_databases,
)
from orchestrator.poller import HypothesisPoller
from orchestrator.rdagent_client import (
    KIND_FEEDBACK,
    KIND_HYPOTHESIS,
    ArtifactNotFoundError,
    RunArtifacts,
    RunStatus,
)
from orchestrator.state import Directive, StateStore
from tests.test_conversation import (
    RecordingSay,
    StubLauncher,
    lifecycle_script,
    save_directive_script,
    start_research_script,
)
from tests.test_llm import FakeClient
from tests.test_notion_client import FakeResponse, FakeSession
from tests.test_poller import (
    FEEDBACK_CONTENT,
    HYPO_CONTENT,
    SESSION,
    TRACE_ID,
    FakeSlack,
    StubRdAgent,
    interaction,
)
from tests.test_summary import FIXTURE_METRICS, write_qlib_res_csv, write_ret_pkl

THREAD = "1751900000.000100"
CHANNEL = "C0TESTCHAN"

DBS = NotionDatabases(
    research_ideas="db-ideas",
    hypothesis_log="db-hypo",
    backtest_results="db-bt",
    decision_log="db-dec",
    trade_ledger="db-tl",
)

DIRECTIVE = Directive(
    id=1,
    thread_ts=THREAD,
    objective="Test whether 12-1 momentum beats SPY",
    universe_hint="US large caps",
    constraints="long-only, monthly rebalance",
    created_at="2026-07-09T00:00:00+00:00",
)


def page_response(page_id: str) -> FakeResponse:
    return FakeResponse(200, {"object": "page", "id": page_id})


def make_recorder(
    tmp_path: Path, responses: list[FakeResponse]
) -> tuple[NotionRecorder, StateStore, FakeSession]:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    session = FakeSession(responses)
    client = NotionClient(session=session, sleep=lambda _s: None, max_retries=0)
    recorder = NotionRecorder(
        client, DBS, store, permalink=lambda ts: f"https://slack.example/archives/p{ts}"
    )
    return recorder, store, session


def plain_text(prop: dict[str, Any], kind: str) -> str:
    return "".join(part["text"]["content"] for part in prop[kind])


# --- record_idea ---------------------------------------------------------------


def test_record_idea_creates_research_ideas_page(tmp_path: Path) -> None:
    recorder, store, session = make_recorder(tmp_path, [page_response("page-idea")])

    page_id = recorder.record_idea(
        THREAD, raw_idea="momentum on big US names?", directive=DIRECTIVE, universe="us_liquid"
    )

    assert page_id == "page-idea"
    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/pages")
    body = call["json"]
    assert body["parent"] == {"type": "database_id", "database_id": "db-ideas"}
    props = body["properties"]
    assert plain_text(props["Idea"], "title") == DIRECTIVE.objective
    assert plain_text(props["Raw Idea"], "rich_text") == "momentum on big US names?"
    directive_text = plain_text(props["Directive"], "rich_text")
    assert directive_text == directive_details(DIRECTIVE)
    assert DIRECTIVE.objective in directive_text
    assert "long-only, monthly rebalance" in directive_text
    assert plain_text(props["Universe"], "rich_text") == "us_liquid"
    assert props["Status"] == {"select": {"name": "proposed"}}
    assert plain_text(props["Thread TS"], "rich_text") == THREAD
    assert props["Thread"] == {"url": f"https://slack.example/archives/p{THREAD}"}
    # the mapping lets later lifecycle points address the same page
    assert store.get_notion_page("idea", THREAD) == "page-idea"


def test_record_idea_again_updates_the_same_page(tmp_path: Path) -> None:
    recorder, _store, session = make_recorder(
        tmp_path, [page_response("page-idea"), page_response("page-idea")]
    )
    recorder.record_idea(THREAD, raw_idea="raw", directive=DIRECTIVE)

    recorder.record_idea(THREAD, raw_idea="raw v2", directive=DIRECTIVE)

    update = session.calls[1]
    assert update["method"] == "PATCH"
    assert update["url"].endswith("/v1/pages/page-idea")
    props = update["json"]["properties"]
    assert plain_text(props["Raw Idea"], "rich_text") == "raw v2"
    assert "Status" not in props  # a re-save never resets the lifecycle status


def test_record_idea_status_updates_status_and_universe(tmp_path: Path) -> None:
    recorder, store, session = make_recorder(tmp_path, [page_response("page-idea")])
    store.set_notion_page("idea", THREAD, "page-idea")

    recorder.record_idea_status(THREAD, "researching", universe="ai_semis")

    (call,) = session.calls
    assert call["method"] == "PATCH"
    assert call["url"].endswith("/v1/pages/page-idea")
    props = call["json"]["properties"]
    assert props["Status"] == {"select": {"name": "researching"}}
    assert plain_text(props["Universe"], "rich_text") == "ai_semis"


def test_record_idea_status_without_idea_page_is_a_noop(tmp_path: Path) -> None:
    recorder, _store, session = make_recorder(tmp_path, [])
    recorder.record_idea_status(THREAD, "researching")
    assert session.calls == []


# --- record_hypothesis -----------------------------------------------------------


def test_record_hypothesis_links_idea_and_stores_mapping(tmp_path: Path) -> None:
    recorder, store, session = make_recorder(tmp_path, [page_response("page-hypo")])
    store.set_notion_page("idea", THREAD, "page-idea")

    page_id = recorder.record_hypothesis(THREAD, "trace|t1|hypothesis", HYPO_CONTENT)

    assert page_id == "page-hypo"
    (call,) = session.calls
    body = call["json"]
    assert body["parent"] == {"type": "database_id", "database_id": "db-hypo"}
    props = body["properties"]
    assert plain_text(props["Hypothesis"], "title") == HYPO_CONTENT["hypothesis"]
    assert json.loads(plain_text(props["Details"], "rich_text")) == HYPO_CONTENT
    assert props["Action"] == {"select": {"name": "pending"}}
    assert plain_text(props["Interaction Key"], "rich_text") == "trace|t1|hypothesis"
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}
    assert store.get_notion_page("hypothesis", "trace|t1|hypothesis") == "page-hypo"


def test_record_hypothesis_without_idea_page_omits_relation(tmp_path: Path) -> None:
    recorder, _store, session = make_recorder(tmp_path, [page_response("page-hypo")])
    recorder.record_hypothesis(THREAD, "trace|t1|hypothesis", HYPO_CONTENT)
    assert "Idea" not in session.calls[0]["json"]["properties"]


def test_record_hypothesis_action_updates_action_and_operator_input(tmp_path: Path) -> None:
    recorder, store, session = make_recorder(
        tmp_path, [page_response("page-hypo"), page_response("page-hypo")]
    )
    store.set_notion_page("hypothesis", "key-1", "page-hypo")

    recorder.record_hypothesis_action("key-1", "approved")
    recorder.record_hypothesis_action("key-1", "edited", operator_input="use 60d momentum")

    approved, edited = session.calls
    assert approved["method"] == "PATCH"
    assert approved["url"].endswith("/v1/pages/page-hypo")
    assert approved["json"]["properties"] == {"Action": {"select": {"name": "approved"}}}
    props = edited["json"]["properties"]
    assert props["Action"] == {"select": {"name": "edited"}}
    assert plain_text(props["Operator Input"], "rich_text") == "use 60d momentum"


def test_record_hypothesis_action_without_row_is_a_noop(tmp_path: Path) -> None:
    recorder, _store, session = make_recorder(tmp_path, [])
    recorder.record_hypothesis_action("unknown-key", "approved")
    assert session.calls == []


# --- record_backtest -------------------------------------------------------------


def test_record_backtest_payload(tmp_path: Path) -> None:
    recorder, store, session = make_recorder(tmp_path, [page_response("page-bt")])
    store.set_notion_page("idea", THREAD, "page-idea")

    page_id = recorder.record_backtest(
        THREAD,
        title="Experiment t1 — us_liquid",
        metrics=FIXTURE_METRICS,
        sharpe=1.23,
        sota=True,
        workspace_path="/ws/run1",
        universe="us_liquid",
    )

    assert page_id == "page-bt"
    (call,) = session.calls
    body = call["json"]
    assert body["parent"] == {"type": "database_id", "database_id": "db-bt"}
    props = body["properties"]
    assert plain_text(props["Experiment"], "title") == "Experiment t1 — us_liquid"
    assert props["IC"] == {"number": FIXTURE_METRICS["IC"]}
    assert props["ICIR"] == {"number": FIXTURE_METRICS["ICIR"]}
    assert props["Rank IC"] == {"number": FIXTURE_METRICS["Rank IC"]}
    assert props["ARR"] == {
        "number": FIXTURE_METRICS["1day.excess_return_with_cost.annualized_return"]
    }
    assert props["IR"] == {
        "number": FIXTURE_METRICS["1day.excess_return_with_cost.information_ratio"]
    }
    assert props["MDD"] == {
        "number": FIXTURE_METRICS["1day.excess_return_with_cost.max_drawdown"]
    }
    assert props["Sharpe"] == {"number": 1.23}
    assert props["SOTA"] == {"checkbox": True}
    assert plain_text(props["Workspace"], "rich_text") == "/ws/run1"
    assert plain_text(props["Universe"], "rich_text") == "us_liquid"
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}


def test_record_backtest_omits_missing_metrics(tmp_path: Path) -> None:
    recorder, _store, session = make_recorder(tmp_path, [page_response("page-bt")])
    recorder.record_backtest(
        THREAD,
        title="Experiment",
        metrics={"IC": 0.05},
        sharpe=None,
        sota=False,
        workspace_path="/ws",
        universe=None,
    )
    props = session.calls[0]["json"]["properties"]
    assert props["IC"] == {"number": 0.05}
    for absent in ("ICIR", "Rank IC", "ARR", "IR", "MDD", "Sharpe", "Universe"):
        assert absent not in props
    assert props["SOTA"] == {"checkbox": False}


# --- failure isolation -------------------------------------------------------------


class ExplodingSession:
    """A session whose every request fails at the network layer."""

    def request(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("notion is down")


def test_recorder_swallows_every_failure(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    store.set_notion_page("idea", THREAD, "page-idea")
    store.set_notion_page("hypothesis", "key-1", "page-hypo")
    client = NotionClient(session=ExplodingSession(), sleep=lambda _s: None, max_retries=0)
    recorder = NotionRecorder(client, DBS, store)

    # none of these raise — recording never breaks the flow it observes
    assert recorder.record_idea(THREAD, raw_idea="raw", directive=DIRECTIVE) is None
    recorder.record_idea_status(THREAD, "researching")
    assert recorder.record_hypothesis(THREAD, "key-2", HYPO_CONTENT) is None
    recorder.record_hypothesis_action("key-1", "approved")
    assert (
        recorder.record_backtest(
            THREAD,
            title="Experiment",
            metrics={},
            sharpe=None,
            sota=False,
            workspace_path="/ws",
            universe=None,
        )
        is None
    )


def test_broken_permalink_does_not_block_idea_creation(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    session = FakeSession([page_response("page-idea")])
    client = NotionClient(session=session, sleep=lambda _s: None, max_retries=0)

    def permalink(_ts: str) -> str | None:
        raise RuntimeError("slack permalink lookup failed")

    recorder = NotionRecorder(client, DBS, store, permalink=permalink)
    assert recorder.record_idea(THREAD, raw_idea="raw", directive=DIRECTIVE) == "page-idea"
    assert "Thread" not in session.calls[0]["json"]["properties"]


# --- config loading -----------------------------------------------------------------


def test_load_notion_databases_from_repo_config() -> None:
    databases = load_notion_databases()
    for db_id in (
        databases.research_ideas,
        databases.hypothesis_log,
        databases.backtest_results,
        databases.decision_log,
        databases.trade_ledger,
    ):
        assert db_id  # bootstrap has run; ids are committed in config.yaml


def test_load_notion_databases_missing_ids_raises(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("notion:\n  databases:\n    research_ideas: db-1\n")
    with pytest.raises(RecorderConfigError, match="hypothesis_log"):
        load_notion_databases(config)
    with pytest.raises(RecorderConfigError, match="bootstrap_notion"):
        load_notion_databases(tmp_path / "missing.yaml")


# --- lifecycle: conversation core ----------------------------------------------------


def make_core(
    tmp_path: Path, client: FakeClient, responses: list[FakeResponse]
) -> tuple[ConversationCore, StateStore, FakeSession]:
    recorder, store, session = make_recorder(tmp_path, responses)
    core = ConversationCore(
        store=store,
        router=ModelRouter(client=client),
        rdagent=StubLauncher(),
        recorder=recorder,
    )
    return core, store, session


def test_save_directive_records_research_idea(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=save_directive_script())
    core, store, session = make_core(tmp_path, client, [page_response("page-idea")])
    say = RecordingSay()

    core.handle_message(THREAD, "momentum on big US names?", say)

    (call,) = session.calls
    props = call["json"]["properties"]
    # raw idea = the operator's message that led to the save, unedited
    assert plain_text(props["Raw Idea"], "rich_text") == "momentum on big US names?"
    assert plain_text(props["Idea"], "title") == "Test whether 12-1 momentum beats SPY"
    assert "US large caps" in plain_text(props["Directive"], "rich_text")
    assert plain_text(props["Universe"], "rich_text") == "us_liquid"
    assert props["Status"] == {"select": {"name": "proposed"}}
    assert plain_text(props["Thread TS"], "rich_text") == THREAD
    assert store.get_notion_page("idea", THREAD) == "page-idea"
    # the Slack flow was untouched: summary + final reply still posted
    assert len(say.calls) == 2


def test_start_research_records_researching_status(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=start_research_script())
    core, store, session = make_core(tmp_path, client, [page_response("page-idea")])
    store.create_directive(THREAD, objective="Momentum beats SPY")
    store.set_notion_page("idea", THREAD, "page-idea")

    core.handle_message(THREAD, "research it", say=RecordingSay())

    (call,) = session.calls
    assert call["method"] == "PATCH"
    assert call["url"].endswith("/v1/pages/page-idea")
    props = call["json"]["properties"]
    assert props["Status"] == {"select": {"name": "researching"}}
    assert plain_text(props["Universe"], "rich_text") == "us_liquid"


def test_stop_run_records_stopped_status_and_cancelled_hypotheses(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Stopped."))
    core, store, session = make_core(
        tmp_path, client, [page_response("page-hypo"), page_response("page-idea")]
    )
    store.create_run(THREAD, str(StubLauncher.TRACE_FOLDER / "t/1"), universe="us_liquid")
    store.set_notion_page("idea", THREAD, "page-idea")
    store.set_notion_page("hypothesis", "k|1|hypothesis", "page-hypo")
    assert store.add_pending_interaction(THREAD, "k|1|hypothesis", {"content": {}}) is not None

    core.handle_message(THREAD, "stop the run", say=RecordingSay())

    cancelled, stopped = session.calls
    assert cancelled["url"].endswith("/v1/pages/page-hypo")
    assert cancelled["json"]["properties"]["Action"] == {"select": {"name": "cancelled"}}
    assert stopped["url"].endswith("/v1/pages/page-idea")
    assert stopped["json"]["properties"]["Status"] == {"select": {"name": "stopped"}}


def test_resume_run_records_researching_status(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("resume_run", "Resumed."))
    core, store, session = make_core(tmp_path, client, [page_response("page-idea")])
    store.create_directive(THREAD, objective="Momentum beats SPY")
    store.create_run(
        THREAD, str(StubLauncher.TRACE_FOLDER / "t/1"), universe="us_liquid", status="stopped"
    )
    store.set_notion_page("idea", THREAD, "page-idea")

    core.handle_message(THREAD, "resume the run", say=RecordingSay())

    (call,) = session.calls
    assert call["json"]["properties"]["Status"] == {"select": {"name": "researching"}}


# --- lifecycle: hypothesis poller -----------------------------------------------------


def make_poller(
    tmp_path: Path,
    responses: list[FakeResponse],
    locate: Any = None,
) -> tuple[HypothesisPoller, StateStore, FakeSession, StubRdAgent, FakeSlack]:
    recorder, store, session = make_recorder(tmp_path, responses)
    store.create_run(THREAD, SESSION, universe="us_liquid")
    store.set_notion_page("idea", THREAD, "page-idea")
    rd = StubRdAgent()
    slack = FakeSlack()
    kwargs: dict[str, Any] = {"recorder": recorder}
    if locate is not None:
        kwargs["locate"] = locate
    poller = HypothesisPoller(store, rd, slack, CHANNEL, **kwargs)
    return poller, store, session, rd, slack


def test_posted_hypothesis_records_hypothesis_log_row(tmp_path: Path) -> None:
    poller, store, session, rd, slack = make_poller(tmp_path, [page_response("page-hypo")])
    pending = interaction(KIND_HYPOTHESIS, HYPO_CONTENT)
    rd.pending_by_trace[TRACE_ID] = [pending]

    assert poller.poll_once() == 1

    assert len(slack.posts) == 1  # Slack post still happened
    (call,) = session.calls
    props = call["json"]["properties"]
    assert call["json"]["parent"]["database_id"] == "db-hypo"
    assert plain_text(props["Hypothesis"], "title") == HYPO_CONTENT["hypothesis"]
    assert plain_text(props["Interaction Key"], "rich_text") == pending.key
    assert props["Action"] == {"select": {"name": "pending"}}
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}
    assert store.get_notion_page("hypothesis", pending.key) == "page-hypo"


def test_approve_records_approved_action(tmp_path: Path) -> None:
    poller, store, session, rd, _slack = make_poller(
        tmp_path, [page_response("page-hypo"), page_response("page-hypo")]
    )
    pending = interaction(KIND_HYPOTHESIS, HYPO_CONTENT)
    rd.pending_by_trace[TRACE_ID] = [pending]
    poller.poll_once()
    (row,) = store.list_pending_interactions(THREAD, status="pending")

    poller.approve(row.id, RecordingSay())

    update = session.calls[1]
    assert update["method"] == "PATCH"
    assert update["url"].endswith("/v1/pages/page-hypo")
    assert update["json"]["properties"] == {"Action": {"select": {"name": "approved"}}}


def test_edit_reply_records_edited_action_with_operator_input(tmp_path: Path) -> None:
    poller, store, session, rd, _slack = make_poller(
        tmp_path, [page_response("page-hypo"), page_response("page-hypo")]
    )
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    poller.poll_once()
    (row,) = store.list_pending_interactions(THREAD, status="pending")
    poller.request_edit(row.id, RecordingSay())

    assert poller.consume_edit_reply(THREAD, "use 60d momentum instead", RecordingSay())

    props = session.calls[1]["json"]["properties"]
    assert props["Action"] == {"select": {"name": "edited"}}
    assert plain_text(props["Operator Input"], "rich_text") == "use 60d momentum instead"


def fixture_artifacts(tmp_path: Path) -> RunArtifacts:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_qlib_res_csv(workspace / "qlib_res.csv")
    write_ret_pkl(workspace / "ret.pkl")
    return RunArtifacts(
        workspace_path=workspace,
        qlib_res_csv=workspace / "qlib_res.csv",
        ret_pkl=workspace / "ret.pkl",
        source_pkl=workspace / "source.pkl",
    )


def test_feedback_records_backtest_results_row(tmp_path: Path) -> None:
    artifacts = fixture_artifacts(tmp_path)
    poller, _store, session, rd, _slack = make_poller(
        tmp_path, [page_response("page-bt")], locate=lambda _p: artifacts
    )
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_FEEDBACK, FEEDBACK_CONTENT, ts="t9")]

    poller.poll_once()

    assert rd.submitted == [(TRACE_ID, FEEDBACK_CONTENT)]  # auto-ack unaffected
    (call,) = session.calls
    body = call["json"]
    assert body["parent"]["database_id"] == "db-bt"
    props = body["properties"]
    assert plain_text(props["Experiment"], "title") == "Experiment t9 — us_liquid"
    assert props["IC"] == {"number": FIXTURE_METRICS["IC"]}
    assert props["MDD"] == {
        "number": FIXTURE_METRICS["1day.excess_return_with_cost.max_drawdown"]
    }
    assert isinstance(props["Sharpe"]["number"], float)  # derived from ret.pkl
    assert props["SOTA"] == {"checkbox": True}  # FEEDBACK_CONTENT decision=True
    assert plain_text(props["Workspace"], "rich_text") == str(artifacts.workspace_path)
    assert plain_text(props["Universe"], "rich_text") == "us_liquid"
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}


def test_feedback_without_artifacts_skips_backtest_row(tmp_path: Path) -> None:
    def locate(_p: Any) -> RunArtifacts:
        raise ArtifactNotFoundError("nothing yet")

    poller, _store, session, rd, _slack = make_poller(tmp_path, [], locate=locate)
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_FEEDBACK, FEEDBACK_CONTENT)]

    poller.poll_once()

    assert rd.submitted == [(TRACE_ID, FEEDBACK_CONTENT)]  # ack still went through
    assert session.calls == []


def test_completion_records_terminal_idea_status(tmp_path: Path) -> None:
    def locate(_p: Any) -> RunArtifacts:
        raise ArtifactNotFoundError("gone")

    poller, store, session, rd, slack = make_poller(
        tmp_path, [page_response("page-idea")], locate=locate
    )
    rd.status_by_trace[TRACE_ID] = RunStatus(finished=True, end_code=0)

    poller.poll_once()

    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"
    assert len(slack.posts) == 1  # completion notice still posted
    (call,) = session.calls
    assert call["method"] == "PATCH"
    assert call["url"].endswith("/v1/pages/page-idea")
    assert call["json"]["properties"] == {"Status": {"select": {"name": "completed"}}}

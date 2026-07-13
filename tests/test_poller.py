"""US-021: hypothesis poller with Approve/Edit/Reject buttons.

Poller behavior is tested against a stubbed rdagent client and a fake Slack
poster; button routing is tested by dispatching real ``block_actions``
payloads through Bolt (harness shared with tests/test_slack_app.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from slack_bolt import App
from slack_bolt.request import BoltRequest

from orchestrator.poller import (
    ACTION_APPROVE,
    ACTION_EDIT,
    ACTION_REJECT,
    EDIT_PROMPT,
    HypothesisPoller,
    edited_payload,
    rejection_payload,
    terminal_status,
)
from orchestrator.rdagent_client import (
    KIND_BASE_FEATURES,
    KIND_FEEDBACK,
    KIND_HYPOTHESIS,
    KIND_INIT_PARAMS,
    ArtifactNotFoundError,
    PendingInteraction,
    RunArtifacts,
    RunStatus,
)
from orchestrator.state import StateStore
from tests.test_slack_app import CHANNEL, dispatch_message, make_app, user_message
from tests.test_summary import write_qlib_res_csv, write_ret_pkl

THREAD = "1751900000.000100"
TRACE_FOLDER = "/home/user/rdq-runs/server_ui/traces"
TRACE_ID = "Finance Whole Pipeline/2026-07-08_10-00-00-000000"
SESSION = f"{TRACE_FOLDER}/{TRACE_ID}"

HYPO_CONTENT: dict[str, Any] = {
    "hypothesis": "Adding 20-day momentum factors improves IC",
    "reason": "Momentum persists in liquid US names",
    "concise_reason": "momentum persists",
    "concise_observation": "obs",
    "concise_justification": "just",
    "concise_knowledge": "know",
    "action": "factor",
}

FEEDBACK_CONTENT: dict[str, Any] = {
    "decision": True,
    "observations": "IC improved",
    "reason": "better",
    "hypothesis_evaluation": "good",
    "new_hypothesis": "",
    "code_change_summary": "",
}


def interaction(kind: str, content: dict[str, Any], ts: str = "t1") -> PendingInteraction:
    return PendingInteraction(trace_id=TRACE_ID, timestamp=ts, kind=kind, content=content)


class StubRdAgent:
    """Stubbed rdagent client: canned pending lists, recorded submits."""

    def __init__(self) -> None:
        self.pending_by_trace: dict[str, list[PendingInteraction]] = {}
        self.status_by_trace: dict[str, RunStatus] = {}
        self.submitted: list[tuple[str, Any]] = []
        self.fail_submit = False
        self.broken_sessions: set[str] = set()

    def pending(self, trace_id: str) -> list[PendingInteraction]:
        return list(self.pending_by_trace.get(trace_id, []))

    def submit(self, trace_id: str, payload: Any) -> None:
        if self.fail_submit:
            raise RuntimeError("server_ui unreachable")
        self.submitted.append((trace_id, payload))

    def trace_id_of(self, session_path: str) -> str:
        if session_path in self.broken_sessions:
            raise RuntimeError(f"not under the trace folder: {session_path}")
        return str(Path(session_path).relative_to(TRACE_FOLDER))

    def status(self, trace_id: str) -> RunStatus:
        return self.status_by_trace.get(trace_id, RunStatus(finished=False))


class FakeSlack:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.fail = False
        self.fail_upload = False

    def chat_postMessage(self, **kwargs: Any) -> None:  # noqa: N802 - slack_sdk casing
        if self.fail:
            raise RuntimeError("slack down")
        self.posts.append(kwargs)

    def files_upload_v2(self, **kwargs: Any) -> None:  # noqa: N802 - slack_sdk casing
        if self.fail or self.fail_upload:
            raise RuntimeError("missing_scope: files:write")
        self.uploads.append(kwargs)


class RecordingSay:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    @property
    def texts(self) -> list[str]:
        return [c.get("text", "") for c in self.calls]


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    s = StateStore(tmp_path / "state.sqlite")
    s.create_run(THREAD, SESSION, universe="us_liquid")
    return s


@pytest.fixture
def rd() -> StubRdAgent:
    return StubRdAgent()


@pytest.fixture
def slack() -> FakeSlack:
    return FakeSlack()


@pytest.fixture
def poller(store: StateStore, rd: StubRdAgent, slack: FakeSlack) -> HypothesisPoller:
    return HypothesisPoller(store, rd, slack, CHANNEL)


def posted_hypothesis_row(store: StateStore):
    rows = store.list_pending_interactions(THREAD, status="pending")
    assert len(rows) == 1
    return rows[0]


# --- polling ----------------------------------------------------------------


def test_poll_posts_hypothesis_with_buttons_to_owning_thread(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    assert poller.poll_once() == 1
    (post,) = slack.posts
    assert post["channel"] == CHANNEL
    assert post["thread_ts"] == THREAD
    assert HYPO_CONTENT["hypothesis"] in post["text"]

    section, actions = post["blocks"]
    assert HYPO_CONTENT["hypothesis"] in section["text"]["text"]
    row = posted_hypothesis_row(store)
    labels = {e["action_id"]: e for e in actions["elements"]}
    assert set(labels) == {ACTION_APPROVE, ACTION_EDIT, ACTION_REJECT}
    assert all(e["value"] == str(row.id) for e in actions["elements"])
    assert row.payload == {"trace_id": TRACE_ID, "kind": KIND_HYPOTHESIS, "content": HYPO_CONTENT}


def test_same_hypothesis_not_posted_twice(
    poller: HypothesisPoller, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    assert poller.poll_once() == 1
    assert poller.poll_once() == 0  # answered requests stay in the stream forever
    assert len(slack.posts) == 1


def test_pending_row_survives_poller_restart(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    poller.poll_once()

    # "restart": fresh store on the same file, fresh poller instance
    store2 = StateStore(store.db_path)
    poller2 = HypothesisPoller(store2, rd, slack, CHANNEL)
    assert poller2.poll_once() == 0  # not re-posted
    rows = store2.list_pending_interactions(THREAD, status="pending")
    assert len(rows) == 1  # ...but still pending and actionable
    assert rows[0].payload["content"] == HYPO_CONTENT
    assert len(slack.posts) == 1


def test_preseeded_interaction_kinds_are_skipped(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [
        interaction(KIND_INIT_PARAMS, {"user_instruction": None}, ts="t1"),
        interaction(
            KIND_BASE_FEATURES, {"features": {}, "feature_validation_msg": ""}, ts="t2"
        ),
    ]
    assert poller.poll_once() == 0
    assert slack.posts == []
    assert rd.submitted == []
    assert store.list_pending_interactions(THREAD) == []


def test_feedback_is_auto_acknowledged_once(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_FEEDBACK, FEEDBACK_CONTENT)]
    poller.poll_once()
    assert rd.submitted == [(TRACE_ID, FEEDBACK_CONTENT)]  # unchanged
    assert slack.posts == []  # no operator buttons for feedback
    rows = store.list_pending_interactions(THREAD, status="auto_approved")
    assert len(rows) == 1

    poller.poll_once()
    assert len(rd.submitted) == 1  # dedup: never submitted twice


def test_feedback_not_acked_while_hypothesis_awaits_operator(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [
        interaction(KIND_HYPOTHESIS, HYPO_CONTENT, ts="t1"),
        interaction(KIND_FEEDBACK, FEEDBACK_CONTENT, ts="t2"),
    ]
    poller.poll_once()
    assert rd.submitted == []  # answers are FIFO; nothing may jump the queue
    assert len(slack.posts) == 1

    # once the hypothesis is resolved, the later feedback is acknowledged
    say = RecordingSay()
    poller.approve(posted_hypothesis_row(store).id, say)
    poller.poll_once()
    assert rd.submitted == [(TRACE_ID, HYPO_CONTENT), (TRACE_ID, FEEDBACK_CONTENT)]


def test_slack_failure_frees_key_so_next_poll_retries(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    slack.fail = True
    assert poller.poll_once() == 0
    assert store.list_pending_interactions(THREAD) == []  # key freed

    slack.fail = False
    assert poller.poll_once() == 1
    assert len(slack.posts) == 1


def test_broken_run_does_not_starve_others(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    broken_session = f"{TRACE_FOLDER}/Finance Whole Pipeline/broken"
    store.create_run("1751900099.000900", broken_session, universe="us_liquid")
    rd.broken_sessions.add(broken_session)
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    poller = HypothesisPoller(store, rd, slack, CHANNEL)
    assert poller.poll_once() == 1  # the healthy run still got its post


# --- run completion (US-022) ---------------------------------------------------


FINISHED_OK = RunStatus(finished=True, end_code=0, error_msg="RD-Agent process has completed.")


def make_artifacts(tmp_path: Path, *, with_ret: bool = True) -> RunArtifacts:
    """A real fixture workspace: qlib_res.csv (+ ret.pkl) on disk."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    csv = write_qlib_res_csv(workspace / "qlib_res.csv")
    ret = write_ret_pkl(workspace / "ret.pkl") if with_ret else None
    return RunArtifacts(
        workspace_path=workspace,
        qlib_res_csv=csv,
        ret_pkl=ret,
        source_pkl=workspace / "source.pkl",
    )


def completion_poller(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, artifacts: RunArtifacts
) -> HypothesisPoller:
    return HypothesisPoller(store, rd, slack, CHANNEL, locate=lambda _session: artifacts)


def test_finished_run_posts_metrics_summary_and_chart(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path))
    poller.poll_once()

    (post,) = slack.posts
    assert post["channel"] == CHANNEL
    assert post["thread_ts"] == THREAD
    for label in ("IC", "ICIR", "Rank IC", "ARR", "IR", "MDD", "Sharpe"):
        assert f"*{label}:*" in post["text"]
    assert "*IC:* 0.0432" in post["text"]
    assert "*MDD:* -8.40%" in post["text"]

    (upload,) = slack.uploads
    assert upload["thread_ts"] == THREAD
    assert upload["filename"] == "equity_curve.png"
    assert upload["file"].startswith(b"\x89PNG\r\n\x1a\n")

    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"


def test_finished_run_skips_interaction_polling(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path))
    assert poller.poll_once() == 0
    assert store.list_pending_interactions(THREAD) == []  # no buttons for a dead run
    assert len(slack.posts) == 1  # the summary only


def test_completion_is_handled_exactly_once(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path))
    poller.poll_once()
    poller.poll_once()  # run is no longer 'running' — nothing reposted
    assert len(slack.posts) == 1
    assert len(slack.uploads) == 1


def test_finished_without_ret_pkl_posts_summary_without_chart(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path, with_ret=False))
    poller.poll_once()
    (post,) = slack.posts
    assert "*IC:* 0.0432" in post["text"]
    assert "*Sharpe:* n/a" in post["text"]
    assert "equity chart unavailable" in post["text"]
    assert slack.uploads == []
    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"


def test_corrupt_ret_pkl_still_posts_metrics_and_completes(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    artifacts = make_artifacts(tmp_path)
    assert artifacts.ret_pkl is not None
    artifacts.ret_pkl.write_bytes(b"garbage")
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, artifacts)
    poller.poll_once()
    (post,) = slack.posts
    assert "*IC:* 0.0432" in post["text"]  # csv metrics survive the bad pkl
    assert "*Sharpe:* n/a" in post["text"]
    assert "equity chart unavailable" in post["text"]
    assert slack.uploads == []
    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"  # no retry loop


def test_chart_upload_failure_still_finalizes_run(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    """A persistent chart-upload failure (e.g. bot token missing files:write)
    must not abort completion — otherwise the poller re-posts the summary
    every cycle forever. The metrics text posts; the run finalizes."""
    slack.fail_upload = True
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path))
    poller.poll_once()
    poller.poll_once()  # run is no longer 'running' — not reposted
    assert len(slack.posts) == 1  # summary text posted exactly once, no loop
    assert "*IC:* 0.0432" in slack.posts[0]["text"]
    assert slack.uploads == []  # upload attempted, swallowed
    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"


def test_failed_run_without_artifacts_reports_honestly(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    # default poller = real locate_artifacts; SESSION does not exist on disk
    rd.status_by_trace[TRACE_ID] = RunStatus(finished=True, end_code=2, error_msg="boom")
    poller.poll_once()
    (post,) = slack.posts
    assert "failed" in post["text"]
    assert "boom" in post["text"]
    assert "No backtest artifacts" in post["text"]
    run = store.get_run(THREAD)
    assert run is not None and run.status == "failed"


def test_stopped_run_is_marked_stopped(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack
) -> None:
    rd.status_by_trace[TRACE_ID] = RunStatus(
        finished=True, end_code=-1, error_msg="RD-Agent process was stopped by user."
    )

    def locate(_session: str | Path) -> RunArtifacts:
        raise ArtifactNotFoundError("no artifacts for a stopped run")

    poller = HypothesisPoller(store, rd, slack, CHANNEL, locate=locate)
    poller.poll_once()
    (post,) = slack.posts
    assert "stopped" in post["text"]
    run = store.get_run(THREAD)
    assert run is not None and run.status == "stopped"


def test_slack_failure_keeps_run_running_for_retry(
    store: StateStore, rd: StubRdAgent, slack: FakeSlack, tmp_path: Path
) -> None:
    rd.status_by_trace[TRACE_ID] = FINISHED_OK
    poller = completion_poller(store, rd, slack, make_artifacts(tmp_path))
    slack.fail = True
    poller.poll_once()  # must not raise (per-run catch) and must not close the run
    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"

    slack.fail = False
    poller.poll_once()
    assert len(slack.posts) == 1
    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"


def test_terminal_status_mapping() -> None:
    assert terminal_status(RunStatus(finished=True, end_code=0)) == "completed"
    assert terminal_status(RunStatus(finished=True, end_code=None)) == "completed"
    assert terminal_status(RunStatus(finished=True, end_code=-1)) == "stopped"
    assert terminal_status(RunStatus(finished=True, end_code=137)) == "failed"


# --- operator actions ---------------------------------------------------------


def start_pending(poller: HypothesisPoller, rd: StubRdAgent, store: StateStore) -> int:
    rd.pending_by_trace[TRACE_ID] = [interaction(KIND_HYPOTHESIS, HYPO_CONTENT)]
    poller.poll_once()
    return posted_hypothesis_row(store).id


def test_approve_submits_hypothesis_unchanged(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent
) -> None:
    row_id = start_pending(poller, rd, store)
    say = RecordingSay()
    poller.approve(row_id, say)
    assert rd.submitted == [(TRACE_ID, HYPO_CONTENT)]
    assert store.get_pending_interaction(row_id).status == "approved"  # type: ignore[union-attr]
    assert any("approved" in t for t in say.texts)
    assert say.calls[-1]["thread_ts"] == THREAD


def test_edit_round_trip_merges_operator_text(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent
) -> None:
    row_id = start_pending(poller, rd, store)
    say = RecordingSay()
    poller.request_edit(row_id, say)
    assert store.get_pending_interaction(row_id).status == "editing"  # type: ignore[union-attr]
    assert EDIT_PROMPT in say.texts

    consumed = poller.consume_edit_reply(THREAD, "Use 60-day momentum instead", say)
    assert consumed is True
    assert rd.submitted == [
        (TRACE_ID, {**HYPO_CONTENT, "hypothesis": "Use 60-day momentum instead"})
    ]
    assert store.get_pending_interaction(row_id).status == "edited"  # type: ignore[union-attr]


def test_consume_edit_reply_without_editing_interaction_is_noop(
    poller: HypothesisPoller, rd: StubRdAgent
) -> None:
    say = RecordingSay()
    assert poller.consume_edit_reply(THREAD, "just chatting", say) is False
    assert rd.submitted == []
    assert say.calls == []


def test_reject_requests_a_new_proposal(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent
) -> None:
    row_id = start_pending(poller, rd, store)
    say = RecordingSay()
    poller.reject(row_id, say)
    (submitted,) = rd.submitted
    trace_id, payload = submitted
    assert trace_id == TRACE_ID
    # same constructor keys (upstream rebuilds via type(hypo)(**payload)) ...
    assert set(payload) == set(HYPO_CONTENT)
    # ... with the hypothesis text replaced by the rejection instruction
    assert "REJECTED" in payload["hypothesis"]
    assert HYPO_CONTENT["hypothesis"] in payload["hypothesis"]
    assert "different" in payload["hypothesis"]
    assert store.get_pending_interaction(row_id).status == "rejected"  # type: ignore[union-attr]


def test_second_click_reports_already_handled(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent
) -> None:
    row_id = start_pending(poller, rd, store)
    say = RecordingSay()
    poller.approve(row_id, say)
    poller.reject(row_id, say)
    assert len(rd.submitted) == 1  # second click submitted nothing
    assert any("already handled" in t for t in say.texts)


def test_unknown_interaction_id_reports_and_submits_nothing(
    poller: HypothesisPoller, rd: StubRdAgent
) -> None:
    say = RecordingSay()
    poller.approve(999, say)
    assert rd.submitted == []
    assert any("Unknown interaction" in t for t in say.texts)


def test_submit_failure_keeps_row_actionable(
    poller: HypothesisPoller, store: StateStore, rd: StubRdAgent
) -> None:
    row_id = start_pending(poller, rd, store)
    say = RecordingSay()
    rd.fail_submit = True
    poller.approve(row_id, say)
    assert store.get_pending_interaction(row_id).status == "pending"  # type: ignore[union-attr]
    assert any("failed" in t for t in say.texts)

    rd.fail_submit = False
    poller.approve(row_id, say)
    assert rd.submitted == [(TRACE_ID, HYPO_CONTENT)]
    assert store.get_pending_interaction(row_id).status == "approved"  # type: ignore[union-attr]


def test_payload_helpers() -> None:
    edited = edited_payload(HYPO_CONTENT, "  new text  ")
    assert edited["hypothesis"] == "new text"
    assert edited["action"] == HYPO_CONTENT["action"]
    rejected = rejection_payload(HYPO_CONTENT)
    assert set(rejected) == set(HYPO_CONTENT)
    assert HYPO_CONTENT["hypothesis"] in rejected["hypothesis"]


# --- Bolt routing (buttons + edit-reply interception) --------------------------


class FakeInteractions:
    """Stub InteractionHandler recording which method Bolt routed to."""

    def __init__(self, consume: bool = False) -> None:
        self.consume = consume
        self.calls: list[tuple[str, Any]] = []

    def approve(self, interaction_id: int, say: Any) -> None:
        self.calls.append(("approve", interaction_id))

    def reject(self, interaction_id: int, say: Any) -> None:
        self.calls.append(("reject", interaction_id))

    def request_edit(self, interaction_id: int, say: Any) -> None:
        self.calls.append(("request_edit", interaction_id))

    def consume_edit_reply(self, thread_ts: str, text: str, say: Any) -> bool:
        self.calls.append(("consume_edit_reply", (thread_ts, text)))
        return self.consume


def dispatch_action(app: App, action_id: str, value: str) -> None:
    body = {
        "type": "block_actions",
        "token": "ignored",
        "api_app_id": "A0APP",
        "team": {"id": "T0TEAM"},
        "user": {"id": "U0USER"},
        "trigger_id": "123.456.789",
        "container": {
            "type": "message",
            "message_ts": "1751900001.000000",
            "channel_id": CHANNEL,
            "is_ephemeral": False,
        },
        "channel": {"id": CHANNEL, "name": "quant-research"},
        "message": {"type": "message", "ts": "1751900001.000000", "thread_ts": THREAD},
        "response_url": "https://hooks.slack.com/actions/T0TEAM/123/xyz",
        "actions": [
            {
                "type": "button",
                "action_id": action_id,
                "block_id": "hypothesis_7",
                "text": {"type": "plain_text", "text": "x"},
                "value": value,
                "action_ts": "1751900002.000000",
            }
        ],
    }
    request = BoltRequest(body=json.dumps(body), mode="socket_mode")
    response = app.dispatch(request)
    assert response.status == 200


@pytest.mark.parametrize(
    ("action_id", "method"),
    [
        (ACTION_APPROVE, "approve"),
        (ACTION_EDIT, "request_edit"),
        (ACTION_REJECT, "reject"),
    ],
)
def test_button_click_routes_to_interaction_handler(
    monkeypatch: pytest.MonkeyPatch, action_id: str, method: str
) -> None:
    interactions = FakeInteractions()
    app, _client, _conversation = make_app(monkeypatch, interactions=interactions)
    dispatch_action(app, action_id, value="7")
    assert interactions.calls == [(method, 7)]


def test_edit_reply_is_consumed_before_the_conversational_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactions = FakeInteractions(consume=True)
    app, _client, conversation = make_app(monkeypatch, interactions=interactions)
    dispatch_message(
        app, user_message("my edit text", ts="1751900010.000200", thread_ts=THREAD)
    )
    assert interactions.calls == [("consume_edit_reply", (THREAD, "my edit text"))]
    assert conversation.calls == []  # the core never saw the edit text


def test_non_edit_message_still_reaches_the_conversational_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactions = FakeInteractions(consume=False)
    app, _client, conversation = make_app(monkeypatch, interactions=interactions)
    dispatch_message(
        app, user_message("regular chat", ts="1751900020.000300", thread_ts=THREAD)
    )
    assert conversation.calls == [(THREAD, "regular chat")]

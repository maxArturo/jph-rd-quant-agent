"""US-039: OneCLI approvals bridge — pending -> Slack buttons -> decision submit.

Client tests run against a FakeSession (mocked OneCLI HTTP); bridge tests use
a stub client + recording Slack poster; app-wiring tests dispatch real Bolt
block_actions requests through App.dispatch() with a mocked WebClient.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from slack_bolt import App
from slack_bolt.request import BoltRequest
from slack_sdk import WebClient

from orchestrator.app import create_app
from orchestrator.approvals import (
    ACTION_ONECLI_APPROVE,
    ACTION_ONECLI_DENY,
    ApprovalRequest,
    ApprovalsApiError,
    ApprovalsBridge,
    OneCliApprovalsClient,
    approval_blocks,
    format_approval_text,
    parse_expiry,
)
from orchestrator.config import DEFAULT_ONECLI_URL, SlackConfig, load_onecli_url

CHANNEL = "C0TESTCHAN"

GATEWAY_URL_RESPONSE = {"url": "http://localhost:10255"}

RAW_REQUEST = {
    "id": "req-abc-123",
    "method": "POST",
    "url": "https://api.alpaca.markets/v2/orders",
    "host": "api.alpaca.markets",
    "path": "/v2/orders",
    "headers": {},
    "bodyPreview": '{"symbol": "AAPL", "qty": "5"}',
    "agent": {"id": "uuid-1", "name": "rdq-exec-live", "externalId": "rdq-exec-live"},
    "createdAt": "2026-07-09T12:00:00.000Z",
    "expiresAt": "2026-07-09T12:03:00.000Z",
}


# --- fakes ------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeSession:
    """Returns queued responses and records every request."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        params: Any = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {"method": method, "url": url, "params": params, "json": json, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError("FakeSession ran out of queued responses")
        return self._responses.pop(0)


def make_client(responses: list[FakeResponse]) -> tuple[OneCliApprovalsClient, FakeSession]:
    session = FakeSession(responses)
    client = OneCliApprovalsClient(base_url="http://onecli.test:10254", session=session)
    return client, session


class StubApprovalsClient:
    """Stands in for OneCliApprovalsClient in bridge tests."""

    def __init__(self) -> None:
        self.base_url = "http://onecli.test:10254"
        self.pending: list[ApprovalRequest] = []
        self.poll_calls: list[tuple[str, ...]] = []
        self.decisions: list[tuple[str, str]] = []
        self.submit_result: bool = True
        self.submit_error: Exception | None = None

    def poll_pending(self, exclude: tuple[str, ...] = ()) -> list[ApprovalRequest]:
        self.poll_calls.append(exclude)
        return list(self.pending)

    def submit_decision(self, request_id: str, decision: str) -> bool:
        if self.submit_error is not None:
            raise self.submit_error
        self.decisions.append((request_id, decision))
        return self.submit_result


class FakeSlack:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.fail_next = False

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("slack down")
        self.posts.append(kwargs)
        return {"ok": True, "ts": f"1751900{len(self.posts):03d}.000000"}


class RecordingSay:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            kwargs = {"text": args[0], **kwargs}
        self.calls.append(kwargs)


def sample_request(**overrides: Any) -> ApprovalRequest:
    return ApprovalRequest.from_api({**RAW_REQUEST, **overrides})


def make_bridge(
    now: datetime | None = None,
) -> tuple[ApprovalsBridge, StubApprovalsClient, FakeSlack]:
    client = StubApprovalsClient()
    slack = FakeSlack()
    fixed_now = now or datetime(2026, 7, 9, 12, 1, 0, tzinfo=timezone.utc)
    bridge = ApprovalsBridge(
        client,  # type: ignore[arg-type]  # structural stand-in
        slack,
        channel_id=CHANNEL,
        now=lambda: fixed_now,
    )
    return bridge, client, slack


# --- client: gateway resolution ----------------------------------------------


def test_gateway_url_resolved_once_and_cached() -> None:
    client, session = make_client(
        [
            FakeResponse(200, GATEWAY_URL_RESPONSE),
            FakeResponse(200, {"requests": [], "timeoutSeconds": 180}),
            FakeResponse(200, {"requests": [], "timeoutSeconds": 180}),
        ]
    )
    client.poll_pending()
    client.poll_pending()
    gateway_calls = [c for c in session.calls if c["url"].endswith("/api/gateway-url")]
    assert len(gateway_calls) == 1
    assert gateway_calls[0]["url"] == "http://onecli.test:10254/api/gateway-url"


def test_explicit_gateway_url_skips_resolution() -> None:
    session = FakeSession([FakeResponse(200, {"requests": [], "timeoutSeconds": 180})])
    client = OneCliApprovalsClient(
        base_url="http://onecli.test:10254",
        gateway_url="http://gw.test:10255/",
        session=session,
    )
    client.poll_pending()
    assert session.calls[0]["url"] == "http://gw.test:10255/api/approvals/pending"


def test_gateway_url_failure_raises() -> None:
    client, _ = make_client([FakeResponse(500)])
    with pytest.raises(ApprovalsApiError, match="gateway URL"):
        client.poll_pending()


# --- client: polling ----------------------------------------------------------


def test_poll_pending_parses_requests() -> None:
    client, session = make_client(
        [
            FakeResponse(200, GATEWAY_URL_RESPONSE),
            FakeResponse(200, {"requests": [RAW_REQUEST], "timeoutSeconds": 180}),
        ]
    )
    requests_out = client.poll_pending()
    assert len(requests_out) == 1
    req = requests_out[0]
    assert req.id == "req-abc-123"
    assert req.method == "POST"
    assert req.host == "api.alpaca.markets"
    assert req.path == "/v2/orders"
    assert req.body_preview == '{"symbol": "AAPL", "qty": "5"}'
    assert req.agent_name == "rdq-exec-live"
    assert req.expires_at == "2026-07-09T12:03:00.000Z"
    poll_call = session.calls[-1]
    assert poll_call["url"] == "http://localhost:10255/api/approvals/pending"
    assert poll_call["params"] is None
    # Client timeout must exceed the server's ~30s long-poll hold.
    assert poll_call["timeout"] > 30


def test_poll_pending_sends_exclude_param() -> None:
    client, session = make_client(
        [
            FakeResponse(200, GATEWAY_URL_RESPONSE),
            FakeResponse(200, {"requests": [], "timeoutSeconds": 180}),
        ]
    )
    client.poll_pending(exclude=("id-1", "id-2"))
    assert session.calls[-1]["params"] == {"exclude": "id-1,id-2"}


def test_poll_pending_http_error_raises() -> None:
    client, _ = make_client([FakeResponse(200, GATEWAY_URL_RESPONSE), FakeResponse(503)])
    with pytest.raises(ApprovalsApiError, match="503"):
        client.poll_pending()


# --- client: decisions ---------------------------------------------------------


def test_submit_approve_posts_decision() -> None:
    client, session = make_client(
        [FakeResponse(200, GATEWAY_URL_RESPONSE), FakeResponse(200, {"ok": True})]
    )
    assert client.submit_decision("req-abc-123", "approve") is True
    call = session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == "http://localhost:10255/api/approvals/req-abc-123/decision"
    assert call["json"] == {"decision": "approve"}


def test_submit_deny_posts_decision() -> None:
    client, session = make_client(
        [FakeResponse(200, GATEWAY_URL_RESPONSE), FakeResponse(200, {"ok": True})]
    )
    assert client.submit_decision("req-abc-123", "deny") is True
    assert session.calls[-1]["json"] == {"decision": "deny"}


def test_submit_410_reports_expired_not_error() -> None:
    client, _ = make_client([FakeResponse(200, GATEWAY_URL_RESPONSE), FakeResponse(410)])
    assert client.submit_decision("req-abc-123", "approve") is False


def test_submit_http_error_raises() -> None:
    client, _ = make_client([FakeResponse(200, GATEWAY_URL_RESPONSE), FakeResponse(500)])
    with pytest.raises(ApprovalsApiError, match="500"):
        client.submit_decision("req-abc-123", "approve")


def test_submit_invalid_decision_rejected_before_http() -> None:
    client, session = make_client([])
    with pytest.raises(ValueError, match="decision"):
        client.submit_decision("req-abc-123", "maybe")
    assert session.calls == []


# --- formatting -----------------------------------------------------------------


def test_format_approval_text_names_agent_and_request() -> None:
    text = format_approval_text(sample_request())
    assert "rdq-exec-live" in text
    assert "`POST api.alpaca.markets/v2/orders`" in text
    assert '"symbol": "AAPL"' in text
    assert "2026-07-09T12:03:00.000Z" in text


def test_approval_blocks_carry_both_buttons_with_request_id() -> None:
    blocks = approval_blocks(sample_request())
    actions = blocks[1]["elements"]
    by_action = {element["action_id"]: element for element in actions}
    assert set(by_action) == {ACTION_ONECLI_APPROVE, ACTION_ONECLI_DENY}
    assert all(element["value"] == "req-abc-123" for element in actions)


def test_parse_expiry_handles_zulu_and_garbage() -> None:
    parsed = parse_expiry("2026-07-09T12:03:00.000Z")
    assert parsed == datetime(2026, 7, 9, 12, 3, 0, tzinfo=timezone.utc)
    assert parse_expiry("not-a-date") is None
    assert parse_expiry("") is None


# --- bridge: pending -> post -----------------------------------------------------


def test_pending_request_posted_to_channel_with_buttons() -> None:
    bridge, client, slack = make_bridge()
    client.pending = [sample_request()]
    assert bridge.poll_once() == 1
    assert len(slack.posts) == 1
    post = slack.posts[0]
    assert post["channel"] == CHANNEL
    assert "api.alpaca.markets" in post["text"]
    action_ids = {el["action_id"] for el in post["blocks"][1]["elements"]}
    assert action_ids == {ACTION_ONECLI_APPROVE, ACTION_ONECLI_DENY}


def test_same_request_not_posted_twice() -> None:
    bridge, client, slack = make_bridge()
    client.pending = [sample_request()]
    bridge.poll_once()
    bridge.poll_once()
    assert len(slack.posts) == 1
    # The second poll excluded the already-posted id.
    assert client.poll_calls[1] == ("req-abc-123",)


def test_failed_slack_post_retries_next_poll() -> None:
    bridge, client, slack = make_bridge()
    client.pending = [sample_request()]
    slack.fail_next = True
    assert bridge.poll_once() == 0
    assert bridge.poll_once() == 1  # not marked posted, so it retries
    assert len(slack.posts) == 1


def test_expired_posted_request_pruned_from_exclude() -> None:
    past_expiry = datetime(2026, 7, 9, 12, 5, 0, tzinfo=timezone.utc)  # after expiresAt
    bridge, client, slack = make_bridge(now=past_expiry)
    client.pending = [sample_request()]
    bridge.poll_once()
    client.pending = []
    bridge.poll_once()
    assert client.poll_calls[1] == ()  # pruned: no longer excluded


# --- bridge: decisions ------------------------------------------------------------


def test_approve_submits_and_confirms_in_thread() -> None:
    bridge, client, _ = make_bridge()
    client.pending = [sample_request()]
    bridge.poll_once()
    say = RecordingSay()
    bridge.approve("req-abc-123", say)
    assert client.decisions == [("req-abc-123", "approve")]
    assert "Approved" in say.calls[0]["text"]
    assert say.calls[0]["thread_ts"]  # threaded under the approval message


def test_deny_submits_and_confirms() -> None:
    bridge, client, _ = make_bridge()
    client.pending = [sample_request()]
    bridge.poll_once()
    say = RecordingSay()
    bridge.deny("req-abc-123", say)
    assert client.decisions == [("req-abc-123", "deny")]
    assert "Denied" in say.calls[0]["text"]


def test_decision_after_restart_still_submits() -> None:
    # Fresh bridge with no posted state (simulated restart): the button click
    # still resolves — the decision endpoint only needs the request id.
    bridge, client, _ = make_bridge()
    say = RecordingSay()
    bridge.approve("req-abc-123", say)
    assert client.decisions == [("req-abc-123", "approve")]
    assert "Approved" in say.calls[0]["text"]
    assert "thread_ts" not in say.calls[0]  # no message ts known post-restart


def test_expired_decision_reports_timeout_denial() -> None:
    bridge, client, _ = make_bridge()
    client.submit_result = False  # gateway said 410 Gone
    say = RecordingSay()
    bridge.approve("req-abc-123", say)
    assert "expired" in say.calls[0]["text"]


def test_submit_failure_points_at_web_ui_fallback() -> None:
    bridge, client, _ = make_bridge()
    client.pending = [sample_request()]
    bridge.poll_once()
    client.submit_error = ApprovalsApiError("POST ... returned HTTP 502")
    say = RecordingSay()
    bridge.deny("req-abc-123", say)
    text = say.calls[0]["text"]
    assert "OneCLI web UI" in text
    assert client.base_url in text
    # Not resolved locally — the buttons remain actionable for a retry.
    client.submit_error = None
    bridge.deny("req-abc-123", say)
    assert client.decisions == [("req-abc-123", "deny")]


# --- config ----------------------------------------------------------------------


def test_load_onecli_url_default_and_override(tmp_path: Any) -> None:
    assert load_onecli_url(env_file=tmp_path / "nope.env", environ={}) == DEFAULT_ONECLI_URL
    env_file = tmp_path / ".env"
    env_file.write_text("ONECLI_URL=http://filehost:1\n")
    assert load_onecli_url(env_file=env_file, environ={}) == "http://filehost:1"
    assert (
        load_onecli_url(env_file=env_file, environ={"ONECLI_URL": "http://envhost:2"})
        == "http://envhost:2"
    )


# --- Bolt wiring -------------------------------------------------------------------


class FakeConversation:
    def handle_message(self, thread_ts: str, text: str, say: Any) -> str:
        return "ok"


class FakeApprovalsHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def approve(self, request_id: str, say: Any) -> None:
        self.calls.append(("approve", request_id))

    def deny(self, request_id: str, say: Any) -> None:
        self.calls.append(("deny", request_id))


def make_app(monkeypatch: pytest.MonkeyPatch) -> tuple[App, FakeApprovalsHandler]:
    client = MagicMock(spec=WebClient)
    client.token = "xoxb-test"
    client.base_url = "https://slack.com/api/"
    client.timeout = 30
    client.ssl = None
    client.proxy = None
    client.headers = {}
    client.logger = logging.getLogger("test-webclient")
    client.retry_handlers = []
    handler = FakeApprovalsHandler()
    app = create_app(
        SlackConfig(bot_token="xoxb-test", app_token="xapp-test", channel_id=CHANNEL),
        FakeConversation(),
        approvals=handler,
        client=client,
        token_verification_enabled=False,
        process_before_response=True,
    )
    monkeypatch.setattr("slack_bolt.app.app.WebClient", lambda **_kwargs: client)
    return app, handler


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
        "message": {"type": "message", "ts": "1751900001.000000"},
        "response_url": "https://hooks.slack.com/actions/T0TEAM/123/xyz",
        "actions": [
            {
                "type": "button",
                "action_id": action_id,
                "block_id": "onecli_approval_req-abc-123",
                "text": {"type": "plain_text", "text": "x"},
                "value": "req-abc-123",
                "action_ts": "1751900002.000000",
            }
        ],
    }
    request = BoltRequest(body=json.dumps(body), mode="socket_mode")
    response = app.dispatch(request)
    assert response.status == 200


def test_bolt_routes_approve_button_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, handler = make_app(monkeypatch)
    dispatch_action(app, ACTION_ONECLI_APPROVE, "req-abc-123")
    assert handler.calls == [("approve", "req-abc-123")]


def test_bolt_routes_deny_button_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, handler = make_app(monkeypatch)
    dispatch_action(app, ACTION_ONECLI_DENY, "req-abc-123")
    assert handler.calls == [("deny", "req-abc-123")]


def test_load_max_hypotheses_default_env_and_validation(tmp_path: Any) -> None:
    from orchestrator.config import (
        DEFAULT_MAX_HYPOTHESES,
        ConfigError,
        load_max_hypotheses,
    )

    missing = tmp_path / "nope.env"
    assert load_max_hypotheses(env_file=missing, environ={}) == DEFAULT_MAX_HYPOTHESES
    assert load_max_hypotheses(env_file=missing, environ={"RDQ_MAX_HYPOTHESES": "3"}) == 3

    env_file = tmp_path / "budget.env"
    env_file.write_text("RDQ_MAX_HYPOTHESES=7\n")
    assert load_max_hypotheses(env_file=env_file, environ={}) == 7
    # process env wins over the file
    assert load_max_hypotheses(env_file=env_file, environ={"RDQ_MAX_HYPOTHESES": "4"}) == 4

    import pytest as _pytest

    with _pytest.raises(ConfigError):
        load_max_hypotheses(env_file=missing, environ={"RDQ_MAX_HYPOTHESES": "many"})
    with _pytest.raises(ConfigError):
        load_max_hypotheses(env_file=missing, environ={"RDQ_MAX_HYPOTHESES": "0"})

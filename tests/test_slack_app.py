"""US-006: Slack Bolt Socket Mode skeleton — config loading and message routing.

Routing tests dispatch real Bolt requests through App.dispatch() with a mocked
WebClient, so they exercise Bolt's event routing (channel message -> handler,
thread reply targets thread_ts) without any network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from slack_bolt import App
from slack_bolt.request import BoltRequest
from slack_sdk import WebClient

from orchestrator.app import create_app, handle_message
from orchestrator.config import (
    ConfigError,
    SlackConfig,
    load_slack_config,
    parse_env_file,
)

CHANNEL = "C0TESTCHAN"
CONFIG = SlackConfig(bot_token="xoxb-test", app_token="xapp-test", channel_id=CHANNEL)


class FakeConversation:
    """Stub MessageResponder: records calls and echoes like the real core."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def handle_message(self, thread_ts: str, text: str, say: Any) -> str:
        self.calls.append((thread_ts, text))
        reply = f"Received: {text}"
        say(text=reply, thread_ts=thread_ts)
        return reply


# --- config loading -------------------------------------------------------


def write_env(tmp_path: Path, content: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(content)
    return env_file


def test_load_config_from_env_file(tmp_path: Path) -> None:
    env_file = write_env(
        tmp_path,
        "# comment\n"
        "SLACK_OAUTH_TOKEN=xoxb-abc\n"
        "export SLACK_SOCKET_TOKEN='xapp-def'\n"
        'SLACK_CHANNEL_ID="C123"\n',
    )
    cfg = load_slack_config(env_file=env_file, environ={})
    assert cfg == SlackConfig(bot_token="xoxb-abc", app_token="xapp-def", channel_id="C123")


def test_process_environ_overrides_env_file(tmp_path: Path) -> None:
    env_file = write_env(
        tmp_path,
        "SLACK_OAUTH_TOKEN=xoxb-file\nSLACK_SOCKET_TOKEN=xapp-file\nSLACK_CHANNEL_ID=CFILE\n",
    )
    cfg = load_slack_config(env_file=env_file, environ={"SLACK_CHANNEL_ID": "CENV"})
    assert cfg.channel_id == "CENV"
    assert cfg.bot_token == "xoxb-file"


@pytest.mark.parametrize(
    "missing", ["SLACK_OAUTH_TOKEN", "SLACK_SOCKET_TOKEN", "SLACK_CHANNEL_ID"]
)
def test_missing_variable_raises_named_error(tmp_path: Path, missing: str) -> None:
    values = {
        "SLACK_OAUTH_TOKEN": "xoxb-a",
        "SLACK_SOCKET_TOKEN": "xapp-b",
        "SLACK_CHANNEL_ID": "C1",
    }
    del values[missing]
    env_file = write_env(tmp_path, "".join(f"{k}={v}\n" for k, v in values.items()))
    with pytest.raises(ConfigError, match=missing):
        load_slack_config(env_file=env_file, environ={})


def test_wrong_token_prefixes_rejected(tmp_path: Path) -> None:
    env_file = write_env(
        tmp_path,
        "SLACK_OAUTH_TOKEN=xapp-swapped\nSLACK_SOCKET_TOKEN=xoxb-swapped\nSLACK_CHANNEL_ID=C1\n",
    )
    with pytest.raises(ConfigError, match="xoxb-"):
        load_slack_config(env_file=env_file, environ={})


def test_parse_env_file_missing_file_is_empty(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "nope.env") == {}


# --- message routing through Bolt ----------------------------------------


def make_app(
    monkeypatch: pytest.MonkeyPatch,
    interactions: Any | None = None,
    promotions: Any | None = None,
) -> tuple[App, MagicMock, FakeConversation]:
    client = MagicMock(spec=WebClient)
    client.token = CONFIG.bot_token
    # Instance attributes Bolt's _init_context reads off the singleton client
    # (spec= only mirrors class-level attributes, so set them explicitly).
    client.base_url = "https://slack.com/api/"
    client.timeout = 30
    client.ssl = None
    client.proxy = None
    client.headers = {}
    client.logger = logging.getLogger("test-webclient")
    client.retry_handlers = []
    # process_before_response=True runs listeners synchronously inside
    # dispatch(), so assertions after dispatch never race a worker thread.
    conversation = FakeConversation()
    app = create_app(
        CONFIG,
        conversation,
        interactions=interactions,
        promotions=promotions,
        client=client,
        token_verification_enabled=False,
        process_before_response=True,
    )
    # Bolt >=1.15 constructs a NEW WebClient per request in _init_context, so
    # say()/context.client would bypass an injected mock and hit the network.
    # Patch the symbol Bolt instantiates so the per-request client IS the mock.
    # (Patched after App() — its constructor isinstance-checks the same symbol.)
    monkeypatch.setattr("slack_bolt.app.app.WebClient", lambda **_kwargs: client)
    return app, client, conversation


def dispatch_message(app: App, event: dict[str, Any]) -> None:
    body = {
        "token": "ignored",
        "team_id": "T0TEAM",
        "api_app_id": "A0APP",
        "type": "event_callback",
        "event_id": "Ev0000000001",
        "event_time": int(event["ts"].split(".")[0]) if "ts" in event else 0,
        "event": {"type": "message", **event},
    }
    request = BoltRequest(body=json.dumps(body), mode="socket_mode")
    response = app.dispatch(request)
    assert response.status == 200


def user_message(text: str, ts: str, thread_ts: str | None = None, **extra: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "channel": CHANNEL,
        "user": "U0USER",
        "text": text,
        "ts": ts,
        "channel_type": "channel",
        **extra,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def test_channel_message_reaches_handler_and_replies_in_new_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, conversation = make_app(monkeypatch)
    dispatch_message(app, user_message("momentum idea", ts="1751900000.000100"))
    # the conversational core received the message keyed by its new thread
    assert conversation.calls == [("1751900000.000100", "momentum idea")]
    client.chat_postMessage.assert_called_once()
    kwargs = client.chat_postMessage.call_args.kwargs
    assert kwargs["channel"] == CHANNEL
    # reply threads onto the triggering message
    assert kwargs["thread_ts"] == "1751900000.000100"
    assert "momentum idea" in kwargs["text"]


def test_thread_message_reply_targets_existing_thread_ts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, conversation = make_app(monkeypatch)
    dispatch_message(
        app,
        user_message("follow-up", ts="1751900010.000200", thread_ts="1751900000.000100"),
    )
    # core is keyed by the thread, not the message ts
    assert conversation.calls == [("1751900000.000100", "follow-up")]
    kwargs = client.chat_postMessage.call_args.kwargs
    assert kwargs["thread_ts"] == "1751900000.000100"  # the thread, not the message ts


def test_message_in_other_channel_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, conversation = make_app(monkeypatch)
    event = user_message("hello", ts="1751900020.000300")
    event["channel"] = "C0OTHER"
    dispatch_message(app, event)
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [{"bot_id": "B0BOT"}, {"subtype": "message_changed"}, {"subtype": "channel_join"}],
)
def test_bot_and_subtype_messages_are_ignored(
    extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, conversation = make_app(monkeypatch)
    dispatch_message(app, user_message("noise", ts="1751900030.000400", **extra))
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


# --- handler unit behavior (no Bolt machinery) ----------------------------


def test_handle_message_returns_false_without_reply_for_foreign_channel() -> None:
    say = MagicMock()
    conversation = FakeConversation()
    replied = handle_message(
        {"channel": "C0OTHER", "text": "x", "ts": "1.2"},
        say,
        channel_id=CHANNEL,
        conversation=conversation,
    )
    assert replied is False
    assert conversation.calls == []
    say.assert_not_called()


def test_handle_message_replies_true_for_valid_message() -> None:
    say = MagicMock()
    conversation = FakeConversation()
    replied = handle_message(
        {"channel": CHANNEL, "text": "x", "ts": "1.2", "user": "U1"},
        say,
        channel_id=CHANNEL,
        conversation=conversation,
    )
    assert replied is True
    assert conversation.calls == [("1.2", "x")]
    assert say.call_args.kwargs["thread_ts"] == "1.2"

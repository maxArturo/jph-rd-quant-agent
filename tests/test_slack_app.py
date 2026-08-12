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
    load_trusted_bot_ids,
    parse_env_file,
)

CHANNEL = "C0TESTCHAN"
CONFIG = SlackConfig(bot_token="xoxb-test", app_token="xapp-test", channel_id=CHANNEL)
BOT_USER = "U0TRADING"
TRUSTED_BOT = "B0CLAUDE"


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


# --- SLACK_LIVE_CHANNEL_ID (US-004) ---------------------------------------

BASE_ENV = "SLACK_OAUTH_TOKEN=xoxb-a\nSLACK_SOCKET_TOKEN=xapp-b\nSLACK_CHANNEL_ID=C1\n"


def test_live_channel_absent_yields_none(tmp_path: Path) -> None:
    cfg = load_slack_config(env_file=write_env(tmp_path, BASE_ENV), environ={})
    assert cfg.live_channel_id is None
    # Identical to a pre-live config: the field just defaults to None.
    assert cfg == SlackConfig(bot_token="xoxb-a", app_token="xapp-b", channel_id="C1")


@pytest.mark.parametrize("empty", ["", "   "])
def test_live_channel_empty_yields_none(tmp_path: Path, empty: str) -> None:
    env_file = write_env(tmp_path, BASE_ENV + f"SLACK_LIVE_CHANNEL_ID={empty}\n")
    assert load_slack_config(env_file=env_file, environ={}).live_channel_id is None
    cfg = load_slack_config(
        env_file=write_env(tmp_path, BASE_ENV), environ={"SLACK_LIVE_CHANNEL_ID": empty}
    )
    assert cfg.live_channel_id is None


def test_live_channel_read_from_env_file(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, BASE_ENV + "SLACK_LIVE_CHANNEL_ID=C2LIVE\n")
    assert load_slack_config(env_file=env_file, environ={}).live_channel_id == "C2LIVE"


def test_live_channel_process_environ_overrides_env_file(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, BASE_ENV + "SLACK_LIVE_CHANNEL_ID=CFILE\n")
    cfg = load_slack_config(env_file=env_file, environ={"SLACK_LIVE_CHANNEL_ID": "CENV"})
    assert cfg.live_channel_id == "CENV"


def test_live_channel_equal_to_paper_channel_rejected(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, BASE_ENV + "SLACK_LIVE_CHANNEL_ID=C1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_slack_config(env_file=env_file, environ={})
    assert "SLACK_LIVE_CHANNEL_ID" in str(excinfo.value)
    assert "SLACK_CHANNEL_ID" in str(excinfo.value)


# --- message routing through Bolt ----------------------------------------


def make_app(
    monkeypatch: pytest.MonkeyPatch,
    interactions: Any | None = None,
    promotions: Any | None = None,
    trusted_bot_ids: frozenset[str] = frozenset(),
    bot_user_id: str | None = None,
    config: SlackConfig = CONFIG,
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
        config,
        conversation,
        interactions=interactions,
        promotions=promotions,
        client=client,
        token_verification_enabled=False,
        process_before_response=True,
        trusted_bot_ids=trusted_bot_ids,
        bot_user_id=bot_user_id,
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


# --- trusted bots (RDQ_TRUSTED_BOT_IDS, e.g. Claude in Slack) ---------------


def trusted_app(monkeypatch: pytest.MonkeyPatch) -> tuple[App, MagicMock, FakeConversation]:
    return make_app(
        monkeypatch, trusted_bot_ids=frozenset({TRUSTED_BOT}), bot_user_id=BOT_USER
    )


@pytest.mark.parametrize(
    "extra",
    [
        # Claude posts either as its app user with bot_id set...
        {"bot_id": TRUSTED_BOT, "user": "U0CLAUDE"},
        # ...or as subtype bot_message with a username override and no user.
        {"bot_id": TRUSTED_BOT, "subtype": "bot_message", "username": "Claude [run]"},
    ],
)
def test_trusted_bot_mentioning_us_is_handled(
    extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, conversation = make_app(
        monkeypatch, trusted_bot_ids=frozenset({TRUSTED_BOT}), bot_user_id=BOT_USER
    )
    event = user_message(f"<@{BOT_USER}> today's directive", ts="1751900040.000500")
    event.pop("user")
    event.update(extra)
    dispatch_message(app, event)
    assert conversation.calls == [("1751900040.000500", f"<@{BOT_USER}> today's directive")]
    client.chat_postMessage.assert_called_once()


def test_trusted_bot_without_mention_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # The loop brake: never answer a trusted bot's status chatter unprompted.
    app, client, conversation = trusted_app(monkeypatch)
    event = user_message("nightly digest: all fine", ts="1751900050.000600")
    event.pop("user")
    event.update({"bot_id": TRUSTED_BOT, "subtype": "bot_message"})
    dispatch_message(app, event)
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


def test_trusted_bot_pipe_style_mention_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, conversation = trusted_app(monkeypatch)
    event = user_message(f"<@{BOT_USER}|trading_agent> start", ts="1751900060.000700")
    event.pop("user")
    event.update({"bot_id": TRUSTED_BOT, "subtype": "bot_message"})
    dispatch_message(app, event)
    assert conversation.calls == [("1751900060.000700", f"<@{BOT_USER}|trading_agent> start")]


def test_trusted_bot_edited_message_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, conversation = trusted_app(monkeypatch)
    event = user_message(f"<@{BOT_USER}> edited", ts="1751900070.000800")
    event.pop("user")
    event.update({"bot_id": TRUSTED_BOT, "subtype": "message_changed"})
    dispatch_message(app, event)
    assert conversation.calls == []


def test_trusted_bot_ignored_without_known_bot_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No bot_user_id (misconfigured startup) — fail closed, no bot passes.
    app, client, conversation = make_app(
        monkeypatch, trusted_bot_ids=frozenset({TRUSTED_BOT}), bot_user_id=None
    )
    event = user_message("<@U0ANY> hello", ts="1751900080.000900")
    event.pop("user")
    event["bot_id"] = TRUSTED_BOT
    dispatch_message(app, event)
    assert conversation.calls == []


def test_own_messages_ignored_even_if_own_bot_id_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Misconfiguration guard: our own bot_id on the allowlist must not let
    # the bot converse with itself (its posts carry user == bot_user_id).
    app, client, conversation = make_app(
        monkeypatch, trusted_bot_ids=frozenset({"B0SELF"}), bot_user_id=BOT_USER
    )
    event = user_message(f"<@{BOT_USER}> echo", ts="1751900090.001000")
    event.update({"bot_id": "B0SELF", "user": BOT_USER})
    dispatch_message(app, event)
    assert conversation.calls == []


def test_load_trusted_bot_ids_default_empty(tmp_path: Path) -> None:
    assert load_trusted_bot_ids(tmp_path / "nope.env", environ={}) == frozenset()


def test_load_trusted_bot_ids_parses_csv_with_whitespace(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, "RDQ_TRUSTED_BOT_IDS=B0AAA, B0BBB ,,\n")
    assert load_trusted_bot_ids(env_file, environ={}) == frozenset({"B0AAA", "B0BBB"})


def test_load_trusted_bot_ids_environ_overrides_file(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, "RDQ_TRUSTED_BOT_IDS=B0FILE\n")
    assert load_trusted_bot_ids(env_file, environ={"RDQ_TRUSTED_BOT_IDS": "B0ENV"}) == frozenset(
        {"B0ENV"}
    )


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


# --- dual-channel routing (US-006) -----------------------------------------

LIVE_CHANNEL = "C0LIVECHAN"
CONFIG_LIVE = SlackConfig(
    bot_token="xoxb-test",
    app_token="xapp-test",
    channel_id=CHANNEL,
    live_channel_id=LIVE_CHANNEL,
)


def live_message(text: str, ts: str, **extra: Any) -> dict[str, Any]:
    event = user_message(text, ts, **extra)
    event["channel"] = LIVE_CHANNEL
    return event


def test_live_channel_message_is_actionable_and_replies_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, conversation = make_app(monkeypatch, config=CONFIG_LIVE)
    dispatch_message(app, live_message("promote to live", ts="1751900100.000100"))
    assert conversation.calls == [("1751900100.000100", "promote to live")]
    kwargs = client.chat_postMessage.call_args.kwargs
    # the reply lands in the originating (live) channel, threaded on the message
    assert kwargs["channel"] == LIVE_CHANNEL
    assert kwargs["thread_ts"] == "1751900100.000100"


def test_paper_channel_still_handled_when_live_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, conversation = make_app(monkeypatch, config=CONFIG_LIVE)
    dispatch_message(app, user_message("momentum idea", ts="1751900110.000200"))
    assert conversation.calls == [("1751900110.000200", "momentum idea")]
    assert client.chat_postMessage.call_args.kwargs["channel"] == CHANNEL


def test_unknown_channel_ignored_when_live_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client, conversation = make_app(monkeypatch, config=CONFIG_LIVE)
    event = user_message("hello", ts="1751900120.000300")
    event["channel"] = "C0OTHER"
    dispatch_message(app, event)
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


def test_live_channel_ignored_when_live_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Paper freeze: with live_channel_id unset the app hears only the paper channel.
    app, client, conversation = make_app(monkeypatch)
    dispatch_message(app, live_message("hello", ts="1751900130.000400"))
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


def test_bot_and_subtype_messages_ignored_in_live_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All non-channel filtering applies identically to the live channel.
    app, client, conversation = make_app(monkeypatch, config=CONFIG_LIVE)
    dispatch_message(app, live_message("noise", ts="1751900140.000500", bot_id="B0BOT"))
    dispatch_message(
        app, live_message("edited", ts="1751900150.000600", subtype="message_changed")
    )
    assert conversation.calls == []
    client.chat_postMessage.assert_not_called()


def test_handle_message_accepts_live_channel_via_parameter() -> None:
    say = MagicMock()
    conversation = FakeConversation()
    replied = handle_message(
        {"channel": LIVE_CHANNEL, "text": "x", "ts": "1.2", "user": "U1"},
        say,
        channel_id=CHANNEL,
        conversation=conversation,
        live_channel_id=LIVE_CHANNEL,
    )
    assert replied is True
    assert conversation.calls == [("1.2", "x")]

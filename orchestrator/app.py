"""Slack Bolt Socket Mode app (US-006 skeleton, US-009 conversational core).

Connects to Slack over Socket Mode (no inbound port) and routes messages from
the configured #quant-research channel to the conversational core, which
refines raw ideas into saved research directives and replies in-thread.

Run: ``.venv/bin/python -m orchestrator.app`` (needs SLACK_* in .env; see
orchestrator/config.py). Anthropic auth is injected by the OneCLI proxy, so
start it under ``onecli run --agent rdq-orchestrator``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from slack_bolt import App, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from orchestrator.config import SlackConfig, load_slack_config

logger = logging.getLogger(__name__)


class MessageResponder(Protocol):
    """What the app needs from the conversational core (see ConversationCore)."""

    def handle_message(self, thread_ts: str, text: str, say: Say) -> str: ...


def _is_actionable_user_message(event: dict[str, Any], channel_id: str) -> bool:
    """True for plain user messages in the target channel (top-level or in-thread)."""
    if event.get("channel") != channel_id:
        return False
    if event.get("subtype"):  # message_changed, bot_message, channel_join, ...
        return False
    if event.get("bot_id"):  # never reply to ourselves or other bots
        return False
    return bool(event.get("text") and event.get("ts"))


def handle_message(
    event: dict[str, Any], say: Say, channel_id: str, conversation: MessageResponder
) -> bool:
    """Route one message event to the conversational core. Returns True if handled.

    Replies target the message's thread: for a top-level message the reply
    starts a thread on it (thread_ts = its ts); for a threaded message the
    reply stays in that thread (thread_ts = the event's thread_ts).
    """
    if not _is_actionable_user_message(event, channel_id):
        return False
    thread_ts = event.get("thread_ts") or event["ts"]
    conversation.handle_message(thread_ts, event["text"], say)
    return True


def create_app(
    config: SlackConfig,
    conversation: MessageResponder,
    client: WebClient | None = None,
    token_verification_enabled: bool = True,
    process_before_response: bool = False,
) -> App:
    """Build the Bolt app with the message handler registered.

    ``client``, ``token_verification_enabled`` and ``process_before_response``
    exist for tests (inject a mocked WebClient, skip the auth.test call, run
    listeners synchronously inside dispatch()). Keep process_before_response
    False in production: handlers call Claude (slow) and Slack retries events
    not acked within ~3s — Bolt's default acks first, then runs the listener.
    """
    app = App(
        token=config.bot_token,
        client=client,
        token_verification_enabled=token_verification_enabled,
        process_before_response=process_before_response,
        # Socket Mode: no request-signature verification (no inbound HTTP)
        request_verification_enabled=False,
    )

    @app.event("message")
    def _on_message(event: dict[str, Any], say: Say) -> None:
        handle_message(event, say, config.channel_id, conversation)

    return app


def main() -> None:
    # Heavy imports stay here so tests importing this module don't pay for them.
    from orchestrator.conversation import ConversationCore
    from orchestrator.llm import ModelRouter
    from orchestrator.state import StateStore

    logging.basicConfig(level=logging.INFO)
    config = load_slack_config()
    conversation = ConversationCore(store=StateStore(), router=ModelRouter())
    app = create_app(config, conversation)
    logger.info("starting Socket Mode connection (channel %s)", config.channel_id)
    SocketModeHandler(app, config.app_token).start()


if __name__ == "__main__":
    main()

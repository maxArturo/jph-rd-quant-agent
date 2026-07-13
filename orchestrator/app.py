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


class InteractionHandler(Protocol):
    """What the app needs from the hypothesis poller (see HypothesisPoller)."""

    def approve(self, interaction_id: int, say: Say) -> None: ...

    def reject(self, interaction_id: int, say: Say) -> None: ...

    def request_edit(self, interaction_id: int, say: Say) -> None: ...

    def consume_edit_reply(self, thread_ts: str, text: str, say: Say) -> bool: ...


class PromotionHandler(Protocol):
    """What the app needs from the promotion flow (see PromotionFlow)."""

    def request_promotion(self, thread_ts: str, say: Say) -> None: ...

    def confirm_promotion(self, thread_ts: str, say: Say) -> None: ...

    def cancel_promotion(self, thread_ts: str, say: Say) -> None: ...


class ApprovalsHandler(Protocol):
    """What the app needs from the OneCLI approvals bridge (see ApprovalsBridge)."""

    def approve(self, request_id: str, say: Say) -> None: ...

    def deny(self, request_id: str, say: Say) -> None: ...


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
    event: dict[str, Any],
    say: Say,
    channel_id: str,
    conversation: MessageResponder,
    interactions: InteractionHandler | None = None,
) -> bool:
    """Route one message event to the conversational core. Returns True if handled.

    Replies target the message's thread: for a top-level message the reply
    starts a thread on it (thread_ts = its ts); for a threaded message the
    reply stays in that thread (thread_ts = the event's thread_ts).

    When the thread has a hypothesis in the Edit round-trip, the message is
    the operator's edit text and is consumed by the poller instead of the
    conversational core.
    """
    if not _is_actionable_user_message(event, channel_id):
        return False
    thread_ts = event.get("thread_ts") or event["ts"]
    if interactions is not None and interactions.consume_edit_reply(
        thread_ts, event["text"], say
    ):
        return True
    conversation.handle_message(thread_ts, event["text"], say)
    return True


def create_app(
    config: SlackConfig,
    conversation: MessageResponder,
    interactions: InteractionHandler | None = None,
    promotions: PromotionHandler | None = None,
    approvals: ApprovalsHandler | None = None,
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
        handle_message(event, say, config.channel_id, conversation, interactions)

    if interactions is not None:
        # Local alias: pyright does not carry the None-narrowing into closures.
        handler = interactions
        # Late import keeps the action-id constants next to their handlers.
        from orchestrator.poller import ACTION_APPROVE, ACTION_EDIT, ACTION_REJECT

        def _interaction_id(action: dict[str, Any]) -> int:
            return int(action["value"])

        @app.action(ACTION_APPROVE)
        def _on_approve(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            handler.approve(_interaction_id(action), say)

        @app.action(ACTION_EDIT)
        def _on_edit(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            handler.request_edit(_interaction_id(action), say)

        @app.action(ACTION_REJECT)
        def _on_reject(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            handler.reject(_interaction_id(action), say)

    if promotions is not None:
        promoter = promotions
        from orchestrator.promotion import (
            ACTION_PROMOTE,
            ACTION_PROMOTE_CANCEL,
            ACTION_PROMOTE_CONFIRM,
        )

        @app.action(ACTION_PROMOTE)
        def _on_promote(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            promoter.request_promotion(str(action["value"]), say)

        @app.action(ACTION_PROMOTE_CONFIRM)
        def _on_promote_confirm(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            promoter.confirm_promotion(str(action["value"]), say)

        @app.action(ACTION_PROMOTE_CANCEL)
        def _on_promote_cancel(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            promoter.cancel_promotion(str(action["value"]), say)

    if approvals is not None:
        approver = approvals
        from orchestrator.approvals import ACTION_ONECLI_APPROVE, ACTION_ONECLI_DENY

        @app.action(ACTION_ONECLI_APPROVE)
        def _on_onecli_approve(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            approver.approve(str(action["value"]), say)

        @app.action(ACTION_ONECLI_DENY)
        def _on_onecli_deny(ack: Any, action: dict[str, Any], say: Say) -> None:
            ack()
            approver.deny(str(action["value"]), say)

    return app


def main() -> None:
    # Heavy imports stay here so tests importing this module don't pay for them.
    from orchestrator.approvals import ApprovalsBridge, OneCliApprovalsClient
    from orchestrator.config import load_onecli_url
    from orchestrator.conversation import ConversationCore
    from orchestrator.llm import ModelRouter
    from orchestrator.notion_client import NotionClient
    from orchestrator.notion_recorder import (
        NotionRecorder,
        RecorderConfigError,
        load_notion_databases,
    )
    from orchestrator.poller import HypothesisPoller
    from orchestrator.promotion import PromotionFlow
    from orchestrator.rdagent_client import RdAgentClient
    from orchestrator.state import StateStore

    logging.basicConfig(level=logging.INFO)
    config = load_slack_config()
    store = StateStore()
    rdagent = RdAgentClient()
    # One WebClient shared by Bolt and the background poller (which posts
    # outside any Bolt request context, so it needs the client directly).
    web_client = WebClient(token=config.bot_token)

    recorder = None
    try:
        databases = load_notion_databases()
    except RecorderConfigError as exc:
        logger.warning("Notion recording disabled: %s", exc)
    else:

        def _permalink(thread_ts: str) -> str | None:
            response = web_client.chat_getPermalink(
                channel=config.channel_id, message_ts=thread_ts
            )
            return response.get("permalink")

        recorder = NotionRecorder(NotionClient(), databases, store, permalink=_permalink)

    poller = HypothesisPoller(
        store, rdagent, slack=web_client, channel_id=config.channel_id, recorder=recorder
    )
    promotions = PromotionFlow(store, recorder=recorder)
    conversation = ConversationCore(
        store=store,
        router=ModelRouter(),
        rdagent=rdagent,
        recorder=recorder,
        # Spoken decisions ride the same handlers as the buttons (US-044).
        interactions=poller,
        promotions=promotions,
    )
    approvals = ApprovalsBridge(
        OneCliApprovalsClient(base_url=load_onecli_url()),
        slack=web_client,
        channel_id=config.channel_id,
    )
    app = create_app(
        config,
        conversation,
        interactions=poller,
        promotions=promotions,
        approvals=approvals,
        client=web_client,
    )
    poller.start()
    approvals.start()
    logger.info("starting Socket Mode connection (channel %s)", config.channel_id)
    handler = SocketModeHandler(app, config.app_token)
    # Slack must never route through the OneCLI proxy (docs/decisions.md
    # 2026-07-08), but slack_sdk loads HTTPS_PROXY from the env and ignores
    # NO_PROXY — under `onecli run` the websocket tunnels through the proxy,
    # which drops long-lived connections and leaves the bot deaf. Force
    # direct connections for both the websocket and the Web API client.
    handler.client.proxy = None
    web_client.proxy = None
    handler.start()


if __name__ == "__main__":
    main()

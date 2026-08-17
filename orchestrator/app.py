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


class ApprovalsHandler(Protocol):
    """What the app needs from the OneCLI approvals bridge (see ApprovalsBridge)."""

    def approve(self, request_id: str, say: Say) -> None: ...

    def deny(self, request_id: str, say: Say) -> None: ...


def _mentions(text: str, user_id: str) -> bool:
    """True when the raw Slack text @mentions the given user id."""
    return f"<@{user_id}>" in text or f"<@{user_id}|" in text


def _is_actionable_user_message(
    event: dict[str, Any],
    channel_id: str,
    trusted_bot_ids: frozenset[str] = frozenset(),
    bot_user_id: str | None = None,
) -> bool:
    """True for plain user messages in the target channel (top-level or in-thread).

    Bot-authored messages are ignored — except messages from a trusted bot id
    (RDQ_TRUSTED_BOT_IDS, e.g. Claude in Slack) that explicitly @mention our
    bot user. The mention gate is the loop brake: trusted bots post status
    digests and relayed notes in this channel all day, and answering anything
    they didn't address to us risks two bots talking to each other forever.
    Trusted-bot posts arrive in two shapes — as their app user with bot_id
    set, or as subtype "bot_message" with a username override — so only those
    two subtypes pass.
    """
    if event.get("channel") != channel_id:
        return False
    if not (event.get("text") and event.get("ts")):
        return False
    bot_id = event.get("bot_id")
    if bot_id and bot_id in trusted_bot_ids:
        if event.get("subtype") not in (None, "bot_message"):
            return False
        if bot_user_id is None or event.get("user") == bot_user_id:
            return False  # never answer ourselves, whatever the allowlist says
        return _mentions(event["text"], bot_user_id)
    if event.get("subtype"):  # message_changed, bot_message, channel_join, ...
        return False
    if bot_id:  # never reply to ourselves or other bots
        return False
    return True


def handle_message(
    event: dict[str, Any],
    say: Say,
    channel_id: str,
    conversation: MessageResponder,
    trusted_bot_ids: frozenset[str] = frozenset(),
    bot_user_id: str | None = None,
) -> bool:
    """Route one message event to the conversational core. Returns True if handled.

    Replies target the message's thread: for a top-level message the reply
    starts a thread on it (thread_ts = its ts); for a threaded message the
    reply stays in that thread (thread_ts = the event's thread_ts).
    """
    if not _is_actionable_user_message(event, channel_id, trusted_bot_ids, bot_user_id):
        return False
    thread_ts = event.get("thread_ts") or event["ts"]
    conversation.handle_message(thread_ts, event["text"], say)
    return True


def create_app(
    config: SlackConfig,
    conversation: MessageResponder,
    approvals: ApprovalsHandler | None = None,
    client: WebClient | None = None,
    token_verification_enabled: bool = True,
    process_before_response: bool = False,
    trusted_bot_ids: frozenset[str] = frozenset(),
    bot_user_id: str | None = None,
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
        handle_message(
            event,
            say,
            config.channel_id,
            conversation,
            trusted_bot_ids=trusted_bot_ids,
            bot_user_id=bot_user_id,
        )

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
    from orchestrator.config import load_onecli_url, load_trusted_bot_ids
    from orchestrator.conversation import ConversationCore
    from orchestrator.llm import ModelRouter
    from orchestrator.notion_client import NotionClient
    from orchestrator.notion_recorder import (
        NotionRecorder,
        RecorderConfigError,
        load_notion_databases,
    )
    from orchestrator.promotion import PromotionFlow
    from orchestrator.rdagent_client import RdAgentClient
    from orchestrator.state import StateStore

    logging.basicConfig(level=logging.INFO)
    config = load_slack_config()
    store = StateStore()
    rdagent = RdAgentClient()
    # One WebClient shared by Bolt and the background threads (approvals
    # bridge, run reaper — they post outside any Bolt request context, so
    # they need the client directly). proxy=None immediately: slack_sdk loads
    # HTTPS_PROXY from the env and ignores NO_PROXY, and the auth.test below
    # runs before handler setup.
    web_client = WebClient(token=config.bot_token)
    web_client.proxy = None

    # Our own bot user id, for the trusted-bot mention gate (a trusted bot
    # message only counts as a directive when it @mentions this user).
    bot_user_id = web_client.auth_test().get("user_id")
    trusted_bot_ids = load_trusted_bot_ids()
    if trusted_bot_ids:
        logger.info(
            "trusted bot ids %s may address the bot (user id %s)",
            sorted(trusted_bot_ids),
            bot_user_id,
        )

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

    # locate understands both backends: fetched GPU traces (remapped pickled
    # paths, last-SOTA candidate) and classic server_ui traces.
    from orchestrator.gpu_backend import GpuBackend, locate_run_artifacts

    promotions = PromotionFlow(store, recorder=recorder, locate=locate_run_artifacts)
    from orchestrator.run_memory import build_digest_details

    gpu = GpuBackend()
    conversation = ConversationCore(
        store=store,
        router=ModelRouter(),
        rdagent=rdagent,
        recorder=recorder,
        # Promotion is conversational (US-044): promote_run/confirm_promotion
        # tools drive the PromotionFlow handlers directly.
        promotions=promotions,
        gpu=gpu,
        # US-015: run-history digest injected into RDQ_USER_INSTRUCTION at
        # start_research (never raises, never stalls a launch).
        digest_builder=lambda: build_digest_details(store.db_path),
    )
    approvals = ApprovalsBridge(
        OneCliApprovalsClient(base_url=load_onecli_url()),
        slack=web_client,
        channel_id=config.channel_id,
    )
    # US-021: finalize GPU run rows whose pipeline unit died (SIGKILL, reboot)
    # so a stranded 'running' row can't permanently brick its Slack thread.
    from orchestrator.run_reaper import GpuRunReaper

    reaper = GpuRunReaper(
        store,
        slack=web_client,
        channel_id=config.channel_id,
        unit_active=gpu.unit_active,
    )
    app = create_app(
        config,
        conversation,
        approvals=approvals,
        client=web_client,
        trusted_bot_ids=trusted_bot_ids,
        bot_user_id=bot_user_id,
    )
    approvals.start()
    reaper.start()
    logger.info("starting Socket Mode connection (channel %s)", config.channel_id)
    handler = SocketModeHandler(app, config.app_token)
    # Slack must never route through the OneCLI proxy (docs/decisions.md
    # 2026-07-08), but slack_sdk loads HTTPS_PROXY from the env and ignores
    # NO_PROXY — under `onecli run` the websocket tunnels through the proxy,
    # which drops long-lived connections and leaves the bot deaf. Force
    # direct connections for the websocket too (web_client is already direct).
    handler.client.proxy = None
    handler.start()


if __name__ == "__main__":
    main()

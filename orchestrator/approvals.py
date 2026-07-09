"""OneCLI approvals bridge: pending credential approvals -> Slack buttons (US-039).

The OneCLI gateway can hold a proxied request open pending human approval when
an approval rule matches its host. This bridge long-polls the gateway's
pending list and posts each held request to the #quant-research channel as
Approve / Deny buttons; the operator's click is submitted back to OneCLI as
the decision. It is plumbing for the FUTURE live-trading gate — no approval
rules exist (or are ever created) for paper hosts, so today the pending list
is empty (docs/decisions.md 2026-07-09).

API surface (verified live against onecli 2.2.0; matches @onecli-sh/sdk's
ApprovalClient — see docs/decisions.md for the probe transcript):

- The approvals endpoints live on the GATEWAY url (port 10255 here), not the
  management API at :10254. Resolve it once via
  ``GET {ONECLI_URL}/api/gateway-url`` -> ``{"url": ...}``.
- ``GET {gateway}/api/approvals/pending[?exclude=id,id]`` long-polls (the
  server holds the connection up to ~30s) and returns
  ``{"requests": [...], "timeoutSeconds": N}``. Requests carry camelCase
  fields: id, method, url, host, path, bodyPreview, agent{id,name,externalId},
  createdAt, expiresAt.
- ``POST {gateway}/api/approvals/{id}/decision`` with
  ``{"decision": "approve"|"deny"}``. 410 means the request already timed out
  server-side — tolerated (the gateway denied it by expiry).

Fallback path: a decision can always be made manually in the OneCLI web UI at
the management URL, so any submit failure here tells the operator to decide
there — the Slack post degrades to notification-only, never a lost gate.

Restart semantics: posted-approval state is deliberately in-memory only.
Pending approvals expire in ~3 minutes (timeoutSeconds), so persisting them
buys nothing; after a restart the next poll re-lists anything still pending
and it is re-posted (both copies' buttons stay valid — the decision endpoint
only needs the request id).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)

DEFAULT_ONECLI_URL = "http://127.0.0.1:10254"
# The server holds a pending poll up to ~30s; the client timeout must exceed it.
DEFAULT_POLL_TIMEOUT_SECONDS = 35.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0
# How long the bridge sleeps after a failed poll before retrying.
DEFAULT_ERROR_BACKOFF_SECONDS = 5.0

# Block Kit action ids (app.py registers a Bolt listener per id).
ACTION_ONECLI_APPROVE = "onecli_approve"
ACTION_ONECLI_DENY = "onecli_deny"

APPROVE = "approve"
DENY = "deny"

# Slack section blocks cap text at 3000 chars.
_MAX_SECTION_TEXT = 2900

WEB_UI_FALLBACK = (
    "You can still decide it manually in the OneCLI web UI"
    " (the management console at {onecli_url})."
)


class ApprovalsApiError(RuntimeError):
    """The OneCLI approvals API returned an unexpected response."""


@dataclass(frozen=True)
class ApprovalRequest:
    """One credentialed request held by the gateway pending a human decision."""

    id: str
    method: str
    host: str
    path: str
    body_preview: str | None
    agent_name: str
    agent_identifier: str | None
    created_at: str
    expires_at: str

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ApprovalRequest:
        agent = data.get("agent") or {}
        return cls(
            id=str(data["id"]),
            method=str(data.get("method", "")),
            host=str(data.get("host", "")),
            path=str(data.get("path", "")),
            body_preview=data.get("bodyPreview"),
            agent_name=str(agent.get("name", "unknown")),
            agent_identifier=agent.get("externalId"),
            created_at=str(data.get("createdAt", "")),
            expires_at=str(data.get("expiresAt", "")),
        )


def parse_expiry(expires_at: str) -> datetime | None:
    """Parse the gateway's ISO-8601 expiry; None when unparsable (never prune)."""
    if not expires_at:
        return None
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class OneCliApprovalsClient:
    """Minimal typed client for the gateway's approvals endpoints.

    Talks DIRECTLY to the local OneCLI management/gateway URLs (this is
    management traffic, not a proxied credential call — no ``onecli run``
    wrapper, no injected secrets; the local API is unauthenticated on this
    box, matching the SDK's optional-Bearer behavior).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_ONECLI_URL,
        gateway_url: str | None = None,
        session: Any | None = None,
        poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._gateway_url = gateway_url.rstrip("/") if gateway_url else None
        if session is None:
            session = requests.Session()
            # Local management traffic must never route through the credential
            # proxy that `onecli run` injects via HTTP(S)_PROXY env vars.
            session.trust_env = False
        self.session = session
        self.poll_timeout = poll_timeout
        self.request_timeout = request_timeout

    def gateway_url(self) -> str:
        """Resolve (once) and cache the gateway URL from the management API."""
        if self._gateway_url is None:
            url = f"{self.base_url}/api/gateway-url"
            response = self.session.request("GET", url, timeout=self.request_timeout)
            if response.status_code != 200:
                raise ApprovalsApiError(
                    f"GET {url} returned HTTP {response.status_code} — cannot resolve"
                    " the OneCLI gateway URL for approvals polling."
                )
            self._gateway_url = str(response.json()["url"]).rstrip("/")
        return self._gateway_url

    def poll_pending(self, exclude: tuple[str, ...] = ()) -> list[ApprovalRequest]:
        """Long-poll the pending list; ``exclude`` suppresses already-posted ids."""
        url = f"{self.gateway_url()}/api/approvals/pending"
        params = {"exclude": ",".join(exclude)} if exclude else None
        response = self.session.request("GET", url, params=params, timeout=self.poll_timeout)
        if response.status_code != 200:
            raise ApprovalsApiError(
                f"GET {url} returned HTTP {response.status_code} — approvals poll failed."
            )
        payload = response.json()
        return [ApprovalRequest.from_api(item) for item in payload.get("requests", [])]

    def submit_decision(self, request_id: str, decision: str) -> bool:
        """Submit approve/deny for one request.

        Returns False when the gateway answered 410 Gone — the request already
        expired server-side (denied by timeout), which callers report rather
        than treat as a failure.
        """
        if decision not in (APPROVE, DENY):
            raise ValueError(f"decision must be {APPROVE!r} or {DENY!r}, got {decision!r}")
        url = f"{self.gateway_url()}/api/approvals/{request_id}/decision"
        response = self.session.request(
            "POST", url, json={"decision": decision}, timeout=self.request_timeout
        )
        if response.status_code == 410:
            return False
        if not 200 <= response.status_code < 300:
            raise ApprovalsApiError(
                f"POST {url} returned HTTP {response.status_code} — decision not delivered."
            )
        return True


def format_approval_text(request: ApprovalRequest) -> str:
    """Slack mrkdwn body for a pending credential approval."""
    agent = request.agent_name
    if request.agent_identifier and request.agent_identifier != agent:
        agent += f" ({request.agent_identifier})"
    lines = [
        ":rotating_light: *OneCLI credential approval requested*",
        f"*Agent:* {agent}",
        f"*Request:* `{request.method} {request.host}{request.path}`",
    ]
    if request.body_preview:
        lines.append(f"*Body:* ```{request.body_preview}```")
    if request.expires_at:
        lines.append(f"*Expires:* {request.expires_at} (denied by timeout if ignored)")
    text = "\n".join(lines)
    if len(text) > _MAX_SECTION_TEXT:
        text = text[: _MAX_SECTION_TEXT - 1] + "…"
    return text


def approval_blocks(request: ApprovalRequest) -> list[dict[str, Any]]:
    """Block Kit blocks: request summary + Approve/Deny buttons (value = request id)."""

    def button(label: str, action_id: str, style: str) -> dict[str, Any]:
        return {
            "type": "button",
            "text": {"type": "plain_text", "text": label},
            "action_id": action_id,
            "value": request.id,
            "style": style,
        }

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": format_approval_text(request)},
        },
        {
            "type": "actions",
            "block_id": f"onecli_approval_{request.id}",
            "elements": [
                button("Approve", ACTION_ONECLI_APPROVE, style="primary"),
                button("Deny", ACTION_ONECLI_DENY, style="danger"),
            ],
        },
    ]


class ApprovalsPoster(Protocol):
    """The WebClient method the bridge posts through."""

    def chat_postMessage(self, **kwargs: Any) -> Any: ...  # noqa: N802 - slack_sdk casing


class SayFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class _PostedApproval:
    request: ApprovalRequest
    message_ts: str | None


class ApprovalsBridge:
    """Long-polls OneCLI for pending approvals and relays operator decisions.

    Share one instance per process: the background thread runs ``poll_once``
    in a loop, and app.py routes the ``onecli_approve``/``onecli_deny`` button
    clicks to ``approve``/``deny``. Decisions are submitted by request id
    alone, so a click on a message posted before a restart still works.
    """

    def __init__(
        self,
        client: OneCliApprovalsClient,
        slack: ApprovalsPoster,
        channel_id: str,
        error_backoff: float = DEFAULT_ERROR_BACKOFF_SECONDS,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._client = client
        self._slack = slack
        self._channel_id = channel_id
        self._error_backoff = error_backoff
        self._now = now
        # request id -> posted state; pruned once the request expires or is decided.
        self._posted: dict[str, _PostedApproval] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- background loop ----------------------------------------------------

    def start(self) -> None:
        """Start the daemon polling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="onecli-approvals-bridge", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # A poll in flight blocks up to poll_timeout; the thread is a
            # daemon, so don't hold process shutdown hostage to it.
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # The pending endpoint long-polls server-side, so the loop
                # needs no extra sleep on the happy path.
                self.poll_once()
            except Exception:  # noqa: BLE001 - the bridge must survive any poll failure
                logger.exception("OneCLI approvals poll failed")
                self._stop.wait(self._error_backoff)

    # -- polling ------------------------------------------------------------

    def poll_once(self) -> int:
        """One poll cycle; returns how many new approvals were posted."""
        self._prune_expired()
        exclude = tuple(self._posted)
        posted = 0
        for request in self._client.poll_pending(exclude=exclude):
            if request.id in self._posted:
                continue  # gateway ignored (or predates) the exclude param
            posted += self._post_approval(request)
        return posted

    def _post_approval(self, request: ApprovalRequest) -> int:
        try:
            response = self._slack.chat_postMessage(
                channel=self._channel_id,
                text=(
                    "OneCLI credential approval requested:"
                    f" {request.method} {request.host}{request.path}"
                    f" by {request.agent_name}"
                ),
                blocks=approval_blocks(request),
            )
        except Exception:  # noqa: BLE001 - not recorded as posted, so the next poll retries
            logger.exception("failed to post OneCLI approval %s to Slack", request.id)
            return 0
        # slack_sdk's SlackResponse and plain dicts both expose .get().
        message_ts = response.get("ts") if hasattr(response, "get") else None
        self._posted[request.id] = _PostedApproval(request=request, message_ts=message_ts)
        logger.info(
            "posted OneCLI approval %s (%s %s%s) to channel %s",
            request.id,
            request.method,
            request.host,
            request.path,
            self._channel_id,
        )
        return 1

    def _prune_expired(self) -> None:
        """Forget posted approvals past their expiry (gateway denied by timeout)."""
        now = self._now()
        for request_id in list(self._posted):
            expiry = parse_expiry(self._posted[request_id].request.expires_at)
            if expiry is not None and expiry < now:
                del self._posted[request_id]

    # -- operator actions (Bolt listeners call these) ------------------------

    def approve(self, request_id: str, say: SayFn) -> None:
        self._decide(request_id, APPROVE, say)

    def deny(self, request_id: str, say: SayFn) -> None:
        self._decide(request_id, DENY, say)

    def _decide(self, request_id: str, decision: str, say: SayFn) -> None:
        thread_ts = None
        posted = self._posted.get(request_id)
        if posted is not None:
            thread_ts = posted.message_ts

        def reply(text: str) -> None:
            if thread_ts:
                say(text=text, thread_ts=thread_ts)
            else:
                say(text=text)

        try:
            delivered = self._client.submit_decision(request_id, decision)
        except (ApprovalsApiError, requests.RequestException) as exc:
            logger.exception("OneCLI decision submit failed for %s", request_id)
            fallback = WEB_UI_FALLBACK.format(onecli_url=self._client.base_url)
            reply(f":warning: Could not deliver the decision to OneCLI ({exc}). {fallback}")
            return
        self._posted.pop(request_id, None)
        if not delivered:
            reply(
                ":hourglass: That approval already expired in OneCLI — the request"
                " was denied by timeout before your decision arrived."
            )
            return
        if decision == APPROVE:
            reply(":white_check_mark: Approved — OneCLI is releasing the request.")
        else:
            reply(":no_entry: Denied — OneCLI is rejecting the request.")

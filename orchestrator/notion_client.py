"""Thin Notion API client: raw HTTP through the OneCLI proxy.

All requests are bare HTTPS: no Authorization header appears anywhere in this
module (a test greps for it). The OneCLI proxy injects the Notion bearer token
via a connector integration ("JPH NanoClaw Connection") when the process runs
under `onecli run --agent rdq-orchestrator` — the token is NOT a vaulted
secret, so it never shows in `onecli secrets list` (docs/decisions.md
2026-07-08).

Every request carries `Notion-Version: 2022-06-28`.

Notion is eventually consistent: a page created a moment ago may be missing
from a database query. `query_db_until()` is the read-after-write path — it
retries the query with bounded backoff until a caller predicate is satisfied.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import requests

BASE_URL = "https://api.notion.com"
NOTION_VERSION = "2022-06-28"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_CONSISTENCY_RETRIES = 5
DEFAULT_CONSISTENCY_BACKOFF_SECONDS = 0.5
DEFAULT_PAGE_SIZE = 100

# Statuses Notion documents as retryable (conflict_error + transient 5xx).
# 429 is also retried, honoring Retry-After.
TRANSIENT_STATUSES = frozenset({409, 500, 502, 503, 504})


class NotionError(RuntimeError):
    """Base error for Notion client failures."""


class NotionAuthError(NotionError):
    """401/403 from Notion: the OneCLI proxy did not inject a valid token."""


class NotionRateLimitError(NotionError):
    """429 from Notion persisted beyond the retry budget."""


class NotionConsistencyError(NotionError):
    """A read-after-write query never satisfied its predicate within budget."""


class NotionClient:
    """Minimal typed client for the Notion API through the OneCLI proxy.

    429/409/5xx responses are retried with bounded backoff; 401/403 raise
    NotionAuthError with the fix spelled out.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        consistency_retries: int = DEFAULT_CONSISTENCY_RETRIES,
        consistency_backoff: float = DEFAULT_CONSISTENCY_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.consistency_retries = consistency_retries
        self.consistency_backoff = consistency_backoff
        self._sleep = sleep

    def create_page(
        self,
        parent: dict[str, Any],
        properties: dict[str, Any],
        children: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/pages: create a page under a page or database parent."""
        payload: dict[str, Any] = {"parent": parent, "properties": properties}
        if children is not None:
            payload["children"] = children
        return self._request("POST", "/v1/pages", payload)

    def query_db(
        self,
        database_id: str,
        filter: dict[str, Any] | None = None,  # noqa: A002 - Notion API field name
        sorts: list[dict[str, Any]] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        """POST /v1/databases/{id}/query: return ALL matching rows (paginates)."""
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"page_size": page_size}
            if filter is not None:
                payload["filter"] = filter
            if sorts is not None:
                payload["sorts"] = sorts
            if cursor is not None:
                payload["start_cursor"] = cursor
            page = self._request("POST", f"/v1/databases/{database_id}/query", payload)
            page_results = page.get("results", [])
            if not isinstance(page_results, list):
                raise NotionError(
                    f"expected a 'results' list from database query, got: {page_results!r:.200}"
                )
            results.extend(page_results)
            cursor = page.get("next_cursor")
            if not page.get("has_more") or cursor is None:
                return results

    def update_page(
        self,
        page_id: str,
        properties: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        """PATCH /v1/pages/{id}: update properties and/or the archived flag."""
        payload: dict[str, Any] = {}
        if properties is not None:
            payload["properties"] = properties
        if archived is not None:
            payload["archived"] = archived
        if not payload:
            raise ValueError("update_page needs properties and/or archived")
        return self._request("PATCH", f"/v1/pages/{page_id}", payload)

    def query_db_until(
        self,
        database_id: str,
        predicate: Callable[[list[dict[str, Any]]], bool],
        filter: dict[str, Any] | None = None,  # noqa: A002 - Notion API field name
        sorts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Read-after-write query: retry until predicate(results) is true.

        Notion is eventually consistent — a just-created page can be absent
        from an immediate query. Retries with bounded exponential backoff;
        raises NotionConsistencyError once the budget is exhausted.
        """
        attempt = 0
        while True:
            results = self.query_db(database_id, filter=filter, sorts=sorts)
            if predicate(results):
                return results
            if attempt >= self.consistency_retries:
                raise NotionConsistencyError(
                    f"database {database_id} query never satisfied the predicate "
                    f"after {self.consistency_retries} retries "
                    f"({len(results)} rows on the last attempt) — either the write "
                    "never landed or consistency lag exceeded the backoff budget"
                )
            self._sleep(self.consistency_backoff * (2**attempt))
            attempt += 1

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            response = self.session.request(
                method,
                url,
                json=payload,
                headers={"Notion-Version": NOTION_VERSION},
                timeout=self.timeout,
            )
            status = getattr(response, "status_code", None)
            if status == 429 or status in TRANSIENT_STATUSES:
                if attempt >= self.max_retries:
                    if status == 429:
                        raise NotionRateLimitError(
                            f"Notion kept returning 429 for {method} {path} after "
                            f"{self.max_retries} retries; back off and retry later"
                        )
                    raise NotionError(
                        f"Notion kept returning HTTP {status} for {method} {path} "
                        f"after {self.max_retries} retries: {_error_detail(response)}"
                    )
                self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue
            if status in (401, 403):
                raise NotionAuthError(
                    f"Notion returned {status} for {method} {path}: no valid token was "
                    "injected. Run this process under `onecli run --agent "
                    "rdq-orchestrator` — Notion auth comes from the OneCLI connector "
                    "integration (NOT a vaulted secret; it never shows in `onecli "
                    "secrets list`). If it still fails, re-check the connector in the "
                    "OneCLI web UI and that the target page/database is shared with "
                    "the integration (docs/decisions.md 2026-07-08). "
                    f"Body: {_error_detail(response)}"
                )
            if status is None or not 200 <= int(status) < 300:
                raise NotionError(
                    f"Notion returned HTTP {status} for {method} {path}: {_error_detail(response)}"
                )
            body = response.json()
            if not isinstance(body, dict):
                raise NotionError(
                    f"expected a JSON object from {method} {path}, got: {str(body)[:200]}"
                )
            return body

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After") if response.headers else None
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass  # non-numeric Retry-After: fall back to exponential
        return DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt)


def _error_detail(response: Any) -> str:
    """Notion error bodies are {'object': 'error', 'code': ..., 'message': ...}."""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict) and ("code" in body or "message" in body):
        return f"{body.get('code', 'unknown_code')}: {body.get('message', '')}"[:300]
    return str(getattr(response, "text", ""))[:300]

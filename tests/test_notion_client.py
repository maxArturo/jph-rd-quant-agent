"""Unit tests for orchestrator/notion_client.py (mocked HTTP)."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import orchestrator.notion_client as notion_module
from orchestrator.notion_client import (
    BASE_URL,
    NOTION_VERSION,
    NotionAuthError,
    NotionClient,
    NotionConsistencyError,
    NotionError,
    NotionRateLimitError,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

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
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {"method": method, "url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError("FakeSession ran out of queued responses")
        return self._responses.pop(0)


def make_client(
    responses: list[FakeResponse], **kwargs: Any
) -> tuple[NotionClient, FakeSession, list[float]]:
    session = FakeSession(responses)
    sleeps: list[float] = []
    client = NotionClient(session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


PAGE_OBJECT = {"object": "page", "id": "page-123"}


def query_result(rows: list[dict[str, Any]], has_more: bool = False, cursor: str | None = None):
    return FakeResponse(
        200, {"object": "list", "results": rows, "has_more": has_more, "next_cursor": cursor}
    )


# ---------------------------------------------------------------- create_page


def test_create_page_posts_payload_with_version_header() -> None:
    client, session, _ = make_client([FakeResponse(200, PAGE_OBJECT)])
    parent = {"page_id": "parent-1"}
    properties = {"title": {"title": [{"text": {"content": "Idea"}}]}}

    result = client.create_page(parent, properties)

    assert result == PAGE_OBJECT
    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"] == f"{BASE_URL}/v1/pages"
    assert call["json"] == {"parent": parent, "properties": properties}
    assert call["headers"] == {"Notion-Version": NOTION_VERSION}


def test_create_page_includes_children_when_given() -> None:
    client, session, _ = make_client([FakeResponse(200, PAGE_OBJECT)])
    children = [{"object": "block", "type": "paragraph"}]

    client.create_page({"page_id": "p"}, {}, children=children)

    assert session.calls[0]["json"]["children"] == children


# ------------------------------------------------------------------- query_db


def test_query_db_returns_rows_and_sends_filter_and_sorts() -> None:
    rows = [{"id": "row-1"}, {"id": "row-2"}]
    client, session, _ = make_client([query_result(rows)])
    filter_ = {"property": "Status", "select": {"equals": "Active"}}
    sorts = [{"timestamp": "created_time", "direction": "descending"}]

    assert client.query_db("db-1", filter=filter_, sorts=sorts) == rows

    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"] == f"{BASE_URL}/v1/databases/db-1/query"
    assert call["json"] == {"page_size": 100, "filter": filter_, "sorts": sorts}


def test_query_db_paginates_through_next_cursor() -> None:
    client, session, _ = make_client(
        [
            query_result([{"id": "row-1"}], has_more=True, cursor="cur-2"),
            query_result([{"id": "row-2"}]),
        ]
    )

    assert client.query_db("db-1") == [{"id": "row-1"}, {"id": "row-2"}]

    first, second = session.calls
    assert "start_cursor" not in first["json"]
    assert second["json"]["start_cursor"] == "cur-2"


def test_query_db_omits_filter_and_sorts_when_unset() -> None:
    client, session, _ = make_client([query_result([])])

    assert client.query_db("db-1") == []
    assert session.calls[0]["json"] == {"page_size": 100}


# ---------------------------------------------------------------- update_page


def test_update_page_patches_properties() -> None:
    client, session, _ = make_client([FakeResponse(200, PAGE_OBJECT)])
    properties = {"Status": {"select": {"name": "Done"}}}

    result = client.update_page("page-123", properties=properties)

    assert result == PAGE_OBJECT
    (call,) = session.calls
    assert call["method"] == "PATCH"
    assert call["url"] == f"{BASE_URL}/v1/pages/page-123"
    assert call["json"] == {"properties": properties}
    assert call["headers"] == {"Notion-Version": NOTION_VERSION}


def test_update_page_can_archive() -> None:
    client, session, _ = make_client([FakeResponse(200, PAGE_OBJECT)])

    client.update_page("page-123", archived=True)

    assert session.calls[0]["json"] == {"archived": True}


def test_update_page_requires_something_to_update() -> None:
    client, session, _ = make_client([])

    with pytest.raises(ValueError, match="properties and/or archived"):
        client.update_page("page-123")
    assert session.calls == []


# -------------------------------------------------------- retry / error paths


def test_429_honors_retry_after_then_succeeds() -> None:
    client, session, sleeps = make_client(
        [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(200, PAGE_OBJECT),
        ]
    )

    assert client.create_page({"page_id": "p"}, {}) == PAGE_OBJECT
    assert sleeps == [7.0]
    assert len(session.calls) == 2


def test_429_exhausted_raises_rate_limit_error() -> None:
    client, _, sleeps = make_client([FakeResponse(429) for _ in range(3)], max_retries=2)

    with pytest.raises(NotionRateLimitError, match="after 2 retries"):
        client.create_page({"page_id": "p"}, {})
    assert sleeps == [1.0, 2.0]  # exponential fallback without Retry-After


def test_transient_conflict_retried_then_succeeds() -> None:
    client, session, sleeps = make_client(
        [
            FakeResponse(409, {"object": "error", "code": "conflict_error", "message": "retry"}),
            FakeResponse(503),
            FakeResponse(200, PAGE_OBJECT),
        ]
    )

    assert client.update_page("page-123", archived=False) == PAGE_OBJECT
    assert sleeps == [1.0, 2.0]
    assert len(session.calls) == 3


def test_transient_exhausted_raises_with_error_detail() -> None:
    responses = [
        FakeResponse(503, {"object": "error", "code": "service_unavailable", "message": "down"})
        for _ in range(2)
    ]
    client, _, _ = make_client(responses, max_retries=1)

    with pytest.raises(NotionError, match="service_unavailable"):
        client.query_db("db-1")


def test_401_raises_auth_error_with_actionable_message() -> None:
    client, _, _ = make_client(
        [FakeResponse(401, {"object": "error", "code": "unauthorized", "message": "no token"})]
    )

    with pytest.raises(NotionAuthError) as excinfo:
        client.query_db("db-1")
    message = str(excinfo.value)
    assert "rdq-orchestrator" in message
    assert "connector" in message
    assert "unauthorized" in message


def test_other_http_error_raises_notion_error_with_body_detail() -> None:
    client, _, _ = make_client(
        [FakeResponse(400, {"object": "error", "code": "validation_error", "message": "bad prop"})]
    )

    with pytest.raises(NotionError, match="validation_error: bad prop"):
        client.create_page({"page_id": "p"}, {})


def test_non_json_error_body_falls_back_to_text() -> None:
    client, _, _ = make_client([FakeResponse(400, text="<html>gateway</html>")])

    with pytest.raises(NotionError, match="gateway"):
        client.query_db("db-1")


# ------------------------------------------------- read-after-write retries


def test_query_db_until_retries_until_predicate_satisfied() -> None:
    client, session, sleeps = make_client(
        [
            query_result([]),  # write not visible yet
            query_result([]),
            query_result([{"id": "row-1"}]),
        ]
    )

    rows = client.query_db_until("db-1", lambda results: len(results) > 0)

    assert rows == [{"id": "row-1"}]
    assert len(session.calls) == 3
    assert sleeps == [0.5, 1.0]  # bounded exponential backoff between reads


def test_query_db_until_exhausted_raises_consistency_error() -> None:
    client, session, sleeps = make_client(
        [query_result([]) for _ in range(3)], consistency_retries=2
    )

    with pytest.raises(NotionConsistencyError, match="db-1"):
        client.query_db_until("db-1", lambda results: bool(results))
    assert len(session.calls) == 3  # initial try + 2 retries, then gives up
    assert sleeps == [0.5, 1.0]


def test_query_db_until_passes_filter_through() -> None:
    filter_ = {"property": "Name", "title": {"equals": "X"}}
    client, session, _ = make_client([query_result([{"id": "r"}])])

    client.query_db_until("db-1", lambda results: True, filter=filter_)

    assert session.calls[0]["json"]["filter"] == filter_


# ----------------------------------------------------------- auth invariants


def test_no_authorization_header_is_ever_sent() -> None:
    client, session, _ = make_client(
        [FakeResponse(200, PAGE_OBJECT), query_result([]), FakeResponse(200, PAGE_OBJECT)]
    )
    client.create_page({"page_id": "p"}, {})
    client.query_db("db-1")
    client.update_page("page-123", archived=True)

    for call in session.calls:
        header_names = {name.lower() for name in (call["headers"] or {})}
        assert "authorization" not in header_names


def test_module_source_never_mentions_authorization_header() -> None:
    source = inspect.getsource(notion_module)
    assert "Authorization" not in source.replace("no Authorization header", "")

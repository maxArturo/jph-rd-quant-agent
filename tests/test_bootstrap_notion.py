"""Unit tests for ops/bootstrap_notion.py (mocked HTTP via FakeSession)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ops.bootstrap_notion import (
    CONFIG_KEYS,
    IDEAS_TITLE,
    LIVE_CONFIG_KEYS,
    LIVE_PAGE_INTRO,
    LIVE_PAGE_TITLE,
    LiveBootstrapError,
    bootstrap,
    bootstrap_live,
    database_properties,
    live_database_properties,
    main,
    write_config,
)
from orchestrator.notion_client import NOTION_VERSION, NotionClient
from tests.test_notion_client import FakeResponse, FakeSession

PARENT = "3979b1a4-36cf-8046-baa5-cc14c1ca7665"
GRANDPARENT = "grandparent-page-id"
LIVE_PAGE_ID = "page-live"

ALL_TITLES = list(CONFIG_KEYS)
LIVE_TITLES = list(LIVE_CONFIG_KEYS)


def child_db_block(title: str, block_id: str) -> dict[str, Any]:
    return {"object": "block", "id": block_id, "type": "child_database",
            "child_database": {"title": title}}


def children_response(
    blocks: list[dict[str, Any]], has_more: bool = False, cursor: str | None = None
) -> FakeResponse:
    return FakeResponse(
        200, {"object": "list", "results": blocks, "has_more": has_more, "next_cursor": cursor}
    )


def created_db_response(db_id: str) -> FakeResponse:
    return FakeResponse(200, {"object": "database", "id": db_id})


def make_client(responses: list[FakeResponse]) -> tuple[NotionClient, FakeSession]:
    session = FakeSession(responses)
    return NotionClient(session=session, sleep=lambda _s: None), session


# ---------------------------------------------------------------- client API


def test_list_child_databases_filters_and_paginates() -> None:
    page1 = children_response(
        [child_db_block("Research Ideas", "db-ideas"),
         {"object": "block", "id": "b1", "type": "paragraph"}],
        has_more=True,
        cursor="cur-2",
    )
    page2 = children_response([child_db_block("Trade Ledger", "db-ledger")])
    client, session = make_client([page1, page2])
    found = client.list_child_databases(PARENT)
    assert found == {"Research Ideas": "db-ideas", "Trade Ledger": "db-ledger"}
    assert [c["method"] for c in session.calls] == ["GET", "GET"]
    assert "start_cursor=cur-2" in session.calls[1]["url"]
    assert all(c["headers"] == {"Notion-Version": NOTION_VERSION} for c in session.calls)


def test_create_database_payload() -> None:
    client, session = make_client([created_db_response("db-new")])
    result = client.create_database(PARENT, "Decision Log", {"Decision": {"title": {}}})
    assert result["id"] == "db-new"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/databases")
    assert call["json"]["parent"] == {"type": "page_id", "page_id": PARENT}
    assert call["json"]["title"][0]["text"]["content"] == "Decision Log"
    assert call["json"]["properties"] == {"Decision": {"title": {}}}


# ----------------------------------------------------------------- bootstrap


def test_fresh_bootstrap_creates_all_five_ideas_first() -> None:
    responses = [children_response([])] + [
        created_db_response(f"db-{i}") for i in range(len(ALL_TITLES))
    ]
    client, session = make_client(responses)
    outcome = bootstrap(client, PARENT)

    creates = [c for c in session.calls if c["method"] == "POST"]
    assert len(creates) == len(ALL_TITLES)
    created_titles = [c["json"]["title"][0]["text"]["content"] for c in creates]
    assert created_titles[0] == IDEAS_TITLE
    assert sorted(created_titles) == sorted(ALL_TITLES)
    assert set(outcome) == set(ALL_TITLES)
    assert all(info["action"] == "created" for info in outcome.values())

    # Every relation property points at the Research Ideas database id.
    ideas_id = outcome[IDEAS_TITLE]["id"]
    relation_count = 0
    for call in creates:
        for prop in call["json"]["properties"].values():
            if "relation" in prop:
                relation_count += 1
                assert prop["relation"]["database_id"] == ideas_id
                assert "single_property" in prop["relation"]
    assert relation_count == 3  # Hypothesis Log, Backtest Results, Decision Log


def test_rerun_is_idempotent_no_creates() -> None:
    blocks = [child_db_block(t, f"db-{i}") for i, t in enumerate(ALL_TITLES)]
    client, session = make_client([children_response(blocks)])
    outcome = bootstrap(client, PARENT)
    assert [c["method"] for c in session.calls] == ["GET"]  # no POST /v1/databases
    assert all(info["action"] == "exists" for info in outcome.values())
    assert outcome[IDEAS_TITLE]["id"] == "db-0"


def test_partial_rerun_creates_only_missing_and_links_existing_ideas() -> None:
    blocks = [child_db_block(IDEAS_TITLE, "db-ideas-existing"),
              child_db_block("Trade Ledger", "db-ledger-existing")]
    missing = [t for t in ALL_TITLES if t not in (IDEAS_TITLE, "Trade Ledger")]
    responses = [children_response(blocks)] + [
        created_db_response(f"db-new-{i}") for i in range(len(missing))
    ]
    client, session = make_client(responses)
    outcome = bootstrap(client, PARENT)

    creates = [c for c in session.calls if c["method"] == "POST"]
    created_titles = {c["json"]["title"][0]["text"]["content"] for c in creates}
    assert created_titles == set(missing)
    assert outcome[IDEAS_TITLE] == {"id": "db-ideas-existing", "action": "exists"}
    # Relations in the newly created databases point at the EXISTING ideas db.
    for call in creates:
        for prop in call["json"]["properties"].values():
            if "relation" in prop:
                assert prop["relation"]["database_id"] == "db-ideas-existing"


def test_schemas_match_reference_doc_property_names() -> None:
    doc = Path(__file__).resolve().parent.parent / "docs" / "reference" / "notion-schema.md"
    text = doc.read_text()
    schemas = {**database_properties("db-ideas"), **live_database_properties()}
    for title, props in schemas.items():
        assert f"## {title}" in text
        for prop_name in props:
            assert f"| {prop_name} " in text, f"{prop_name} missing from {title} table in {doc}"


# ------------------------------------------------------------ live bootstrap


def live_page_object(page_id: str = LIVE_PAGE_ID, title: str = LIVE_PAGE_TITLE) -> dict[str, Any]:
    return {
        "object": "page",
        "id": page_id,
        "properties": {
            "title": {
                "type": "title",
                "title": [{"type": "text", "plain_text": title, "text": {"content": title}}],
            }
        },
    }


def search_response(pages: list[dict[str, Any]]) -> FakeResponse:
    return FakeResponse(
        200, {"object": "list", "results": pages, "has_more": False, "next_cursor": None}
    )


def paper_page_response(parent: dict[str, Any]) -> FakeResponse:
    return FakeResponse(200, {"object": "page", "id": PARENT, "parent": parent})


def callout_block(block_id: str = "b-callout") -> dict[str, Any]:
    return {"object": "block", "id": block_id, "type": "callout"}


def test_live_schemas_identical_to_paper_counterparts() -> None:
    paper = database_properties("db-ideas")
    live = live_database_properties()
    assert live["Trade Ledger (Live)"] == paper["Trade Ledger"]
    assert live["Account Snapshots (Live)"] == paper["Account Snapshots"]
    # No relation property anywhere: the live schemas never embed an ideas id.
    for props in live.values():
        assert not any("relation" in prop for prop in props.values())


def test_live_fresh_bootstrap_creates_sibling_page_and_both_databases() -> None:
    responses = [
        search_response([]),  # no page shared yet
        paper_page_response({"type": "page_id", "page_id": GRANDPARENT}),
        FakeResponse(200, {"object": "page", "id": LIVE_PAGE_ID}),  # create_page
        children_response([]),  # live page has no databases yet
        created_db_response("db-ledger-live"),
        created_db_response("db-snapshots-live"),
    ]
    client, session = make_client(responses)
    live_page, outcome = bootstrap_live(client, PARENT)

    assert live_page == {"id": LIVE_PAGE_ID, "action": "created"}
    search = session.calls[0]
    assert search["url"].endswith("/v1/search")
    assert search["json"]["query"] == LIVE_PAGE_TITLE
    assert search["json"]["filter"] == {"value": "page", "property": "object"}

    create_page = session.calls[2]
    assert create_page["url"].endswith("/v1/pages")
    assert create_page["json"]["parent"] == {"type": "page_id", "page_id": GRANDPARENT}
    title = create_page["json"]["properties"]["title"]["title"][0]["text"]["content"]
    assert title == LIVE_PAGE_TITLE
    (callout,) = create_page["json"]["children"]
    assert callout["type"] == "callout"
    assert callout["callout"]["rich_text"][0]["text"]["content"] == LIVE_PAGE_INTRO
    assert "REAL-MONEY" in LIVE_PAGE_INTRO

    db_creates = [c for c in session.calls if c["url"].endswith("/v1/databases")]
    assert [c["json"]["title"][0]["text"]["content"] for c in db_creates] == LIVE_TITLES
    paper = database_properties("db-ideas")
    for call, source in zip(db_creates, ("Trade Ledger", "Account Snapshots"), strict=True):
        assert call["json"]["parent"] == {"type": "page_id", "page_id": LIVE_PAGE_ID}
        assert call["json"]["properties"] == paper[source]
    assert all(outcome[t] == {"id": f"db-{k}", "action": "created"}
               for t, k in (("Trade Ledger (Live)", "ledger-live"),
                            ("Account Snapshots (Live)", "snapshots-live")))


def test_live_rerun_adopts_page_and_databases_without_creates() -> None:
    blocks = [
        callout_block(),
        child_db_block("Trade Ledger (Live)", "db-ledger-live"),
        child_db_block("Account Snapshots (Live)", "db-snapshots-live"),
    ]
    responses = [
        search_response([live_page_object()]),
        children_response(blocks),  # callout check
        children_response(blocks),  # child-database listing
    ]
    client, session = make_client(responses)
    live_page, outcome = bootstrap_live(client, PARENT)

    assert live_page == {"id": LIVE_PAGE_ID, "action": "exists"}
    assert all(info["action"] == "exists" for info in outcome.values())
    assert all(c["method"] != "PATCH" for c in session.calls)
    posts = [c for c in session.calls if c["method"] == "POST"]
    assert all(c["url"].endswith("/v1/search") for c in posts)  # nothing created


def test_live_adoption_appends_missing_intro_callout() -> None:
    db_blocks = [
        child_db_block("Trade Ledger (Live)", "db-ledger-live"),
        child_db_block("Account Snapshots (Live)", "db-snapshots-live"),
    ]
    responses = [
        search_response([live_page_object()]),
        children_response(db_blocks),  # no callout on the operator's page
        FakeResponse(200, {"object": "list", "results": []}),  # append_children
        children_response(db_blocks),
    ]
    client, session = make_client(responses)
    live_page, _outcome = bootstrap_live(client, PARENT)

    assert live_page["action"] == "exists"
    append = session.calls[2]
    assert append["method"] == "PATCH"
    assert append["url"].endswith(f"/v1/blocks/{LIVE_PAGE_ID}/children")
    (callout,) = append["json"]["children"]
    assert callout["callout"]["rich_text"][0]["text"]["content"] == LIVE_PAGE_INTRO


def test_live_fuzzy_search_match_is_not_adopted_and_workspace_parent_refuses() -> None:
    # Notion search is fuzzy: the PAPER page matches the live-title query.
    # It must not be adopted — and with a workspace-level paper page no
    # sibling can be created, so the operator is pointed at task D.
    responses = [
        search_response([live_page_object(page_id=PARENT, title="Automated AI Quant Investment")]),
        paper_page_response({"type": "workspace", "workspace": True}),
    ]
    client, session = make_client(responses)
    with pytest.raises(LiveBootstrapError, match="task D"):
        bootstrap_live(client, PARENT)
    assert all(not c["url"].endswith("/v1/pages") or c["method"] == "GET" for c in session.calls)


# -------------------------------------------------------------- write_config


def outcome_fixture() -> dict[str, dict[str, str]]:
    return {t: {"id": f"db-{i}", "action": "created"} for i, t in enumerate(ALL_TITLES)}


def test_write_config_fresh_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, PARENT, outcome_fixture())
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["notion"]["parent_page_id"] == PARENT
    assert loaded["notion"]["databases"] == {
        CONFIG_KEYS[t]: f"db-{i}" for i, t in enumerate(ALL_TITLES)
    }


def test_write_config_preserves_other_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"slack": {"channel": "C123"}}))
    write_config(config_path, PARENT, outcome_fixture())
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["slack"] == {"channel": "C123"}
    assert set(loaded["notion"]["databases"]) == set(CONFIG_KEYS.values())


def test_write_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        write_config(config_path, PARENT, outcome_fixture())


def test_write_config_live_keys_and_live_parent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    outcome = outcome_fixture()
    outcome["Trade Ledger (Live)"] = {"id": "db-tl-live", "action": "created"}
    outcome["Account Snapshots (Live)"] = {"id": "db-as-live", "action": "created"}
    write_config(config_path, PARENT, outcome, live_parent_page_id=LIVE_PAGE_ID)
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["notion"]["live_parent_page_id"] == LIVE_PAGE_ID
    assert loaded["notion"]["databases"]["trade_ledger_live"] == "db-tl-live"
    assert loaded["notion"]["databases"]["account_snapshots_live"] == "db-as-live"
    assert loaded["notion"]["databases"]["trade_ledger"] == "db-4"  # paper keys intact


def test_write_config_paper_rerun_preserves_live_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "notion": {
                    "parent_page_id": PARENT,
                    "live_parent_page_id": LIVE_PAGE_ID,
                    "databases": {
                        "trade_ledger_live": "db-tl-live",
                        "account_snapshots_live": "db-as-live",
                    },
                }
            }
        )
    )
    write_config(config_path, PARENT, outcome_fixture())  # paper-only rerun
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["notion"]["live_parent_page_id"] == LIVE_PAGE_ID
    assert loaded["notion"]["databases"]["trade_ledger_live"] == "db-tl-live"
    assert loaded["notion"]["databases"]["account_snapshots_live"] == "db-as-live"
    assert set(loaded["notion"]["databases"]) == set(CONFIG_KEYS.values()) | set(
        LIVE_CONFIG_KEYS.values()
    )


# ---------------------------------------------------------------------- main


def test_main_end_to_end_with_mocked_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    responses = [children_response([])] + [
        created_db_response(f"db-{i}") for i in range(len(ALL_TITLES))
    ]
    session = FakeSession(responses)
    monkeypatch.setattr(
        "ops.bootstrap_notion.NotionClient",
        lambda: NotionClient(session=session, sleep=lambda _s: None),
    )
    config_path = tmp_path / "config.yaml"
    assert main(["--parent-page-id", PARENT, "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "created" in out and str(config_path) in out
    loaded = yaml.safe_load(config_path.read_text())
    assert len(loaded["notion"]["databases"]) == 7
    assert "live_parent_page_id" not in loaded["notion"]  # paper-only run


def test_main_live_end_to_end_with_mocked_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paper_blocks = [child_db_block(t, f"db-{i}") for i, t in enumerate(ALL_TITLES)]
    responses = [
        children_response(paper_blocks),  # paper bootstrap: everything exists
        search_response([]),
        paper_page_response({"type": "page_id", "page_id": GRANDPARENT}),
        FakeResponse(200, {"object": "page", "id": LIVE_PAGE_ID}),
        children_response([]),
        created_db_response("db-tl-live"),
        created_db_response("db-as-live"),
    ]
    session = FakeSession(responses)
    monkeypatch.setattr(
        "ops.bootstrap_notion.NotionClient",
        lambda: NotionClient(session=session, sleep=lambda _s: None),
    )
    config_path = tmp_path / "config.yaml"
    assert main(["--parent-page-id", PARENT, "--config", str(config_path), "--live"]) == 0
    out = capsys.readouterr().out
    assert LIVE_PAGE_TITLE in out
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["notion"]["live_parent_page_id"] == LIVE_PAGE_ID
    assert loaded["notion"]["databases"]["trade_ledger_live"] == "db-tl-live"
    assert loaded["notion"]["databases"]["account_snapshots_live"] == "db-as-live"
    assert len(loaded["notion"]["databases"]) == 9


def test_main_live_reports_failure_when_page_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paper_blocks = [child_db_block(t, f"db-{i}") for i, t in enumerate(ALL_TITLES)]
    responses = [
        children_response(paper_blocks),
        search_response([]),
        paper_page_response({"type": "workspace", "workspace": True}),
    ]
    session = FakeSession(responses)
    monkeypatch.setattr(
        "ops.bootstrap_notion.NotionClient",
        lambda: NotionClient(session=session, sleep=lambda _s: None),
    )
    config_path = tmp_path / "config.yaml"
    assert main(["--parent-page-id", PARENT, "--config", str(config_path), "--live"]) == 1
    assert "task D" in capsys.readouterr().err
    assert not config_path.exists()  # nothing half-written

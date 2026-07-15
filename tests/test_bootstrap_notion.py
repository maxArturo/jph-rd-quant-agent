"""Unit tests for ops/bootstrap_notion.py (mocked HTTP via FakeSession)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ops.bootstrap_notion import (
    CONFIG_KEYS,
    IDEAS_TITLE,
    bootstrap,
    database_properties,
    main,
    write_config,
)
from orchestrator.notion_client import NOTION_VERSION, NotionClient
from tests.test_notion_client import FakeResponse, FakeSession

PARENT = "3979b1a4-36cf-8046-baa5-cc14c1ca7665"

ALL_TITLES = list(CONFIG_KEYS)


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
    for title, props in database_properties("db-ideas").items():
        assert f"## {title}" in text
        for prop_name in props:
            assert f"| {prop_name} " in text, f"{prop_name} missing from {title} table in {doc}"


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
    assert len(loaded["notion"]["databases"]) == 6

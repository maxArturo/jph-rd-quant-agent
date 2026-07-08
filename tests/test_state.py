"""Unit tests for orchestrator/state.py (US-007)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.state import DuplicateRunError, StateStore


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.sqlite"


@pytest.fixture()
def store(db_path: Path) -> StateStore:
    return StateStore(db_path)


# -- migration ---------------------------------------------------------------


def test_startup_creates_db_file_and_tables(db_path: Path) -> None:
    assert not db_path.exists()
    StateStore(db_path)
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"directives", "runs", "promoted_strategy", "pending_interactions"} <= names


def test_migration_is_idempotent(db_path: Path) -> None:
    store = StateStore(db_path)
    store.create_directive("111.222", "momentum in semis")
    store.migrate()  # explicit re-run
    StateStore(db_path)  # second startup on the same file
    assert StateStore(db_path).get_directive("111.222") is not None


# -- directives ---------------------------------------------------------------


def test_directive_create_and_fetch_by_thread(store: StateStore) -> None:
    created = store.create_directive(
        "111.222",
        objective="find momentum factors in semis",
        universe_hint="semiconductors",
        constraints="long-only",
    )
    fetched = store.get_directive("111.222")
    assert fetched == created
    assert fetched is not None and fetched.universe_hint == "semiconductors"


def test_get_directive_returns_latest_for_thread(store: StateStore) -> None:
    store.create_directive("111.222", "first idea")
    store.create_directive("111.222", "refined idea")
    fetched = store.get_directive("111.222")
    assert fetched is not None and fetched.objective == "refined idea"


def test_get_directive_missing_thread_returns_none(store: StateStore) -> None:
    assert store.get_directive("999.000") is None


# -- runs ----------------------------------------------------------------------


def test_run_create_and_fetch_by_thread(store: StateStore) -> None:
    created = store.create_run(
        "111.222", session_path="/logs/run1", universe="us_liquid"
    )
    fetched = store.get_run("111.222")
    assert fetched == created
    assert fetched is not None and fetched.status == "running"


def test_duplicate_run_for_thread_is_rejected(store: StateStore) -> None:
    store.create_run("111.222", session_path="/logs/run1")
    with pytest.raises(DuplicateRunError) as exc_info:
        store.create_run("111.222", session_path="/logs/run2")
    assert exc_info.value.existing.session_path == "/logs/run1"


def test_update_run_status(store: StateStore) -> None:
    store.create_run("111.222", session_path="/logs/run1")
    updated = store.update_run_status("111.222", "stopped")
    assert updated.status == "stopped"
    fetched = store.get_run("111.222")
    assert fetched is not None and fetched.status == "stopped"


def test_update_run_status_missing_thread_raises(store: StateStore) -> None:
    with pytest.raises(KeyError):
        store.update_run_status("999.000", "stopped")


def test_list_runs_filters_by_status(store: StateStore) -> None:
    store.create_run("1.1", session_path="/logs/a")
    store.create_run("2.2", session_path="/logs/b")
    store.update_run_status("2.2", "finished")
    running = store.list_runs(status="running")
    assert [run.thread_ts for run in running] == ["1.1"]
    assert len(store.list_runs()) == 2


def test_delete_run_frees_thread(store: StateStore) -> None:
    store.create_run("1.1", session_path="/logs/a")
    store.delete_run("1.1")
    assert store.get_run("1.1") is None
    store.create_run("1.1", session_path="/logs/b")  # no DuplicateRunError


# -- promoted strategy -----------------------------------------------------------


def test_promoted_strategy_empty_initially(store: StateStore) -> None:
    assert store.get_promoted_strategy() is None


def test_promoted_strategy_set_and_get(store: StateStore) -> None:
    config = {"topk": 30, "n_drop": 3}
    store.set_promoted_strategy("/workspaces/abc", config)
    fetched = store.get_promoted_strategy()
    assert fetched is not None
    assert fetched.workspace_path == "/workspaces/abc"
    assert fetched.config == config


def test_promoted_strategy_replace_keeps_single_row(store: StateStore, db_path: Path) -> None:
    store.set_promoted_strategy("/workspaces/old", {"topk": 30})
    store.set_promoted_strategy("/workspaces/new", {"topk": 50})
    fetched = store.get_promoted_strategy()
    assert fetched is not None and fetched.workspace_path == "/workspaces/new"
    assert fetched.config == {"topk": 50}
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM promoted_strategy").fetchone()
    assert count == 1


# -- pending interactions ---------------------------------------------------------


def test_pending_interaction_add_and_list(store: StateStore) -> None:
    added = store.add_pending_interaction(
        "111.222", "hypo-1", {"hypothesis": "momentum works"}
    )
    assert added is not None and added.status == "pending"
    listed = store.list_pending_interactions()
    assert listed == [added]


def test_pending_interaction_dedup_on_key(store: StateStore) -> None:
    assert store.add_pending_interaction("111.222", "hypo-1", {"a": 1}) is not None
    assert store.add_pending_interaction("111.222", "hypo-1", {"a": 1}) is None
    assert len(store.list_pending_interactions()) == 1


def test_pending_interaction_resolve(store: StateStore) -> None:
    added = store.add_pending_interaction("111.222", "hypo-1", {"a": 1})
    assert added is not None
    resolved = store.resolve_pending_interaction(added.id, "approved")
    assert resolved.status == "approved"
    assert resolved.resolved_at is not None
    assert store.list_pending_interactions() == []  # default lists only 'pending'
    assert store.list_pending_interactions(status="approved") == [resolved]


def test_pending_interaction_list_filters_by_thread(store: StateStore) -> None:
    store.add_pending_interaction("1.1", "k1", {})
    store.add_pending_interaction("2.2", "k2", {})
    listed = store.list_pending_interactions(thread_ts="1.1")
    assert [item.interaction_key for item in listed] == ["k1"]


def test_resolve_missing_interaction_raises(store: StateStore) -> None:
    with pytest.raises(KeyError):
        store.resolve_pending_interaction(42, "approved")


# -- restart survival --------------------------------------------------------------


def test_state_survives_store_restart(db_path: Path) -> None:
    store = StateStore(db_path)
    store.create_directive("111.222", "idea", universe_hint="semis")
    store.create_run("111.222", session_path="/logs/run1", universe="custom_semis")
    store.set_promoted_strategy("/workspaces/abc", {"topk": 30})
    pending = store.add_pending_interaction("111.222", "hypo-1", {"h": "x"})
    assert pending is not None

    reopened = StateStore(db_path)  # simulates process restart
    directive = reopened.get_directive("111.222")
    assert directive is not None and directive.objective == "idea"
    run = reopened.get_run("111.222")
    assert run is not None and run.universe == "custom_semis"
    promoted = reopened.get_promoted_strategy()
    assert promoted is not None and promoted.workspace_path == "/workspaces/abc"
    assert reopened.list_pending_interactions("111.222") == [pending]

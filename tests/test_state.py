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


# -- thread universes (US-023) -------------------------------------------------


def test_thread_universe_propose_get_confirm(store: StateStore) -> None:
    assert store.get_thread_universe("111.222") is None
    proposed = store.propose_thread_universe("111.222", "ai_semis", ["NVDA", "AMD"])
    assert proposed.name == "ai_semis"
    assert proposed.tickers == ("NVDA", "AMD")
    assert proposed.status == "proposed"

    confirmed = store.confirm_thread_universe("111.222")
    assert confirmed.status == "confirmed"
    assert confirmed.tickers == ("NVDA", "AMD")


def test_thread_universe_repropose_resets_to_proposed(store: StateStore) -> None:
    store.propose_thread_universe("111.222", "ai_semis", ["NVDA", "AMD"])
    store.confirm_thread_universe("111.222")
    replaced = store.propose_thread_universe("111.222", "ai_chips", ["NVDA", "AVGO"])
    assert replaced.name == "ai_chips"
    assert replaced.status == "proposed"
    # still a single row per thread
    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM universes WHERE thread_ts = '111.222'"
        ).fetchone()[0]
    assert count == 1


def test_thread_universe_confirm_without_proposal_raises(store: StateStore) -> None:
    with pytest.raises(KeyError):
        store.confirm_thread_universe("999.999")


def test_thread_universe_delete_and_restart_survival(db_path: Path) -> None:
    store = StateStore(db_path)
    store.propose_thread_universe("111.222", "ai_semis", ["NVDA", "AMD"])
    reopened = StateStore(db_path)
    survived = reopened.get_thread_universe("111.222")
    assert survived is not None and survived.name == "ai_semis"
    reopened.delete_thread_universe("111.222")
    assert reopened.get_thread_universe("111.222") is None


def test_run_universe_tickers_roundtrip(store: StateStore) -> None:
    run = store.create_run(
        "111.222", "/logs/run1", universe="ai_semis", universe_tickers=["NVDA", "AMD"]
    )
    assert run.universe_tickers == ("NVDA", "AMD")
    fetched = store.get_run("111.222")
    assert fetched is not None and fetched.universe_tickers == ("NVDA", "AMD")

    bare = store.create_run("333.444", "/logs/run2", universe="us_liquid")
    assert bare.universe_tickers is None
    fetched_bare = store.get_run("333.444")
    assert fetched_bare is not None and fetched_bare.universe_tickers is None


def test_migration_adds_universe_tickers_to_legacy_db(db_path: Path) -> None:
    """DBs created before US-023 lack runs.universe_tickers; migrate() retrofits it."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE runs (thread_ts TEXT PRIMARY KEY, session_path TEXT NOT NULL,"
            " status TEXT NOT NULL, universe TEXT, created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('111.222', '/logs/run1', 'running', 'us_liquid',"
            " '2026-01-01', '2026-01-01')"
        )
    store = StateStore(db_path)  # migration runs here
    legacy = store.get_run("111.222")
    assert legacy is not None and legacy.universe_tickers is None
    store.create_run("333.444", "/logs/run2", universe_tickers=["NVDA"])
    fresh = store.get_run("333.444")
    assert fresh is not None and fresh.universe_tickers == ("NVDA",)


def test_run_supervised_roundtrip_and_default(store: StateStore) -> None:
    run = store.create_run("111.222", "/logs/run1", supervised=True)
    assert run.supervised is True
    fetched = store.get_run("111.222")
    assert fetched is not None and fetched.supervised is True

    bare = store.create_run("333.444", "/logs/run2")
    assert bare.supervised is False  # autonomous is the default (US-045)
    fetched_bare = store.get_run("333.444")
    assert fetched_bare is not None and fetched_bare.supervised is False


def test_migration_adds_supervised_to_legacy_db(db_path: Path) -> None:
    """DBs created before US-045 lack runs.supervised; legacy runs go autonomous."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE runs (thread_ts TEXT PRIMARY KEY, session_path TEXT NOT NULL,"
            " status TEXT NOT NULL, universe TEXT, universe_tickers TEXT,"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('111.222', '/logs/run1', 'running', 'us_liquid',"
            " NULL, '2026-01-01', '2026-01-01')"
        )
    store = StateStore(db_path)  # migration runs here
    legacy = store.get_run("111.222")
    assert legacy is not None and legacy.supervised is False


def test_list_interactions_returns_all_statuses_oldest_first(store: StateStore) -> None:
    first = store.add_pending_interaction("t1", "k1", {"kind": "hypothesis"})
    second = store.add_pending_interaction("t1", "k2", {"kind": "feedback"})
    store.add_pending_interaction("t2", "k3", {"kind": "hypothesis"})  # other thread
    assert first is not None and second is not None
    store.resolve_pending_interaction(first.id, "auto_approved")

    rows = store.list_interactions("t1")
    assert [r.interaction_key for r in rows] == ["k1", "k2"]
    assert [r.status for r in rows] == ["auto_approved", "pending"]

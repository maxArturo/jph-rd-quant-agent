"""Unit tests for orchestrator/state.py (US-007)."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
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


# -- WAL mode + busy timeout (loop-hardening US-001) ---------------------------


def test_connections_use_wal_and_busy_timeout(store: StateStore) -> None:
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30_000


def test_existing_delete_mode_db_converts_to_wal_with_rows_intact(db_path: Path) -> None:
    StateStore(db_path).create_directive("111.222", "seeded before WAL")
    # Force the file back to the legacy rollback-journal mode, as the live DB
    # was before this change.
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
    store = StateStore(db_path)
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    fetched = store.get_directive("111.222")
    assert fetched is not None and fetched.objective == "seeded before WAL"


def test_concurrent_connections_write_without_lock_errors(store: StateStore) -> None:
    # Each helper call opens its own connection, so parallel threads exercise
    # genuinely concurrent writers against the same file.
    errors: list[Exception] = []

    def write(worker: int) -> None:
        try:
            for i in range(25):
                store.create_directive(f"{worker}.{i}", f"idea {worker}-{i}")
        except Exception as exc:  # noqa: BLE001 - surface any lock error
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    with closing(sqlite3.connect(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM directives").fetchone()[0]
    assert count == 100


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


def test_run_backend_defaults_to_server_ui_and_persists_gpu(store: StateStore) -> None:
    store.create_run("1.1", session_path="/logs/a")
    store.create_run("2.2", session_path="/status.json", backend="gpu")
    legacy = store.get_run("1.1")
    gpu = store.get_run("2.2")
    assert legacy is not None and legacy.backend == "server_ui"
    assert gpu is not None and gpu.backend == "gpu"


def test_update_run_session_path_repoints_fetched_trace(store: StateStore) -> None:
    store.create_run("3.3", session_path="/status.json", backend="gpu")
    updated = store.update_run_session_path("3.3", "/results/us_quant/log/2026-08-06")
    assert updated.session_path == "/results/us_quant/log/2026-08-06"
    with pytest.raises(KeyError):
        store.update_run_session_path("999.000", "/x")


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


def test_create_run_replace_failed_replaces_a_failed_row(store: StateStore) -> None:
    """US-021: a reaped run row can be replaced atomically by a fresh start."""
    store.create_run("1.1", session_path="/status.json", backend="gpu")
    store.update_run_status("1.1", "failed")
    replacement = store.create_run(
        "1.1", session_path="/status2.json", backend="gpu", replace_failed=True
    )
    fetched = store.get_run("1.1")
    assert fetched == replacement
    assert fetched is not None and fetched.status == "running"
    assert fetched.session_path == "/status2.json"


def test_create_run_replace_failed_still_rejects_a_live_row(store: StateStore) -> None:
    store.create_run("1.1", session_path="/status.json", backend="gpu")
    with pytest.raises(DuplicateRunError) as exc_info:
        store.create_run("1.1", session_path="/status2.json", replace_failed=True)
    assert exc_info.value.existing.session_path == "/status.json"
    fetched = store.get_run("1.1")
    assert fetched is not None and fetched.status == "running"


def test_create_run_default_still_rejects_a_failed_row(store: StateStore) -> None:
    store.create_run("1.1", session_path="/status.json", backend="gpu")
    store.update_run_status("1.1", "failed")
    with pytest.raises(DuplicateRunError):
        store.create_run("1.1", session_path="/status2.json")


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


# -- promotion history (loop-hardening US-005) -------------------------------------


def test_promote_appends_history_row(store: StateStore) -> None:
    store.set_promoted_strategy(
        "/workspaces/abc", {"topk": 30}, source="auto_gate", gate_verdict={"pass": True}
    )
    history = store.list_promotion_history()
    assert len(history) == 1
    entry = history[0]
    assert entry.workspace_path == "/workspaces/abc"
    assert entry.config == {"topk": 30}
    assert entry.source == "auto_gate"
    assert entry.gate_verdict == {"pass": True}
    assert entry.replaced_workspace is None
    assert entry.promoted_at != ""


def test_history_records_replaced_workspace(store: StateStore) -> None:
    store.set_promoted_strategy("/workspaces/old", {"topk": 30}, source="cli")
    store.set_promoted_strategy("/workspaces/new", {"topk": 50}, source="conversation")
    newest, oldest = store.list_promotion_history()  # newest first
    assert newest.workspace_path == "/workspaces/new"
    assert newest.replaced_workspace == "/workspaces/old"
    assert newest.source == "conversation"
    assert oldest.replaced_workspace is None
    # the single-row pointer still tracks the latest promotion
    promoted = store.get_promoted_strategy()
    assert promoted is not None and promoted.workspace_path == "/workspaces/new"


def test_history_default_source_is_cli_with_null_verdict(store: StateStore) -> None:
    store.set_promoted_strategy("/workspaces/abc", {"topk": 30})
    (entry,) = store.list_promotion_history()
    assert entry.source == "cli"
    assert entry.gate_verdict is None


def test_promote_rejects_unknown_source(store: StateStore) -> None:
    with pytest.raises(ValueError, match="unknown promotion source"):
        store.set_promoted_strategy("/workspaces/abc", {}, source="button_mash")
    assert store.list_promotion_history() == []
    assert store.get_promoted_strategy() is None


def test_list_promotion_history_limit(store: StateStore) -> None:
    for i in range(4):
        store.set_promoted_strategy(f"/workspaces/{i}", {"topk": i})
    recent = store.list_promotion_history(limit=2)
    assert [e.workspace_path for e in recent] == ["/workspaces/3", "/workspaces/2"]


def test_migration_backfills_legacy_promoted_row(db_path: Path) -> None:
    """DBs promoted before US-005 lack promotion_history; the existing promoted
    row becomes history row 1 with source 'cli'."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE promoted_strategy (id INTEGER PRIMARY KEY CHECK (id = 1),"
            " workspace_path TEXT NOT NULL, config TEXT NOT NULL,"
            " promoted_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO promoted_strategy VALUES"
            " (1, '/workspaces/legacy', '{\"topk\": 30}', '2026-08-01T00:00:00+00:00')"
        )
    store = StateStore(db_path)  # migration runs here
    (entry,) = store.list_promotion_history()
    assert entry.workspace_path == "/workspaces/legacy"
    assert entry.config == {"topk": 30}
    assert entry.promoted_at == "2026-08-01T00:00:00+00:00"
    assert entry.source == "cli"
    assert entry.gate_verdict is None and entry.replaced_workspace is None


def test_backfill_is_idempotent_across_remigrations(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE promoted_strategy (id INTEGER PRIMARY KEY CHECK (id = 1),"
            " workspace_path TEXT NOT NULL, config TEXT NOT NULL,"
            " promoted_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO promoted_strategy VALUES"
            " (1, '/workspaces/legacy', '{}', '2026-08-01T00:00:00+00:00')"
        )
    store = StateStore(db_path)
    store.migrate()  # explicit re-run
    StateStore(db_path)  # second startup on the same file
    assert len(store.list_promotion_history()) == 1
    # later promotions grow the history; further migrations still add nothing
    store.set_promoted_strategy("/workspaces/new", {"topk": 50})
    store.migrate()
    assert len(store.list_promotion_history()) == 2


def test_fresh_db_without_promotion_backfills_nothing(store: StateStore) -> None:
    assert store.list_promotion_history() == []
    store.migrate()
    assert store.list_promotion_history() == []


# -- pending interactions (legacy, read-only since US-027) -------------------------


def _seed_legacy_interaction(
    db_path: Path, thread_ts: str, key: str, status: str = "pending"
) -> None:
    """Insert a poller-era row the way the deleted write path used to.

    The poller was removed in US-027; historic rows must stay readable, so
    the seed goes through raw SQL exactly like a legacy on-disk DB.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO pending_interactions (thread_ts, interaction_key, payload,"
            " status, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_ts, key, '{"kind": "hypothesis"}', status, "2026-07-08T10:00:00+00:00"),
        )


def test_legacy_interaction_rows_remain_readable(store: StateStore, db_path: Path) -> None:
    _seed_legacy_interaction(db_path, "111.222", "hypo-1", status="approved")
    _seed_legacy_interaction(db_path, "111.222", "hypo-2", status="pending")
    _seed_legacy_interaction(db_path, "333.444", "hypo-3", status="rejected")

    rows = store.list_interactions("111.222")
    assert [r.interaction_key for r in rows] == ["hypo-1", "hypo-2"]  # oldest first
    assert [r.status for r in rows] == ["approved", "pending"]
    assert rows[0].payload == {"kind": "hypothesis"}
    assert store.list_interactions("999.999") == []


# -- restart survival --------------------------------------------------------------


def test_state_survives_store_restart(db_path: Path) -> None:
    store = StateStore(db_path)
    store.create_directive("111.222", "idea", universe_hint="semis")
    store.create_run("111.222", session_path="/logs/run1", universe="custom_semis")
    store.set_promoted_strategy("/workspaces/abc", {"topk": 30})

    reopened = StateStore(db_path)  # simulates process restart
    directive = reopened.get_directive("111.222")
    assert directive is not None and directive.objective == "idea"
    run = reopened.get_run("111.222")
    assert run is not None and run.universe == "custom_semis"
    promoted = reopened.get_promoted_strategy()
    assert promoted is not None and promoted.workspace_path == "/workspaces/abc"


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



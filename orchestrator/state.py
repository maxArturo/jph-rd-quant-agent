"""SQLite state store for the orchestrator.

Persists what must survive a process restart: refined research directives,
thread-to-run mappings, the single promoted strategy (plus its append-only
promotion history), and legacy poller-era interaction rows (read-only since
US-027). The schema migration is idempotent (plain ``CREATE ... IF NOT
EXISTS``) and runs on every startup.

Concurrency model: each helper opens a short-lived connection, so a
``StateStore`` instance is safe to share across threads (the Bolt handlers
and background threads never share a sqlite3 connection). The database
runs in WAL mode with a 30s busy timeout so writers in other processes
(the transient GPU pipeline unit, CLI promotes) queue instead of failing
with 'database is locked'.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "state.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS directives (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_ts   TEXT NOT NULL,
    objective   TEXT NOT NULL,
    universe_hint TEXT,
    constraints TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_directives_thread_ts ON directives (thread_ts);

CREATE TABLE IF NOT EXISTS runs (
    thread_ts    TEXT PRIMARY KEY,
    session_path TEXT NOT NULL,
    status       TEXT NOT NULL,
    universe     TEXT,
    universe_tickers TEXT,
    supervised   INTEGER NOT NULL DEFAULT 0,
    backend      TEXT NOT NULL DEFAULT 'server_ui',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- Per-thread custom universe (US-023): proposed by set_universe, flipped to
-- 'confirmed' after the operator approves and the data work succeeds.
CREATE TABLE IF NOT EXISTS universes (
    thread_ts  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    tickers    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'proposed',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Single-row table: id is constrained to 1 so a second strategy can only
-- ever replace the first, never coexist with it.
CREATE TABLE IF NOT EXISTS promoted_strategy (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    workspace_path TEXT NOT NULL,
    config         TEXT NOT NULL,
    promoted_at    TEXT NOT NULL
);

-- Append-only promotion audit trail (US-005): every set_promoted_strategy
-- call appends a row here, while promoted_strategy above stays the current
-- pointer. Rows are never updated or deleted, so what-replaced-what is
-- always reconstructable.
CREATE TABLE IF NOT EXISTS promotion_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_path     TEXT NOT NULL,
    config             TEXT NOT NULL,
    promoted_at        TEXT NOT NULL,
    source             TEXT NOT NULL
        CHECK (source IN ('auto_gate', 'conversation', 'cli')),
    gate_verdict       TEXT,
    replaced_workspace TEXT
);

-- Notion page-id mappings (US-027): which Notion page records a given
-- lifecycle object (kind 'idea' keyed by thread_ts, kind 'hypothesis' keyed
-- by interaction_key), so later lifecycle points update the same page.
CREATE TABLE IF NOT EXISTS notion_pages (
    kind       TEXT NOT NULL,
    key        TEXT NOT NULL,
    page_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, key)
);

CREATE TABLE IF NOT EXISTS pending_interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_ts       TEXT NOT NULL,
    interaction_key TEXT NOT NULL UNIQUE,
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_interactions_status
    ON pending_interactions (status);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Directive:
    id: int
    thread_ts: str
    objective: str
    universe_hint: str | None
    constraints: str | None
    created_at: str


@dataclass(frozen=True)
class Run:
    thread_ts: str
    session_path: str
    status: str
    universe: str | None
    created_at: str
    updated_at: str
    universe_tickers: tuple[str, ...] | None = None
    # Supervised runs gate each hypothesis on operator buttons (the pre-US-045
    # flow); unsupervised runs auto-approve and stop on their own budget.
    supervised: bool = False
    # 'server_ui' = legacy local trace (poller-era rows); 'gpu' = remote
    # burst-droplet run driven by ops/gpu_pipeline (session_path holds the
    # pipeline status file until fetch rewrites it to the fetched trace dir).
    backend: str = "server_ui"


@dataclass(frozen=True)
class ThreadUniverse:
    thread_ts: str
    name: str
    tickers: tuple[str, ...]
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class PromotedStrategy:
    workspace_path: str
    config: dict[str, Any]
    promoted_at: str


# Who drove a promotion: the auto-promotion gate, a conversational (Slack)
# confirm, or a CLI invocation (ops/promote_fetched.py, rollback).
PROMOTION_SOURCES = ("auto_gate", "conversation", "cli")


@dataclass(frozen=True)
class PromotionHistoryEntry:
    id: int
    workspace_path: str
    config: dict[str, Any]
    promoted_at: str
    source: str
    gate_verdict: dict[str, Any] | None
    replaced_workspace: str | None


@dataclass(frozen=True)
class PendingInteraction:
    id: int
    thread_ts: str
    interaction_key: str
    payload: dict[str, Any]
    status: str
    created_at: str
    resolved_at: str | None


def _directive_from_row(row: sqlite3.Row) -> Directive:
    return Directive(
        id=row["id"],
        thread_ts=row["thread_ts"],
        objective=row["objective"],
        universe_hint=row["universe_hint"],
        constraints=row["constraints"],
        created_at=row["created_at"],
    )


def _run_from_row(row: sqlite3.Row) -> Run:
    tickers = row["universe_tickers"]
    return Run(
        thread_ts=row["thread_ts"],
        session_path=row["session_path"],
        status=row["status"],
        universe=row["universe"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        universe_tickers=None if tickers is None else tuple(json.loads(tickers)),
        supervised=bool(row["supervised"]),
        backend=row["backend"],
    )


def _universe_from_row(row: sqlite3.Row) -> ThreadUniverse:
    return ThreadUniverse(
        thread_ts=row["thread_ts"],
        name=row["name"],
        tickers=tuple(json.loads(row["tickers"])),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _history_from_row(row: sqlite3.Row) -> PromotionHistoryEntry:
    verdict = row["gate_verdict"]
    return PromotionHistoryEntry(
        id=row["id"],
        workspace_path=row["workspace_path"],
        config=json.loads(row["config"]),
        promoted_at=row["promoted_at"],
        source=row["source"],
        gate_verdict=None if verdict is None else json.loads(verdict),
        replaced_workspace=row["replaced_workspace"],
    )


def _interaction_from_row(row: sqlite3.Row) -> PendingInteraction:
    return PendingInteraction(
        id=row["id"],
        thread_ts=row["thread_ts"],
        interaction_key=row["interaction_key"],
        payload=json.loads(row["payload"]),
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )


class DuplicateRunError(RuntimeError):
    """A run already exists for this thread (one active run per thread)."""

    def __init__(self, existing: Run):
        super().__init__(f"thread {existing.thread_ts} already has a run: {existing.session_path}")
        self.existing = existing


class StateStore:
    """Thread-safe accessor for orchestrator/state.sqlite."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # timeout=30 doubles as PRAGMA busy_timeout=30000: concurrent writers
        # (orchestrator threads, the transient GPU pipeline unit, CLI
        # promotes) block instead of raising 'database is locked'.
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL lets readers proceed during a write. The mode is persistent
            # in the DB file, so this is a cheap no-op after the first switch
            # (and safely converts a pre-existing delete-mode DB in place).
            conn.execute("PRAGMA journal_mode = WAL")
            with conn:  # commit on success, rollback on exception
                yield conn

    def migrate(self) -> None:
        """Create the schema. Idempotent — safe to run on every startup."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Column added in US-023; CREATE IF NOT EXISTS skips existing DBs,
            # so retrofit them with a guarded ALTER (also idempotent).
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
            if "universe_tickers" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN universe_tickers TEXT")
            # Column added in US-045: pre-existing runs become unsupervised
            # (autonomous is the new default behavior).
            if "supervised" not in columns:
                conn.execute(
                    "ALTER TABLE runs ADD COLUMN supervised INTEGER NOT NULL DEFAULT 0"
                )
            # GPU burst-worker backend (2026-08-06): pre-existing runs are all
            # server_ui traces.
            if "backend" not in columns:
                conn.execute(
                    "ALTER TABLE runs ADD COLUMN backend TEXT NOT NULL DEFAULT 'server_ui'"
                )
            # Backfill (US-005): a DB promoted before promotion_history existed
            # gets its current pointer as history row 1. Provenance of that
            # promotion is unknowable, so it is labeled 'cli' (every promotion
            # to date was a human-driven path). History is append-only and
            # never emptied, so the empty-table guard makes this idempotent.
            has_history = conn.execute(
                "SELECT 1 FROM promotion_history LIMIT 1"
            ).fetchone()
            if has_history is None:
                promoted = conn.execute(
                    "SELECT workspace_path, config, promoted_at"
                    " FROM promoted_strategy WHERE id = 1"
                ).fetchone()
                if promoted is not None:
                    conn.execute(
                        "INSERT INTO promotion_history (workspace_path, config,"
                        " promoted_at, source, gate_verdict, replaced_workspace)"
                        " VALUES (?, ?, ?, 'cli', NULL, NULL)",
                        (
                            promoted["workspace_path"],
                            promoted["config"],
                            promoted["promoted_at"],
                        ),
                    )

    # -- directives ---------------------------------------------------------

    def create_directive(
        self,
        thread_ts: str,
        objective: str,
        universe_hint: str | None = None,
        constraints: str | None = None,
    ) -> Directive:
        now = _utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO directives (thread_ts, objective, universe_hint, constraints,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_ts, objective, universe_hint, constraints, now),
            )
            row_id = cur.lastrowid
        assert row_id is not None
        return Directive(
            id=row_id,
            thread_ts=thread_ts,
            objective=objective,
            universe_hint=universe_hint,
            constraints=constraints,
            created_at=now,
        )

    def get_directive(self, thread_ts: str) -> Directive | None:
        """Latest directive for a thread (a thread may refine its idea repeatedly)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM directives WHERE thread_ts = ? ORDER BY id DESC LIMIT 1",
                (thread_ts,),
            ).fetchone()
        return None if row is None else _directive_from_row(row)

    # -- runs ----------------------------------------------------------------

    def create_run(
        self,
        thread_ts: str,
        session_path: str,
        universe: str | None = None,
        status: str = "running",
        universe_tickers: Sequence[str] | None = None,
        supervised: bool = False,
        backend: str = "server_ui",
        replace_failed: bool = False,
    ) -> Run:
        now = _utcnow()
        tickers = None if universe_tickers is None else tuple(universe_tickers)
        run = Run(
            thread_ts=thread_ts,
            session_path=session_path,
            status=status,
            universe=universe,
            created_at=now,
            updated_at=now,
            universe_tickers=tickers,
            supervised=supervised,
            backend=backend,
        )
        try:
            with self._connect() as conn:
                if replace_failed:
                    # US-021: a reaped (failed) run must not brick its thread —
                    # drop it in the same transaction so the insert either
                    # replaces exactly a failed row or hits the PK (a
                    # concurrent non-failed row still raises DuplicateRunError).
                    conn.execute(
                        "DELETE FROM runs WHERE thread_ts = ? AND status = 'failed'",
                        (thread_ts,),
                    )
                conn.execute(
                    "INSERT INTO runs (thread_ts, session_path, status, universe,"
                    " universe_tickers, supervised, backend, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        thread_ts,
                        session_path,
                        status,
                        universe,
                        None if tickers is None else json.dumps(list(tickers)),
                        int(supervised),
                        backend,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.get_run(thread_ts)
            assert existing is not None
            raise DuplicateRunError(existing) from exc
        return run

    def get_run(self, thread_ts: str) -> Run | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE thread_ts = ?", (thread_ts,)
            ).fetchone()
        return None if row is None else _run_from_row(row)

    def list_runs(self, status: str | None = None) -> list[Run]:
        query = "SELECT * FROM runs"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created_at", params).fetchall()
        return [_run_from_row(row) for row in rows]

    def update_run_status(self, thread_ts: str, status: str) -> Run:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE thread_ts = ?",
                (status, _utcnow(), thread_ts),
            )
            if cur.rowcount == 0:
                raise KeyError(f"no run for thread {thread_ts}")
        run = self.get_run(thread_ts)
        assert run is not None
        return run

    def update_run_session_path(self, thread_ts: str, session_path: str) -> Run:
        """Repoint a run at its (fetched) trace dir — the GPU pipeline calls
        this after `fetch` so promotion can locate artifacts locally."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE runs SET session_path = ?, updated_at = ? WHERE thread_ts = ?",
                (session_path, _utcnow(), thread_ts),
            )
            if cur.rowcount == 0:
                raise KeyError(f"no run for thread {thread_ts}")
        run = self.get_run(thread_ts)
        assert run is not None
        return run

    def delete_run(self, thread_ts: str) -> None:
        """Free a thread for a new run (e.g. after a failed or abandoned one)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE thread_ts = ?", (thread_ts,))

    # -- thread universes (US-023) --------------------------------------------

    def propose_thread_universe(
        self, thread_ts: str, name: str, tickers: Sequence[str]
    ) -> ThreadUniverse:
        """Upsert the thread's universe proposal (re-proposing resets to 'proposed')."""
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO universes (thread_ts, name, tickers, status, created_at,"
                " updated_at) VALUES (?, ?, ?, 'proposed', ?, ?)"
                " ON CONFLICT (thread_ts) DO UPDATE SET name = excluded.name,"
                " tickers = excluded.tickers, status = 'proposed',"
                " updated_at = excluded.updated_at",
                (thread_ts, name, json.dumps(list(tickers)), now, now),
            )
        universe = self.get_thread_universe(thread_ts)
        assert universe is not None
        return universe

    def get_thread_universe(self, thread_ts: str) -> ThreadUniverse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM universes WHERE thread_ts = ?", (thread_ts,)
            ).fetchone()
        return None if row is None else _universe_from_row(row)

    def confirm_thread_universe(self, thread_ts: str) -> ThreadUniverse:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE universes SET status = 'confirmed', updated_at = ?"
                " WHERE thread_ts = ?",
                (_utcnow(), thread_ts),
            )
            if cur.rowcount == 0:
                raise KeyError(f"no universe proposal for thread {thread_ts}")
        universe = self.get_thread_universe(thread_ts)
        assert universe is not None
        return universe

    def delete_thread_universe(self, thread_ts: str) -> None:
        """Drop the thread's universe (falls back to the built-in default)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM universes WHERE thread_ts = ?", (thread_ts,))

    # -- promoted strategy ----------------------------------------------------

    def set_promoted_strategy(
        self,
        workspace_path: str,
        config: dict[str, Any],
        source: str = "cli",
        gate_verdict: dict[str, Any] | None = None,
    ) -> PromotedStrategy:
        """Replace THE promoted strategy (single row; any previous one is overwritten).

        Every call also appends a promotion_history row (US-005) recording who
        drove it (``source``), the gate verdict when one exists, and which
        workspace was replaced — pointer flip and audit row commit atomically.
        """
        if source not in PROMOTION_SOURCES:
            raise ValueError(
                f"unknown promotion source {source!r} (expected one of {PROMOTION_SOURCES})"
            )
        now = _utcnow()
        with self._connect() as conn:
            previous = conn.execute(
                "SELECT workspace_path FROM promoted_strategy WHERE id = 1"
            ).fetchone()
            conn.execute(
                "INSERT INTO promoted_strategy (id, workspace_path, config, promoted_at)"
                " VALUES (1, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET workspace_path = excluded.workspace_path,"
                " config = excluded.config, promoted_at = excluded.promoted_at",
                (workspace_path, json.dumps(config), now),
            )
            conn.execute(
                "INSERT INTO promotion_history (workspace_path, config, promoted_at,"
                " source, gate_verdict, replaced_workspace) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    workspace_path,
                    json.dumps(config),
                    now,
                    source,
                    None if gate_verdict is None else json.dumps(gate_verdict),
                    None if previous is None else previous["workspace_path"],
                ),
            )
        return PromotedStrategy(workspace_path=workspace_path, config=config, promoted_at=now)

    def get_promoted_strategy(self) -> PromotedStrategy | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_path, config, promoted_at FROM promoted_strategy WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return PromotedStrategy(
            workspace_path=row["workspace_path"],
            config=json.loads(row["config"]),
            promoted_at=row["promoted_at"],
        )

    def list_promotion_history(self, limit: int | None = None) -> list[PromotionHistoryEntry]:
        """Promotion audit rows, newest first (optionally the most recent ``limit``)."""
        query = "SELECT * FROM promotion_history ORDER BY id DESC"
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_history_from_row(row) for row in rows]

    # -- notion page mappings (US-027) -------------------------------------------

    def set_notion_page(self, kind: str, key: str, page_id: str) -> None:
        """Remember which Notion page records (kind, key) — upsert."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notion_pages (kind, key, page_id, created_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT (kind, key) DO UPDATE SET page_id = excluded.page_id",
                (kind, key, page_id, _utcnow()),
            )

    def get_notion_page(self, kind: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT page_id FROM notion_pages WHERE kind = ? AND key = ?",
                (kind, key),
            ).fetchone()
        return None if row is None else row["page_id"]

    # -- pending interactions (legacy, read-only) --------------------------------
    #
    # The hypothesis poller that wrote this table was removed in US-027 (the
    # GPU pipeline drives runs end-to-end). The table and this reader stay so
    # historic server_ui-era rows remain inspectable — never add a write path
    # or a destructive migration for it.

    def list_interactions(self, thread_ts: str) -> list[PendingInteraction]:
        """Every recorded interaction of a thread, any status, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_interactions WHERE thread_ts = ? ORDER BY id",
                (thread_ts,),
            ).fetchall()
        return [_interaction_from_row(row) for row in rows]

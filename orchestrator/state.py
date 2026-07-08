"""SQLite state store for the orchestrator.

Persists what must survive a process restart: refined research directives,
thread-to-run mappings, the single promoted strategy, and pending operator
interactions. The schema migration is idempotent (plain ``CREATE ... IF NOT
EXISTS``) and runs on every startup.

Concurrency model: each helper opens a short-lived connection, so a
``StateStore`` instance is safe to share across threads (the Bolt handlers
and the background poller never share a sqlite3 connection).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
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
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- Single-row table: id is constrained to 1 so a second strategy can only
-- ever replace the first, never coexist with it.
CREATE TABLE IF NOT EXISTS promoted_strategy (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    workspace_path TEXT NOT NULL,
    config         TEXT NOT NULL,
    promoted_at    TEXT NOT NULL
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


@dataclass(frozen=True)
class PromotedStrategy:
    workspace_path: str
    config: dict[str, Any]
    promoted_at: str


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
    return Run(
        thread_ts=row["thread_ts"],
        session_path=row["session_path"],
        status=row["status"],
        universe=row["universe"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:  # commit on success, rollback on exception
                yield conn

    def migrate(self) -> None:
        """Create the schema. Idempotent — safe to run on every startup."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

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
    ) -> Run:
        now = _utcnow()
        run = Run(
            thread_ts=thread_ts,
            session_path=session_path,
            status=status,
            universe=universe,
            created_at=now,
            updated_at=now,
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO runs (thread_ts, session_path, status, universe, created_at,"
                    " updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (thread_ts, session_path, status, universe, now, now),
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

    def delete_run(self, thread_ts: str) -> None:
        """Free a thread for a new run (e.g. after a failed or abandoned one)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE thread_ts = ?", (thread_ts,))

    # -- promoted strategy ----------------------------------------------------

    def set_promoted_strategy(
        self, workspace_path: str, config: dict[str, Any]
    ) -> PromotedStrategy:
        """Replace THE promoted strategy (single row; any previous one is overwritten)."""
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO promoted_strategy (id, workspace_path, config, promoted_at)"
                " VALUES (1, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET workspace_path = excluded.workspace_path,"
                " config = excluded.config, promoted_at = excluded.promoted_at",
                (workspace_path, json.dumps(config), now),
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

    # -- pending interactions ---------------------------------------------------

    def add_pending_interaction(
        self, thread_ts: str, interaction_key: str, payload: dict[str, Any]
    ) -> PendingInteraction | None:
        """Insert a pending interaction; return None if the key already exists (dedup)."""
        now = _utcnow()
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO pending_interactions (thread_ts, interaction_key, payload,"
                    " status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                    (thread_ts, interaction_key, json.dumps(payload), now),
                )
                row_id = cur.lastrowid
        except sqlite3.IntegrityError:
            return None
        assert row_id is not None
        return PendingInteraction(
            id=row_id,
            thread_ts=thread_ts,
            interaction_key=interaction_key,
            payload=payload,
            status="pending",
            created_at=now,
            resolved_at=None,
        )

    def list_pending_interactions(
        self, thread_ts: str | None = None, status: str = "pending"
    ) -> list[PendingInteraction]:
        query = "SELECT * FROM pending_interactions WHERE status = ?"
        params: list[str] = [status]
        if thread_ts is not None:
            query += " AND thread_ts = ?"
            params.append(thread_ts)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY id", params).fetchall()
        return [_interaction_from_row(row) for row in rows]

    def resolve_pending_interaction(self, interaction_id: int, status: str) -> PendingInteraction:
        """Mark an interaction resolved (status e.g. 'approved', 'edited', 'rejected')."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE pending_interactions SET status = ?, resolved_at = ? WHERE id = ?",
                (status, _utcnow(), interaction_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"no pending interaction with id {interaction_id}")
            row = conn.execute(
                "SELECT * FROM pending_interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return _interaction_from_row(row)

"""Run-history digest for new research runs (US-014).

``build_digest`` composes a compact, deterministic digest of prior runs so
the research LLM stops re-proposing rejected ideas. Sources, in order of
preference:

- The Notion Strategy Notes database: the most recent rows' machine-readable
  ``run_summary`` JSON (written by ops/notion_summary.py, US-013) gives per
  run the directive, every hypothesis tried with its outcome, and the winner.
- Local state (orchestrator/state.sqlite): the incumbent section always comes
  from here (promoted workspace artifacts), and runs degrade to local
  ``runs``/``directives`` data (directive + status only) when Notion is
  unreachable or a row has no parseable JSON.

The digest must NEVER raise and NEVER stall a launch: every failure degrades,
and total Notion time is budgeted (default 15s, checked before each request)
after which remaining rows degrade to local data. Output is deterministic for
the same inputs (no clocks in the text) and truncates oldest-first at
``max_chars`` (default 4000).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from orchestrator.state import DEFAULT_DB_PATH, StateStore

DIGEST_ROWS = 10
DIGEST_MAX_CHARS = 4000
NOTION_BUDGET_SECONDS = 15.0

NO_PRIOR_RUNS = "No prior runs recorded."
HEADER = "Run-history digest (prior research runs, newest first):"
NO_INCUMBENT = "Incumbent (currently promoted strategy): none"
OMITTED_NOTE = "older runs omitted to fit the digest budget"

# Per-item clips keep single entries from eating the whole budget.
_DIRECTIVE_CHARS = 240
_HYPOTHESIS_CHARS = 160
_FACTOR_NAMES_SHOWN = 12

# Keep the default client snappy: the digest is launch-path code, so one slow
# Notion request must not eat the whole budget in retries.
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_RETRIES = 1

_METRIC_LABELS = ("IC", "ARR", "MDD", "IR")
_OUTCOME_ORDER = ("SOTA", "rejected", "failed")


def build_digest(
    db_path: Path = DEFAULT_DB_PATH,
    client: Any | None = None,
    *,
    max_rows: int = DIGEST_ROWS,
    max_chars: int = DIGEST_MAX_CHARS,
    notion_budget: float = NOTION_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Best-effort digest of prior runs + incumbent. Never raises."""
    try:
        return _build(db_path, client, max_rows, max_chars, notion_budget, clock)
    except Exception:  # noqa: BLE001 — run memory must never block a launch
        return NO_PRIOR_RUNS


def _build(
    db_path: Path,
    client: Any | None,
    max_rows: int,
    max_chars: int,
    notion_budget: float,
    clock: Callable[[], float],
) -> str:
    store = _open_store(db_path)
    local = _local_entries(store, max_rows)
    incumbent = _incumbent_section(store)

    deadline = clock() + notion_budget
    rows = _notion_rows(client, max_rows, deadline, clock)

    entries: list[str] = []
    if rows:
        for info, summary in rows:
            if summary is not None:
                entries.append(_summary_entry(summary))
            else:
                directive = info.get("directive")
                status = _local_status(local, directive)
                entries.append(_degraded_entry(info.get("run_date"), status, directive))
    else:
        # Notion down, over budget, or no notes yet: local runs/directives.
        entries = [
            _degraded_entry(item["date"], item["status"], item["directive"]) for item in local
        ]

    if not entries:
        if incumbent == NO_INCUMBENT:
            return NO_PRIOR_RUNS
        return _clamp(f"{HEADER}\n\n{incumbent}\n\n{NO_PRIOR_RUNS}", max_chars)

    return _assemble(incumbent, entries, max_chars)


def _normalize(directive: str) -> str:
    return " ".join(directive.split())


def _local_status(local: list[dict[str, Any]], directive: str | None) -> str | None:
    """Status of the local run whose directive matches a Notion row's.

    Directive texts diverge in length across records — the Notion row carries
    the full run instruction (clipped at 2000 chars by Notion) while the local
    objective may be a shorter prefix of it — so match on either being a
    whitespace-normalized prefix of the other. Newest matching run wins
    (``local`` is newest-first).
    """
    key = _normalize(directive or "")
    if not key:
        return None
    for item in local:
        candidate = _normalize(item["directive"] or "")
        if candidate and (candidate.startswith(key) or key.startswith(candidate)):
            return item["status"]
    return None


def _open_store(db_path: Path) -> StateStore | None:
    """Read-only opener: StateStore(path) CREATES the db, so guard is_file."""
    try:
        if not Path(db_path).is_file():
            return None
        return StateStore(Path(db_path))
    except Exception:  # noqa: BLE001 — unreadable state degrades to no history
        return None


# -- Notion side ---------------------------------------------------------------


def _notion_rows(
    client: Any | None,
    max_rows: int,
    deadline: float,
    clock: Callable[[], float],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]] | None:
    """(row info, parsed run_summary | None) newest first; None = no Notion.

    A per-row problem (missing/garbled JSON, children fetch failure, budget
    exhausted) degrades that row's summary to None; only a failure to get the
    row list at all returns None (callers fall back to local state).
    """
    try:
        if client is None:
            from orchestrator.notion_client import NotionClient

            client = NotionClient(
                timeout=_DEFAULT_TIMEOUT_SECONDS, max_retries=_DEFAULT_MAX_RETRIES
            )
        from ops.notion_summary import load_notes_database_id

        database_id = load_notes_database_id()
        if clock() >= deadline:
            return None
        rows = client.query_db(
            database_id,
            sorts=[{"property": "Run Date", "direction": "descending"}],
            page_size=max_rows,
        )[:max_rows]
    except Exception:  # noqa: BLE001 — Notion down = local fallback, never a raise
        return None

    from ops.notion_summary import parse_run_summary

    out: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for row in rows:
        info = _row_info(row)
        summary: dict[str, Any] | None = None
        if clock() < deadline:
            try:
                summary = parse_run_summary(client.list_block_children(str(row.get("id"))))
            except Exception:  # noqa: BLE001 — this row degrades, the rest still count
                summary = None
        out.append((info, summary))
    return out


def _row_info(row: dict[str, Any]) -> dict[str, Any]:
    properties = row.get("properties") or {}
    return {
        "directive": _rich_text_value(properties.get("Directive")),
        "run_date": _date_value(properties.get("Run Date")),
    }


def _rich_text_value(prop: dict[str, Any] | None) -> str | None:
    parts: list[str] = []
    for rich in (prop or {}).get("rich_text") or []:
        content = (rich.get("text") or {}).get("content")
        if content is None:
            content = rich.get("plain_text")
        parts.append(content or "")
    text = "".join(parts)
    return text or None


def _date_value(prop: dict[str, Any] | None) -> str | None:
    start = ((prop or {}).get("date") or {}).get("start")
    return str(start) if start else None


# -- local state side ----------------------------------------------------------


def _local_entries(store: StateStore | None, max_rows: int) -> list[dict[str, Any]]:
    """Newest-first directive + status per local run; empty on any failure."""
    if store is None:
        return []
    try:
        runs = store.list_runs()  # ordered by created_at ascending
    except Exception:  # noqa: BLE001 — unreadable runs degrade to no history
        return []
    entries: list[dict[str, Any]] = []
    for run in reversed(runs[-max_rows:]):
        directive: str | None = None
        try:
            record = store.get_directive(run.thread_ts)
            directive = record.objective if record else None
        except Exception:  # noqa: BLE001 — a lost directive degrades one entry
            directive = None
        entries.append(
            {"date": run.created_at[:10], "status": run.status, "directive": directive}
        )
    return entries


def _incumbent_section(store: StateStore | None) -> str:
    if store is None:
        return NO_INCUMBENT
    try:
        promoted = store.get_promoted_strategy()
    except Exception:  # noqa: BLE001 — unreadable pointer reads as no incumbent
        return NO_INCUMBENT
    if promoted is None:
        return NO_INCUMBENT
    from ops.gpu_trace import (
        workspace_factors,
        workspace_metrics,
        workspace_model,
        workspace_window,
    )

    lines = [
        "Incumbent (currently promoted strategy):",
        f"  workspace: {Path(promoted.workspace_path).name}",
        f"  promoted_at: {promoted.promoted_at}",
    ]
    universe = promoted.config.get("universe") if isinstance(promoted.config, dict) else None
    if universe:
        lines.append(f"  universe: {universe}")
    lines.append(f"  model: {workspace_model(promoted.workspace_path) or 'unknown'}")
    factors = workspace_factors(promoted.workspace_path)
    if factors:
        shown = ", ".join(factors[:_FACTOR_NAMES_SHOWN])
        more = len(factors) - _FACTOR_NAMES_SHOWN
        suffix = f" (+{more} more)" if more > 0 else ""
        lines.append(f"  factors ({len(factors)}): {shown}{suffix}")
    metrics = _metrics_line(workspace_metrics(promoted.workspace_path) or {})
    if metrics:
        lines.append(f"  metrics: {metrics}")
    window = workspace_window(promoted.workspace_path)
    if window:
        lines.append(f"  test window: {window[0]} → {window[1]}")
    return "\n".join(lines)


# -- rendering -----------------------------------------------------------------


def _metrics_line(metrics: dict[str, Any]) -> str:
    parts = [
        f"{label} {metrics[label]:.4f}"
        for label in _METRIC_LABELS
        if isinstance(metrics.get(label), (int, float))
    ]
    return " · ".join(parts)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _entry_header(
    run_date: str | None,
    status: str | None,
    directive: str | None,
    universe: str | None = None,
) -> str:
    bits = [run_date or "date unknown", status or "status unknown"]
    if universe:
        bits.append(universe)
    body = _clip(directive, _DIRECTIVE_CHARS) if directive else "(open-ended)"
    return f"[{' | '.join(bits)}] directive: {body}"


def _summary_entry(summary: dict[str, Any]) -> str:
    status = summary.get("status")
    universe = summary.get("universe")
    universe_name = universe.get("name") if isinstance(universe, dict) else None
    lines = [
        _entry_header(
            summary.get("run_date"),
            str(status) if status else None,
            summary.get("directive"),
            str(universe_name) if universe_name else None,
        )
    ]
    winner = summary.get("winner")
    if isinstance(winner, dict):
        loop = winner.get("loop")
        loop_text = f" (loop {loop})" if loop is not None else ""
        hypothesis = _clip(str(winner.get("hypothesis") or "n/a"), _HYPOTHESIS_CHARS)
        metrics = _metrics_line(winner.get("metrics") or {})
        suffix = f" — {metrics}" if metrics else ""
        lines.append(f"  winner{loop_text}: {hypothesis}{suffix}")
    else:
        lines.append("  winner: none")
    hypotheses = [h for h in summary.get("hypotheses") or [] if isinstance(h, dict)]
    if hypotheses:
        counts = {outcome: 0 for outcome in _OUTCOME_ORDER}
        for hypothesis in hypotheses:
            outcome = str(hypothesis.get("outcome"))
            if outcome in counts:
                counts[outcome] += 1
        lines.append(
            "  hypotheses: "
            + " / ".join(f"{counts[outcome]} {outcome}" for outcome in _OUTCOME_ORDER)
        )
        for hypothesis in hypotheses:
            outcome = hypothesis.get("outcome") or "unknown"
            text = _clip(str(hypothesis.get("hypothesis") or "n/a"), _HYPOTHESIS_CHARS)
            lines.append(f"    - {outcome}: {text}")
    return "\n".join(lines)


def _degraded_entry(run_date: str | None, status: str | None, directive: str | None) -> str:
    return _entry_header(run_date, status, directive) + " (no run summary available)"


def _assemble(incumbent: str, entries: list[str], max_chars: int) -> str:
    """Header + incumbent + entries, dropping OLDEST entries to fit max_chars."""
    omitted = 0
    while True:
        parts = [HEADER, "", incumbent, ""]
        parts.append("\n\n".join(entries))
        if omitted:
            parts.append(f"\n(+{omitted} {OMITTED_NOTE})")
        text = "\n".join(parts)
        if len(text) <= max_chars or not entries:
            return _clamp(text, max_chars)
        entries = entries[:-1]
        omitted += 1


def _clamp(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

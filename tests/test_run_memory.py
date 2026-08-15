"""Offline tests for the run-history digest builder (orchestrator/run_memory.py).

Notion is mocked at the client boundary (a stub with query_db /
list_block_children); the run_summary blocks the stub serves are produced by
the REAL writer (ops.notion_summary.build_run_summary + run_summary_blocks),
so the digest reads exactly what US-013 writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ops.notion_summary import build_run_summary, run_summary_blocks
from orchestrator.run_memory import (
    HEADER,
    MEMORY_DELIMITER,
    NO_INCUMBENT,
    NO_PRIOR_RUNS,
    OMITTED_NOTE,
    Digest,
    build_digest,
    build_digest_details,
    compose_instruction,
    split_instruction,
)
from orchestrator.state import StateStore
from tests.test_gpu_trace import write_factors, write_metrics, write_ret

MODEL_CONF = """\
market: us_liquid
task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
"""


class StubNotion:
    """Duck-typed NotionClient: query_db + list_block_children only."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        children: dict[str, list[dict[str, Any]]] | None = None,
        fail_query: bool = False,
        on_children: Any = None,
    ) -> None:
        self.rows = rows or []
        self.children = children or {}
        self.fail_query = fail_query
        self.on_children = on_children
        self.queries: list[dict[str, Any]] = []
        self.children_calls: list[str] = []

    def query_db(
        self,
        database_id: str,
        filter: Any = None,  # noqa: A002 - mirrors the real client
        sorts: Any = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        if self.fail_query:
            raise RuntimeError("notion unreachable")
        self.queries.append({"database_id": database_id, "sorts": sorts, "page_size": page_size})
        return list(self.rows)

    def list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        if self.on_children is not None:
            self.on_children()
        self.children_calls.append(block_id)
        if block_id not in self.children:
            raise RuntimeError(f"no children for {block_id}")
        return self.children[block_id]


def make_context(
    run_date: str,
    directive: str,
    status: str = "completed",
    winner_hypothesis: str = "steady momentum beats churn",
    rejected_hypothesis: str = "low-vol reversal helps",
) -> dict[str, Any]:
    """Pipeline-shaped context (gpu_pipeline.build_notion_context keys)."""
    return {
        "run_date": run_date,
        "run_status": status,
        "directive": directive,
        "universe": "us_liquid",
        "instrument_hash": "6fbafedc13ed9a52",
        "test_end": "2026-06-12",
        "confirmation_window": ["2026-06-15", "2026-08-13"],
        "loops": [
            {
                "loop": 0,
                "action": "factor",
                "hypothesis": rejected_hypothesis,
                "decision": False,
                "metrics": {"IC": 0.001},
            },
            {
                "loop": 1,
                "action": "factor",
                "hypothesis": winner_hypothesis,
                "decision": True,
                "metrics": {"IC": 0.02},
            },
        ],
        "candidate": {
            "loop": 1,
            "hypothesis": winner_hypothesis,
            "metrics": {"IC": 0.02, "ARR": 0.31, "MDD": -0.11, "IR": 1.5},
            "factors": ["mom_20"],
            "window": ["2016-01-01", "2026-06-12"],
        },
    }


def summary_children(context: dict[str, Any]) -> list[dict[str, Any]]:
    return run_summary_blocks(build_run_summary(context))


def make_row(row_id: str, directive: str | None = None, run_date: str | None = None) -> dict:
    properties: dict[str, Any] = {}
    if directive is not None:
        properties["Directive"] = {
            "rich_text": [{"type": "text", "text": {"content": directive}}]
        }
    if run_date is not None:
        properties["Run Date"] = {"date": {"start": run_date}}
    return {"id": row_id, "properties": properties}


def make_incumbent_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws" / "c9587797deadbeef"
    workspace.mkdir(parents=True)
    (workspace / "conf_sota_factors_model.yaml").write_text(MODEL_CONF)
    write_metrics(workspace, ic=0.0186, arr=0.71)
    write_ret(workspace)
    write_factors(workspace)
    return workspace


def seed_store(
    tmp_path: Path,
    runs: list[tuple[str, str, str]] | None = None,
    promoted_workspace: Path | None = None,
) -> Path:
    """state.sqlite with (thread_ts, directive, status) runs + optional incumbent."""
    db_path = tmp_path / "state.sqlite"
    store = StateStore(db_path)
    for thread_ts, directive, status in runs or []:
        store.create_directive(thread_ts, directive)
        store.create_run(thread_ts, f"/runs/{thread_ts}", status=status, backend="gpu")
    if promoted_workspace is not None:
        store.set_promoted_strategy(
            str(promoted_workspace),
            {"universe": "us_liquid", "topk": 20, "n_drop": 3},
        )
    return db_path


# ------------------------------------------------------------- normal digest


class TestNormalDigest:
    def test_composes_summaries_and_incumbent(self, tmp_path: Path) -> None:
        workspace = make_incumbent_workspace(tmp_path)
        db_path = seed_store(
            tmp_path,
            runs=[("1.1", "find momentum alpha", "completed")],
            promoted_workspace=workspace,
        )
        newest = make_context(
            "2026-08-14", "try downside-share factors", winner_hypothesis="downside share wins"
        )
        older = make_context("2026-08-10", "find momentum alpha")
        client = StubNotion(
            rows=[make_row("row-new", run_date="2026-08-14"), make_row("row-old")],
            children={"row-new": summary_children(newest), "row-old": summary_children(older)},
        )

        digest = build_digest(db_path, client)

        assert digest.startswith(HEADER)
        # Incumbent section from local state + workspace artifacts.
        assert "workspace: c9587797deadbeef" in digest
        assert "model: LGBModel" in digest
        assert "factors (2): alpha_one, beta_two" in digest
        assert "IC 0.0186" in digest
        assert "test window: 2025-01-02 → 2026-07-10" in digest
        # Both runs, newest first.
        assert digest.index("try downside-share factors") < digest.index("find momentum alpha")
        assert "[2026-08-14 | completed | us_liquid]" in digest
        assert "winner (loop 1): downside share wins" in digest
        assert "IR 1.5000" in digest
        # Rejected ideas are listed — the whole point of the digest.
        assert "rejected: low-vol reversal helps" in digest
        assert "hypotheses: 1 SOTA / 1 rejected / 0 failed" in digest

    def test_queries_newest_rows_first(self, tmp_path: Path) -> None:
        client = StubNotion(rows=[])
        build_digest(tmp_path / "absent.sqlite", client, max_rows=7)

        (query,) = client.queries
        assert query["sorts"] == [{"property": "Run Date", "direction": "descending"}]
        assert query["page_size"] == 7

    def test_deterministic_given_same_inputs(self, tmp_path: Path) -> None:
        db_path = seed_store(tmp_path, runs=[("1.1", "objective one", "completed")])
        context = make_context("2026-08-14", "objective one")

        digests = [
            build_digest(
                db_path,
                StubNotion(rows=[make_row("r1")], children={"r1": summary_children(context)}),
            )
            for _ in range(2)
        ]

        assert digests[0] == digests[1]


# --------------------------------------------------------------- degradation


class TestDegradation:
    def test_missing_json_degrades_to_directive_and_status(self, tmp_path: Path) -> None:
        db_path = seed_store(tmp_path, runs=[("1.1", "explore value factors", "completed")])
        paragraph = {"type": "paragraph", "paragraph": {"rich_text": []}}
        client = StubNotion(
            rows=[make_row("row-1", directive="explore value factors", run_date="2026-08-12")],
            children={"row-1": [paragraph]},
        )

        digest = build_digest(db_path, client)

        # Status matched from the local run row; no run details invented.
        assert "[2026-08-12 | completed] directive: explore value factors" in digest
        assert "(no run summary available)" in digest
        assert "winner" not in digest

    def test_status_matches_when_notion_directive_extends_local(self, tmp_path: Path) -> None:
        # Live shape (2026-08-15): the Notion row's Directive is the full run
        # instruction; the local objective is a shorter prefix of it.
        local_objective = "Test whether tilting the incumbent score by downside share wins"
        db_path = seed_store(tmp_path, runs=[("1.1", local_objective, "stopped")])
        client = StubNotion(
            rows=[make_row("row-1", directive=local_objective + " — full protocol: " + "y" * 300)],
            children={"row-1": []},
        )

        digest = build_digest(db_path, client)

        assert "| stopped]" in digest

    def test_children_fetch_failure_degrades_that_row_only(self, tmp_path: Path) -> None:
        good = make_context("2026-08-14", "good run directive")
        client = StubNotion(
            rows=[make_row("row-good"), make_row("row-bad", directive="bad run directive")],
            children={"row-good": summary_children(good)},  # row-bad raises
        )

        digest = build_digest(tmp_path / "absent.sqlite", client)

        assert "winner (loop 1)" in digest
        assert "bad run directive (no run summary available)" in digest

    def test_notion_down_falls_back_to_local_runs(self, tmp_path: Path) -> None:
        db_path = seed_store(
            tmp_path,
            runs=[
                ("1.1", "older local objective", "failed"),
                ("2.2", "newer local objective", "completed"),
            ],
        )
        digest = build_digest(db_path, StubNotion(fail_query=True))

        assert digest.startswith(HEADER)
        assert digest.index("newer local objective") < digest.index("older local objective")
        assert "completed] directive: newer local objective (no run summary available)" in digest
        assert "failed] directive: older local objective (no run summary available)" in digest

    def test_never_raises_on_hostile_inputs(self, tmp_path: Path) -> None:
        garbage_db = tmp_path / "state.sqlite"
        garbage_db.write_bytes(b"not a sqlite file")

        digest = build_digest(garbage_db, object())  # client lacks every method

        assert digest == NO_PRIOR_RUNS

    def test_never_creates_state_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "absent.sqlite"
        build_digest(db_path, StubNotion(rows=[]))
        assert not db_path.exists()


# -------------------------------------------------------------------- budget


class TestNotionBudget:
    def test_budget_cutoff_degrades_remaining_rows(self, tmp_path: Path) -> None:
        contexts = [
            make_context(f"2026-08-1{i}", f"directive number {i}") for i in (3, 2, 1)
        ]
        rows = [make_row(f"row-{i}", directive=f"directive number {i}") for i in (3, 2, 1)]
        children = {
            f"row-{i}": summary_children(context)
            for i, context in zip((3, 2, 1), contexts, strict=True)
        }
        clock = [0.0]

        def advance() -> None:
            clock[0] += 10.0

        client = StubNotion(rows=rows, children=children, on_children=advance)
        digest = build_digest(
            tmp_path / "absent.sqlite",
            client,
            notion_budget=15.0,
            clock=lambda: clock[0],
        )

        # Rows 3 and 2 fetched inside the budget; row 1 degraded, never fetched.
        assert client.children_calls == ["row-3", "row-2"]
        assert "winner (loop 1)" in digest
        assert "directive number 1 (no run summary available)" in digest

    def test_exhausted_budget_skips_notion_entirely(self, tmp_path: Path) -> None:
        db_path = seed_store(tmp_path, runs=[("1.1", "local objective", "completed")])
        client = StubNotion(rows=[make_row("row-1")])

        digest = build_digest(db_path, client, notion_budget=0.0)

        assert client.queries == []
        assert "local objective (no run summary available)" in digest


# ---------------------------------------------------------------- truncation


class TestTruncation:
    def test_truncates_oldest_first_at_max_chars(self, tmp_path: Path) -> None:
        count = 6
        rows = [make_row(f"row-{i}") for i in range(count)]
        children = {
            f"row-{i}": summary_children(
                make_context(f"2026-08-{10 + count - i}", f"unique directive {i}")
            )
            for i in range(count)
        }
        client = StubNotion(rows=rows, children=children)

        digest = build_digest(tmp_path / "absent.sqlite", client, max_chars=900)

        assert len(digest) <= 900
        assert "unique directive 0" in digest  # newest kept
        assert f"unique directive {count - 1}" not in digest  # oldest dropped
        assert OMITTED_NOTE in digest

    def test_default_stays_within_max_chars(self, tmp_path: Path) -> None:
        rows = [make_row(f"row-{i}") for i in range(10)]
        children = {
            f"row-{i}": summary_children(
                make_context(f"2026-07-{10 + i}", "directive " + "x" * 400)
            )
            for i in range(10)
        }
        digest = build_digest(tmp_path / "absent.sqlite", StubNotion(rows=rows, children=children))
        assert len(digest) <= 4000


# -------------------------------------------------- digest details (US-015)


class TestDigestDetails:
    def test_counts_included_entries(self, tmp_path: Path) -> None:
        rows = [make_row("row-a"), make_row("row-b")]
        children = {
            "row-a": summary_children(make_context("2026-08-14", "directive a")),
            "row-b": summary_children(make_context("2026-08-10", "directive b")),
        }
        client = StubNotion(rows=rows, children=children)

        details = build_digest_details(tmp_path / "absent.sqlite", client)

        assert details.runs == 2
        # The text is exactly what build_digest returns for the same inputs.
        assert details.text == build_digest(
            tmp_path / "absent.sqlite", StubNotion(rows=rows, children=children)
        )

    def test_truncation_reduces_count(self, tmp_path: Path) -> None:
        count = 6
        rows = [make_row(f"row-{i}") for i in range(count)]
        children = {
            f"row-{i}": summary_children(
                make_context(f"2026-08-{10 + count - i}", f"unique directive {i}")
            )
            for i in range(count)
        }
        client = StubNotion(rows=rows, children=children)

        details = build_digest_details(tmp_path / "absent.sqlite", client, max_chars=900)

        assert 0 < details.runs < count
        assert OMITTED_NOTE in details.text

    def test_fallback_counts_zero(self, tmp_path: Path) -> None:
        garbage_db = tmp_path / "state.sqlite"
        garbage_db.write_bytes(b"not a sqlite file")
        assert build_digest_details(garbage_db, object()) == Digest(NO_PRIOR_RUNS, 0)

    def test_incumbent_without_history_counts_zero(self, tmp_path: Path) -> None:
        workspace = make_incumbent_workspace(tmp_path)
        db_path = seed_store(tmp_path, promoted_workspace=workspace)

        details = build_digest_details(db_path, StubNotion(rows=[]))

        assert details.runs == 0
        assert "workspace: c9587797deadbeef" in details.text


# --------------------------------------- instruction composition (US-015)


DIRECTIVE = "Test whether 12-1 momentum beats SPY\nConstraints: long-only"
DIGEST_TEXT = f"{HEADER}\n\n{NO_INCUMBENT}\n\n[2026-08-14 | completed] directive: try things"


class TestComposeInstruction:
    def test_directive_first_then_delimiter_then_digest(self) -> None:
        composed = compose_instruction(DIRECTIVE, DIGEST_TEXT)
        assert composed == DIRECTIVE + MEMORY_DELIMITER + DIGEST_TEXT
        assert composed.startswith(DIRECTIVE)

    def test_digest_is_trimmed_to_fit_never_the_directive(self) -> None:
        digest = "x" * 500
        composed = compose_instruction(DIRECTIVE, digest, max_chars=200)
        assert len(composed) <= 200
        assert composed.startswith(DIRECTIVE + MEMORY_DELIMITER)
        assert composed.endswith("x")  # a digest prefix survived

    def test_oversized_directive_is_never_truncated(self) -> None:
        directive = "d" * 300
        composed = compose_instruction(directive, DIGEST_TEXT, max_chars=200)
        assert composed == directive  # intact even beyond max_chars; digest dropped

    def test_empty_or_placeholder_digest_composes_to_bare_directive(self) -> None:
        assert compose_instruction(DIRECTIVE, None) == DIRECTIVE
        assert compose_instruction(DIRECTIVE, "") == DIRECTIVE
        assert compose_instruction(DIRECTIVE, NO_PRIOR_RUNS) == DIRECTIVE
        assert compose_instruction(DIRECTIVE, f"  {NO_PRIOR_RUNS}  ") == DIRECTIVE

    def test_split_round_trips(self) -> None:
        composed = compose_instruction(DIRECTIVE, DIGEST_TEXT)
        assert split_instruction(composed) == (DIRECTIVE, DIGEST_TEXT)

    def test_split_of_bare_directive_has_no_digest(self) -> None:
        assert split_instruction(DIRECTIVE) == (DIRECTIVE, None)


# --------------------------------------------------------------- empty state


class TestEmptyState:
    def test_empty_everything_yields_no_prior_runs(self, tmp_path: Path) -> None:
        digest = build_digest(tmp_path / "absent.sqlite", StubNotion(rows=[]))
        assert digest == NO_PRIOR_RUNS

    def test_incumbent_without_history_still_shown(self, tmp_path: Path) -> None:
        workspace = make_incumbent_workspace(tmp_path)
        db_path = seed_store(tmp_path, promoted_workspace=workspace)

        digest = build_digest(db_path, StubNotion(rows=[]))

        assert NO_PRIOR_RUNS in digest
        assert "workspace: c9587797deadbeef" in digest
        assert NO_INCUMBENT not in digest

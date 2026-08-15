"""Notion Decision Log rows for promotions from every source (US-012).

    onecli run --agent rdq-orchestrator -- .venv/bin/python -m ops.promotion_decision \
        --payload /path/to/payload.json

The conversational promotion path (orchestrator/promotion.py) writes its
Decision Log row in-process — the orchestrator already runs with the Notion
bearer injected. The two ops-side paths (``ops.promote_fetched`` CLI and the
GPU pipeline's auto-promotion) do NOT: the bearer only injects under
``onecli run --agent rdq-orchestrator``. So they build a small payload and
call :func:`record_via_onecli`, which re-enters this module as a subprocess
under that identity (the same pattern as gpu_pipeline's notion_writeup).

The actual write still goes through ``NotionRecorder.record_decision`` — the
Decision Log keeps exactly one writing component (one-writer-per-DB,
docs/reference/notion-schema.md); this module only assembles the payload and
provides the identity hop.

Payload keys (built by :func:`build_payload` from a PromotionResult):
source, workspace, market, topk, n_drop, metrics, replaced_workspace,
promoted_at, thread_ts, gate_verdict (GateVerdict.to_dict() or None),
gate_error, forced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ops.promotion_gate import gate_summary_line

if TYPE_CHECKING:
    from ops.promote_fetched import PromotionResult
    from orchestrator.notion_client import NotionClient

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_AGENT = "rdq-orchestrator"


class PromotionDecisionError(RuntimeError):
    """The Decision Log row could not be written."""


def build_payload(
    result: PromotionResult,
    *,
    source: str,
    gate_verdict: Mapping[str, Any] | None = None,
    gate_error: str | None = None,
    forced: bool = False,
    thread_ts: str | None = None,
) -> dict[str, Any]:
    """Everything the Decision Log row needs, JSON-safe (rides a subprocess)."""
    return {
        "source": source,
        "workspace": result.workspace,
        "market": result.market,
        "topk": result.topk,
        "n_drop": result.n_drop,
        "metrics": dict(result.metrics),
        "replaced_workspace": result.replaced_workspace,
        "promoted_at": result.promoted_at,
        "thread_ts": thread_ts,
        "gate_verdict": dict(gate_verdict) if gate_verdict is not None else None,
        "gate_error": gate_error,
        "forced": bool(forced),
    }


def decision_title(payload: Mapping[str, Any]) -> str:
    tag = Path(str(payload["workspace"])).name[:8] or "unknown"
    return f"Promote '{tag}' to paper trading"


def decision_details(payload: Mapping[str, Any]) -> str:
    """The row's Details text: workspace, key metrics, replacement, gate line."""
    from orchestrator.summary import METRIC_SPECS

    forced = bool(payload.get("forced"))
    lines = [
        f"Source: {payload['source']}" + (" (forced)" if forced else ""),
        f"Workspace: {payload['workspace']}",
        f"Universe: {payload['market']}",
        f"TopkDropoutStrategy: topk={payload['topk']}, n_drop={payload['n_drop']}",
    ]
    metrics: Mapping[str, Any] = payload.get("metrics") or {}
    for label, _keys, _style in METRIC_SPECS:
        value = metrics.get(label)
        if value is not None:
            lines.append(f"{label}: {value:.4f}")
    replaced = payload.get("replaced_workspace")
    lines.append(f"Replaced: {replaced}" if replaced else "Replaced: none (first promotion)")
    lines.append(
        gate_summary_line(
            payload.get("gate_verdict"), forced=forced, error=payload.get("gate_error")
        )
    )
    return "\n".join(lines)


def write_decision(
    payload: Mapping[str, Any],
    *,
    db_path: Path | None = None,
    config_path: Path | None = None,
    notion: NotionClient | None = None,
) -> str | None:
    """Write the row through NotionRecorder (the Decision Log's one writer).

    Needs the Notion bearer in this process — run under
    ``onecli run --agent rdq-orchestrator`` (or inject ``notion`` in tests).
    Returns the page id, or None when the recorder swallowed a write failure.
    """
    from orchestrator.notion_recorder import (
        DEFAULT_CONFIG_PATH,
        NotionRecorder,
        load_notion_databases,
    )
    from orchestrator.state import DEFAULT_DB_PATH, StateStore

    databases = load_notion_databases(config_path or DEFAULT_CONFIG_PATH)
    db = Path(db_path if db_path is not None else DEFAULT_DB_PATH).expanduser()
    if not db.is_file():
        # StateStore(path) would CREATE the DB — never do that from a
        # bookkeeping path (same guard as sweep.py / promote_fetched).
        raise PromotionDecisionError(
            f"{db} does not exist — run from the deployed checkout (~/rd-agent-q)"
        )
    if notion is None:
        from orchestrator.notion_client import NotionClient

        notion = NotionClient()
    recorder = NotionRecorder(notion, databases, StateStore(db))
    return recorder.record_decision(
        title=decision_title(payload),
        decision_type="promotion",
        details=decision_details(payload),
        thread_ts=payload.get("thread_ts"),
    )


def record_via_onecli(payload: Mapping[str, Any], *, timeout: float = 120.0) -> bool:
    """Write the row from a process WITHOUT the Notion bearer (CLI, pipeline).

    Re-runs this module under ``onecli run --agent rdq-orchestrator`` so the
    proxy injects the bearer — the ops-side analog of notion_writeup. Never
    raises: a failed write reports to stderr and returns False (the promotion
    already happened; the caller prints the manual fallback).
    """
    fd, raw_path = tempfile.mkstemp(prefix="rdq-promotion-decision-", suffix=".json")
    payload_path = Path(raw_path)
    try:
        with open(fd, "w") as handle:
            json.dump(dict(payload), handle)
        result = subprocess.run(
            [
                "onecli",
                "run",
                "--agent",
                ORCHESTRATOR_AGENT,
                "--",
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "-m",
                "ops.promotion_decision",
                "--payload",
                str(payload_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"decision log write failed to launch ({exc})", file=sys.stderr)
        return False
    finally:
        payload_path.unlink(missing_ok=True)
    if result.returncode == 0:
        return True
    tail = (result.stderr or result.stdout).strip().splitlines()[-1:]
    print(f"decision log write failed: {' '.join(tail)}", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="payload JSON path")
    parser.add_argument("--db", type=Path, default=None, help="orchestrator state.sqlite")
    args = parser.parse_args(argv)
    payload = json.loads(args.payload.read_text())
    try:
        page_id = write_decision(payload, db_path=args.db)
    except Exception as exc:  # noqa: BLE001 — one actionable line for the wrapping caller
        print(f"decision log write failed: {exc}", file=sys.stderr)
        return 1
    if page_id is None:
        print("decision log write failed (see log output above)", file=sys.stderr)
        return 1
    print(page_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

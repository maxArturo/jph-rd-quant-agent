"""Roll back the promoted strategy to a prior promotion_history entry.

    .venv/bin/python -m ops.rollback_promotion [--to <workspace>] [--yes]

Why this exists (US-006): promotions are append-only audited (US-005), so
undoing a bad promotion is itself a promotion. This command re-promotes the
previous history entry (or a named one via --to) using the config recorded
when that workspace was last promoted, re-runs its pred-refresh snapshot,
and appends a NEW promotion_history row (source 'cli') — history rows are
never updated or deleted.

Safety rails (same spirit as ops.promote_fetched):
- dry-run by default; --yes actually writes;
- refuses when state.sqlite doesn't exist (never creates the DB);
- refuses when the target workspace directory no longer exists (a swept
  workspace cannot be re-promoted — its pred.pkl and snapshot are gone);
- the snapshot re-run happens BEFORE the pointer flip, so a snapshot failure
  leaves the current promotion untouched;
- re-snapshotting regenerates conf_pred_refresh.yaml from the run's own logs,
  which overwrites an operator-pinned market (e.g. a frozen *_promoted_*
  universe) — the overwrite is announced, and --keep-snapshot preserves the
  workspace's existing snapshot files instead.

Verify afterwards:  .venv/bin/python -m execution.pred_refresh --no-slack
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from execution.pred_refresh import PredRefreshError, snapshot_pred_refresh
from orchestrator.state import (
    DEFAULT_DB_PATH,
    PromotionHistoryEntry,
    StateStore,
)

PRED_REFRESH_CONF = "conf_pred_refresh.yaml"


class RollbackError(RuntimeError):
    """Validation failure — nothing was written."""


def _resolved(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def select_target(
    history: list[PromotionHistoryEntry],
    current_workspace: str | None,
    to: Path | None,
) -> PromotionHistoryEntry:
    """The history entry to re-promote (newest matching row).

    Default: the most recent entry whose workspace differs from the current
    pointer — 'the previous promotion'. With ``to``, the most recent entry
    for that workspace (only past promotions can be rolled back to; history
    carries the config the rebalancer needs).
    """
    if not history:
        raise RollbackError("promotion history is empty — nothing to roll back to")
    if to is not None:
        wanted = to.expanduser().resolve()
        for entry in history:
            if _resolved(entry.workspace_path) == wanted:
                if current_workspace is not None and _resolved(current_workspace) == wanted:
                    raise RollbackError(
                        f"{entry.workspace_path} is already the promoted strategy"
                    )
                return entry
        raise RollbackError(
            f"{to} never appears in promotion history — only past promotions "
            "can be rolled back to (their config is recorded there)"
        )
    for entry in history:
        if current_workspace is None or _resolved(entry.workspace_path) != _resolved(
            current_workspace
        ):
            return entry
    raise RollbackError(
        "nothing to roll back to: every history entry points at the currently "
        "promoted workspace"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--to",
        type=Path,
        default=None,
        help="workspace of the history entry to restore (default: the previous promotion)",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="orchestrator state.sqlite"
    )
    parser.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="keep the workspace's existing pred-refresh snapshot (preserves an "
        "operator-pinned market) instead of regenerating it",
    )
    parser.add_argument("--yes", action="store_true", help="actually write (default is dry-run)")
    parser.add_argument("--no-slack", action="store_true")
    args = parser.parse_args(argv)

    if not args.db.expanduser().is_file():
        print(
            f"REFUSED: {args.db} does not exist — run this from the deployed checkout "
            "(~/rd-agent-q), never let it create a fresh DB",
            file=sys.stderr,
        )
        return 1
    store = StateStore(args.db.expanduser())
    current = store.get_promoted_strategy()
    try:
        entry = select_target(
            store.list_promotion_history(),
            None if current is None else current.workspace_path,
            args.to,
        )
    except RollbackError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    workspace = _resolved(entry.workspace_path)
    if not workspace.is_dir():
        print(
            f"REFUSED: workspace {entry.workspace_path} no longer exists on disk "
            "(likely swept) — its predictions and snapshot are gone, so it cannot "
            "be re-promoted; pick another history entry with --to",
            file=sys.stderr,
        )
        return 1

    print(f"rolling back to history entry #{entry.id} ({entry.source}, {entry.promoted_at})")
    print(f"  restore: {entry.workspace_path}")
    print(f"  config: {json.dumps(entry.config)[:300]}")
    if current is not None:
        print(f"  replaces: {current.workspace_path}")
    if not args.keep_snapshot and (workspace / PRED_REFRESH_CONF).is_file():
        print(
            "  NOTE: the restored workspace already has a pred-refresh snapshot; "
            "re-snapshotting regenerates it from the run's own logs, so any "
            "operator-pinned market (e.g. a frozen *_promoted_* universe) is lost "
            "— use --keep-snapshot to preserve it."
        )

    if not args.yes:
        print("dry-run (no --yes): nothing written")
        return 0

    if args.keep_snapshot:
        print("snapshot kept (existing conf_pred_refresh.yaml untouched)")
    else:
        try:
            conf_path, env_path, params_path = snapshot_pred_refresh(workspace)
        except PredRefreshError as exc:
            print(
                f"REFUSED: pred-refresh snapshot failed ({exc}) — the current "
                "promotion is untouched",
                file=sys.stderr,
            )
            return 1
        print(f"snapshot written: {conf_path.name}, {env_path.name}, {params_path.name}")

    promoted = store.set_promoted_strategy(
        entry.workspace_path,
        entry.config,
        source="cli",
        gate_verdict={
            "action": "rollback",
            "restored_history_id": entry.id,
            "rolled_back_from": None if current is None else current.workspace_path,
        },
    )
    print(f"promoted_strategy row set at {promoted.promoted_at}")

    notice = (
        f":rewind: Rolled back the promoted strategy to `{workspace.name[:8]}` "
        f"(history entry #{entry.id}, originally promoted {entry.promoted_at}"
        f"{', replacing ' + Path(current.workspace_path).name[:8] if current else ''}). "
        "Verify with `python -m execution.pred_refresh --no-slack`."
    )
    if args.no_slack:
        print(notice, file=sys.stderr)
    else:
        try:
            from execution.rebalance import slack_notifier

            slack_notifier()(notice)
        except Exception as exc:  # noqa: BLE001 — the rollback already happened
            print(f"slack notice failed ({exc}): {notice}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Promoted-strategy loader: the rebalancer's entrypoint check (US-033).

The nightly rebalancer (US-034) must never trade without a deliberate
operator promotion. ``load_promoted_strategy()`` reads THE single
``promoted_strategy`` row from the orchestrator's SQLite state and raises
``NoPromotedStrategyError`` when the state database, the row, or the pinned
workspace is missing — the rebalancer treats that as abort-without-trading.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.state import DEFAULT_DB_PATH, PromotedStrategy, StateStore

PROMOTE_HINT = (
    "promote a completed research run from Slack (Promote button on the run"
    " summary) before rebalancing"
)


class NoPromotedStrategyError(RuntimeError):
    """No tradable promoted strategy exists; the rebalancer must not run."""


def load_promoted_strategy(db_path: Path = DEFAULT_DB_PATH) -> PromotedStrategy:
    """Return the promoted strategy, or refuse when none can be traded.

    Refusals: state DB absent (never create the orchestrator's database from
    the execution side), no promoted_strategy row, or the pinned workspace
    directory gone from disk (a retention sweep or manual delete — the signal
    extraction would have nothing to read).
    """
    db_path = Path(db_path).expanduser()
    if not db_path.is_file():
        raise NoPromotedStrategyError(
            f"orchestrator state database not found at {db_path} — {PROMOTE_HINT}"
        )
    promoted = StateStore(db_path).get_promoted_strategy()
    if promoted is None:
        raise NoPromotedStrategyError(f"no promoted strategy exists — {PROMOTE_HINT}")
    workspace = Path(promoted.workspace_path).expanduser()
    if not workspace.is_dir():
        raise NoPromotedStrategyError(
            f"promoted workspace is missing on disk: {workspace} — re-promote a"
            " run whose workspace still exists"
        )
    return promoted

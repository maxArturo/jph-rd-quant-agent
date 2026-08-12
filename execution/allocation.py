"""Live equity allocation config for the live rebalancer (US-002).

The live account trades only a capped slice of its equity: the live
rebalance pipeline scales the equity it hands to ``diff.compute_orders`` by
``live_equity_allocation_pct / 100`` (buying-power capping still uses the
account's REAL buying power — scaling happens in rebalance.py, never here).

The percentage lives in ``execution/allocation.live.json`` with the same
strictness as ``limits.live.json``/``breaker.live.json``: the one required
key must be present, unknown keys are refused, and the value must satisfy
0 < pct <= 100. There is no paper counterpart — the paper rebalancer trades
full equity and must stay byte-for-byte unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ALLOCATION_PATH = Path(__file__).resolve().parent / "allocation.live.json"

_CONFIG_KEYS = ("live_equity_allocation_pct",)


class AllocationConfigError(RuntimeError):
    """execution/allocation.live.json is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class LiveAllocation:
    live_equity_allocation_pct: float


def load_live_allocation(path: Path | str = ALLOCATION_PATH) -> LiveAllocation:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AllocationConfigError(f"allocation config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AllocationConfigError(f"allocation config {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise AllocationConfigError(f"allocation config {path} must hold a JSON object")
    unknown = sorted(set(raw) - set(_CONFIG_KEYS))
    if unknown:
        raise AllocationConfigError(
            f"allocation config {path} has unknown keys: {', '.join(unknown)}"
        )
    missing = sorted(set(_CONFIG_KEYS) - set(raw))
    if missing:
        raise AllocationConfigError(
            f"allocation config {path} is missing keys: {', '.join(missing)}"
        )
    value = raw["live_equity_allocation_pct"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AllocationConfigError(
            f"allocation config {path}: live_equity_allocation_pct must be a number, "
            f"got {value!r}"
        )
    pct = float(value)
    if not 0 < pct <= 100:
        raise AllocationConfigError(
            f"allocation config {path}: live_equity_allocation_pct must satisfy "
            f"0 < pct <= 100, got {value!r}"
        )
    return LiveAllocation(live_equity_allocation_pct=pct)

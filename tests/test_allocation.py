"""Tests for execution/allocation.py (US-002 live equity allocation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.allocation import (
    ALLOCATION_PATH,
    AllocationConfigError,
    LiveAllocation,
    load_live_allocation,
)


def write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "allocation.live.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_committed_config_loads() -> None:
    allocation = load_live_allocation(ALLOCATION_PATH)
    assert ALLOCATION_PATH.name == "allocation.live.json"
    assert allocation == LiveAllocation(live_equity_allocation_pct=10.0)


def test_value_coerced_to_float(tmp_path: Path) -> None:
    path = write(tmp_path, {"live_equity_allocation_pct": 25})
    allocation = load_live_allocation(path)
    assert isinstance(allocation.live_equity_allocation_pct, float)
    assert allocation.live_equity_allocation_pct == 25.0


def test_full_allocation_at_boundary_allowed(tmp_path: Path) -> None:
    path = write(tmp_path, {"live_equity_allocation_pct": 100})
    assert load_live_allocation(path).live_equity_allocation_pct == 100.0


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AllocationConfigError, match="not found"):
        load_live_allocation(tmp_path / "absent.json")


def test_invalid_json(tmp_path: Path) -> None:
    path = write(tmp_path, "{nope")
    with pytest.raises(AllocationConfigError, match="not valid JSON"):
        load_live_allocation(path)


def test_non_object(tmp_path: Path) -> None:
    path = write(tmp_path, [10])
    with pytest.raises(AllocationConfigError, match="JSON object"):
        load_live_allocation(path)


def test_unknown_key_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {"live_equity_allocation_pct": 10, "max_bananas": 3},
    )
    with pytest.raises(AllocationConfigError, match="unknown keys: max_bananas"):
        load_live_allocation(path)


def test_missing_key_refused(tmp_path: Path) -> None:
    path = write(tmp_path, {})
    with pytest.raises(
        AllocationConfigError, match="missing keys: live_equity_allocation_pct"
    ):
        load_live_allocation(path)


@pytest.mark.parametrize("bad", ["10", None, True, False])
def test_non_numeric_refused(tmp_path: Path, bad: object) -> None:
    path = write(tmp_path, {"live_equity_allocation_pct": bad})
    with pytest.raises(AllocationConfigError, match="must be a number"):
        load_live_allocation(path)


@pytest.mark.parametrize("bad", [0, -5, 100.001, 250])
def test_out_of_range_refused(tmp_path: Path, bad: float) -> None:
    path = write(tmp_path, {"live_equity_allocation_pct": bad})
    with pytest.raises(AllocationConfigError, match="0 < pct <= 100"):
        load_live_allocation(path)

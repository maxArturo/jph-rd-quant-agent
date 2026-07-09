"""Unit tests for execution/order_gate.py (US-030).

Each limit gets a boundary pair (pass exactly at the limit, fail just over)
plus an assertion that the rejection message names the violated limit key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.alpaca_client import Account, Position
from execution.order_gate import (
    LIMITS_PATH,
    GateResult,
    Limits,
    LimitsConfigError,
    OrderGateError,
    ProposedOrder,
    evaluate_orders,
    load_limits,
)


def make_account(equity: float = 100_000.0) -> Account:
    return Account(
        id="acct-1",
        status="ACTIVE",
        currency="USD",
        equity=equity,
        cash=equity,
        buying_power=equity * 2,
    )


def make_position(symbol: str, qty: float, price: float) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        side="long" if qty >= 0 else "short",
        avg_entry_price=price,
        current_price=price,
        market_value=qty * price,
    )


def make_limits(**overrides: float | int) -> Limits:
    base: dict[str, float | int] = {
        "max_order_notional_usd": 10_000.0,
        "max_position_pct_equity": 10.0,
        "max_day_orders": 100,
        "max_total_positions": 5,
    }
    base.update(overrides)
    return Limits(
        max_order_notional_usd=float(base["max_order_notional_usd"]),
        max_position_pct_equity=float(base["max_position_pct_equity"]),
        max_day_orders=int(base["max_day_orders"]),
        max_total_positions=int(base["max_total_positions"]),
    )


def gate(
    orders: list[ProposedOrder],
    account: Account | None = None,
    positions: list[Position] | None = None,
    day_orders_placed: int = 0,
    limits: Limits | None = None,
) -> GateResult:
    return evaluate_orders(
        orders,
        account if account is not None else make_account(),
        positions if positions is not None else [],
        day_orders_placed,
        limits if limits is not None else make_limits(),
    )


def write_limits(tmp_path: Path, **overrides: object) -> Path:
    raw: dict[str, object] = {
        "max_order_notional_usd": 10000,
        "max_position_pct_equity": 10,
        "max_day_orders": 120,
        "max_total_positions": 60,
    }
    raw.update(overrides)
    raw = {k: v for k, v in raw.items() if v is not None}
    path = tmp_path / "limits.paper.json"
    path.write_text(json.dumps(raw))
    return path


# ---------------------------------------------------------------- limits file


def test_committed_limits_file_loads() -> None:
    limits = load_limits()
    assert LIMITS_PATH.name == "limits.paper.json"
    assert limits.max_order_notional_usd > 0
    assert limits.max_position_pct_equity > 0
    assert limits.max_day_orders > 0
    assert limits.max_total_positions > 0


def test_load_limits_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LimitsConfigError, match="not found"):
        load_limits(tmp_path / "nope.json")


def test_load_limits_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "limits.paper.json"
    path.write_text("{not json")
    with pytest.raises(LimitsConfigError, match="not valid JSON"):
        load_limits(path)


def test_load_limits_missing_key(tmp_path: Path) -> None:
    path = write_limits(tmp_path, max_day_orders=None)
    with pytest.raises(LimitsConfigError, match="missing keys: max_day_orders"):
        load_limits(path)


def test_load_limits_unknown_key(tmp_path: Path) -> None:
    path = write_limits(tmp_path, max_live_orders=5)
    with pytest.raises(LimitsConfigError, match="unknown keys: max_live_orders"):
        load_limits(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_order_notional_usd", 0),
        ("max_order_notional_usd", -1),
        ("max_position_pct_equity", 0),
        ("max_day_orders", 0),
        ("max_day_orders", 2.5),
        ("max_day_orders", True),
        ("max_total_positions", -3),
    ],
)
def test_load_limits_rejects_bad_values(tmp_path: Path, key: str, value: object) -> None:
    path = write_limits(tmp_path, **{key: value})
    with pytest.raises(LimitsConfigError, match=key):
        load_limits(path)


# ------------------------------------------------------------- ProposedOrder


def test_proposed_order_validation() -> None:
    with pytest.raises(ValueError, match="side"):
        ProposedOrder(symbol="AAPL", side="hold", qty=1, limit_price=10.0)
    with pytest.raises(ValueError, match="qty"):
        ProposedOrder(symbol="AAPL", side="buy", qty=0, limit_price=10.0)
    with pytest.raises(ValueError, match="limit_price"):
        ProposedOrder(symbol="AAPL", side="buy", qty=1, limit_price=0.0)


def test_proposed_order_notional() -> None:
    order = ProposedOrder(symbol="AAPL", side="buy", qty=100, limit_price=100.5)
    assert order.notional == pytest.approx(10_050.0)


# ----------------------------------------------------- max_order_notional_usd


def test_order_notional_passes_at_limit() -> None:
    result = gate([ProposedOrder(symbol="AAPL", side="buy", qty=100, limit_price=100.0)])
    assert result.ok
    assert len(result.approved) == 1


def test_order_notional_fails_just_over() -> None:
    result = gate([ProposedOrder(symbol="AAPL", side="buy", qty=100, limit_price=100.01)])
    assert not result.ok
    [rejection] = result.rejections
    assert rejection.limit_name == "max_order_notional_usd"
    assert "max_order_notional_usd" in rejection.message
    assert "$10,001.00" in rejection.message
    assert "$10,000.00" in rejection.message


def test_order_notional_applies_to_sells_too() -> None:
    positions = [make_position("AAPL", 200, 100.0)]
    result = gate(
        [ProposedOrder(symbol="AAPL", side="sell", qty=150, limit_price=100.0)],
        positions=positions,
    )
    [rejection] = result.rejections
    assert rejection.limit_name == "max_order_notional_usd"


# --------------------------------------------------- max_position_pct_equity

WIDE = {"max_order_notional_usd": 1_000_000.0}  # keep notional out of the way


def test_position_pct_passes_at_cap() -> None:
    # equity 100k, 10% cap => $10k position allowed exactly
    result = gate(
        [ProposedOrder(symbol="AAPL", side="buy", qty=100, limit_price=100.0)],
        limits=make_limits(**WIDE),
    )
    assert result.ok


def test_position_pct_fails_just_over() -> None:
    result = gate(
        [ProposedOrder(symbol="AAPL", side="buy", qty=101, limit_price=100.0)],
        limits=make_limits(**WIDE),
    )
    [rejection] = result.rejections
    assert rejection.limit_name == "max_position_pct_equity"
    assert "max_position_pct_equity" in rejection.message
    assert "10%" in rejection.message
    assert "$10,100.00" in rejection.message
    assert "$10,000.00" in rejection.message


def test_position_pct_counts_existing_position() -> None:
    positions = [make_position("AAPL", 50, 100.0)]
    at_cap = gate(
        [ProposedOrder(symbol="AAPL", side="buy", qty=50, limit_price=100.0)],
        positions=positions,
        limits=make_limits(**WIDE),
    )
    assert at_cap.ok
    over = gate(
        [ProposedOrder(symbol="AAPL", side="buy", qty=51, limit_price=100.0)],
        positions=positions,
        limits=make_limits(**WIDE),
    )
    assert [r.limit_name for r in over.rejections] == ["max_position_pct_equity"]


def test_position_pct_allows_trimming_oversized_position() -> None:
    # A position already over the cap must remain sellable.
    positions = [make_position("AAPL", 200, 100.0)]  # $20k > $10k cap
    result = gate(
        [ProposedOrder(symbol="AAPL", side="sell", qty=50, limit_price=100.0)],
        positions=positions,
        limits=make_limits(**WIDE),
    )
    assert result.ok


def test_position_pct_checks_short_side_absolute_value() -> None:
    # Selling past flat into a short bigger than the cap increases |exposure|.
    positions = [make_position("AAPL", 10, 100.0)]
    result = gate(
        [ProposedOrder(symbol="AAPL", side="sell", qty=250, limit_price=100.0)],
        positions=positions,
        limits=make_limits(**WIDE),
    )
    assert [r.limit_name for r in result.rejections] == ["max_position_pct_equity"]


def test_position_pct_cumulative_within_batch() -> None:
    orders = [
        ProposedOrder(symbol="AAPL", side="buy", qty=60, limit_price=100.0),
        ProposedOrder(symbol="AAPL", side="buy", qty=60, limit_price=100.0),
    ]
    result = gate(orders, limits=make_limits(**WIDE))
    assert len(result.approved) == 1
    assert [r.limit_name for r in result.rejections] == ["max_position_pct_equity"]


# ------------------------------------------------------------ max_day_orders


def small(symbol: str = "AAPL") -> ProposedOrder:
    return ProposedOrder(symbol=symbol, side="buy", qty=1, limit_price=10.0)


def test_day_orders_passes_at_limit() -> None:
    result = gate([small()], day_orders_placed=99)  # order #100 of 100
    assert result.ok


def test_day_orders_fails_just_over() -> None:
    result = gate([small()], day_orders_placed=100)
    [rejection] = result.rejections
    assert rejection.limit_name == "max_day_orders"
    assert "max_day_orders" in rejection.message
    assert "day order 101" in rejection.message
    assert "100 already placed today" in rejection.message


def test_day_orders_cumulative_within_batch() -> None:
    result = gate([small("AAPL"), small("MSFT"), small("NVDA")], day_orders_placed=98)
    assert len(result.approved) == 2
    [rejection] = result.rejections
    assert rejection.limit_name == "max_day_orders"
    assert rejection.order.symbol == "NVDA"


def test_rejected_orders_do_not_consume_day_budget() -> None:
    # First order dies on notional; the second still fits as day order #100.
    orders = [
        ProposedOrder(symbol="AAPL", side="buy", qty=2000, limit_price=100.0),
        small("MSFT"),
    ]
    result = gate(orders, day_orders_placed=99)
    assert [o.symbol for o in result.approved] == ["MSFT"]
    assert [r.limit_name for r in result.rejections] == ["max_order_notional_usd"]


def test_negative_day_orders_placed_rejected() -> None:
    with pytest.raises(ValueError, match="day_orders_placed"):
        gate([small()], day_orders_placed=-1)


# ------------------------------------------------------- max_total_positions


def four_positions() -> list[Position]:
    return [make_position(s, 10, 50.0) for s in ("AAPL", "MSFT", "NVDA", "AMZN")]


def test_total_positions_passes_at_limit() -> None:
    result = gate([small("GOOG")], positions=four_positions())  # 5th of 5
    assert result.ok


def test_total_positions_fails_just_over() -> None:
    positions = [*four_positions(), make_position("GOOG", 10, 50.0)]
    result = gate([small("TSLA")], positions=positions)
    [rejection] = result.rejections
    assert rejection.limit_name == "max_total_positions"
    assert "max_total_positions" in rejection.message
    assert "position 6" in rejection.message
    assert "5 cap" in rejection.message


def test_adding_to_held_position_never_violates_count() -> None:
    positions = [*four_positions(), make_position("GOOG", 10, 50.0)]  # at cap
    result = gate([small("GOOG")], positions=positions)
    assert result.ok


def test_full_exit_frees_a_slot_within_batch() -> None:
    positions = [*four_positions(), make_position("GOOG", 10, 50.0)]  # at cap
    orders = [
        ProposedOrder(symbol="GOOG", side="sell", qty=10, limit_price=50.0),
        small("TSLA"),
    ]
    result = gate(orders, positions=positions)
    assert result.ok
    assert [o.symbol for o in result.approved] == ["GOOG", "TSLA"]


def test_partial_exit_does_not_free_a_slot() -> None:
    positions = [*four_positions(), make_position("GOOG", 10, 50.0)]  # at cap
    orders = [
        ProposedOrder(symbol="GOOG", side="sell", qty=5, limit_price=50.0),
        small("TSLA"),
    ]
    result = gate(orders, positions=positions)
    assert [o.symbol for o in result.approved] == ["GOOG"]
    assert [r.limit_name for r in result.rejections] == ["max_total_positions"]


# ------------------------------------------------------------------- results


def test_empty_batch_is_ok() -> None:
    result = gate([])
    assert result.ok
    assert result.approved == []


def test_first_violation_wins_in_documented_order() -> None:
    # Violates both the notional cap and the position pct cap; the notional
    # check runs first per the module docstring.
    order = ProposedOrder(symbol="AAPL", side="buy", qty=500, limit_price=100.0)
    result = gate([order])
    assert [r.limit_name for r in result.rejections] == ["max_order_notional_usd"]


def test_raise_for_rejections_lists_every_message() -> None:
    orders = [
        ProposedOrder(symbol="AAPL", side="buy", qty=2000, limit_price=100.0),
        small("MSFT"),
        ProposedOrder(symbol="NVDA", side="buy", qty=1500, limit_price=100.0),
    ]
    result = gate(orders)
    with pytest.raises(OrderGateError) as excinfo:
        result.raise_for_rejections()
    message = str(excinfo.value)
    assert "rejected 2 of 3" in message
    assert message.count("max_order_notional_usd") == 2
    assert "AAPL" in message and "NVDA" in message


def test_raise_for_rejections_noop_when_clean() -> None:
    gate([small()]).raise_for_rejections()

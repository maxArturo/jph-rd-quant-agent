"""Deterministic pre-trade limit gate for the paper rebalancer.

Every proposed order is checked against execution/limits.paper.json BEFORE
anything is submitted, using a fresh account/positions snapshot the caller
passes in — this module does no HTTP and holds no state.

Limits (all four keys required in the JSON file):

- max_order_notional_usd: per-order qty * limit_price cap.
- max_position_pct_equity: cap on the projected |position value| as a
  percent of snapshot equity (10 means 10%). Position value is marked at the
  order's limit price (marketable limits track the market). Only orders that
  INCREASE the absolute position size are checked — trimming an already
  oversized position must stay legal.
- max_day_orders: orders already placed today (caller-supplied count) plus
  orders approved from this batch.
- max_total_positions: distinct nonzero positions after the order applies;
  only an order that opens a new position can violate it.

The batch is evaluated sequentially with cumulative projection: an approved
order counts toward the day-order budget, the projected position book, and
the per-symbol exposure that later orders in the same batch are judged
against. Rejected orders are excluded from the projection (they will never
be submitted).

Per order, checks run in a fixed sequence — max_order_notional_usd,
max_day_orders, max_total_positions, max_position_pct_equity — and the first
violation is the one reported. Boundary semantics: a value exactly AT a
limit passes; strictly over fails. Every rejection message starts with the
violated limit's JSON key.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution.alpaca_client import Account, Position

LIMITS_PATH = Path(__file__).resolve().parent / "limits.paper.json"

_FLOAT_LIMITS = ("max_order_notional_usd", "max_position_pct_equity")
_INT_LIMITS = ("max_day_orders", "max_total_positions")

_QTY_EPSILON = 1e-9  # projected quantities inside this band count as flat


class OrderGateError(RuntimeError):
    """One or more proposed orders violated the configured paper limits."""


class LimitsConfigError(OrderGateError):
    """execution/limits.paper.json is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Limits:
    max_order_notional_usd: float
    max_position_pct_equity: float
    max_day_orders: int
    max_total_positions: int


def load_limits(path: Path | str = LIMITS_PATH) -> Limits:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise LimitsConfigError(f"limits file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LimitsConfigError(f"limits file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise LimitsConfigError(f"limits file {path} must hold a JSON object")
    known = set(_FLOAT_LIMITS) | set(_INT_LIMITS)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise LimitsConfigError(f"limits file {path} has unknown keys: {', '.join(unknown)}")
    missing = sorted(known - set(raw))
    if missing:
        raise LimitsConfigError(f"limits file {path} is missing keys: {', '.join(missing)}")
    values: dict[str, Any] = {}
    for key in _FLOAT_LIMITS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise LimitsConfigError(
                f"limits file {path}: {key} must be a positive number, got {value!r}"
            )
    for key in _INT_LIMITS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LimitsConfigError(
                f"limits file {path}: {key} must be a positive integer, got {value!r}"
            )
    for key in _FLOAT_LIMITS:
        values[key] = float(raw[key])
    for key in _INT_LIMITS:
        values[key] = raw[key]
    return Limits(**values)


@dataclass(frozen=True)
class ProposedOrder:
    """A not-yet-submitted marketable-limit order, as US-032's diff emits."""

    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    limit_price: float

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty!r}")
        if self.limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {self.limit_price!r}")

    @property
    def notional(self) -> float:
        return self.qty * self.limit_price

    def describe(self) -> str:
        return f"{self.side} {self.qty:g} {self.symbol} @ {self.limit_price:.2f}"


@dataclass(frozen=True)
class GateRejection:
    order: ProposedOrder
    limit_name: str  # the violated key in limits.paper.json
    message: str


@dataclass(frozen=True)
class GateResult:
    approved: list[ProposedOrder]
    rejections: list[GateRejection]

    @property
    def ok(self) -> bool:
        return not self.rejections

    def raise_for_rejections(self) -> None:
        """Raise OrderGateError listing every rejection (no-op when clean)."""
        if self.rejections:
            total = len(self.approved) + len(self.rejections)
            details = "\n".join(r.message for r in self.rejections)
            raise OrderGateError(
                f"order gate rejected {len(self.rejections)} of {total} proposed orders:\n{details}"
            )


def evaluate_orders(
    orders: Sequence[ProposedOrder],
    account: Account,
    positions: Sequence[Position],
    day_orders_placed: int,
    limits: Limits,
) -> GateResult:
    """Check a proposed order batch against the paper limits.

    ``day_orders_placed`` is how many orders were already submitted today
    (the caller counts them from a fresh ``list_orders`` snapshot). Returns
    every order as approved or rejected; never raises for limit violations
    (use ``GateResult.raise_for_rejections`` to abort on any rejection).
    """
    if day_orders_placed < 0:
        raise ValueError(f"day_orders_placed must be >= 0, got {day_orders_placed}")
    projected_qty: dict[str, float] = {p.symbol: p.qty for p in positions}
    position_cap = account.equity * limits.max_position_pct_equity / 100.0
    approved: list[ProposedOrder] = []
    rejections: list[GateRejection] = []

    def reject(order: ProposedOrder, limit_name: str, message: str) -> None:
        rejections.append(GateRejection(order=order, limit_name=limit_name, message=message))

    for order in orders:
        old_qty = projected_qty.get(order.symbol, 0.0)
        signed_qty = order.qty if order.side == "buy" else -order.qty
        new_qty = old_qty + signed_qty
        if abs(new_qty) < _QTY_EPSILON:
            new_qty = 0.0
        new_value = new_qty * order.limit_price

        if order.notional > limits.max_order_notional_usd:
            reject(
                order,
                "max_order_notional_usd",
                f"max_order_notional_usd: {order.describe()} notional "
                f"${order.notional:,.2f} exceeds the "
                f"${limits.max_order_notional_usd:,.2f} per-order cap",
            )
            continue

        order_number = day_orders_placed + len(approved) + 1
        if order_number > limits.max_day_orders:
            reject(
                order,
                "max_day_orders",
                f"max_day_orders: {order.describe()} would be day order "
                f"{order_number}, over the {limits.max_day_orders} cap "
                f"({day_orders_placed} already placed today, "
                f"{len(approved)} approved earlier in this batch)",
            )
            continue

        opens_position = old_qty == 0.0 and new_qty != 0.0
        if opens_position:
            count_after = sum(1 for q in projected_qty.values() if q != 0.0) + 1
            if count_after > limits.max_total_positions:
                reject(
                    order,
                    "max_total_positions",
                    f"max_total_positions: {order.describe()} would open position "
                    f"{count_after}, over the {limits.max_total_positions} cap",
                )
                continue

        if abs(new_qty) > abs(old_qty) and abs(new_value) > position_cap:
            reject(
                order,
                "max_position_pct_equity",
                f"max_position_pct_equity: {order.describe()} projects "
                f"{order.symbol} to ${abs(new_value):,.2f}, over "
                f"{limits.max_position_pct_equity:g}% of equity "
                f"${account.equity:,.2f} (cap ${position_cap:,.2f})",
            )
            continue

        approved.append(order)
        projected_qty[order.symbol] = new_qty

    return GateResult(approved=approved, rejections=rejections)

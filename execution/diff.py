"""Order diff: current positions + target weights -> marketable-limit orders.

Pure money-touching arithmetic, same discipline as execution/order_gate.py:
no HTTP, no state. The caller passes a fresh account/positions snapshot, the
target weights (execution/signal.py TargetBook.weights), and a reference
price per symbol; it gets back the exact order list to submit.

Rules (each is unit-tested in tests/test_diff.py):

* Sizing / share rounding: target share count = floor(weight * equity /
  ref_price) — whole shares, rounded DOWN so a fill can never exceed the
  target notional. Non-exit trade quantities are the delta floored TOWARD
  ZERO to whole shares (a sub-share delta trades nothing). Full exits sell
  the position's exact quantity, fractional or not, so nothing is left
  behind.
* Full exit: a held symbol absent from the targets is closed completely
  (a short is bought back). Exits are never skipped, however small — they
  free position slots and leave no orphans.
* New entry: a targeted symbol with no current position is bought whenever
  it affords at least one whole share; the min-notional skip does NOT apply
  (per US-032 it covers rebalance-only deltas).
* Rebalance skip: when the symbol is both held and targeted, a trade whose
  notional (floored qty * ref_price) is strictly below
  min_rebalance_notional_usd is skipped. Exactly AT the threshold trades
  (same boundary convention as the order gate).
* Marketable-limit pricing: buy limit = ref_price * (1 + offset_pct/100)
  rounded UP to the cent; sell limit = ref_price * (1 - offset_pct/100)
  rounded DOWN to the cent — priced through the market in both directions so
  the order fills like a market order but with a bounded worst price.
* Ordering: all sells first (alphabetical), then all buys (alphabetical) —
  sells free the cash the buys spend.

Failure policy: every bad input (missing/nonpositive price, nonpositive
equity, weights invalid or summing over 1, duplicate positions) raises
DiffError before any order list exists — no partial output, ever.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from execution.alpaca_client import Account, Position
from execution.order_gate import ProposedOrder

DEFAULT_MIN_REBALANCE_NOTIONAL_USD = 200.0
DEFAULT_LIMIT_OFFSET_PCT = 0.5  # marketable-limit distance from the reference price

_QTY_EPSILON = 1e-9  # quantities inside this band count as flat
_CENT_EPSILON = 1e-6  # keeps exact-cent prices from bumping a cent on float error


class DiffError(RuntimeError):
    """Any condition that must abort the diff (no partial order list)."""


@dataclass(frozen=True)
class SkippedDelta:
    """A target/position delta deliberately not traded, for dry-run output."""

    symbol: str
    reason: str  # "below_min_notional" | "zero_shares"
    message: str


@dataclass(frozen=True)
class DiffResult:
    orders: list[ProposedOrder]  # sells first, then buys; alphabetical within each
    skipped: list[SkippedDelta]


def marketable_limit_price(
    ref_price: float, side: str, offset_pct: float = DEFAULT_LIMIT_OFFSET_PCT
) -> float:
    """Limit price priced through the market by offset_pct, rounded to cents.

    Buys round UP to the cent, sells round DOWN — always at least as
    aggressive as the raw offset, never less marketable.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if not math.isfinite(ref_price) or ref_price <= 0:
        raise DiffError(f"reference price must be a positive number, got {ref_price!r}")
    if not math.isfinite(offset_pct) or offset_pct < 0:
        raise DiffError(f"limit_offset_pct must be >= 0, got {offset_pct!r}")
    if side == "buy":
        cents = math.ceil(ref_price * (1 + offset_pct / 100.0) * 100 - _CENT_EPSILON)
    else:
        cents = math.floor(ref_price * (1 - offset_pct / 100.0) * 100 + _CENT_EPSILON)
    price = cents / 100.0
    if price <= 0:
        raise DiffError(
            f"sell limit for ref price ${ref_price} rounds to ${price:.2f}; "
            f"cannot price a marketable sell"
        )
    return price


def _ref_price(symbol: str, prices: Mapping[str, float]) -> float:
    price = prices.get(symbol)
    if price is None:
        raise DiffError(f"no reference price for {symbol}; cannot size its order")
    if isinstance(price, bool) or not isinstance(price, int | float):
        raise DiffError(f"reference price for {symbol} must be a number, got {price!r}")
    price = float(price)
    if not math.isfinite(price) or price <= 0:
        raise DiffError(f"reference price for {symbol} must be positive, got {price!r}")
    return price


def _validate_targets(targets: Mapping[str, float]) -> None:
    total = 0.0
    for symbol, weight in targets.items():
        if not symbol or not isinstance(symbol, str):
            raise DiffError(f"target symbols must be non-empty strings, got {symbol!r}")
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise DiffError(f"target weight for {symbol} must be a number, got {weight!r}")
        if not math.isfinite(weight) or weight <= 0 or weight > 1:
            raise DiffError(f"target weight for {symbol} must be in (0, 1], got {weight!r}")
        total += float(weight)
    if total > 1 + 1e-6:
        raise DiffError(f"target weights sum to {total:g}, over 1.0 — refusing a levered book")


def _current_book(positions: Sequence[Position]) -> dict[str, float]:
    current: dict[str, float] = {}
    for position in positions:
        if position.symbol in current:
            raise DiffError(f"duplicate position rows for {position.symbol} in snapshot")
        if abs(position.qty) >= _QTY_EPSILON:
            current[position.symbol] = position.qty
    return current


def compute_orders(
    targets: Mapping[str, float],
    account: Account,
    positions: Sequence[Position],
    prices: Mapping[str, float],
    min_rebalance_notional_usd: float = DEFAULT_MIN_REBALANCE_NOTIONAL_USD,
    limit_offset_pct: float = DEFAULT_LIMIT_OFFSET_PCT,
) -> DiffResult:
    """Diff the current book against target weights into marketable-limit orders.

    ``prices`` maps every symbol that is held or targeted to its reference
    price (latest close or quote — the caller decides the source). See the
    module docstring for sizing, rounding, skip, and ordering rules.
    """
    if not math.isfinite(account.equity) or account.equity <= 0:
        raise DiffError(f"account equity must be positive, got {account.equity!r}")
    if not math.isfinite(min_rebalance_notional_usd) or min_rebalance_notional_usd < 0:
        raise DiffError(
            f"min_rebalance_notional_usd must be >= 0, got {min_rebalance_notional_usd!r}"
        )
    _validate_targets(targets)
    current = _current_book(positions)

    sells: list[ProposedOrder] = []
    buys: list[ProposedOrder] = []
    skipped: list[SkippedDelta] = []

    def emit(symbol: str, side: str, qty: float, price: float) -> None:
        order = ProposedOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            limit_price=marketable_limit_price(price, side, limit_offset_pct),
        )
        (sells if side == "sell" else buys).append(order)

    # Full exits: held but not targeted. Exact quantity, never skipped.
    for symbol in sorted(set(current) - set(targets)):
        qty = current[symbol]
        side = "sell" if qty > 0 else "buy"  # closing a short buys it back
        emit(symbol, side, abs(qty), _ref_price(symbol, prices))

    for symbol in sorted(targets):
        price = _ref_price(symbol, prices)
        cur_qty = current.get(symbol, 0.0)
        target_qty = math.floor(targets[symbol] * account.equity / price + _QTY_EPSILON)
        delta = target_qty - cur_qty
        trade_qty = math.floor(abs(delta) + _QTY_EPSILON)
        if trade_qty == 0:
            if cur_qty == 0.0:
                skipped.append(
                    SkippedDelta(
                        symbol=symbol,
                        reason="zero_shares",
                        message=(
                            f"{symbol}: target weight {targets[symbol]:g} affords zero whole "
                            f"shares at ${price:,.2f} on ${account.equity:,.2f} equity"
                        ),
                    )
                )
            continue  # held book already within one share of target: no-op
        trade_notional = trade_qty * price
        if cur_qty != 0.0 and trade_notional < min_rebalance_notional_usd:
            skipped.append(
                SkippedDelta(
                    symbol=symbol,
                    reason="below_min_notional",
                    message=(
                        f"{symbol}: rebalance delta ${trade_notional:,.2f} is below the "
                        f"${min_rebalance_notional_usd:,.2f} min-notional threshold"
                    ),
                )
            )
            continue
        emit(symbol, "buy" if delta > 0 else "sell", float(trade_qty), price)

    sells.sort(key=lambda o: o.symbol)
    buys.sort(key=lambda o: o.symbol)
    return DiffResult(orders=sells + buys, skipped=skipped)

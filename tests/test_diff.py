"""Unit tests for execution/diff.py (US-032).

Fixture positions + targets must produce the EXACT expected order list —
this is the code that decides what gets bought and sold. Covers: full exit
(long and short), new entry, rebalance deltas (at / below the min-notional
threshold), share and limit-price rounding, ordering, and every DiffError
input guard.
"""

from __future__ import annotations

import pytest

from execution.alpaca_client import Account, Position
from execution.diff import (
    DiffError,
    SkippedDelta,
    compute_orders,
    marketable_limit_price,
)
from execution.order_gate import Limits, ProposedOrder, evaluate_orders

EQUITY = 100_000.0

PRICES = {
    "AAPL": 200.0,
    "GME": 25.0,
    "MSFT": 400.0,
    "NVDA": 100.0,
    "TSLA": 250.0,
    "XOM": 100.0,
}


def make_account(equity: float = EQUITY) -> Account:
    return Account(
        id="acct-1",
        status="ACTIVE",
        currency="USD",
        equity=equity,
        cash=equity,
        buying_power=equity * 2,
    )


def make_position(symbol: str, qty: float, price: float | None = None) -> Position:
    price = PRICES[symbol] if price is None else price
    return Position(
        symbol=symbol,
        qty=qty,
        side="long" if qty >= 0 else "short",
        avg_entry_price=price,
        current_price=price,
        market_value=qty * price,
    )


# ---------------------------------------------------------------- golden diff


class TestGoldenDiff:
    """One realistic book: exit + entries + rebalance + skip, exact output."""

    def result(self):
        targets = {"AAPL": 0.25, "MSFT": 0.25, "NVDA": 0.25, "TSLA": 0.25}
        positions = [
            make_position("AAPL", 100),  # rebalance up: target 125
            make_position("NVDA", 251),  # trim by 1 = $100 -> below min notional
            make_position("XOM", 50),  # full exit
        ]
        return compute_orders(targets, make_account(), positions, PRICES)

    def test_exact_order_list(self) -> None:
        assert self.result().orders == [
            # sells first (alphabetical), then buys (alphabetical)
            ProposedOrder(symbol="XOM", side="sell", qty=50.0, limit_price=99.50),
            ProposedOrder(symbol="AAPL", side="buy", qty=25.0, limit_price=201.00),
            ProposedOrder(symbol="MSFT", side="buy", qty=62.0, limit_price=402.00),
            ProposedOrder(symbol="TSLA", side="buy", qty=100.0, limit_price=251.25),
        ]

    def test_small_rebalance_skipped_with_reason(self) -> None:
        assert self.result().skipped == [
            SkippedDelta(
                symbol="NVDA",
                reason="below_min_notional",
                message=(
                    "NVDA: rebalance delta $100.00 is below the "
                    "$200.00 min-notional threshold"
                ),
            )
        ]

    def test_orders_pass_the_gate(self) -> None:
        """The diff emits orders the order gate accepts (shared ProposedOrder)."""
        limits = Limits(
            max_order_notional_usd=50_000.0,
            max_position_pct_equity=50.0,
            max_day_orders=100,
            max_total_positions=10,
        )
        result = evaluate_orders(
            self.result().orders,
            make_account(),
            [make_position("AAPL", 100), make_position("NVDA", 251), make_position("XOM", 50)],
            day_orders_placed=0,
            limits=limits,
        )
        assert result.ok and len(result.approved) == 4


# ---------------------------------------------------------------- full exits


class TestFullExit:
    def test_long_exit_sells_exact_fractional_qty(self) -> None:
        result = compute_orders({}, make_account(), [make_position("GME", 10.5)], PRICES)
        assert result.orders == [
            ProposedOrder(symbol="GME", side="sell", qty=10.5, limit_price=24.87)
        ]

    def test_short_exit_buys_back(self) -> None:
        result = compute_orders({}, make_account(), [make_position("GME", -30)], PRICES)
        assert result.orders == [
            ProposedOrder(symbol="GME", side="buy", qty=30.0, limit_price=25.13)
        ]

    def test_exit_never_skipped_even_below_min_notional(self) -> None:
        result = compute_orders(
            {}, make_account(), [make_position("GME", 1)], PRICES, min_rebalance_notional_usd=200.0
        )
        assert len(result.orders) == 1 and result.skipped == []

    def test_flat_position_row_is_ignored(self) -> None:
        result = compute_orders({}, make_account(), [make_position("GME", 0.0)], PRICES)
        assert result.orders == [] and result.skipped == []


# ---------------------------------------------------------------- entries


class TestNewEntry:
    def test_entry_floors_target_shares(self) -> None:
        # 0.1 * 100_000 / 400 = exactly 25 shares (float-safe floor)
        result = compute_orders({"MSFT": 0.1}, make_account(), [], PRICES)
        assert result.orders == [
            ProposedOrder(symbol="MSFT", side="buy", qty=25.0, limit_price=402.00)
        ]

    def test_entry_rounds_down_never_up(self) -> None:
        # 0.1 * 100_000 / 250 = 40 shares; 0.0999 -> 39.96 -> floor 39
        result = compute_orders({"TSLA": 0.0999}, make_account(), [], PRICES)
        assert result.orders[0].qty == 39.0

    def test_entry_ignores_min_notional(self) -> None:
        # one whole GME share = $25 notional, far below the $200 threshold
        result = compute_orders(
            {"GME": 0.00026}, make_account(), [], PRICES, min_rebalance_notional_usd=200.0
        )
        assert result.orders == [
            ProposedOrder(symbol="GME", side="buy", qty=1.0, limit_price=25.13)
        ]
        assert result.skipped == []

    def test_entry_affording_zero_shares_is_skipped_with_reason(self) -> None:
        result = compute_orders({"MSFT": 0.000001}, make_account(), [], PRICES)
        assert result.orders == []
        assert len(result.skipped) == 1
        skip = result.skipped[0]
        assert skip.symbol == "MSFT" and skip.reason == "zero_shares"
        assert "zero whole shares" in skip.message


# ---------------------------------------------------------------- rebalances


class TestRebalance:
    def test_delta_exactly_at_min_notional_trades(self) -> None:
        # trim NVDA 252 -> 250: 2 shares * $100 = $200 = the threshold exactly
        result = compute_orders(
            {"NVDA": 0.25},
            make_account(),
            [make_position("NVDA", 252)],
            PRICES,
            min_rebalance_notional_usd=200.0,
        )
        assert result.orders == [
            ProposedOrder(symbol="NVDA", side="sell", qty=2.0, limit_price=99.50)
        ]
        assert result.skipped == []

    def test_delta_just_below_min_notional_skipped(self) -> None:
        result = compute_orders(
            {"NVDA": 0.25},
            make_account(),
            [make_position("NVDA", 251)],
            PRICES,
            min_rebalance_notional_usd=200.0,
        )
        assert result.orders == []
        assert result.skipped[0].reason == "below_min_notional"

    def test_zero_delta_is_silent_noop(self) -> None:
        result = compute_orders(
            {"AAPL": 0.25}, make_account(), [make_position("AAPL", 125)], PRICES
        )
        assert result.orders == [] and result.skipped == []

    def test_sub_share_delta_is_silent_noop(self) -> None:
        # target 125, held 124.6 -> delta 0.4 floors to zero shares
        result = compute_orders(
            {"AAPL": 0.25}, make_account(), [make_position("AAPL", 124.6)], PRICES
        )
        assert result.orders == [] and result.skipped == []

    def test_fractional_holding_delta_floors_toward_zero(self) -> None:
        # target 125, held 123.5 -> delta 1.5 -> buy 1 whole share
        result = compute_orders(
            {"AAPL": 0.25},
            make_account(),
            [make_position("AAPL", 123.5)],
            PRICES,
            min_rebalance_notional_usd=0.0,
        )
        assert result.orders == [
            ProposedOrder(symbol="AAPL", side="buy", qty=1.0, limit_price=201.00)
        ]

    def test_short_to_long_is_one_crossing_buy(self) -> None:
        # held -10, target 0.2 * 100_000 / 200 = 100 -> buy 110 in one order
        result = compute_orders(
            {"AAPL": 0.2}, make_account(), [make_position("AAPL", -10)], PRICES
        )
        assert result.orders == [
            ProposedOrder(symbol="AAPL", side="buy", qty=110.0, limit_price=201.00)
        ]

    def test_targeted_tiny_weight_on_held_name_trims_via_min_notional_rule(self) -> None:
        # target affords 0 shares but 40 are held -> sell 40 ($1000 >= min)
        result = compute_orders(
            {"GME": 0.000001}, make_account(), [make_position("GME", 40)], PRICES
        )
        assert result.orders == [
            ProposedOrder(symbol="GME", side="sell", qty=40.0, limit_price=24.87)
        ]


# ---------------------------------------------------------------- ordering


class TestOrdering:
    def test_sells_before_buys_alphabetical_within(self) -> None:
        targets = {"AAPL": 0.3, "NVDA": 0.1, "TSLA": 0.3}
        positions = [
            make_position("XOM", 10),  # exit sell
            make_position("AAPL", 10),  # rebalance buy (target 150)
            make_position("NVDA", 500),  # rebalance sell (target 100)
            make_position("GME", -5),  # exit buy (short cover)
        ]
        result = compute_orders(targets, make_account(), positions, PRICES)
        assert [(o.side, o.symbol) for o in result.orders] == [
            ("sell", "NVDA"),
            ("sell", "XOM"),
            ("buy", "AAPL"),
            ("buy", "GME"),
            ("buy", "TSLA"),
        ]


# ---------------------------------------------------------------- limit price


class TestMarketableLimitPrice:
    def test_buy_rounds_up_to_cent(self) -> None:
        # 10.01 * 1.005 = 10.06005 -> 10.07
        assert marketable_limit_price(10.01, "buy", 0.5) == 10.07

    def test_sell_rounds_down_to_cent(self) -> None:
        # 10.01 * 0.995 = 9.95995 -> 9.95
        assert marketable_limit_price(10.01, "sell", 0.5) == 9.95

    def test_exact_cent_does_not_bump(self) -> None:
        assert marketable_limit_price(400.0, "buy", 0.5) == 402.00
        assert marketable_limit_price(100.0, "sell", 0.5) == 99.50

    def test_zero_offset_returns_ref_price(self) -> None:
        assert marketable_limit_price(123.45, "buy", 0.0) == 123.45
        assert marketable_limit_price(123.45, "sell", 0.0) == 123.45

    def test_sell_price_rounding_to_zero_raises(self) -> None:
        with pytest.raises(DiffError, match="rounds to"):
            marketable_limit_price(0.004, "sell", 0.5)

    def test_bad_side_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="side"):
            marketable_limit_price(100.0, "hold", 0.5)

    def test_negative_offset_raises(self) -> None:
        with pytest.raises(DiffError, match="limit_offset_pct"):
            marketable_limit_price(100.0, "buy", -0.5)


# ---------------------------------------------------------------- input guards


class TestDiffErrors:
    def test_missing_price_for_target(self) -> None:
        with pytest.raises(DiffError, match="no reference price for ZZZZ"):
            compute_orders({"ZZZZ": 0.5}, make_account(), [], PRICES)

    def test_missing_price_for_exit(self) -> None:
        position = make_position("GME", 10)
        with pytest.raises(DiffError, match="no reference price for GME"):
            compute_orders({}, make_account(), [position], {"AAPL": 200.0})

    def test_nonpositive_price(self) -> None:
        with pytest.raises(DiffError, match="must be positive"):
            compute_orders({"AAPL": 0.5}, make_account(), [], {"AAPL": 0.0})

    def test_nonpositive_equity(self) -> None:
        with pytest.raises(DiffError, match="equity"):
            compute_orders({"AAPL": 0.5}, make_account(equity=0.0), [], PRICES)

    def test_duplicate_position_rows(self) -> None:
        positions = [make_position("GME", 10), make_position("GME", 5)]
        with pytest.raises(DiffError, match="duplicate position rows for GME"):
            compute_orders({}, make_account(), positions, PRICES)

    def test_weight_over_one(self) -> None:
        with pytest.raises(DiffError, match=r"in \(0, 1\]"):
            compute_orders({"AAPL": 1.5}, make_account(), [], PRICES)

    def test_nonpositive_weight(self) -> None:
        with pytest.raises(DiffError, match=r"in \(0, 1\]"):
            compute_orders({"AAPL": -0.1}, make_account(), [], PRICES)

    def test_weights_summing_over_one(self) -> None:
        targets = {"AAPL": 0.6, "MSFT": 0.6}
        with pytest.raises(DiffError, match="levered"):
            compute_orders(targets, make_account(), [], PRICES)

    def test_weights_summing_to_exactly_one_ok(self) -> None:
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        result = compute_orders(targets, make_account(), [], PRICES)
        assert len(result.orders) == 2

    def test_negative_min_notional(self) -> None:
        with pytest.raises(DiffError, match="min_rebalance_notional_usd"):
            compute_orders({}, make_account(), [], PRICES, min_rebalance_notional_usd=-1.0)

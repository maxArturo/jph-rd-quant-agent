"""Nightly rebalance pipeline (US-034; paper by default, ``--live`` for the
real-money account, US-013).

Chains the tested execution pieces end-to-end, in the PRD's order:

    market-calendar check -> promoted strategy load -> signal extraction
    -> order diff -> buying-power cap -> order gate -> circuit breaker
    -> submit -> poll fills

``--live`` switches the whole chain to the live account in one flag: the
LIVE promoted slot (``load_promoted_strategy_live`` — never the paper row),
``AlpacaClient`` against the live host with ``allow_live=True`` (this
entrypoint is one of the few callers allowed to pass it), limits.live.json +
breaker.live.json with breaker state under ``~/rdq-data/breaker-live/``, the
live Slack channel for every notice, and client_order_id prefix
``rdq-live-``. The equity handed to ``diff.compute_orders`` is scaled by
``live_equity_allocation_pct / 100`` (allocation.live.json) so the live
account trades only its allocation slice; the buying-power cap and the order
gate keep seeing the REAL account snapshot. Everything else — abort
contract, halt semantics, exit codes — is identical to paper, and without
``--live`` behavior is byte-for-byte unchanged.

Live Notion routing (US-014): ``--live`` writes each order to Trade Ledger
(Live) and the daily row to Account Snapshots (Live) — never the paper
databases — and REFUSES to run when the live database ids are missing from
orchestrator/config.yaml (run ``ops/bootstrap_notion.py --live``); there is
no ``--no-notion`` escape hatch in live mode. The live daily summary names
LIVE in its header, states the allocation pct in force, and — when the paper
account also wrote its snapshot row for the same day — appends a one-line
paper-vs-live day-P/L comparison (advisory and best-effort: absent paper
data simply omits the line).

The buying-power cap (cap_buys_to_buying_power) defers buys the account
snapshot's buying_power cannot fund — Alpaca reserves qty * limit_price when
it accepts a buy and credits sells only on fill, so a pre-open batch sized
off equity can 403 (40310000) mid-submission otherwise. Deferred buys appear
as skipped lines in the plan and WARNING lines in the daily summary;
tomorrow's diff re-proposes them once today's sells settle.

Abort policy: every failure posts to the Slack channel and exits nonzero —
except a halt-file breaker trip, which posts a "halted" notice and exits 0
(an operator halt is a deliberate state, not an error). Gate and breaker
rejections post the daily summary (equity + each violated limit) as their
notice; earlier failures post the plain abort message. Days that reach the
end post the daily summary too: orders placed, fills, rejections ("none"),
account equity, breaker state — and every submitted order lands in the Notion Trade Ledger
(execution/ledger.py, US-035), created at submit time and updated with its
terminal fill or rejection after the poll. ``--dry-run`` runs
the full chain through the gate and breaker, prints the exact order list,
and exits 0 without submitting anything. Note the breaker's clean-pass
high-water-mark update still happens on a dry run (recording an equity peak
is state the kill switch should have either way).

Every day the pipeline obtained a broker snapshot also writes one Account
Snapshots row to Notion (execution/account_log.py, US-047) — equity, cash,
previous completed day's P/L, order counts, outcome — on traded, no-trade,
gate-rejected, breaker-tripped and halted days alike; dry runs and earlier
aborts write nothing.

Sources of truth wired here (each decided in its own story):

* Trading-day check: the Alpaca market calendar (GET /v2/calendar) — the
  qlib store calendar ends at the last built bar, so it cannot say whether
  *today* trades.
* topk/n_drop: the promoted strategy's pinned config (the operator confirmed
  those exact values when promoting; US-033).
* Reference prices: the qlib store's latest raw close (stored adjusted close
  / stored factor), falling back to the position snapshot's current_price
  for held symbols missing from the store. No price -> abort.
* "Today" for the day-order count and traded notional: the order's
  submitted_at converted to America/New_York — Alpaca timestamps are UTC and
  the trading day is Eastern.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from execution import allocation as allocation_mod
from execution import breaker as breaker_mod
from execution import order_gate as order_gate_mod
from execution.account_log import AccountSnapshotLog, previous_day_pnl
from execution.allocation import AllocationConfigError, LiveAllocation, load_live_allocation
from execution.alpaca_client import (
    LIVE_BASE_URL,
    Account,
    AlpacaClient,
    AlpacaError,
    Order,
    Position,
)
from execution.breaker import (
    Breaker,
    BreakerError,
    BreakerReason,
    BreakerStateError,
    load_breaker_config,
)
from execution.diff import (
    DEFAULT_LIMIT_OFFSET_PCT,
    DEFAULT_MIN_REBALANCE_NOTIONAL_USD,
    DiffError,
    DiffResult,
    SkippedDelta,
    compute_orders,
)
from execution.ledger import TradeLedger
from execution.order_gate import (
    Limits,
    OrderGateError,
    ProposedOrder,
    evaluate_orders,
    load_limits,
)
from execution.promoted import (
    NoPromotedStrategyError,
    load_promoted_strategy,
    load_promoted_strategy_live,
)
from execution.signal import SignalError, StrategyParams, extract_targets
from orchestrator.live_status import LiveSnapshot
from orchestrator.state import DEFAULT_DB_PATH

MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_STORE_PATH = Path("~/.qlib/qlib_data/us_data")

DEFAULT_POLL_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_ORDER_LIST_LIMIT = 500  # Alpaca's max page size for GET /v2/orders

# Order statuses that will never fill further (Alpaca v2 lifecycle).
TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "stopped", "done_for_day"}
)

Notify = Callable[[str], None]


class RebalanceError(RuntimeError):
    """Any pipeline-level condition that must abort without trading."""


# Every known abort reason. Anything else is a bug and crashes loudly
# (after telling the operator) instead of being folded into exit 1.
_ABORT_ERRORS = (
    RebalanceError,
    NoPromotedStrategyError,
    SignalError,
    DiffError,
    OrderGateError,
    BreakerError,
    AlpacaError,
    AllocationConfigError,  # live mode: unusable allocation.live.json
)


def submitted_market_date(order: Order) -> dt.date | None:
    """The America/New_York date an order was submitted, or None if unknown."""
    if not order.submitted_at:
        return None
    raw = order.submitted_at.replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp.astimezone(MARKET_TZ).date()


def orders_submitted_on(orders: Iterable[Order], day: dt.date) -> list[Order]:
    """Orders whose submitted_at falls on ``day`` in market (Eastern) time."""
    return [order for order in orders if submitted_market_date(order) == day]


def day_traded_notional(orders: Iterable[Order]) -> float:
    """Dollars actually traded across the given orders (|filled qty| * avg price)."""
    total = 0.0
    for order in orders:
        if order.filled_qty and order.filled_avg_price is not None:
            total += abs(order.filled_qty) * order.filled_avg_price
    return total


def latest_store_price(store_path: Path, symbol: str) -> float | None:
    """Latest raw price from the qlib store: last adjusted close / last factor.

    (Stored closes are adjusted = raw * factor, so dividing by the stored
    factor recovers the raw close — the same identity data/make_universe.py
    uses.) Returns None when the store has no usable bins for the symbol.
    """
    import numpy as np

    feature_dir = store_path.expanduser() / "features" / symbol.lower()
    close_path = feature_dir / "close.day.bin"
    factor_path = feature_dir / "factor.day.bin"
    if not close_path.is_file() or not factor_path.is_file():
        return None
    close = np.fromfile(close_path, dtype="<f")
    factor = np.fromfile(factor_path, dtype="<f")
    if len(close) < 2 or len(factor) < 2:  # element 0 is the calendar-index header
        return None
    last_close = float(close[-1])
    last_factor = float(factor[-1])
    if not math.isfinite(last_close) or not math.isfinite(last_factor):
        return None
    if last_close <= 0 or last_factor <= 0:
        return None
    return last_close / last_factor


def build_reference_prices(
    store_path: Path,
    symbols: Iterable[str],
    positions: Sequence[Position],
) -> dict[str, float]:
    """Reference price for every held-or-targeted symbol, or abort.

    Store price first (deterministic pre-open close), position snapshot
    current_price as the fallback for held names the store no longer carries.
    """
    snapshot_prices = {
        p.symbol: p.current_price
        for p in positions
        if p.current_price is not None and p.current_price > 0
    }
    prices: dict[str, float] = {}
    missing: list[str] = []
    for symbol in sorted(set(symbols)):
        price = latest_store_price(store_path, symbol)
        if price is None:
            price = snapshot_prices.get(symbol)
        if price is None:
            missing.append(symbol)
        else:
            prices[symbol] = price
    if missing:
        raise RebalanceError(
            f"no reference price for {', '.join(missing)}: not in the qlib store at "
            f"{store_path} and no current_price in the positions snapshot — refresh "
            "the store before trading"
        )
    return prices


def assert_trading_day(client: AlpacaClient, as_of: dt.date) -> None:
    """Abort unless as_of is a trading day per the Alpaca market calendar."""
    days = client.get_calendar(as_of, as_of)
    if not any(day.date == as_of.isoformat() for day in days):
        raise RebalanceError(
            f"market closed: {as_of} is not a trading day per the Alpaca calendar"
        )


def _strategy_params(config: Mapping[str, object]) -> StrategyParams | None:
    """StrategyParams from the promoted config, or None to re-read the conf.

    Promotion (US-033) always pins topk/n_drop; the None fallback only covers
    a promoted row written before that convention.
    """
    topk = config.get("topk")
    n_drop = config.get("n_drop")
    if topk is None or n_drop is None:
        return None
    if (
        isinstance(topk, bool)
        or isinstance(n_drop, bool)
        or not isinstance(topk, int)
        or not isinstance(n_drop, int)
    ):
        raise RebalanceError(
            f"promoted strategy config has non-integer topk/n_drop: "
            f"topk={topk!r}, n_drop={n_drop!r} — re-promote the run"
        )
    return StrategyParams(topk=topk, n_drop=n_drop)


def format_plan(diff: DiffResult, as_of: dt.date, dry_run: bool) -> str:
    """Human-readable order list (the dry-run contract output)."""
    header = f"rebalance plan for {as_of}" + (" (dry run — nothing submitted)" if dry_run else "")
    lines = [header]
    if diff.orders:
        lines += [f"  {order.describe()}" for order in diff.orders]
        total = sum(order.notional for order in diff.orders)
        lines.append(f"  {len(diff.orders)} orders, ${total:,.2f} total notional")
    else:
        lines.append("  no orders — book already on target")
    for skip in diff.skipped:
        lines.append(f"  skipped: {skip.message}")
    return "\n".join(lines)


def cap_buys_to_buying_power(
    orders: Sequence[ProposedOrder], buying_power: float
) -> tuple[list[ProposedOrder], list[SkippedDelta]]:
    """Defer buys the snapshot's buying power cannot cover; sells always pass.

    Alpaca reserves qty * limit_price the moment a buy limit order is
    accepted and does NOT credit unfilled sells, so a pre-open batch is
    funded solely by the account snapshot's buying_power — sell proceeds
    only exist after fills (the 2026-07-23 mid-batch 40310000 abort). Buys
    keep the diff's deterministic order and are taken while they fit
    (exactly-at-limit passes, strictly-over defers, matching the gate's
    boundary semantics); each deferred buy comes back as a SkippedDelta so
    the plan and summary report it. Tomorrow's diff re-proposes whatever
    was deferred once today's sells have settled.
    """
    kept: list[ProposedOrder] = []
    deferred: list[SkippedDelta] = []
    remaining = buying_power
    for order in orders:
        if order.side == "buy":
            if order.notional > remaining:
                deferred.append(
                    SkippedDelta(
                        symbol=order.symbol,
                        reason="insufficient_buying_power",
                        message=(
                            f"{order.symbol}: {order.describe()} deferred — notional "
                            f"${order.notional:,.2f} exceeds the ${remaining:,.2f} "
                            "buying power left (unfilled sells do not fund pre-open "
                            "buys; tomorrow's diff re-proposes it)"
                        ),
                    )
                )
                continue
            remaining -= order.notional
        kept.append(order)
    return kept, deferred


def submit_orders(
    client: AlpacaClient,
    orders: Sequence[ProposedOrder],
    as_of: dt.date,
    ledger: TradeLedger | None = None,
    id_prefix: str = "rdq",
    account_label: str = "paper",
) -> list[Order]:
    """Submit every proposed order as a day marketable-limit order.

    client_order_id is deterministic per (day, side, symbol) so an accidental
    same-day rerun is rejected by Alpaca's uniqueness check instead of
    doubling the book (live mode passes ``id_prefix="rdq-live"``, keeping the
    two accounts' id spaces distinct). Each order is recorded in the Trade
    Ledger right after its POST succeeds, so a mid-batch failure still leaves
    a row for every order that is live.
    """
    submitted: list[Order] = []
    for order in orders:
        client_order_id = f"{id_prefix}-{as_of.isoformat()}-{order.side}-{order.symbol}"
        try:
            placed = client.place_order(
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                order_type="limit",
                time_in_force="day",
                limit_price=order.limit_price,
                client_order_id=client_order_id,
            )
        except AlpacaError as exc:
            raise RebalanceError(
                f"order submission failed after {len(submitted)} of {len(orders)} orders "
                f"went in (failed on {order.describe()}): {exc} — check the "
                f"{account_label} account before rerunning; submitted orders are live"
            ) from exc
        submitted.append(placed)
        if ledger is not None:
            ledger.record_submitted(placed, as_of)
    return submitted


def poll_fills(
    client: AlpacaClient,
    order_ids: Sequence[str],
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Order]:
    """Poll until every order reaches a terminal status or the timeout lapses.

    Returns the last snapshot either way — pre-open submissions legitimately
    stay unfilled until the market opens, so a timeout is reported, not
    raised.
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds!r}")
    wanted = set(order_ids)
    max_polls = max(int(timeout_seconds // interval_seconds), 1)
    snapshot: list[Order] = []
    for poll in range(max_polls):
        rows = client.list_orders(status="all", limit=_ORDER_LIST_LIMIT)
        snapshot = [row for row in rows if row.id in wanted]
        done = {row.id for row in snapshot if row.status in TERMINAL_ORDER_STATUSES}
        if wanted <= done:
            return snapshot
        if poll < max_polls - 1:
            sleep(interval_seconds)
    return snapshot


def fill_summary(submitted: Sequence[Order], final: Sequence[Order]) -> str:
    """One line per order with its final status, plus a fill count header."""
    by_id = {order.id: order for order in final}
    lines: list[str] = []
    filled = 0
    for order in submitted:
        latest = by_id.get(order.id, order)
        if latest.status == "filled":
            filled += 1
            price = f"{latest.filled_avg_price:,.2f}" if latest.filled_avg_price else "?"
            lines.append(
                f"  {latest.side} {latest.filled_qty:g} {latest.symbol}: filled @ ${price}"
            )
        else:
            lines.append(
                f"  {latest.side} {latest.qty or 0:g} {latest.symbol}: {latest.status} "
                f"({latest.filled_qty:g} filled)"
            )
    header = f"{filled}/{len(submitted)} orders filled"
    if filled < len(submitted):
        header += " (day limit orders may still fill after market open)"
    return "\n".join([header, *lines])


def breaker_state_line(breaker: Breaker) -> str:
    """One line describing the breaker's current state, for the daily summary.

    Deliberately never raises: a corrupt high-water-mark file is reported in
    the line itself (the breaker's own check aborts the run separately).
    """
    if breaker.halted:
        note = breaker.halt_note
        detail = f" — {note}" if note else ""
        return f"breaker: HALTED{detail} (resume trading to lift it)"
    try:
        high_water_mark = breaker.high_water_mark
    except BreakerStateError as exc:
        return f"breaker: STATE ERROR — {exc}"
    if high_water_mark is None:
        return "breaker: normal (no high-water mark recorded yet)"
    return f"breaker: normal (high-water mark ${high_water_mark:,.2f})"


class PaperSnapshotSource(Protocol):
    """Latest paper Account Snapshots row, for the paper-vs-live comparison.

    Satisfied by ``orchestrator.live_status.LiveStatusReader`` pointed at the
    PAPER Account Snapshots database (the row schemas are identical).
    """

    def latest_snapshot(self) -> LiveSnapshot | None: ...


def _pnl_text(profit_loss: float | None, pct: float | None) -> str:
    """`$+1,250.50 (+1.25%)` — or `n/a` when the P/L figure is missing."""
    if profit_loss is None:
        return "n/a"
    text = f"${profit_loss:+,.2f}"
    if pct is not None:
        text += f" ({pct * 100:+.2f}%)"
    return text


def paper_vs_live_pnl_line(
    paper_snapshots: PaperSnapshotSource | None,
    client: AlpacaClient,
    as_of: dt.date,
) -> str | None:
    """Same-day paper-vs-live day-P/L line for the live summary (US-014).

    Present only when the paper account also wrote its Account Snapshots row
    for ``as_of`` (both accounts traded the same morning). Advisory and
    best-effort end to end: a missing/stale paper row omits the line, a
    Notion read failure omits it with a stderr warning, and a live
    portfolio-history outage degrades the live side to n/a — nothing here
    may ever block or fail a live summary.
    """
    if paper_snapshots is None:
        return None
    try:
        paper = paper_snapshots.latest_snapshot()
    except Exception as exc:  # noqa: BLE001 - advisory; a Notion outage must not block
        print(f"WARNING: paper snapshot read failed: {exc}", file=sys.stderr)
        return None
    if paper is None or paper.date != as_of.isoformat():
        return None
    live_day = None
    try:
        live_day = previous_day_pnl(
            client.get_portfolio_history(period="1M", timeframe="1D"), as_of
        )
    except AlpacaError as exc:
        print(f"WARNING: live portfolio history fetch failed: {exc}", file=sys.stderr)
    live_pl, live_pct = (
        (live_day.profit_loss, live_day.profit_loss_pct) if live_day else (None, None)
    )
    return (
        f"paper vs live ({as_of} snapshots): paper day P/L "
        f"{_pnl_text(paper.day_pl, paper.day_pl_pct)}, live day P/L "
        f"{_pnl_text(live_pl, live_pct)}"
    )


def format_daily_summary(
    as_of: dt.date,
    equity: float,
    submitted: Sequence[Order],
    final: Sequence[Order],
    rejections: Sequence[str] = (),
    no_trade_note: str | None = None,
    ledger_failures: Sequence[str] = (),
    breaker_state: str | None = None,
    warnings: Sequence[str] = (),
    allocation_pct: float | None = None,
    paper_vs_live: str | None = None,
) -> str:
    """The daily Slack digest: orders placed, fills, rejections, equity.

    Posted on every day the pipeline reaches the gate — traded days carry the
    fill report, no-trade days say why, and gate/breaker-rejection days list
    each rejection (those days still exit nonzero; this is the notice).
    ``breaker_state`` is breaker_state_line()'s output (US-038).

    ``allocation_pct`` marks a LIVE summary (US-014): the header says LIVE
    and an allocation line states the slice sizing was computed against;
    ``paper_vs_live`` is paper_vs_live_pnl_line()'s output, appended last.
    With both None (every paper call) the output is byte-for-byte today's.
    """
    live_tag = "LIVE " if allocation_pct is not None else ""
    lines = [
        f"daily {live_tag}rebalance summary ({as_of})",
        f"account equity: ${equity:,.2f}",
    ]
    if allocation_pct is not None:
        lines.append(
            f"live allocation: {allocation_pct:g}% of equity"
            f" (sizing against ${equity * allocation_pct / 100.0:,.2f})"
        )
    lines.append(f"orders placed: {len(submitted)}")
    if submitted:
        lines.append(fill_summary(submitted, final))
    elif no_trade_note:
        lines.append(f"  {no_trade_note}")
    if rejections:
        lines.append("gate/breaker rejections:")
        lines += [f"  {reason}" for reason in rejections]
    else:
        lines.append("gate/breaker rejections: none")
    if breaker_state is not None:
        lines.append(breaker_state)
    for failure in ledger_failures:
        lines.append(f"WARNING: Trade Ledger write failed — {failure}")
    for warning in warnings:
        lines.append(f"WARNING: {warning}")
    if paper_vs_live:
        lines.append(paper_vs_live)
    return "\n".join(lines)


def _record_snapshot(
    snapshots: AccountSnapshotLog | None,
    client: AlpacaClient,
    as_of: dt.date,
    account: Account,
    positions: Sequence[Position],
    outcome: str,
    breaker: Breaker,
    orders_placed: int = 0,
    orders_filled: int = 0,
    note: str = "",
) -> list[str]:
    """Write the day's Account Snapshot row (US-047); returns WARNING lines.

    Best-effort end to end: the portfolio-history fetch (previous completed
    day's P/L) degrades to an empty Day P/L rather than blocking the row, and
    write failures come back as summary warnings, never exceptions.
    """
    if snapshots is None:
        return []
    history = None
    try:
        history = client.get_portfolio_history(period="1M", timeframe="1D")
    except AlpacaError as exc:
        snapshots.failures.append(f"portfolio history fetch: {exc}")
    snapshots.record_daily(
        as_of,
        account,
        positions,
        outcome,
        orders_placed=orders_placed,
        orders_filled=orders_filled,
        breaker_state=breaker_state_line(breaker),
        history=history,
        note=note,
    )
    return [f"Account Snapshot: {failure}" for failure in snapshots.failures]


def _safe_notify(notify: Notify, text: str) -> None:
    """Post to Slack best-effort: a chat outage must not mask the real outcome."""
    try:
        notify(text)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, reported on stderr
        print(f"WARNING: Slack notification failed: {exc}", file=sys.stderr)


def universe_divergence_warnings(
    config: Mapping[str, Any], target_weights: Mapping[str, float]
) -> list[str]:
    """WARNING lines when target holdings fall outside the pinned universe.

    An empty/absent ``universe_tickers`` skips the check (older promotions,
    or an unreadable instruments file at promote time). Purely advisory —
    the caller must never abort on it: pred.pkl is the backtested ground
    truth; a divergence means the promotion RECORDED the wrong universe.
    """
    tickers = config.get("universe_tickers")
    if not tickers:
        return []
    allowed = {str(t).strip().upper() for t in tickers}
    outside = sorted(s for s in target_weights if s.strip().upper() not in allowed)
    if not outside:
        return []
    shown = ", ".join(outside[:15]) + (", …" if len(outside) > 15 else "")
    return [
        f"universe_divergence: {len(outside)}/{len(target_weights)} target holding(s)"
        f" outside the promoted universe '{config.get('universe')}' ({shown}) —"
        " predictions come from a different market than the promotion recorded;"
        " re-promote to fix the pinned universe"
    ]


def run_rebalance(
    client: AlpacaClient,
    notify: Notify,
    dry_run: bool = False,
    as_of: dt.date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    store_path: Path = DEFAULT_STORE_PATH,
    limits: Limits | None = None,
    breaker: Breaker | None = None,
    min_rebalance_notional_usd: float = DEFAULT_MIN_REBALANCE_NOTIONAL_USD,
    limit_offset_pct: float = DEFAULT_LIMIT_OFFSET_PCT,
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    ledger: TradeLedger | None = None,
    snapshots: AccountSnapshotLog | None = None,
    live: bool = False,
    allocation: LiveAllocation | None = None,
    paper_snapshots: PaperSnapshotSource | None = None,
) -> int:
    """Run the full rebalance chain; returns the process exit code.

    0 = traded (or dry-ran, or nothing to trade, or operator halt);
    1 = aborted without trading (the reason is posted to Slack and stderr).

    ``live=True`` (US-013) reads the LIVE promoted slot, defaults ``limits``/
    ``breaker`` to the live guardrail files and live breaker state paths,
    scales the diff's equity by the live allocation pct (``allocation``, or
    allocation.live.json when None), and stamps ``rdq-live-`` order ids. The
    caller supplies a live-host client, a live-channel ``notify``, LIVE-
    database ``ledger``/``snapshots`` recorders, and ``paper_snapshots`` (a
    reader over the PAPER Account Snapshots database) for the same-day
    paper-vs-live comparison line (US-014).
    """
    if as_of is None:
        as_of = dt.datetime.now(MARKET_TZ).date()
    try:
        # 1. Market calendar: is as_of a trading day at all?
        assert_trading_day(client, as_of)

        # 2. Promoted strategy: the only thing that may ever trade. Live mode
        # reads the independent live slot — never the paper row.
        promoted = (load_promoted_strategy_live if live else load_promoted_strategy)(db_path)
        workspace = Path(promoted.workspace_path).expanduser()
        params = _strategy_params(promoted.config)

        # 3. One fresh broker snapshot feeds signal, diff, gate, and breaker.
        account = client.get_account()
        positions = client.get_positions()
        todays_orders = orders_submitted_on(
            client.list_orders(status="all", limit=_ORDER_LIST_LIMIT), as_of
        )

        # 4. Signal: pred.pkl -> fresh equal-weight targets (stale pred aborts).
        book = extract_targets(
            workspace,
            [p.symbol for p in positions],
            params=params,
            as_of=as_of,
            calendar_path=store_path.expanduser() / "calendars" / "day.txt",
        )

        # 4b. Universe sanity: targets outside the promoted universe_tickers
        # mean pred.pkl was generated from a different market than the
        # promotion recorded (the 2026-08-05 ai_deployers incident). Warn,
        # never abort — pred.pkl IS the backtested strategy; only the pinned
        # label diverged.
        universe_warnings = universe_divergence_warnings(promoted.config, book.weights)

        # 5. Diff: targets vs positions -> marketable-limit order list. Live
        # mode trades only its allocation slice: the equity the diff sizes
        # against is scaled by live_equity_allocation_pct / 100 (loaded at
        # call time via the module attr so tests can repoint the file). The
        # buying-power cap and the gate below keep the REAL account snapshot.
        prices = build_reference_prices(
            store_path, set(book.weights) | {p.symbol for p in positions}, positions
        )
        if live:
            if allocation is None:
                allocation = load_live_allocation(allocation_mod.ALLOCATION_PATH)
            diff_account = dataclasses.replace(
                account,
                equity=account.equity * allocation.live_equity_allocation_pct / 100.0,
            )
        else:
            diff_account = account

        # US-014: live summaries add the allocation pct in force and, when
        # the paper account also wrote a same-day snapshot row, a one-line
        # paper-vs-live day-P/L comparison. Resolved lazily at post time —
        # exactly one summary path runs per invocation, and dry runs (which
        # post no summary) never touch Notion or portfolio history.
        summary_day = as_of

        def _live_summary_kwargs() -> dict[str, Any]:
            if not live or allocation is None:
                return {}
            return {
                "allocation_pct": allocation.live_equity_allocation_pct,
                "paper_vs_live": paper_vs_live_pnl_line(paper_snapshots, client, summary_day),
            }

        diff = compute_orders(
            book.weights,
            diff_account,
            positions,
            prices,
            min_rebalance_notional_usd=min_rebalance_notional_usd,
            limit_offset_pct=limit_offset_pct,
        )

        # 5b. Buying-power cap: defer buys the snapshot's buying_power cannot
        # cover (Alpaca reserves qty*limit at accept, credits sells only on
        # fill) instead of 403ing mid-batch with orders already live.
        kept_orders, deferred = cap_buys_to_buying_power(diff.orders, account.buying_power)
        if deferred:
            diff = dataclasses.replace(
                diff, orders=kept_orders, skipped=[*diff.skipped, *deferred]
            )
        deferred_warnings = [*universe_warnings, *(skip.message for skip in deferred)]

        # 6. Gate: any rejected order aborts the whole batch. The rejection
        # notice is the day's summary (equity + every violated limit). Live
        # mode defaults to the live guardrail files and the live breaker
        # state paths (call-time module attrs, monkeypatchable in tests).
        if limits is None:
            limits = load_limits(order_gate_mod.LIVE_LIMITS_PATH) if live else load_limits()
        if breaker is None:
            breaker = (
                Breaker(
                    load_breaker_config(breaker_mod.LIVE_CONFIG_PATH),
                    halt_file=breaker_mod.LIVE_HALT_FILE,
                    high_water_mark_file=breaker_mod.LIVE_HWM_FILE,
                )
                if live
                else Breaker(load_breaker_config())
            )
        gate = evaluate_orders(diff.orders, account, positions, len(todays_orders), limits)
        if gate.rejections:
            rejection_messages = [r.message for r in gate.rejections]
            summary = format_daily_summary(
                as_of,
                account.equity,
                [],
                [],
                rejections=rejection_messages,
                no_trade_note="order gate rejected the batch — nothing submitted",
                breaker_state=breaker_state_line(breaker),
                warnings=universe_warnings
                + _record_snapshot(
                    snapshots,
                    client,
                    as_of,
                    account,
                    positions,
                    "gate_rejected",
                    breaker,
                    note="; ".join(rejection_messages),
                ),
                **_live_summary_kwargs(),
            )
            _safe_notify(notify, summary)
            print(summary, file=sys.stderr)
            return 1

        # 7. Breaker: halt file / daily notional / drawdown kill switch.
        trip = breaker.check(account.equity, day_traded_notional(todays_orders))
        if trip is not None:
            if trip.reason is BreakerReason.HALT_FILE:
                message = f"rebalance halted ({as_of}): {trip.message}"
                for warning in _record_snapshot(
                    snapshots, client, as_of, account, positions, "halted", breaker,
                    note=trip.message,
                ):
                    message += f"\nWARNING: {warning}"
                _safe_notify(notify, message)
                print(message)
                return 0
            summary = format_daily_summary(
                as_of,
                account.equity,
                [],
                [],
                rejections=[trip.message],
                no_trade_note="circuit breaker tripped — nothing submitted",
                breaker_state=breaker_state_line(breaker),
                warnings=universe_warnings
                + _record_snapshot(
                    snapshots, client, as_of, account, positions, "breaker_tripped",
                    breaker, note=trip.message,
                ),
                **_live_summary_kwargs(),
            )
            _safe_notify(notify, summary)
            print(summary, file=sys.stderr)
            return 1

        plan = format_plan(diff, as_of, dry_run)
        print(plan)
        if dry_run:
            return 0
        if not diff.orders:
            no_trade_note = (
                "no orders — every buy deferred for buying power"
                if deferred
                else "no orders — book already on target"
            )
            summary = format_daily_summary(
                as_of,
                account.equity,
                [],
                [],
                no_trade_note=no_trade_note,
                breaker_state=breaker_state_line(breaker),
                warnings=deferred_warnings
                + _record_snapshot(
                    snapshots, client, as_of, account, positions, "no_trade", breaker,
                    note=no_trade_note,
                ),
                **_live_summary_kwargs(),
            )
            _safe_notify(notify, summary)
            return 0

        # 8-9. Submit (ledger row per live order), then poll fills until
        # terminal or timeout, then record each final fill/rejection.
        submitted = submit_orders(
            client,
            diff.orders,
            as_of,
            ledger=ledger,
            id_prefix="rdq-live" if live else "rdq",
            account_label="live" if live else "paper",
        )
        final = poll_fills(
            client,
            [order.id for order in submitted],
            timeout_seconds=poll_timeout_seconds,
            interval_seconds=poll_interval_seconds,
            sleep=sleep,
        )
        if ledger is not None:
            by_id = {order.id: order for order in final}
            for order in submitted:
                ledger.record_final(by_id.get(order.id, order), as_of)
        summary = format_daily_summary(
            as_of,
            account.equity,
            submitted,
            final,
            ledger_failures=ledger.failures if ledger is not None else (),
            breaker_state=breaker_state_line(breaker),
            warnings=deferred_warnings
            + _record_snapshot(
                snapshots,
                client,
                as_of,
                account,
                positions,
                "traded",
                breaker,
                orders_placed=len(submitted),
                orders_filled=sum(1 for order in final if order.status == "filled"),
            ),
            **_live_summary_kwargs(),
        )
        _safe_notify(notify, summary)
        print(summary)
        return 0
    except _ABORT_ERRORS as exc:
        message = f"rebalance aborted ({as_of}): {exc}"
        _safe_notify(notify, message)
        print(message, file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected bug: tell the operator, then crash loudly
        _safe_notify(notify, f"rebalance CRASHED ({as_of}): {exc!r}")
        raise


def slack_notifier(live: bool = False) -> Notify:
    """Channel notifier from the repo Slack config (raises ConfigError if unset).

    ``live=True`` posts to SLACK_LIVE_CHANNEL_ID instead of the paper
    research channel — every live-mode notice must land in the live channel —
    and refuses (ConfigError) when that var is unset. Plain slack_sdk
    WebClient — no Bolt, no Socket Mode; the rebalancer only posts. Slack
    traffic must bypass the OneCLI proxy (NO_PROXY=slack.com in the service
    unit, per the orchestrator convention).
    """
    from slack_sdk import WebClient

    from orchestrator.config import ConfigError, load_slack_config

    config = load_slack_config()
    if live:
        channel = config.live_channel_id
        if not channel:
            raise ConfigError(
                "SLACK_LIVE_CHANNEL_ID is not set — the live rebalancer refuses to "
                "run without the live Slack channel for its notices"
            )
    else:
        channel = config.channel_id
    web_client = WebClient(token=config.bot_token)

    def notify(text: str) -> None:
        web_client.chat_postMessage(channel=channel, text=text)

    return notify


def stderr_notifier() -> Notify:
    """--no-slack fallback for supervised local runs: notices go to stderr."""

    def notify(text: str) -> None:
        print(f"[notify] {text}", file=sys.stderr)

    return notify


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nightly rebalance: promoted strategy -> orders -> fills (paper by default)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the full chain incl. gate and breaker, print the order list, submit nothing",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="trade the LIVE account: live promoted slot, live guardrails and breaker "
        "state, the live Alpaca host, the live Slack channel, rdq-live- order ids",
    )
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="YYYY-MM-DD trading day (default: today in America/New_York)",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_SECONDS)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="print notices to stderr instead of Slack (supervised local runs)",
    )
    parser.add_argument(
        "--no-notion",
        action="store_true",
        help="skip Trade Ledger writes (supervised local runs; live runs must record)",
    )
    args = parser.parse_args(argv)

    from orchestrator.config import ConfigError

    if args.live and args.no_notion:
        print(
            "ERROR: --live requires Notion recording — every live fill must land in "
            "Trade Ledger (Live); --no-notion is a paper-only escape hatch.",
            file=sys.stderr,
        )
        return 1

    if args.no_slack:
        notify = stderr_notifier()
    else:
        try:
            notify = slack_notifier(live=args.live)
        except ConfigError as exc:
            print(
                f"ERROR: {exc}\nRefusing to trade without a Slack channel for failure "
                "notices; pass --no-slack for a supervised local run.",
                file=sys.stderr,
            )
            return 1

    ledger: TradeLedger | None = None
    snapshots: AccountSnapshotLog | None = None
    paper_snapshots: PaperSnapshotSource | None = None
    if not args.no_notion:
        from orchestrator.notion_client import NotionClient
        from orchestrator.notion_recorder import RecorderConfigError, load_notion_databases

        try:
            databases = load_notion_databases()
        except RecorderConfigError as exc:
            hint = (
                "run ops/bootstrap_notion.py first — live runs must record"
                if args.live
                else "pass --no-notion for a supervised local run"
            )
            print(
                f"ERROR: {exc}\nRefusing to trade without a Trade Ledger to record "
                f"orders; {hint}.",
                file=sys.stderr,
            )
            return 1
        notion = NotionClient()
        if args.live:
            # US-014: live fills/snapshots go ONLY to the (Live) databases.
            # Missing live ids refuse outright — no --no-notion escape hatch.
            if not databases.trade_ledger_live or not databases.account_snapshots_live:
                print(
                    "ERROR: live Notion database ids are missing from "
                    "orchestrator/config.yaml (notion.databases.trade_ledger_live / "
                    "account_snapshots_live).\nRefusing to trade live without "
                    "Trade Ledger (Live) to record orders; run "
                    "ops/bootstrap_notion.py --live first.",
                    file=sys.stderr,
                )
                return 1
            from orchestrator.live_status import LiveStatusReader

            ledger = TradeLedger(notion, databases.trade_ledger_live)
            snapshots = AccountSnapshotLog(notion, databases.account_snapshots_live)
            # The same-day paper-vs-live summary line reads the PAPER
            # snapshots database (identical row schema; read-only).
            paper_snapshots = LiveStatusReader(notion, snapshots_db=databases.account_snapshots)
        else:
            ledger = TradeLedger(notion, databases.trade_ledger)
            snapshots = AccountSnapshotLog(notion, databases.account_snapshots)

    client = AlpacaClient(LIVE_BASE_URL, allow_live=True) if args.live else AlpacaClient()
    return run_rebalance(
        client,
        notify,
        dry_run=args.dry_run,
        as_of=args.as_of,
        db_path=args.db_path,
        store_path=args.store,
        poll_timeout_seconds=args.poll_timeout,
        poll_interval_seconds=args.poll_interval,
        ledger=ledger,
        snapshots=snapshots,
        live=args.live,
        paper_snapshots=paper_snapshots,
    )


if __name__ == "__main__":
    sys.exit(main())

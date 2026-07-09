"""Emergency flatten: cancel every open order, close every paper position (US-040).

Sequence (see ops/runbook.md for when to reach for this):

1. cancel all open orders (DELETE /v2/orders) and poll until none remain —
   Alpaca refuses to liquidate a symbol that still has an open order,
2. close every position (DELETE /v2/positions/{symbol}; a short is closed
   with a buy),
3. poll GET /v2/positions until it returns an empty list, and only then
   report success.

Exit codes: 0 = positions confirmed empty; 1 = flatten submitted but
positions were NOT confirmed empty within the timeout (the usual cause is a
closed market — the liquidation market orders sit unfilled until the next
open; rerun the script after the open to confirm); 2 = the flatten itself
could not run (auth/HTTP failure).

NOTE for reconciliation: the liquidation orders this script submits are
placed outside the rebalancer, so they have no Trade Ledger rows —
ops/reconcile.py will (correctly) flag them as missing ledger rows for this
date. That is the audit trail working, not a bug; note the flatten in the
Decision Log.

Run through the OneCLI proxy (paper credentials are injected; never in code):

    onecli run --agent rdq-exec-paper -- .venv/bin/python -m ops.flatten
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from execution.alpaca_client import AlpacaClient, AlpacaError

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
# GET /v2/orders page cap; one page is plenty — the rebalancer never has
# more than a handful of working orders.
OPEN_ORDERS_LIMIT = 500


def _poll_until_empty(
    fetch_remaining: Callable[[], list[str]],
    what: str,
    out: Callable[[str], None],
    sleep: Callable[[float], None],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> list[str]:
    """Poll ``fetch_remaining`` until it returns [] or the timeout elapses.

    Returns the final remaining list ([] = confirmed empty). Time is counted
    in poll intervals rather than wall clock so tests can inject a no-op
    sleep.
    """
    attempts = max(1, int(timeout_seconds / poll_interval_seconds))
    remaining = fetch_remaining()
    for _ in range(attempts):
        if not remaining:
            return []
        out(f"waiting: {len(remaining)} {what} remaining ({', '.join(sorted(remaining))})")
        sleep(poll_interval_seconds)
        remaining = fetch_remaining()
    return remaining


def run_flatten(
    client: AlpacaClient,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> int:
    """Cancel + close + verify. Returns the process exit code."""
    # Step 1: cancel every open order (liquidation is refused while a
    # symbol has a working order).
    open_orders = client.list_orders(status="open", limit=OPEN_ORDERS_LIMIT)
    if open_orders:
        for order in open_orders:
            out(f"open order {order.id}: {order.side} {order.qty} {order.symbol}")
        cancelled = client.cancel_all_orders()
        out(f"cancel requested for {len(cancelled)} order(s)")
        still_open = _poll_until_empty(
            lambda: [o.id for o in client.list_orders(status="open", limit=OPEN_ORDERS_LIMIT)],
            "open order(s)",
            out,
            sleep,
            timeout_seconds,
            poll_interval_seconds,
        )
        if still_open:
            out(
                f"FAIL: {len(still_open)} order(s) still open after {timeout_seconds:g}s; "
                "not submitting liquidations against symbols with working orders"
            )
            return 1
        out("all open orders cancelled")
    else:
        out("no open orders")

    # Step 2: liquidate every position.
    positions = client.get_positions()
    if positions:
        for position in positions:
            close_order = client.close_position(position.symbol)
            out(
                f"closing {position.symbol}: {position.side} {position.qty} -> "
                f"{close_order.side} order {close_order.id} ({close_order.status})"
            )
    else:
        out("no open positions")

    # Step 3: confirm flat via GET /v2/positions.
    remaining = _poll_until_empty(
        lambda: [p.symbol for p in client.get_positions()],
        "position(s)",
        out,
        sleep,
        timeout_seconds,
        poll_interval_seconds,
    )
    if remaining:
        out(
            f"FAIL: positions NOT confirmed empty after {timeout_seconds:g}s: "
            f"{', '.join(sorted(remaining))}. If the market is closed the liquidation "
            "orders fill at the next open — rerun this script after the open to confirm."
        )
        return 1
    out("OK: /v2/positions confirmed empty — account is flat")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cancel all open orders and close all paper positions."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="how long to wait for cancels/liquidations to settle (default %(default)s)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="seconds between settlement polls (default %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.poll_interval_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds and --poll-interval-seconds must be positive")

    try:
        return run_flatten(
            AlpacaClient(),
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except AlpacaError as exc:
        print(f"flatten failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

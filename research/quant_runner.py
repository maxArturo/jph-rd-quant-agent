"""fin_quant CLI wrapper that seeds the loop's user instruction (directive).

Why: plain CLI runs (`rdagent fin_quant`, the GPU-worker path) have no
user-interaction channel — `QuantRDLoop._interact_init_params` only runs when
the IPC queues exist, so CLI runs ignore the directive entirely.

This wrapper replicates ``rdagent.app.qlib_rd_loop.quant.main`` for a FRESH
run and seeds ``loop.plan["user_instruction"]`` before the loop starts — the
exact key the proposal step reads (`components/proposal/__init__.py`,
``plan.get("user_instruction")``). The pinned upstream tree stays unmodified
(research/CLAUDE.md rule).

Usage (normally via ops/run_us_quant.sh with RDQ_USER_INSTRUCTION set):

    .venv/bin/python -m research.quant_runner --loop-n 10 \
        --user-instruction "Focus on volume/price divergence factors"
"""

from __future__ import annotations

import argparse
import asyncio


def build_loop(user_instruction: str | None):
    """Fresh QuantRDLoop with the instruction seeded into its plan."""
    # Heavy import (full rdagent app graph) — keep it out of module import
    # time so offline tests can stub this function.
    from rdagent.app.qlib_rd_loop.quant import QUANT_PROP_SETTING, QuantRDLoop

    loop = QuantRDLoop(QUANT_PROP_SETTING)
    if user_instruction:
        loop.plan["user_instruction"] = user_instruction
    return loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-n", type=int, default=None)
    parser.add_argument("--all-duration", default=None)
    parser.add_argument("--user-instruction", default=None)
    args = parser.parse_args(argv)

    loop = build_loop(args.user_instruction)
    asyncio.run(loop.run(step_n=None, loop_n=args.loop_n, all_duration=args.all_duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

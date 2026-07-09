# Notion database schemas

The system's durable record lives in five Notion databases under one parent
page ("Automated AI Quant Investment",
`3979b1a4-36cf-8046-baa5-cc14c1ca7665`). `ops/bootstrap_notion.py` creates
them (idempotently, matched by title) and writes their IDs into
`orchestrator/config.yaml`. The property schemas below are the source of
truth — the bootstrap script's `database_properties()` spec must match this
document (tests/test_bootstrap_notion.py cross-checks the property names).

Everything a run produced should be reconstructable from Notion alone
(US-027); every order and fill should be auditable from the Trade Ledger
(US-035, US-037).

## One writer per database

Each database has exactly ONE writing component. Anything else may read, but
never writes — this keeps write paths testable, makes reconciliation
meaningful (a ledger row can only have come from the rebalancer), and avoids
concurrent-edit conflicts (Notion 409s on concurrent saves).

| Database         | Sole writer                                        |
| ---------------- | -------------------------------------------------- |
| Research Ideas   | orchestrator conversation core (directive tools)   |
| Hypothesis Log   | orchestrator hypothesis poller                     |
| Backtest Results | orchestrator poller (run-completion path)          |
| Decision Log     | orchestrator operator tools (promote, halt/resume) |
| Trade Ledger     | execution rebalancer                               |

Humans may add comments or extra pages in the workspace, but must not edit
rows in these databases — treat them as append-mostly logs owned by code.

## Research Ideas

One row per research directive (one per Slack thread).

| Property  | Type      | Notes                                                         |
| --------- | --------- | ------------------------------------------------------------- |
| Idea      | title     | Short human title for the idea                                 |
| Raw Idea  | rich_text | The operator's original message, unedited                      |
| Directive | rich_text | Refined directive (objective + constraints) saved by the bot   |
| Universe  | rich_text | Universe name the run uses (us_liquid or a custom name)        |
| Status    | select    | proposed / researching / stopped / completed / failed / promoted |
| Thread    | url       | Slack permalink to the owning thread                           |
| Thread TS | rich_text | Slack thread_ts — join key to the SQLite `runs` table          |

## Hypothesis Log

One row per hypothesis the research loop proposed, plus the operator's action
on it. Linked to its idea.

| Property        | Type      | Notes                                                        |
| --------------- | --------- | ------------------------------------------------------------ |
| Hypothesis      | title     | One-line hypothesis text                                      |
| Idea            | relation  | → Research Ideas                                              |
| Details         | rich_text | Full hypothesis payload (reason, spec) as posted to Slack     |
| Action          | select    | pending / approved / edited / rejected / auto_approved / cancelled |
| Operator Input  | rich_text | Operator's edit text (empty unless Action = edited)           |
| Interaction Key | rich_text | server_ui interaction key — join to `pending_interactions`    |

## Backtest Results

One row per completed experiment/loop with its headline metrics.

| Property   | Type      | Notes                                              |
| ---------- | --------- | -------------------------------------------------- |
| Experiment | title     | e.g. "loop 3 — momentum factors"                    |
| Idea       | relation  | → Research Ideas                                    |
| IC         | number    | from qlib_res.csv                                   |
| ICIR       | number    |                                                     |
| Rank IC    | number    |                                                     |
| ARR        | number    | annualized return (excess, with cost)               |
| IR         | number    | information ratio                                   |
| MDD        | number    | max drawdown (negative)                             |
| Sharpe     | number    | derived from ret.pkl (qlib logs no Sharpe — US-022) |
| SOTA       | checkbox  | loop marked state-of-the-art by RD-Agent            |
| Workspace  | rich_text | workspace path holding pred.pkl / artifacts         |
| Universe   | rich_text | universe the backtest ran on                        |

## Decision Log

One row per deliberate operator/system decision that changes what trades.

| Property   | Type      | Notes                                             |
| ---------- | --------- | ------------------------------------------------- |
| Decision   | title     | e.g. "Promote momentum-v2 to paper trading"        |
| Type       | select    | promotion / halt / resume / universe / other       |
| Details    | rich_text | What was decided and why (config, metrics quoted)  |
| Idea       | relation  | → Research Ideas (when the decision concerns one)  |
| Decided At | date      | when the decision took effect                      |

## Trade Ledger

One row per order the rebalancer submitted, updated with its terminal fill or
rejection. Reconciled against Alpaca order history by `ops/reconcile.py`.

| Property         | Type      | Notes                                                    |
| ---------------- | --------- | -------------------------------------------------------- |
| Order            | title     | human line, e.g. "2026-07-09 BUY 10 AAPL"                 |
| Order ID         | rich_text | Alpaca order id — reconciliation key                      |
| Symbol           | rich_text |                                                           |
| Side             | select    | buy / sell                                                |
| Qty              | number    | submitted share quantity                                  |
| Limit Price      | number    | marketable-limit price                                    |
| Status           | select    | submitted / filled / partially_filled / rejected / cancelled / expired |
| Filled Qty       | number    |                                                           |
| Filled Avg Price | number    |                                                           |
| Submitted At     | date      |                                                           |
| Notes            | rich_text | gate/breaker context, rejection reasons                   |

## Conventions

- Relations are single-direction (`single_property`) so the target database
  gets no synced back-reference property — Research Ideas' schema stays
  exactly as written here regardless of how many databases point at it.
- Select option sets above are the initial vocabulary; Notion auto-creates
  new options on write, so adding a status later needs no schema migration.
- Database IDs are configuration, not secrets: they live in
  `orchestrator/config.yaml` (committed). Auth is injected by the OneCLI
  proxy (connector integration — see docs/decisions.md 2026-07-08).

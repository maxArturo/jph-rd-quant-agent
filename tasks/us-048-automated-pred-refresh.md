# US-048: Automated daily prediction refresh for the promoted strategy

**Status:** implemented (2026-07-20) — see docs/decisions.md 2026-07-20 entry.
`execution/pred_refresh.py` + promote-time snapshot in
`orchestrator/promotion.py` + `ops/rdq-pred-refresh.{service,timer}` (06:45
ET, enabled). Remaining: the 5-consecutive-green-days milestone check below.

**Description:** As an operator, I want the promoted strategy's predictions
regenerated automatically every trading morning, so the nightly rebalance
trades unattended instead of aborting "predictions stale" every day after
promotion.

## Context / why now

- The freshness gate (`execution/signal.py::assert_fresh`) requires the pred
  cross-section to be >= the store's last trading day. `rdq-data-refresh.timer`
  (10:30 UTC) advances the store every morning, so a promoted workspace's
  static `pred.pkl` goes stale one day after any refresh — 2026-07-15's
  rebalance aborted exactly this way after 2026-07-14 traded on a manually
  refreshed pred.
- PLAN.md line 189 always intended this step ("re-run the promoted workspace
  with `test_end=today`"); it is the last unimplemented piece of the US-020
  "unattended paper trading" milestone.
- The manual procedure has now worked twice (2026-07-14 and 2026-07-15, see
  docs/decisions.md and the pred-refresh runbook steps below) and is
  mechanical enough to automate.

## Design decisions to confirm at implementation

1. **Re-train vs re-predict.** The proven path (docker `qrun` on a
   SignalRecord-only conf) RE-FITS the model daily (~13 min GRU CPU) — the
   traded model is a fresh stochastic re-fit, not the exact promoted weights.
   The alternative (load the recorder's `params.pkl`, call `model.predict`
   on a dataset extended to today) keeps the promoted weights exactly, which
   is arguably what "promote" means, but needs per-model-class plumbing.
   **Recommendation:** ship the proven re-fit first (accept drift, log the
   new run's IC in the summary), leave exact-weights re-predict as a
   follow-up story.
2. **Where it runs.** A separate `rdq-pred-refresh.service` + timer between
   data refresh and rebalance (e.g. 10:45 UTC) keeps the rebalancer
   deterministic-and-fast and isolates docker failures; putting it inside
   `rebalance.py` couples a 13-min GPU-less train to the trading path.
   **Recommendation:** separate ExecStart chained via the timer, with the
   rebalance's existing stale-pred abort as the backstop when the refresh
   fails.

## Acceptance criteria

- [x] Promotion snapshots everything inference needs: at promote time, copy
      the SOTA conf to `conf_pred_refresh.yaml` (records reduced to
      SignalRecord only) AND persist the rendered jinja context (the
      "Render the template with the context" dict + `num_features` from the
      training log) alongside it — no log archaeology at refresh time.
      Backfill for the currently promoted workspace (5d19a1fb…, done
      manually 2026-07-14).
- [x] `execution/pred_refresh.py` (or `ops/`): loads the promoted strategy
      (reuse `load_promoted_strategy()` — refuse when none), renders/env-passes
      the stored context with `test_end=<today NY>`, runs the workspace's
      `conf_pred_refresh.yaml` via the local_qlib docker image with the same
      mounts as QTDockerEnv (`-v WS:/workspace/qlib_workspace -v ~/.qlib:/root/.qlib`,
      `PYTHONPATH`, `MLFLOW_ALLOW_FILE_STORE=true`), then `chmod -R 777` the
      new mlruns entries (QTDockerEnv parity).
- [x] Post-run self-check: locate the newest pred.pkl (`signal.locate_pred`)
      and assert `assert_fresh` passes for today before exiting 0; exit
      nonzero + Slack notify otherwise (same notifier conventions as the
      rebalancer, `--no-slack` for supervised runs).
- [x] `ops/rdq-pred-refresh.{service,timer}`: runs after
      `rdq-data-refresh.service` and comfortably before the 12:00 UTC
      rebalance; skips cleanly (exit 0, "no promoted strategy" notice) when
      nothing is promoted; failure alerting matches the other rdq units.
- [x] The SignalRecord-only conf sidesteps the qlib end-of-calendar
      IndexError (no PortAnaRecord backtest), so `test_end=today` is safe
      even when the store ends yesterday — regression-tested with a tmp-dir
      store fixture (reuse `write_bins`/`write_calendar` helpers).
- [x] Old mlruns pred runs in the promoted workspace are pruned or bounded
      (newest-mtime-wins already picks the fresh one; avoid unbounded disk
      growth — coordinate with the rdq-sweep retention job).
- [ ] Milestone check: 5 consecutive trading days with zero manual steps —
      data refresh → pred refresh → rebalance all green in Slack.

## Non-goals

- Exact-weights re-predict from `params.pkl` (follow-up if re-fit drift
  proves problematic).
- Refreshing non-promoted workspaces.
- Any change to the freshness gate semantics — it stays the backstop.

# PRD: Loop Hardening — Auto-Promotion Gate, Run Memory, Honest Evaluation, Ops Fixes, server_ui Removal

## Introduction

The 2026-08-14 repo audit found that rd-agent-q executes single research runs and nightly trading well, but falls short of its goal — *continuously surfacing best-of-breed trading strategies* — in three ways: nothing promotes automatically or audits promotions, every run starts with zero memory of prior runs, and "best" is measured against a burned-in test window on a look-ahead-biased universe with no feedback from realized performance. Separately, several operational safety nets are broken or missing (the GPU billing watchdog has never worked, a second concurrent run destroys the first's droplet, no unit failure alerts), and the legacy server_ui stack is dead weight.

This PRD hardens the loop end to end. Research proposal cadence is already handled (the trusted Claude bot proposes new research every weekday via Slack); this PRD makes what happens *after* a run completes autonomous, honest, and auditable.

Decisions locked in with the operator (2026-08-14):

- **Auto-promote on gate pass.** When a completed run's candidate passes the codified comparison gate, the system promotes it without waiting for a human. Slack is notified after the fact with the full comparison and rollback instructions.
- **Rolling TEST_END + confirmation window** replaces the fixed 2026-07-10 test end.
- **Alert + auto-halt** (via the existing breaker halt file) when live/paper performance severely diverges from backtest expectation. No auto-rollback.
- **Point-in-time ADV universe filter now**; delisted-names backfill is a documented gap for a future PRD.
- **No off-box backup work** — daily DigitalOcean droplet snapshots of this box cover disaster recovery.
- **Remove the server_ui stack entirely.** The operator never used the web UI and prefers Notion reporting.

## Goals

- Every promotion (auto, conversational, or CLI) is recorded in a promotion history with a working rollback command, and writes a Notion Decision Log row.
- A completed GPU run is compared to the incumbent by code, not eyeballs: parity (window, universe, costs, selection params) is *enforced*, the pass/fail criteria are explicit and configurable, and a pass auto-promotes.
- Each new run receives a digest of prior runs (directives, hypotheses tried, outcomes, current incumbent) so it stops re-deriving rejected ideas.
- Candidate selection can no longer overfit a fixed test window: TEST_END rolls forward per run, and promotion requires confirmation on a recent holdout slice the hypothesis search never saw.
- A daily tracker compares realized paper/live performance to the promoted strategy's backtest expectation, alerts on drift, and halts trading on severe breach.
- The `us_liquid` universe is built point-in-time (per-date membership spans), removing the "liquid today ⇒ in sample since 2016" look-ahead.
- The known operational holes are closed: watchdog PATH, GPU run mutual exclusion, unit failure alerts, scheduled health/reconcile, absolute pred staleness bound, SQLite WAL, env file permissions, automatic GPU snapshot use, stale run-row reaper.
- The server_ui stack (service, Flask UI, poller, Block Kit approval flow) is deleted; docs describe only the GPU path.

## User Stories

Stories are ordered so each phase is independently shippable. Phase A (audit + gate) should land before Phase C's gate extensions.

---

### Phase A — Promotion audit trail and comparison gate

### US-001: Promotion history table and rollback command
**Description:** As an operator, I want every promotion appended to a history table and a one-command rollback, so a bad promotion is reversible without hand-editing SQLite.

**Acceptance Criteria:**
- [ ] New `promotion_history` table in `orchestrator/state.py` (append-only: workspace_path, config JSON, promoted_at, source ∈ {auto_gate, conversation, cli}, gate_verdict JSON nullable, replaced_workspace nullable). `set_promoted_strategy` appends a row on every call; the existing single-row `promoted_strategy` table remains the "current" pointer.
- [ ] Migration is idempotent (guarded `CREATE TABLE`, same pattern as existing `migrate()`); existing DBs migrate cleanly — the current promoted row is backfilled as history row 1 with source `cli`.
- [ ] New CLI `python -m ops.rollback_promotion` re-promotes the previous history entry (or `--to <workspace>`), refuses with a clear message if the target workspace directory no longer exists, and itself appends a history row.
- [ ] `ops/sweep.py` protects the workspaces of the last 3 promotion-history entries from deletion.
- [ ] `make check` passes (ruff + pyright + pytest).
- [ ] Tests: history appended on promote, rollback restores previous pointer, rollback refuses missing workspace, sweep protects history workspaces.

### US-002: Codified comparison gate (pure module)
**Description:** As an operator, I want candidate-vs-incumbent comparison to be a pure, tested function with explicit criteria, so "beats the incumbent" is defined by code, not judgment.

**Acceptance Criteria:**
- [ ] New module `ops/promotion_gate.py` with a pure function taking candidate and incumbent metric bundles (each: qlib_res metrics, ret.pkl-derived daily returns, window first/last day, market name, instrument-list hash, topk/n_drop, cost params) and returning a verdict object: `parity_ok`, `pass`, and per-criterion reasons.
- [ ] Parity is **enforced, not advisory**: mismatched test window, market, instrument-list hash, topk/n_drop, or cost params ⇒ `parity_ok=False`, `pass=False`, with the specific mismatch named. (This supersedes the warn-only behavior in `ops/gpu_pipeline.py:_incumbent_lines`.)
- [ ] Default pass criteria (all must hold), values read from `orchestrator/config.yaml` under a new `promotion_gate:` section: candidate information ratio > incumbent IR × **1.05** (5% relative margin — filters noise-level wins that would churn the promoted strategy; margin configurable); candidate MDD no worse than incumbent MDD × 1.25; candidate IC > 0. No promotion when there is no incumbent unless `--allow-first` / config flag is set.
- [ ] Verdict serializes to JSON (stored in `promotion_history.gate_verdict`) and renders to a Slack-ready text block listing each criterion as pass/fail with both values.
- [ ] `make check` passes; unit tests cover: pass, each criterion failing individually, each parity mismatch, missing incumbent.

### US-003: Auto-promotion in the GPU pipeline
**Description:** As an operator, I want a gate-passing candidate promoted automatically at the end of a GPU run, so the loop closes without me clicking anything.

**Acceptance Criteria:**
- [ ] `ops/gpu_pipeline.py` runs the gate after the final summary: on `pass=True` it promotes the fetched candidate workspace via the same code path as `ops/promote_fetched.py` (including the pred-refresh snapshot), with `source=auto_gate` and the verdict stored in history.
- [ ] Snapshot failure blocks auto-promotion (unlike the current warn-and-promote-anyway behavior in `orchestrator/promotion.py`) — a strategy that cannot refresh predictions must not be auto-promoted; the failure is posted to Slack and the run finalizes unpromoted.
- [ ] The Slack final summary includes the full gate verdict block; on promotion it also names the replaced workspace and the exact rollback command; on fail it states which criteria failed.
- [ ] Gate fail or gate error never fails the run: the run finalizes `completed` with the verdict posted.
- [ ] A config kill-switch (`promotion_gate.auto_promote: false`) reverts to report-only mode.
- [ ] `make check` passes; tests cover promote-on-pass, no-promote-on-fail, no-promote-on-parity-mismatch, no-promote-on-snapshot-failure, kill-switch.

### US-004: All promotion paths write the audit trail
**Description:** As an operator, I want CLI and conversational promotions to leave the same records as auto-promotions, so there is one place to answer "what was promoted, when, why, replacing what".

**Acceptance Criteria:**
- [ ] `ops/promote_fetched.py` writes the Notion Decision Log row it currently skips (workspace, key metrics, replaced workspace, gate verdict if computed) and appends to `promotion_history` with `source=cli`. It runs the gate in advisory mode and prints the verdict; `--force` promotes despite a failing verdict but records that in history.
- [ ] The conversational promote path (`orchestrator/promotion.py`) appends to `promotion_history` with `source=conversation` and includes the incumbent's headline metrics (IC / ARR / MDD) and window in its confirmation text — closing the "blind Slack promotion" gap.
- [ ] Notion Decision Log rows include the gate verdict summary line for all three sources.
- [ ] `make check` passes; tests cover Decision Log payload from `promote_fetched`, history rows from both paths, `--force` recording.

---

### Phase B — Run memory

### US-005: Machine-readable run summaries on Notion + digest builder
**Description:** As a researcher (the LLM in the loop), I want a compact digest of prior runs — what was tried, what won, what was rejected, what is currently promoted — read back from the Notion strategy-notes DB, so I stop re-proposing dead ideas and the human-readable record and the machine memory are the same rows.

**Acceptance Criteria:**
- [ ] `ops/notion_summary.py` enriches each run's strategy-note row with a machine-readable `run_summary` JSON: directive objective, hypotheses tried with per-hypothesis outcome (SOTA/rejected/failed), winner metrics (IC/ARR/MDD/IR), test + confirmation windows, universe + instrument-list hash, run status. Stored on the page as a fenced `json` code block (chunked to Notion's ~2000-char rich-text element limit) or a dedicated property — **not** a file attachment (Notion API attachment handling is unreliable through the gateway).
- [ ] New module `orchestrator/run_memory.py` with `build_digest(...) -> str`: queries the strategy-notes DB (most recent N=10 rows), parses each row's `run_summary` JSON, and composes the digest; the incumbent section (promoted workspace's factors, model, headline metrics, window, promoted_at) comes from local state.
- [ ] Fallback chain, never raising and never stalling a launch: Notion unreachable or row missing its JSON ⇒ that run degrades to local `runs`/`directives` data (directive + status only); total Notion budget for digest building ≤ 15s, after which it proceeds with whatever it has; empty history ⇒ short "no prior runs" string.
- [ ] Digest is deterministic given the same inputs, truncates oldest-first at `max_chars` (default 4000).
- [ ] `make check` passes; tests (mocked Notion) cover: normal digest, JSON round-trip through chunked code blocks, missing-JSON degradation, Notion-down fallback, truncation, empty state.

### US-006: Inject digest and incumbent into the run instruction
**Description:** As a researcher, I want the digest prepended to my hypothesis-generation context, so each run builds on the last instead of starting blind.

**Acceptance Criteria:**
- [ ] `GpuBackend.launch` (or its caller in `ConversationCore.start_research`) composes `RDQ_USER_INSTRUCTION` as: operator directive + delimiter + run-history digest, within the existing base64 env mechanism. The directive always comes first and is never truncated in favor of the digest.
- [ ] The composed instruction survives the full path to the hypothesis prompt (`research/quant_runner.py` → `loop.plan["user_instruction"]`) unchanged — verified by a test on the composition function and a launch-args test on `GpuBackend`.
- [ ] Digest injection is skippable per-run (`start_research` tool arg `include_memory: false`, default true) for clean-slate experiments.
- [ ] The Slack run-start message notes that N prior runs were included as context.
- [ ] `make check` passes.

---

### Phase C — Honest evaluation

### US-007: Rolling TEST_END with a reserved confirmation window
**Description:** As an operator, I want each run's test window to roll forward with the data store, holding back a recent slice the hypothesis search never sees, so candidates cannot be selected by overfitting one fixed window.

**Acceptance Criteria:**
- [ ] `ops/gpu_pipeline.py` computes dates at launch instead of relying on the hardcoded default in `ops/run_us_quant.sh`: `CONFIRM_DAYS` (default 42 trading days, configurable) are reserved, so `RDQ_TEST_END = store calendar end − CONFIRM_DAYS`; the confirmation window is `(RDQ_TEST_END, store end]`.
- [ ] The computed TEST_END and confirmation window are recorded in `pipeline_status.json` and shown in the run-start Slack message.
- [ ] `ops/run_us_quant.sh` keeps a hardcoded fallback but the fallback now fails loudly (exit non-zero with message) if the resulting test end is more than 90 days behind the store end — preventing silent regression to a stale window.
- [ ] Existing date-ordering validation (`check_dates`) still passes; `tests/test_services.py` date-sync test updated accordingly.
- [ ] `make check` passes.

### US-008: Confirmation-window evaluation in the gate
**Description:** As an operator, I want the gate to also compare candidate and incumbent on the reserved confirmation window, using exact-weights re-prediction, so promotion requires out-of-search-sample evidence.

**Acceptance Criteria:**
- [ ] Reusing the pred-refresh machinery (`execution/pred_refresh.py` exact-weights re-predict, per US-049), the gate re-predicts **both** candidate and incumbent over the confirmation window and computes portfolio daily returns with the shared backtest params (topk/n_drop/costs).
- [ ] Gate pass additionally requires: candidate confirmation-window IR > incumbent confirmation-window IR — strictly greater, **no margin** on this leg (the 42-day window is too short for a margin to be meaningful; configurable under `promotion_gate:`).
- [ ] Confirmation results (both windows, both strategies) appear in the Slack verdict block and the Notion write-up.
- [ ] If confirmation evaluation fails technically (re-predict error), the gate returns `pass=False` with reason `confirmation_unavailable` — it never silently skips the check.
- [ ] `make check` passes; tests cover confirmation pass, confirmation fail, technical failure ⇒ no promotion.

### US-009: Live-vs-backtest divergence tracker with auto-halt
**Description:** As an operator, I want a daily comparison of realized paper/live performance against the promoted strategy's backtest expectation, with a Slack alert on drift and an automatic trading halt on severe breach, so a decayed strategy cannot keep trading unnoticed.

**Acceptance Criteria:**
- [ ] New module `execution/divergence.py` + `rdq-divergence.{service,timer}` (Mon–Fri, after `rdq-rebalance.service`, e.g. 16:30 ET): loads Alpaca portfolio history since `promoted_at`, loads the promoted workspace's backtest daily-return distribution from `ret.pkl` (σ from the **full test window**; expectation μ_adj = **0.5 × backtest mean** — the haircut acknowledges the documented survivorship inflation of backtest returns; haircut factor configurable under `divergence:`), computes trailing 20-trading-day realized return vs expected `20·μ_adj`, and z-score `(realized − 20·μ_adj) / (σ·√20)`.
- [ ] Alert: z < −2 posts a Slack warning with realized vs expected numbers and the drawdown since promotion.
- [ ] Severe breach: z < −3 **or** drawdown since promotion > backtest MDD × 1.25 ⇒ writes the existing breaker halt file (same file `execution/breaker.py` checks) and posts a `:rotating_light:` Slack notice naming the trigger and the manual clear procedure. Thresholds configurable under a `divergence:` config section.
- [ ] Fewer than 20 trading days since promotion ⇒ posts nothing, exits 0 (warmup).
- [ ] Tracker never trades and never modifies state other than the halt file; failures post a Slack error notice (same pattern as rebalance/pred-refresh).
- [ ] Unit installed by `ops/install_services.sh`, covered by `tests/test_services.py`.
- [ ] `make check` passes; tests cover z-score math, alert threshold, halt threshold, warmup, halt-file write.

### US-010: Point-in-time ADV universe filter
**Description:** As an operator, I want `us_liquid` membership computed per-date from trailing liquidity, so backtests stop admitting names by *today's* liquidity retroactively across a decade.

**Acceptance Criteria:**
- [ ] `data/make_universe.py` gains a point-in-time mode (default for `us_liquid`): for each ticker, membership spans are the date ranges where trailing 20-day ADV ≥ threshold and price ≥ threshold (evaluated monthly to keep spans stable; entry/exit hysteresis of one re-evaluation period to avoid churn).
- [ ] Output uses qlib's instruments span format (`SYMBOL\tSTART\tEND`, multiple rows per symbol allowed); `data/refresh.py`'s universe carry-across preserves spans.
- [ ] The legacy last-window mode remains available behind a flag for the frozen/promoted universe snapshots.
- [ ] Regenerating `us_liquid` does **not** touch `us_liquid_promoted_30.txt` or any promoted-pinned snapshot (existing behavior preserved and now covered by a test).
- [ ] A one-off comparison note is added to `docs/decisions.md`: same incumbent workspace backtested on last-window vs PIT universe, with the headline metric deltas, so the bias magnitude is on record.
- [ ] `make check` passes; tests cover span computation, hysteresis, span-format output, refresh carry-across.

### US-011: Document the delisted-names gap
**Description:** As an operator, I want the remaining survivorship bias documented where decisions are made, so numbers are read with the right caveat until the delisted backfill lands.

**Acceptance Criteria:**
- [ ] `docs/decisions.md` entry describing: what the PIT filter fixed, what remains (no delisted/acquired/bankrupt names in the store, no terminal-return modeling), the expected direction of the bias (backtest ARR flattered), and the sketch of the future fix (FMP delisted endpoint validation, expected-end ticker state, terminal-return modeling, symbol-reuse dedup).
- [ ] The gate's Slack verdict block and the Notion write-up template each carry a one-line standing caveat referencing this entry.

---

### Phase D — Mechanical ops fixes

### US-012: Fix the GPU billing watchdog PATH
**Description:** As an operator, I want the hourly watchdog to actually run `doctl`, so an orphaned GPU droplet cannot bill indefinitely.

**Acceptance Criteria:**
- [ ] `ops/rdq-gpu-watchdog.service` sets `Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin` (matching the injection `orchestrator/gpu_backend.py` already does for pipeline units).
- [ ] `ops/gpu_watchdog.py` fails loudly if `doctl` is not found: posts a Slack notice (it currently crashes before any notify is reachable).
- [ ] `tests/test_services.py` asserts the PATH line on the watchdog unit **and** asserts `gpu_backend.py`'s PATH injection for pipeline units (regression guard for the class of bug).
- [ ] Verified on-box: `systemctl --user start rdq-gpu-watchdog.service` exits 0 with a droplet-state check in the journal (no `FileNotFoundError`).
- [ ] `make check` passes.

### US-013: Global GPU run mutual exclusion
**Description:** As an operator, I want a second research run refused while one is active, so it can never destroy the first run's droplet (today, pipeline #2's provision failure triggers `destroy --force` on run #1's droplet).

**Acceptance Criteria:**
- [ ] A global lock (e.g. `~/rdq-runs/gpu_worker/run.lock` containing thread_ts + unit name, or a check that any `rdq-gpu-run-*` transient unit is active) is acquired by `ops/gpu_pipeline.py` at start and released in `finally`.
- [ ] `ConversationCore.start_research` checks the global lock (not just this thread's run) and refuses with a message naming the active run's thread — replacing the false "queued behind it" text in the duplicate-run message.
- [ ] `stop_run` only acts when the requesting thread owns the lock; otherwise it refuses and names the owning thread.
- [ ] A stale lock (owning unit no longer active) is broken automatically with a Slack note.
- [ ] `make check` passes; tests cover acquire/refuse/release, stale-lock break, stop-run ownership.

### US-014: Unit failure alerts (OnFailure=)
**Description:** As an operator, I want any rdq unit start-failure or crash to reach Slack, so a silently failed data refresh can't feed stale signals downstream.

**Acceptance Criteria:**
- [ ] New templated unit `rdq-notify-failure@.service` that posts "unit %i failed" plus the last ~10 journal lines to the ops Slack channel (via the existing notify mechanism, through the OneCLI proxy).
- [ ] Every `ops/rdq-*.service` gains `OnFailure=rdq-notify-failure@%n.service`.
- [ ] `ops/install_services.sh` installs the template; `tests/test_services.py` asserts the `OnFailure=` line on every service.
- [ ] Verified on-box with a deliberately failing test unit invocation.
- [ ] `make check` passes.

### US-015: Schedule health check and fill reconciliation
**Description:** As an operator, I want `health.sh` and `ops/reconcile.py` on timers, so the audit trail stops being structurally wrong (pre-open orders recorded as `submitted` forever) and unit rot is detected without a human typing commands.

**Acceptance Criteria:**
- [ ] `rdq-reconcile.{service,timer}`: Mon–Fri after market close (e.g. 16:15 ET), runs `ops.reconcile` in a mode that updates Notion Trade Ledger rows to final fill status/qty/price and posts a one-line Slack summary when discrepancies were found (silent when clean).
- [ ] `rdq-health.{service,timer}`: daily, runs `ops/health.sh`; non-zero exit or failed-unit findings post to Slack (leverages US-014's notifier or posts directly).
- [ ] Both installed by `install_services.sh`, asserted by `tests/test_services.py` (calendar, ordering, persistence semantics).
- [ ] `make check` passes.

### US-016: Absolute pred staleness bound
**Description:** As an operator, I want the rebalancer to refuse to trade when the data store itself is stale, so a frozen store calendar can't self-certify week-old predictions as fresh.

**Acceptance Criteria:**
- [ ] `execution/signal.py::assert_fresh` additionally fails when the store's last calendar day is more than N calendar days behind `as_of` (default 5 — tolerates long weekends/holidays; configurable).
- [ ] The failure message distinguishes "predictions stale relative to store" from "store stale relative to today".
- [ ] `execution/pred_refresh.py`'s self-check applies the same bound.
- [ ] `make check` passes; tests cover fresh store, stale store at boundary (5 days) and beyond, holiday-gap tolerance.

### US-017: SQLite WAL mode and busy timeout
**Description:** As an operator, I want `state.sqlite` in WAL mode with a busy timeout, so concurrent writers (orchestrator threads, transient GPU pipeline unit, CLI promotes) can't hit `database is locked`.

**Acceptance Criteria:**
- [ ] `orchestrator/state.py` connection setup executes `PRAGMA journal_mode=WAL` and sets `timeout=30` (or `PRAGMA busy_timeout=30000`).
- [ ] One-time migration is safe on the live DB (WAL switch is persistent; test asserts mode after open).
- [ ] `make check` passes.

### US-018: Env file permissions
**Description:** As an operator, I want the token-bearing env files unreadable by other users.

**Acceptance Criteria:**
- [ ] `.env` and `research/.env` are chmod 600 on-box.
- [ ] `ops/setup` or `ops/health.sh` checks and reports env-file modes so regression is caught by the (now scheduled) health run.
- [ ] `make check` passes.

### US-019: Automatic GPU base-snapshot use with auto-rebake on drift
**Description:** As an operator, I want Slack-launched runs to boot from the baked base snapshot when it is current, and to rebuild it automatically when worker-affecting inputs change, so no run pays the ~20-minute bootstrap tax unnecessarily and no human has to remember to rebake.

**Acceptance Criteria:**
- [ ] A worker-inputs hash is defined: short digest over `research/PINNED_COMMIT`, `ops/gpu_worker/gpu_worker.sh`, the `Makefile` venv/install targets, and a store schema/version marker. The hash is embedded in the snapshot name.
- [ ] At launch, the pipeline selects the latest snapshot whose **hash and region both match**: match ⇒ boot from snapshot; no match ⇒ full bootstrap for this run, and at teardown (before destroy) bake a fresh snapshot tagged with the current hash. Superseded snapshots are pruned, keeping the latest 2.
- [ ] `GpuBackend.launch` plumbs this through so Slack-launched runs get the behavior by default (currently CLI-only `--snapshot`); the run-start Slack message states "booted from snapshot" or "full bootstrap (inputs changed — rebaking)".
- [ ] `latest_snapshot_id` in `ops/gpu_worker/gpu_worker.sh` becomes region-aware (snapshots are regional; a size-plan fallback into another region must not select an image that isn't there).
- [ ] Manual override remains: `--snapshot bake` / `--no-snapshot` pipeline flags, documented in the runbook.
- [ ] Bake failure never fails the run — the run's results are already fetched by teardown; the failure posts a Slack warning and the next run simply bootstraps again.
- [ ] `make check` passes; tests cover hash computation, match/mismatch selection, region-aware selection, prune-keep-2, launch-arg plumbing (mocked `doctl`).

### US-020: Stale run-row reaper
**Description:** As an operator, I want GPU run rows whose pipeline unit died (SIGKILL, reboot) finalized automatically, so a stranded `running` row can't permanently brick its Slack thread.

**Acceptance Criteria:**
- [ ] A periodic check (inside the orchestrator app or the hourly watchdog) finds runs with `status=running`, `backend=gpu` whose transient unit (`rdq-gpu-run-<ts>`) is no longer active, waits one grace period, then marks them `failed` with a Slack note in the run's thread.
- [ ] After reaping, `start_research` works again in that thread (covered by test).
- [ ] `make check` passes; tests cover active-unit-untouched, dead-unit-reaped, grace period.

---

### Phase E — server_ui removal

### US-021: Remove the server_ui research backend
**Description:** As an operator, I want the unused server_ui stack deleted, so the repo has one research path (GPU) and no dead 874-line poller pretending to provide safety brakes.

**Acceptance Criteria:**
- [ ] Deleted: `rdq-research.service` (unit file, install/health references, and the box's out-of-repo `rdq-research.service.d/resources.conf` drop-in), `research/server_ui.py`, `orchestrator/rdagent_client.py`, `orchestrator/poller.py`, and the Block Kit approve/edit/reject + promotion-offer flows they drove. `ConversationCore` no longer takes/uses an `RdAgentClient`.
- [ ] Preserved and verified working after removal: conversational `start_research`/`check_research_status`/`stop_run` for GPU runs, plain-text-reply promotion (US-044 behavior) via `locate_run_artifacts`, `promotion.py`'s conversational path, universe registry/builders, and Notion recording done by the GPU pipeline.
- [ ] `runs` rows with `backend=server_ui` remain readable (history intact); code paths that iterate runs tolerate the legacy backend value.
- [ ] Behavior loss is documented in `docs/decisions.md`: the poller's US-045 brakes (`RDQ_MAX_HYPOTHESES`, identical-error abort, per-hypothesis Notion rows) do not exist on the GPU path; the GPU run's protections are `loop_n`, the 24h cap, and the watchdog. Explicitly note this as accepted.
- [ ] On-box: service stopped, disabled, unit files removed; `systemctl --user status` shows no rdq-research; port 19899 no longer listens.
- [ ] Dead config keys (`RDQ_MAX_HYPOTHESES` if now unused, server_ui URLs) removed from `orchestrator/config.py` and `config.yaml`; `ruff`/`pyright` confirm no dangling imports.
- [ ] `make check` passes; obsolete tests removed with the code they tested.

### US-022: Documentation truth pass
**Description:** As an operator, I want the docs to describe the system that actually runs, so the next person (or agent) doesn't build on stale claims.

**Acceptance Criteria:**
- [ ] `orchestrator/CLAUDE.md` rewritten: GPU is the only research backend; the poller-brakes section replaced by the actual GPU-path protections and the new gate/auto-promotion flow.
- [ ] `README.md` and `PLAN.md` layout/architecture sections updated (server_ui removed, gate + divergence tracker + memory injection added); `ops/gpu_worker/README.md` stale "Promotion caveat" corrected.
- [ ] `ops/runbook.md` gains: rollback procedure, halt-file clear procedure after a divergence halt, snapshot rebake procedure, and drops server_ui sections.
- [ ] Repo-root cruft removed: `selector.log`, `git_ignore_folder/`; `ops/sweep.py` extended to prune repo-root `log/` and `pickle_cache/` (retain last 14 days).

## Functional Requirements

- FR-1: Every write to the promoted-strategy pointer must append a `promotion_history` row recording source, verdict (nullable), and the replaced workspace.
- FR-2: The system must provide a rollback command that re-promotes a prior history entry and refuses when the target workspace is absent.
- FR-3: The comparison gate must hard-fail (no promotion possible) on any parity mismatch: test window, market, instrument-list hash, topk/n_drop, or cost parameters.
- FR-4: The gate's pass criteria (search-window IR × 1.05 margin, MDD bound, IC > 0, confirmation-window IR strictly greater) must be read from config, not hardcoded.
- FR-5: On gate pass, the GPU pipeline must promote automatically, including the pred-refresh snapshot; snapshot failure must abort the promotion.
- FR-6: A config flag must disable auto-promotion (report-only mode) without redeploying code.
- FR-7: All three promotion paths (auto, conversational, CLI) must write the Notion Decision Log.
- FR-8: `GpuBackend.launch` must inject a prior-run digest and incumbent summary into `RDQ_USER_INSTRUCTION`, after the operator directive, skippable per run. The digest is sourced from the Notion strategy-notes DB (machine-readable `run_summary` JSON written by the pipeline on every run), with local-state fallback bounded at 15s so Notion can never block a launch.
- FR-9: The GPU pipeline must compute `RDQ_TEST_END` as store-end minus the configured confirmation reserve; the confirmation window must never be visible to the hypothesis search.
- FR-10: Promotion must require the candidate to beat the incumbent on the confirmation window via exact-weights re-prediction; technical failure of that evaluation must block promotion.
- FR-11: A weekday post-close job must compute realized-vs-backtest divergence against a haircut expectation (μ_adj = 0.5 × backtest mean, σ from the full test window, both configurable); z < −2 alerts, z < −3 or drawdown > backtest-MDD × 1.25 writes the breaker halt file.
- FR-12: `us_liquid` must be built with per-date membership spans from trailing ADV/price; promoted-pinned universe snapshots must be untouched by regeneration.
- FR-13: The watchdog unit must have a PATH including `~/.local/bin`, and a missing `doctl` must produce a Slack notice, not a silent crash.
- FR-14: At most one GPU research run may be active box-wide; a second start must be refused with the owner named; stop must be owner-only.
- FR-15: Every rdq systemd service must declare `OnFailure=` pointing at a Slack notifier unit.
- FR-16: Reconcile and health checks must run on timers; reconcile must bring Notion Trade Ledger rows to final fill status.
- FR-17: `assert_fresh` must bound store-calendar age against the real-world date (default 5 calendar days).
- FR-18: `state.sqlite` must use WAL journal mode with a ≥30s busy timeout.
- FR-19: Slack-launched GPU runs must boot from the latest snapshot whose worker-inputs hash and region match; on mismatch they must fall back to full bootstrap and bake a fresh, hash-tagged snapshot at teardown, pruning superseded ones.
- FR-20: Orphaned `running` GPU run rows must be auto-finalized once their unit is dead, restoring the thread's usability.
- FR-21: The server_ui service, Flask UI, poller, and Block Kit approval flow must be removed; conversational GPU control and promotion must keep working.

## Non-Goals (Out of Scope)

- **Delisted-names backfill / terminal-return modeling** — documented gap (US-011), future PRD.
- **Off-box backup of `state.sqlite`** — daily DO droplet snapshots cover disaster recovery (operator decision 2026-08-14).
- **Scheduling research proposals** — the trusted Claude bot already proposes weekday research via Slack; this PRD does not add a proposal timer.
- **Auto-rollback on divergence** — severe breach halts trading; un-halting and rollback remain human decisions (the rollback *command* from US-001 makes that decision cheap).
- **Walk-forward cross-validation** — rolling TEST_END + confirmation window was chosen instead.
- **Live trading itself** — covered by `tasks/prd-live-trading.md`; this PRD's rigor items are prerequisites, not implementations of it.
- **Widening the research search space** (portfolio construction, alternative data, model families) — worthwhile, but a separate research-quality PRD; this one fixes loop mechanics and honesty of measurement.

## Technical Considerations

- **Gate metric extraction** should reuse `orchestrator/summary.py::METRIC_SPECS` and the existing `ret.pkl` Sharpe derivation — one metrics vocabulary everywhere.
- **Run-summary JSON is written by the pipeline that already writes the Notion page** (`ops/notion_summary.py`), so no new Notion auth surface; the digest reader uses the same client. Chunk the JSON across rich-text elements ≤2000 chars and reassemble on read; include a `schema_version` field so the digest parser can evolve.
- **Confirmation-window re-prediction** reuses the US-049 exact-weights machinery (`execution/pred_refresh.py`); the incumbent side can share the promoted workspace's existing snapshot. Budget ~2–5 min per gate evaluation on the control box.
- **Instrument-list hash**: hash the sorted resolved instrument list at run launch and store it in `pipeline_status.json`, so parity checking doesn't depend on re-reading mutable universe files later.
- **Ordering**: US-001/002 before US-003; US-007 before US-008; US-021 late (after conversational paths are covered by Phase A tests) to catch accidental breakage of kept functionality.
- **The PIT universe change (US-010) alters backtest results for all future runs.** The first post-PIT run's candidate will be compared against an incumbent whose metrics came from the old universe — parity enforcement (instrument-hash) will flag this; the operator should expect one manual/`--force` re-baselining promotion after US-010 lands, and the PRD accepts that.
- **Deploy reminder**: orchestrator changes require `systemctl --user restart rdq-orchestrator` (known deploy gap); US-014's failure alerts do not cover "running stale code" — out of scope here.

## Success Metrics

- A completed GPU run whose candidate beats the incumbent reaches promoted state with zero human actions, and the promotion is fully reconstructable from `promotion_history` + Notion Decision Log.
- Zero promotions possible with mismatched window/universe/costs (previously warn-only).
- Run N+1's hypothesis log shows no verbatim re-proposal of run N's rejected hypotheses (spot-check across 3 consecutive runs).
- The confirmation window is absent from every run's training/test config (auditable from `pipeline_status.json`).
- A simulated 3σ underperformance halts the next rebalance without human action.
- Watchdog: 0 `FileNotFoundError` ticks; a deliberately orphaned droplet is destroyed within 25h.
- After US-021, `rg -l server_ui` returns only docs/history, and `make check` passes with the stack gone.

## Addendum (2026-08-17) — Gate comparability by construction

The first real run through the gate (thread 1786966914.860099) exposed a
structural flaw in US-002's parity design: parity was checked as **equality
of recorded values**, but two of those values drift *by design* — the test
window rolls forward with the store every trading day (US-007), and the PIT
universe membership evolves monthly (US-010). So any candidate launched
after the incumbent's own run day can never match the incumbent's recorded
window, and auto-promotion (US-003's whole point) is structurally
unreachable after the first promotion: every future promotion would need a
manual `--force`. Operator decision 2026-08-17: parity must be
**constructed at gate time** — evaluate both strategies on the same slice of
history before comparing — rather than checked against recorded artifacts.
The confirmation leg (US-008) already works this way; this addendum extends
the same philosophy to the search-window legs.

### US-031: Overlap-window comparison replaces recorded-value parity
**Description:** As an operator, I want the gate's IR and MDD legs computed
from both strategies' daily returns over their *shared* trading days, so two
honestly-measured backtests from different launch days remain comparable and
auto-promotion stays reachable, without weakening any honesty protection.

**Acceptance Criteria:**
- [ ] `ops/promotion_gate.py` gains a pure `align_overlap(candidate,
  incumbent, min_days)` that intersects the two bundles' dated daily-return
  series (`ret.pkl` with dates) and computes each side's annualized IR
  (`ops.confirm_window.annualized_ir` convention) and max drawdown over the
  shared days. Fewer than `promotion_gate.min_overlap_days` (default 126)
  shared days, or missing dated returns on either side, yields an error —
  the IR and MDD criteria then fail as `overlap_unavailable`, never a
  silent skip (same contract as `confirmation_unavailable`).
- [ ] The IR and MDD criteria compare the overlap-computed values (same
  margins as before: IR × `ir_margin`, |MDD| × `mdd_tolerance`); recorded
  qlib_res scalars are no longer used for the incumbent-relative legs. The
  IC leg (candidate-only) still uses the recorded value.
- [ ] Hard parity shrinks to the fields that must never drift silently:
  market, topk, n_drop, cost params. Test-window and instrument-hash
  differences become **drift notes** — rendered in the Slack verdict and
  stored in the verdict JSON — stating what drifted and that the comparison
  used the shared window / each strategy's own deployed universe.
- [ ] `evaluate_gate` requires overlap evidence whenever an incumbent
  exists (mirroring the confirmation-evidence contract); both the pipeline
  auto-gate and `promote_fetched`'s advisory gate supply it.
- [ ] The confirmation leg, reproduction check, IC floor, and all margins
  are unchanged.
- [ ] `make check` passes; tests cover overlap math (IR/MDD on shared days),
  insufficient-overlap refusal, missing-returns refusal, window/universe
  drift notes, hard-parity fields still failing the gate, and the pipeline
  promoting across a window/universe drift when the metrics genuinely pass.

### US-032: Launch-time gate preview
**Description:** As an operator, I want the run-start Slack message to state
when a run structurally cannot end in promotion, so I never spend GPU hours
discovering that at the verdict.

**Acceptance Criteria:**
- [ ] At launch (before the loop starts), the pipeline evaluates the
  launch-knowable blockers: `auto_promote` off, no incumbent with
  `allow_first` off, incumbent workspace missing on disk, incumbent daily
  returns unreadable. Any hit adds a "gate preview: report-only this run —
  <reasons>" line to the run-start message.
- [ ] The preview is advisory and never raises or blocks the launch (runs
  still produce knowledge and run-memory when unpromotable).
- [ ] `make check` passes; tests cover each blocker and the clean case.

## Open Questions

None outstanding. All four were resolved with the operator on 2026-08-14:

1. **Gate margin:** candidate must beat incumbent IR by a 5% relative margin on the search window; strictly-greater (no margin) on the short confirmation window. Config-only tuning; revisit values after ~5 gated runs.
2. **Divergence μ/σ:** σ from the full test window; expectation μ haircut to 50% of backtest mean (configurable), acknowledging the documented survivorship inflation.
3. **Digest durability:** Notion strategy-notes DB is the read-back source; every run's row carries a machine-readable `run_summary` JSON (code block, not attachment); local-state fallback so Notion can never block a launch.
4. **Snapshot rebake:** automatic on drift — worker-inputs hash in the snapshot name; hash mismatch at launch ⇒ full bootstrap + rebake at teardown; no schedule, no manual step.

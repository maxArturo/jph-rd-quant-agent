# PRD: Live Trading — real-money Alpaca account, fully Slack-managed from #live-trading-quant-research

## 1. Introduction / Overview

Extend RD-Agent(Q) from paper-only to real-money trading on Alpaca
(`api.alpaca.markets`). Paper trading continues exactly as today. The live
account gets its own Slack channel (**#live-trading-quant-research**, already
created), its own OneCLI identity (`rdq-exec-live`) holding the live secrets,
its own promoted-strategy slot, tighter software guardrails (limits, breaker,
equity-allocation cap), and its own clearly-labeled Notion reporting.

The live channel is a **full peer** of #quant-research: the same
conversational bot with the same research capabilities (refine a thesis,
start/stop GPU research runs, check status, ask questions, get digests and
charts) plus live-only tools (promote-to-live, demote, halt/resume live,
live account/orders/P&L). Automation is the point — after the operator sends
one promotion message, the system trades daily with no human in the loop,
exactly like paper.

### Operator decisions

Decided 2026-08-10 (supersedes the corresponding 2026-08-05 decisions in the
earlier draft on `worktree-prd-live-trading`):

- **Full research parity in the live channel.** #live-trading-quant-research
  can start, steer, and stop research runs, not just manage the account.
- **Direct-to-live promotion is allowed.** Any promotable run
  (`completed`/`stopped`) can be promoted straight to live from the live
  channel; a paper track record is not a prerequisite. Promoting the current
  paper-promoted strategy remains the default when no run is named.
- **No confirmation step.** A single "promote X to live" message in the live
  channel arms it immediately. The remaining gates are software: promotable
  status, valid config, breaker/halt clear, order gate, allocation cap.
- **Conservative starter guardrails.** Live trades **10% of live equity**;
  max **$500/order**; breaker at **$5,000 daily notional** and **5%
  drawdown** (all operator-editable config, reviewed before go-live).

Carried forward from 2026-08-05 (still in force):

- **Promotion copies, never moves.** The live slot is independent of the
  paper slot. Promoting to live never modifies or clears the paper row;
  when both point at the same workspace, paper-vs-live tracking difference
  is directly observable.
- **Sizing is a percentage of live account equity**, not the whole account.
- **No per-order human approval.** No OneCLI approval rule on the live host
  (an approval rule would deadlock the unattended pre-open rebalancer —
  `orchestrator/CLAUDE.md` warning).
- **One live strategy at a time.** Re-promoting overwrites the slot.

## 2. Goals

- Real orders flow to `api.alpaca.markets` only from the dedicated
  `rdq-exec-live` OneCLI identity; no other identity ever holds live
  secrets, and `ops/check_onecli.sh` proves the isolation in both
  directions.
- The live channel can do everything #quant-research can (research runs on
  GPU workers, thesis steering, status, Q&A, digests, charts, Notion
  write-ups) plus everything live-specific — the operator never needs a
  shell for routine operation.
- Daily live rebalance runs unattended on the same pipeline shape as paper
  (pred refresh → signal → diff → gate → breaker → submit → ledger →
  summary), scaled to `live_equity_allocation_pct` of live equity, with its
  own limits, breaker, and halt state fully independent of paper's.
- The 04:45 ET exact-weights pred refresh (US-049) keeps the live
  workspace's predictions fresh even when paper and live point at different
  workspaces.
- All live reporting lands in Notion databases unmistakably labeled Live;
  `ops/reconcile.py` can audit the live ledger against the live account.
- The operator can halt live trading from Slack in one message, and the
  runbook covers halt/flatten/demote/rotate-keys for real money.
- With `SLACK_LIVE_CHANNEL_ID` unset, the system behaves byte-for-byte as
  today (paper-only, regression-free).

## 3. User Stories

Numbering continues from US-049 (shipped on `main`, `6fe1f4b`). Stories
marked **[OPERATOR-BLOCKED]** cannot complete end-to-end until the
matching operator task in §6 is done; their code and tests ship first
against fakes.

### US-050: AlpacaClient live-host support behind an explicit opt-in
**Description:** As the execution layer, I need `AlpacaClient` to be able to
target `api.alpaca.markets` when — and only when — the caller explicitly
opts in, so the live rebalancer can trade while every existing paper code
path stays incapable of it.

**Acceptance Criteria:**
- [ ] `AlpacaClient(base_url="https://api.alpaca.markets")` still raises
      `ValueError` unless a new flag `allow_live=True` is also passed
      (today's blanket refusal is at `execution/alpaca_client.py:269-274`).
- [ ] Default construction (no args) remains the paper host; `allow_live`
      defaults to `False`; passing `allow_live=True` with the paper URL is
      harmless.
- [ ] The bare-HTTPS invariant is unchanged: no `APCA-*` header appears in
      the module (existing source-grep test in
      `tests/test_alpaca_client.py` still passes).
- [ ] `AlpacaAuthError` messages no longer hardcode "paper" — they name the
      identity appropriate to the host (`rdq-exec-paper` vs `rdq-exec-live`).
- [ ] The module docstring and `execution/CLAUDE.md` "PAPER ONLY" rules are
      rewritten to describe the opt-in and which callers may use it (only
      the live rebalancer path, never the orchestrator).
- [ ] Tests cover: live host refused without flag, accepted with flag,
      paper host unaffected.
- [ ] Full test suite passes.

### US-051: Live limits, breaker, and equity-allocation config
**Description:** As the operator, I want live-specific guardrail numbers so
the live account trades a small, capped slice with tighter tripwires than
paper.

**Acceptance Criteria:**
- [ ] `execution/limits.live.json` exists with the same four required keys
      as paper, starter values: `max_order_notional_usd: 500`,
      `max_position_pct_equity: 10`, `max_day_orders: 60`,
      `max_total_positions: 60`.
- [ ] `execution/breaker.live.json` exists: `max_daily_notional_usd: 5000`,
      `max_drawdown_pct: 5`.
- [ ] A required config value `live_equity_allocation_pct` (starter: `10`)
      scales the equity figure passed into `execution/diff.py` for live
      runs only (the single insertion point is `diff.py:181`, which
      currently uses raw `account.equity`). Paper continues to use full
      equity. Buying-power capping (`cap_buys_to_buying_power`) still uses
      the account's real `buying_power`, never the scaled figure.
- [ ] Live breaker state lives in its own directory
      (`~/rdq-data/breaker-live/` with its own `halt` and
      `high_water_mark.json`): halting paper must not halt live and vice
      versa. `Breaker`'s existing parameterized ctor is used — no fork of
      `breaker.py`.
- [ ] The live high-water mark seeds from the live account's first clean
      pass, moves up only, and a corrupt HWM file refuses to trade
      (`BreakerStateError`) — identical semantics to paper.
- [ ] Loaders refuse unknown keys and missing keys for the live files, same
      as paper (`LimitsConfigError` / `BreakerConfigError`).
- [ ] Tests pass.

### US-052: Live promoted-strategy slot (promote / demote, direct-to-live)
**Description:** As the operator, I want a live promotion slot that can be
pinned from the paper-promoted strategy **or directly from any promotable
run**, and cleared (demoted) independently, so live trading always runs
exactly one deliberately chosen strategy.

**Acceptance Criteria:**
- [ ] `StateStore` gains a single live slot (`promoted_strategy_live` table
      or equivalent — the paper table is `CHECK (id = 1)` single-row,
      `orchestrator/state.py:62-67`; do not relax the paper table), same
      config shape as paper: `universe`, `universe_tickers`, `topk`,
      `n_drop`, `thread_ts`, `session_path`, plus
      `live_equity_allocation_pct` captured at promote time for the audit
      trail (the rebalancer reads the config file, not this copy).
- [ ] Promotion sources: (a) no run named ⇒ copy the current paper-promoted
      row in full, including universe provenance; (b) a promotable run
      (status `completed` or `stopped`, per `PROMOTABLE_STATUSES`) named or
      resolved from the thread ⇒ pin its workspace directly, deriving the
      universe from the workspace conf `market:` line via
      `execution.signal.load_market` and pinning `universe_tickers` from
      the store instruments file — identical provenance rules to paper
      promotion (US-023 semantics in `orchestrator/promotion.py`).
- [ ] Direct-to-live promotion of a workspace that was never
      paper-promoted also writes the pred-refresh snapshot artifacts
      (`conf_pred_refresh.yaml`, `pred_refresh.env`,
      `pred_refresh_params.pkl` via `snapshot_pred_refresh()`), so the
      04:45 refresh can serve it. Promoting an operator-pinned-market
      workspace warns before re-snapshotting, same as
      `ops/promote_fetched.py`.
- [ ] `execution/promoted.py` gains `load_promoted_strategy_live()` with
      the same refusal semantics as paper (no DB / no row / workspace gone
      ⇒ `NoPromotedStrategyError`, live rebalancer aborts without trading).
- [ ] Demoting clears only the live slot; the paper row is never modified
      by any live operation. Re-promoting overwrites the live slot.
- [ ] Every live promote/demote writes a Decision Log entry (existing
      Notion Decision Log, orchestrator remains its sole writer) including
      workspace id, universe + ticker count, allocation pct, and the Slack
      permalink of the triggering message.
- [ ] Tests pass.

### US-053: Dual-channel Slack orchestrator with full research parity
**Description:** As the operator, I want the bot fully functional in
#live-trading-quant-research — research runs and live management alike — so
the live channel is the single place I operate real-money trading from.

**Acceptance Criteria:**
- [ ] New env var `SLACK_LIVE_CHANNEL_ID` in the repo-root `.env`. Absent ⇒
      orchestrator behaves exactly as today (single-channel; no live tools
      registered).
- [ ] `app.py`'s actionable-message check accepts either channel; every
      component that today consumes the scalar `config.channel_id`
      (`app.py`, `poller.py`, `approvals.py`, `promotion.py` button
      handlers) becomes channel-aware. Threads never migrate between
      channels; pending-interaction records are keyed so live and paper
      threads can never collide.
- [ ] All research tools work identically from the live channel:
      `save_directive`, `start_research`, `check_research_status`,
      `stop_run`, `set_universe`/`confirm_universe`, spoken
      approve/reject (US-044), autonomous-mode auto-approval (US-045), loop
      digests, run-completion summary + equity-curve chart, Notion
      write-up. The `runs` table records the originating channel; the
      poller and GPU pipeline post each run's digests to the channel that
      started it.
- [ ] Live-only tools are registered and only invocable from the live
      channel: `promote_to_live` (US-054), `demote_live`,
      `halt_live_trading`, `resume_live_trading`, and live variants of
      `check_account` / `check_orders` / `check_pnl`. Invoking one from
      #quant-research is refused with a pointer to the live channel;
      likewise paper `halt_trading`/`resume_trading`/promotion tools keep
      operating on paper state only, from #quant-research only.
- [ ] `halt_live_trading`/`resume_live_trading` operate on the live breaker
      paths from US-051; the existing paper halt tools are untouched.
- [ ] Live `check_account`/`check_orders`/`check_pnl` are read-only against
      the live account. Decide the read path at implementation time and
      record it in `docs/decisions.md`: either (a) the orchestrator
      identity gains **read-scoped** live access via OneCLI (if the proxy
      can scope by method/path) or (b) the bot reads the Live Notion
      databases and orchestrator state instead, gaining no live host access
      at all. Option (b) is the default if scoping is not practical — the
      orchestrator identity must never be able to place live orders either
      way.
- [ ] `orchestrator/prompts.py` no longer declares "live trading is out of
      scope"; the system prompt explains both channels, that
      #live-trading-quant-research controls a real-money account, and that
      paper and live promotion slots are independent.
- [ ] Paper research notifications, paper rebalance summaries, and paper
      breaker alerts continue to go only to #quant-research; live rebalance
      summaries and live breaker alerts go only to the live channel.
- [ ] Tests pass.

### US-054: One-message live promotion (no confirmation step)
**Description:** As the operator, I want a single message in the live
channel — "promote to live" or "promote run X to live" — to arm live
trading immediately, so operating the system stays handholding-free.

**Acceptance Criteria:**
- [ ] `promote_to_live` resolves its candidate: an explicitly named run, a
      run resolvable from the current thread, or (neither) the current
      paper-promoted strategy. If the reference is ambiguous (matches more
      than one run) or resolves to nothing, the tool refuses and lists what
      it found — it never guesses among multiple candidates.
- [ ] On success it writes the live slot (US-052) and immediately posts an
      armed summary to the live channel: workspace id, source (paper copy
      vs direct run), backtest headline metrics, universe with ticker count
      and any label-vs-conf mismatch call-out, allocation pct, the live
      limits/breaker values in force, and when the next live rebalance
      fires. The summary is confirmation *after the fact* — nothing waits
      on a reply.
- [ ] Refusal paths (nothing written, reason posted): candidate not in
      `PROMOTABLE_STATUSES`; workspace or pred artifacts missing; live
      breaker halted; live limits/breaker/allocation config missing or
      malformed; invoked from any channel other than the live channel.
- [ ] `demote_live` (also single-message) clears the slot and posts what
      was demoted; the next live rebalance aborts with
      `NoPromotedStrategyError` and says so in the live channel.
- [ ] Promote and demote each write the Decision Log entry from US-052.
- [ ] Tests pass.

### US-055: Live Notion reporting **[OPERATOR-BLOCKED — §6 task D]**
**Description:** As the operator, I want all live reporting under a sibling
Notion page unmistakably labeled Live, so paper and live records can never
be confused.

**Acceptance Criteria:**
- [ ] A page titled **"Automated AI Quant Investment — LIVE 🔴"** exists as
      a sibling of the existing paper page, with an intro callout stating
      it reflects a real-money account.
- [ ] `ops/bootstrap_notion.py` (extended, idempotent, matched by title)
      creates **Trade Ledger (Live)** and **Account Snapshots (Live)**
      under it with the same property schemas as their paper counterparts,
      and writes their IDs into `orchestrator/config.yaml` under new keys;
      `NotionDatabases` (`orchestrator/notion_recorder.py:54-62`) gains the
      two fields.
- [ ] `docs/reference/notion-schema.md` documents the live databases and
      their sole writer (the live rebalancer for both), preserving the
      one-writer-per-database rule. Decision Log stays shared, orchestrator
      remains its only writer.
- [ ] Paper page and databases are untouched; bootstrap property-name
      cross-check tests extended to the live databases.
- [ ] Tests pass.

### US-056: Live rebalancer path **[OPERATOR-BLOCKED — §6 tasks A–C]**
**Description:** As the system, I want a live mode of the existing rebalance
pipeline so the live-promoted strategy trades daily under the live identity,
guardrails, and reporting — unattended.

**Acceptance Criteria:**
- [ ] `execution/rebalance.py --live` switches, together: live promoted
      slot (US-052), live client opt-in (US-050), live
      limits/breaker/allocation (US-051), live Notion databases (US-055),
      and the live Slack channel for all posts. Without `--live`, behavior
      is byte-for-byte today's paper behavior (existing tests unmodified
      and passing).
- [ ] Target equity for diffing = live `account.equity` ×
      `live_equity_allocation_pct` / 100; buying-power capping and the 403
      `40310000` classification (business rejection, not auth) behave
      exactly as on paper.
- [ ] `client_order_id` convention gets a live-distinct prefix
      (`rdq-live-<date>-<side>-<symbol>`) so paper and live order ids can
      never collide in cross-account tooling.
- [ ] All `_ABORT_ERRORS` paths abort without trading and post the reason
      to the live channel; the live halt file exits 0; unexpected
      exceptions still notify then re-raise.
- [ ] `universe_divergence_warnings` remain advisory (never abort) and are
      included in the live daily summary on every path, as on paper.
- [ ] Fills land in Trade Ledger (Live); a daily row goes to Account
      Snapshots (Live); the daily summary posted to the live channel
      includes orders, fills, skipped/deferred buys, equity, allocation in
      force, breaker state line, and same-day paper-vs-live P/L when both
      accounts traded.
- [ ] `ops/reconcile.py` and `ops/flatten.py` accept a `--live` flag (live
      identity, live base URL with `allow_live=True`, live ledger DB);
      their exit-code contracts are unchanged.
- [ ] An end-to-end dry run of the `--live` code path against the paper
      account (test-fixture mode: live code path, paper identity/URL)
      reconciles ledger rows with submitted orders before first real use.
- [ ] Tests pass.

### US-057: Pred refresh covers the live workspace
**Description:** As the system, I want the 04:45 ET exact-weights pred
refresh (US-049) to keep the live strategy's predictions fresh even when
paper and live point at different workspaces, so live never trades stale
signals.

**Acceptance Criteria:**
- [ ] `execution/pred_refresh.py` enumerates BOTH promoted slots (paper +
      live), dedupes by workspace path, and refreshes each distinct
      workspace sequentially within the existing timer window.
- [ ] While paper and live share a workspace, exactly one refresh runs.
- [ ] A refresh failure for the live workspace posts to the live channel
      (paper failures keep posting to #quant-research); the stale-pred
      guard stays in place — the live rebalance aborts on stale pred
      exactly as paper does.
- [ ] The staleness notice text no longer hardcodes "paper rebalance"
      (`pred_refresh.py:531`).
- [ ] Tests pass.

### US-058: OneCLI live identity + systemd wiring + isolation proof **[OPERATOR-BLOCKED — §6 tasks A–B]**
**Description:** As the operator, I want the live rebalance scheduled like
paper's and the identity isolation verified by script, so a
misconfiguration is caught by a probe rather than a trade.

**Acceptance Criteria:**
- [ ] `ops/setup_onecli.sh` adds identity `rdq-exec-live` with host
      allowlist `api.alpaca.markets api.notion.com
      financialmodelingprep.com` (no paper host), replacing the current
      hard `die` on the live host with a rule that only `rdq-exec-live` may
      hold it; `ops/CLAUDE.md`'s "never register rdq-exec-live" rule is
      rewritten to "only rdq-exec-live, only via setup_onecli.sh".
- [ ] `ops/check_onecli.sh` proves all four directions: `rdq-exec-live` →
      live host authenticates (2xx); `rdq-exec-paper` → live host fails;
      `rdq-exec-live` → paper host fails; `rdq-orchestrator` → live host
      fails (unless option (a) read-scoping was chosen in US-053, in which
      case the probe asserts read-only scope instead).
- [ ] `rdq-rebalance-live.service`/`.timer` run `rebalance.py --live` under
      `onecli run --agent rdq-exec-live` at **08:10 America/New_York**
      Mon–Fri, `Persistent=false`, `After=` the pred-refresh service, same
      unit conventions as paper's. Files in `ops/`, added to
      `install_services.sh` `UNITS` **and** `ops/health.sh` lists (both
      cross-check tests pass), timer enabled.
- [ ] `ops/health.sh` reports live timer/breaker state alongside paper's.
- [ ] Tests/shellcheck pass.

### US-059: GPU pipeline and write-ups are account-aware
**Description:** As the operator, I want GPU research runs started from
either channel to report correctly, and comparison charts/write-ups to say
which promoted slot they compare against, so nothing mislabels real-money
state.

**Acceptance Criteria:**
- [ ] GPU runs started from the live channel post their loop digests,
      completion summary, chart, and Notion write-up link to the live
      channel (thread-ts and channel travel together through
      `orchestrator/gpu_backend.py`, `ops/gpu_pipeline.py`, and the status
      file).
- [ ] `ops/gpu_pipeline.py:294`'s `"promoted ({ws}, live paper)"` label and
      the candidate-vs-promoted comparison state which slot (paper or
      live) they compare against; when both slots exist and differ, the
      completion summary shows the candidate against both.
- [ ] `ops/notion_summary.py`'s prompt no longer asserts "trades a paper
      account"; it receives and states the account context.
- [ ] `ops/gpu_watchdog.py` alerts reach the channel that owns the run (it
      currently posts to the single channel id).
- [ ] `ConversationCore.gpu` stays `None`-default; only `app.py` wires the
      real backend (droplet-provisioning-from-tests incident, 2026-08-06).
- [ ] Tests pass.

### US-060: Runbook, docs, and go-live procedure
**Description:** As the operator, I want the emergency procedures for real
money written down and rehearsable before the first live order.

**Acceptance Criteria:**
- [ ] `ops/runbook.md` gains a Live section: halt live from Slack; halt
      from the shell (write the live halt file directly, exact path);
      flatten all live positions (`ops/flatten.py --live` one-liner under
      the live identity); demote from Slack; rotate live keys in OneCLI;
      and a "paper incident must not touch live (and vice versa)" note.
      `tests/test_flatten.py`'s required-sections check covers it.
- [ ] The operator checklist (§6 of this PRD) is copied into
      `ops/runbook.md` so it survives outside the PRD.
- [ ] Go-live order of operations documented: §6 tasks A–D → deploy
      US-050..US-059 → `ops/install_services.sh` + daemon-reload + restart
      `rdq-orchestrator` (deployed code is not running code) → review
      guardrail configs → send the one promotion message → observe the
      first live rebalance and run `ops/reconcile.py --live`.
- [ ] `README.md`, `PLAN.md` Phase 6 status, and `docs/decisions.md` record
      the scope change and the 2026-08-10 decisions (including that they
      supersede 2026-08-05's paper-first / two-step-confirm decisions).
- [ ] Tests pass.

## 4. Functional Requirements

- FR-1: Live orders originate exclusively from `rdq-exec-live`; the live
  Alpaca key/secret exist only as that identity's OneCLI secret
  assignments scoped to the `api.alpaca.markets` host pattern. Base-URL
  choice remains credential choice; no `APCA-*` header ever appears in
  repo code.
- FR-2: The orchestrator identity must never be able to place live orders.
  Read access for live status tools is either proxy-scoped read-only or
  replaced by Notion/state reads (US-053 decision point).
- FR-3: The live rebalancer refuses to trade when any of: no live-promoted
  strategy, workspace missing, stale/missing predictions, live halt file
  present, live breaker tripped, gate rejection, or live
  limits/breaker/allocation config missing/malformed.
- FR-4: Live promotion is a single operator message in
  #live-trading-quant-research; it must resolve to exactly one promotable
  candidate or be refused. No confirmation reply is required or waited on.
- FR-5: With `SLACK_LIVE_CHANNEL_ID` unset, the system runs exactly as
  before this feature — paper-only, all existing tests unmodified.
- FR-6: One writer per Notion database: live rebalancer writes Trade
  Ledger (Live) and Account Snapshots (Live); orchestrator remains sole
  writer of the shared Decision Log and idea pages.
- FR-7: Every live order and fill is auditable end-to-end: Trade Ledger
  (Live) row ↔ live Alpaca order id, `ops/reconcile.py --live` clean.
- FR-8: Any live breaker trip or halt posts an alert to
  #live-trading-quant-research and prevents all subsequent live
  submissions until the operator resumes from Slack (or shell).
- FR-9: Research runs are startable, steerable, and stoppable from either
  channel with identical capability; every run's notifications go to the
  channel that started it.
- FR-10: Paper and live guardrail/halt/breaker/HWM state are fully
  disjoint on disk and in behavior.

## 5. Non-Goals (Out of Scope)

- No per-order human approval; no OneCLI approval rule on the live host in
  v1 (see Open Questions for the dead-man's-switch variant).
- No auto-follow: live does NOT re-pin automatically when paper promotes a
  new strategy; changing live requires a new promote-to-live message.
- No more than one live strategy at a time; no capital splitting across
  strategies.
- No intraday trading, options, crypto, shorting, margin, or fractional
  shares (long-only whole-share top-k daily rebalance, matching paper).
- No automatic scaling of `live_equity_allocation_pct` — changing it is a
  deliberate config edit.
- No migration of historical paper records into the Live Notion page.
- Live promotion does not stop, alter, or demote the paper track.

## 6. Operator Tasks (you) — required before the first live trade

**A. Alpaca (live account)**
1. Log in at https://app.alpaca.markets, switch the top-left toggle from
   Paper to **Live**, and complete live brokerage onboarding (identity,
   funding source) if not already done.
2. Fund the account. Remember only `live_equity_allocation_pct` (10%) of
   equity is traded; the breaker caps daily notional at $5,000.
3. Wait for account status **ACTIVE** (dashboard header; verifiable via
   `GET /v2/account` once task B is done).
4. With the Live toggle selected, generate **live** API keys (right-hand
   panel → API Keys). They are distinct from paper keys; the secret shows
   once — paste both straight into OneCLI (task B) and store them nowhere
   else.
5. Recommended account configuration (settable via
   `PATCH /v2/account/configurations` after task B): `no_shorting: true`,
   `max_margin_multiplier: "1"` (cash-like), fractional trading OFF (the
   pipeline trades whole shares).
6. Optional but recommended: enable Alpaca's email trade confirmations as
   an independent audit stream.

**B. OneCLI (live identity)**
1. Run the updated `ops/setup_onecli.sh` (US-058) to create `rdq-exec-live`,
   or create it in the web UI at :10254.
2. Vault the live key id + secret as assignments on `rdq-exec-live`, host
   pattern `api.alpaca.markets` — mirroring the paper assignment shape.
3. Run `ops/check_onecli.sh`; all four isolation probes (US-058) must pass.

**C. Slack**
1. Invite the bot to **#live-trading-quant-research**: `/invite @<bot>`
   (channel already exists).
2. Copy the channel id (channel details → About → bottom) into the
   repo-root `.env` as `SLACK_LIVE_CHANNEL_ID`.
3. After deploy: `systemctl --user restart rdq-orchestrator` — deployed
   code is not running code.

**D. Notion**
1. Ensure the integration used by the orchestrator can create a sibling of
   "Automated AI Quant Investment" — or create an empty page titled exactly
   "Automated AI Quant Investment — LIVE 🔴" and share it with the
   integration; the bootstrap adopts it by title. (Note: Notion app
   connections are granted per-agent in the gateway web UI, no CLI —
   `rdq-exec-live` needs the Notion connection too, for the live ledger
   writes.)

**E. Go-live**
1. Review/adjust `execution/limits.live.json`,
   `execution/breaker.live.json`, and `live_equity_allocation_pct`.
2. In #live-trading-quant-research, send the one promotion message; read
   the armed summary it posts.
3. Be around for the first live rebalance (08:10 ET): check the fill
   summary, Trade Ledger (Live), and run `ops/reconcile.py --live`.
4. During the first week, deliberately halt live from Slack once and
   confirm the next rebalance exits 0 without trading, then resume.

## 7. Technical Considerations

- **US-023 universe provenance is merged and live** (`orchestrator/promotion.py`):
  universe from the workspace conf `market:` line, `universe_tickers`
  pinned from the store instruments file, divergence warnings advisory.
  Live promotion (both sources) reuses these functions — no re-derivation
  logic should be written.
- **Pred refresh is the exact-weights re-predict** (US-049, ~2 min/workspace
  at 04:45 ET), not the old full re-fit — refreshing two workspaces fits
  the window comfortably; the old draft's broad-universe timing concern is
  obsolete. Direct-to-live promotion must create the snapshot artifacts
  the refresh consumes (US-052), because unlike the paper path they may
  not exist yet.
- **Identity isolation is the primary control** and is what makes
  no-confirmation promotion tolerable: paper/orchestrator code paths
  cannot authenticate to the live host at all, the gate/breaker/allocation
  bound the blast radius, and halt-from-Slack is one message.
- The 5xx-never-retry policy on `POST /v2/orders` matters more with real
  money; unchanged. `client_order_id` uniqueness (with the `rdq-live-`
  prefix) remains the same-day-rerun guard.
- `execution/diff.py`, `order_gate.py`, `signal.py`, `breaker.py` stay
  pure and account-agnostic; live-ness enters only through constructor
  args, config paths, and the scaled equity input.
- Channel-awareness is the widest-touch change: `SlackConfig` grows a
  second (optional) channel id; ~8 call sites consume the scalar today.
  The `runs` table needs an originating-channel column; poller, GPU
  pipeline status file, watchdog, and promotion button values must carry
  channel alongside `thread_ts`.
- Live timer at 08:10 ET (paper 08:00) keeps log interleaving readable and
  lets paper act as an informal canary; they are independent — a paper
  abort does not stop live.
- Restart discipline: every orchestrator deploy here requires a unit
  restart; `rdq-rebalance-live.timer` must be **enabled**, not just linked
  (the orchestrator unit was once linked-only and died on reboot).
- Sequencing: US-050→051→052 are dependency-ordered; US-053/054 build on
  052; US-055 is parallel; US-056 needs 050–055; US-057/058/059 are
  parallel after 056's interfaces settle; US-060 last.

## 8. Success Metrics

- One message in the live channel → strategy armed → first live rebalance
  submits only gate-approved orders totaling ≤ 10% of equity, zero manual
  intervention after the promotion message.
- 10 consecutive live sessions with Trade Ledger (Live) reconciling 1:1
  against Alpaca live order history (`ops/reconcile.py --live` clean).
- A GPU research run started from the live channel completes with digests,
  chart, and Notion write-up in the live channel — parity verified.
- Halt-from-Slack verified once, deliberately, in week one (rebalance
  exits 0, no orders).
- `ops/check_onecli.sh` green: live secrets reachable only by
  `rdq-exec-live`, paper unreachable from live and vice versa.
- Paper behavior regression-free: full existing test suite passes
  unmodified with `SLACK_LIVE_CHANNEL_ID` unset.

## 9. Open Questions

- **Dead-man's-switch variant** (deferred): a OneCLI approval rule on
  `api.alpaca.markets` auto-approved by the bridge only while a live
  promotion is armed and the breaker is clear — fails safe if the
  orchestrator is down, at the cost of coupling the rebalance to
  orchestrator uptime. Revisit after the first clean live weeks.
- Should the live daily summary mirror a one-line ping into
  #quant-research, or stay strictly separated? (Default: separated.)
- When paper-promoted ≠ live-promoted for more than N days, should the bot
  nudge in the live channel? (Default: no nudge; `check_account` in the
  live channel reports both slots.)
- Universe divergence stays advisory for live (consistency with paper;
  pred.pkl is the backtested ground truth). Should real money instead
  treat out-of-universe targets as abort-without-trading? PRD keeps it
  advisory.
- US-053's read-path decision (proxy read-scoping vs Notion/state reads)
  is left to implementation; record the choice in `docs/decisions.md`.

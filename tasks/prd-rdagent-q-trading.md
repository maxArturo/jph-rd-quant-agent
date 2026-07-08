# PRD: RD-Agent(Q) Slack-Driven Quant Research & Paper Trading System

**Source plan:** `PLAN.md` (this repo). Read it first — it holds the architecture diagram, the OneCLI mechanics, and the risk table. This PRD turns that plan into implementable, verifiable stories.

## 1. Introduction / Overview

Build a standalone system in this repo (`/home/nanoclaw/rd-agent-q`) that lets a single operator drive quantitative investment research from a dedicated Slack channel and have promoted strategies traded automatically on an **Alpaca paper account**. The research engine is Microsoft's RD-Agent(Q) (pinned commit, driven via its `server_ui` HTTP control plane — no fork). User-facing records (ideas, hypotheses, backtest results, decisions, trades) are written to Notion. All credentials flow through the box-wide OneCLI gateway (`http://127.0.0.1:10254`); no service binds to a public interface (tailnet-only `tailscale serve` for dashboards).

The repo must be **independent of nanoclaw** — nanoclaw is reference material only; any pattern worth keeping is copied into this repo.

**Key decisions (settled):**
- PRD scope: the entire plan, **but live trading is out of scope** (paper only; live enablement gets its own PRD).
- RD-Agent's internal LLM: **Claude via LiteLLM**, with **Voyage AI** for embeddings (`voyage/voyage-3.5-lite` — Anthropic has no embeddings API; no OpenAI anywhere, see docs/decisions.md 2026-07-07). Phase 0 spike verifies JSON-mode; documented fallback is another LiteLLM-supported chat provider.
- **Model tiers — match model to stakes** (exact IDs; do not add date suffixes):
  - `claude-fable-5` — orchestrator conversational layer only (high-stakes judgment: directive refinement, hypothesis relay/edits, promotion and trade-approval context). Always paired with server-side fallback to `claude-opus-4-8` and `stop_reason: "refusal"` handling; requires org data retention ≥30 days.
  - `claude-sonnet-5` — RD-Agent internal loop (`CHAT_MODEL=anthropic/claude-sonnet-5`): high-volume hypothesis/code generation, near-Opus coding quality at ~⅓ Fable's price.
  - `claude-haiku-4-5` — orchestrator utility calls (Slack formatting, log summarization for thread updates, ticker-list extraction).
- Market data: **FMP (Financial Modeling Prep)**, already vaulted in OneCLI box-wide. Bare HTTPS to `financialmodelingprep.com/stable/...`; the proxy injects `?apikey=`. Reference: nanoclaw `container/agent-runner/src/mcp-servers/fmp/server.ts`.

## 2. Goals

- An operator can post an investment idea in `#quant-research` and get a structured research run: hypothesis proposals to approve/edit in-thread, and a backtest summary when the loop completes.
- Ticker/sector-specific ideas map to **per-run custom universes**; default universe is a liquidity-filtered broad US set (`us_liquid`).
- A promoted strategy is rebalanced **nightly, deterministically** into Alpaca paper orders — the LLM never computes order quantities.
- Every research run is reconstructable from Notion alone; every order/fill lands in a Notion Trade Ledger.
- No raw credentials in this repo's code or env files (OneCLI everywhere; documented narrow exception possible for Slack tokens).
- All services bind `127.0.0.1`; human-facing dashboards exposed only via tailnet-only `tailscale serve`.

## 3. User Stories

Ordered = implementation order. Stories marked **[gate]** block everything after them.

### Phase 0 — Environment bring-up

#### US-001: Repo scaffold + pinned RD-Agent install
**Description:** As a developer, I want the repo skeleton and a pinned RD-Agent so all later work has a stable base.

**Acceptance Criteria:**
- [ ] Directory layout from PLAN.md §1 exists (`orchestrator/`, `execution/`, `research/`, `data/`, `ops/`, `docs/reference/`)
- [ ] Python ≥3.10 venv; RD-Agent installed at a pinned commit (recorded in `research/PINNED_COMMIT`)
- [ ] `rdagent health_check` passes (Docker sudo-less, ports free)
- [ ] `README.md` states the standalone constraint and points to PLAN.md

#### US-002: OneCLI identities and secret assignment
**Description:** As a developer, I want per-component OneCLI identities so credential scope is enforced by the gateway, not our code.

**Acceptance Criteria:**
- [ ] Identities registered: `rdq-orchestrator`, `rdq-research`, `rdq-exec-paper` (NOT `rdq-exec-live` — out of scope)
- [ ] Secrets assigned per PLAN.md §1 table (remember: new agents start in `selective` mode with zero secrets)
- [ ] Verification script `ops/check_onecli.sh`: through each identity, a bare `curl` to one endpoint per assigned service returns 200 (e.g. Alpaca `GET /v2/account` for `rdq-exec-paper`, FMP quote for `rdq-research`)
- [ ] Same script confirms `rdq-exec-paper` gets 401 from `api.alpaca.markets` (live host) — isolation proof

#### US-003: [gate] LLM backend spike — Claude via LiteLLM
**Description:** As a developer, I want to verify RD-Agent's loop works with Claude as `CHAT_MODEL` before building on it.

**Acceptance Criteria:**
- [ ] `research/.env` sets `CHAT_MODEL=anthropic/claude-sonnet-5`, `EMBEDDING_MODEL=voyage/voyage-3.5-lite`, and OneCLI-proxy env; placeholder API keys only (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`)
- [ ] A scripted probe exercises RD-Agent's LLM path (JSON-mode hypothesis-shaped prompt) and gets schema-valid JSON back through LiteLLM
- [ ] Outcome recorded in `docs/decisions.md`: Claude confirmed, OR fallback to another LiteLLM-supported chat provider documented with the failing behavior
- [ ] Embedding call verified end-to-end through the OneCLI proxy

#### US-004: [gate] Vanilla RD-Agent loop completes
**Description:** As a developer, I want one unmodified `fin_factor` loop to finish on default (China) data, proving the engine runs on this box.

**Acceptance Criteria:**
- [ ] `onecli run --agent rdq-research -- rdagent fin_factor --loop_n 1` completes without error (first run may download cn_data + build `local_qlib:latest` — allow hours)
- [ ] `qlib_res.csv` exists in the run workspace with non-null IC/annualized-return metrics
- [ ] Trace visible in `rdagent ui` (Streamlit, :19900, bound 127.0.0.1)

### Phase 1 — Slack bot + orchestrator core

#### US-005: Slack app + Socket Mode listener
**Description:** As an operator, I want a bot in `#quant-research` so I can talk to the system without any inbound port.

**Acceptance Criteria:**
- [ ] Slack app created (Socket Mode; scopes: `chat:write`, `channels:history`, `reactions:write`); bot invited to `#quant-research`
- [ ] Bolt-for-Python listener receives channel + thread messages and can reply in-thread
- [ ] Token handling decided and documented: OneCLI injection for `slack.com` verified, or repo-local `.env` fallback (chat tokens only) with rationale in `docs/decisions.md`
- [ ] Tests pass (message-routing unit tests with mocked Slack client)

#### US-006: Claude conversational layer with tool registry
**Description:** As an operator, I want the bot to converse intelligently — refining raw ideas into structured research directives.

**Acceptance Criteria:**
- [ ] Anthropic tool-use loop on `claude-fable-5` (via OneCLI proxy) with a system prompt: quant research desk PM; honest reporting; never trades without explicit approval
- [ ] Fable-specific handling in place: server-side fallback to `claude-opus-4-8` (`fallbacks` param, beta `server-side-fallback-2026-06-01`); `stop_reason == "refusal"` checked before reading `content`; no `thinking` parameter sent (always-on); streaming used for long turns
- [ ] Utility calls (message formatting, log summarization, ticker extraction) routed to `claude-haiku-4-5` via a small model-router helper, not hardcoded per call site
- [ ] Tool registry scaffold exists; first tool `save_directive` produces a structured directive (objective, universe hint, constraints) echoed to the thread
- [ ] SQLite state DB (`orchestrator/state.sqlite`) maps `thread_ts ↔ directive/run`; schema migration runs on startup
- [ ] Conversation context persists across bot restarts (reload from SQLite)
- [ ] Tests pass (tool-dispatch unit tests with mocked Anthropic client)

#### US-007: Orchestrator as a service
**Description:** As an operator, I want the bot supervised so it survives reboots and crashes.

**Acceptance Criteria:**
- [ ] `ops/rdq-orchestrator.service` (systemd user unit) runs the bot via the `rdq-orchestrator` OneCLI identity; `Restart=always`
- [ ] `systemctl --user status rdq-orchestrator` healthy after reboot test
- [ ] Crash-loop visible in journal; a deliberate exception restarts cleanly

### Phase 2 — US market port (FMP data)

#### US-008: [gate] FMP → Qlib US data store
**Description:** As a developer, I want a US equity data store built from FMP so research runs on US names.

**Acceptance Criteria:**
- [ ] `data/build_store.py` pulls daily EOD bars from FMP `/stable/historical-price-eod/full` (bare HTTPS, proxy injects apikey) for a configurable ticker list, ≥10 years where available
- [ ] **Adjustment handling:** adjusted prices used (FMP adjusted variant if the tier has it, else factors computed from splits/dividends endpoints); validated against a known split — e.g. NVDA around 2024-06-10 shows no artificial price cliff after adjustment
- [ ] Backfill is batched, checkpointed, and resumable (kill mid-run → rerun continues, no duplicate/missing days); respects FMP 429s with backoff
- [ ] Output in Qlib bin format at `~/.qlib/qlib_data/us_data` (calendar, instruments, features); `qlib.init(provider_uri=...)` + a `D.features()` smoke query returns sane OHLCV for AAPL
- [ ] Tests pass (adjustment-factor unit tests with fixture split/dividend data)

#### US-009: Universes — `us_liquid` default + custom generator
**Description:** As a developer, I want instrument lists the strategy can safely rank within.

**Acceptance Criteria:**
- [ ] `data/make_universe.py --name <n> --tickers <list|file>` writes a Qlib instruments file and regenerates that universe's factor source h5
- [ ] Built-in universes: `us_liquid` (min ADV + min price filters, roughly top ~1000 liquid names; thresholds in `data/config.yaml`) and `sp500`
- [ ] Rejects tickers absent from the store with a clear listing of the gaps
- [ ] Tests pass (filter logic against fixture data)

#### US-010: US templates + factor source data
**Description:** As a developer, I want RD-Agent's qlib templates pointed at US data without forking upstream.

**Acceptance Criteria:**
- [ ] `research/us_templates/` holds copies of `factor_template/` + `model_template/` with `provider_uri` → us_data, `region: us`, `market: us_liquid`, S&P500 benchmark, A-share `limit_threshold` removed, US costs set
- [ ] `daily_pv_all.h5`/`daily_pv_debug.h5` regenerated from the US store; `FACTOR_CoSTEER_DATA_FOLDER` points at them
- [ ] Date splits set via env (`QLIB_QUANT_TRAIN_START` … `QLIB_QUANT_TEST_END`)
- [ ] A-share-specific prompt language overridden via `APP_TPL` directory (grep `prompts.yaml` for cn-market references)
- [ ] Diff against upstream confined to `research/` (no edits inside the pinned RD-Agent tree)

#### US-011: [gate] `fin_quant` end-to-end on US data
**Description:** As a developer, I want the full quant loop working on US equities — the Phase 2 milestone.

**Acceptance Criteria:**
- [ ] `rdagent fin_quant --loop_n 2` completes on `us_liquid` via `onecli run --agent rdq-research`
- [ ] `qlib_res.csv` metrics are plausible (IC not NaN, |ARR| < 200%, MDD < 0)
- [ ] `mlruns/**/pred.pkl` exists and its index instruments are US tickers
- [ ] Run + findings noted in `docs/decisions.md`

### Phase 3 — Slack ↔ RD-Agent control loop

#### US-012: RD-Agent control-plane service + client
**Description:** As a developer, I want a supervised `server_ui` and a typed client so the orchestrator can drive runs.

**Acceptance Criteria:**
- [ ] `ops/rdq-research.service` runs `rdagent server_ui` on 127.0.0.1:19899 (not exposed via Tailscale; `NO_PROXY` covers localhost + tailnet range)
- [ ] `orchestrator/rdagent_client.py`: `start_run(directive, universe, interaction=True)`, `pending()` (`GET /receive`), `submit(edit)`, `stop()`, `resume(session_path)`; artifact locator resolves a finished loop's `qlib_res.csv`/`ret.pkl`/workspace path from `LOG_TRACE_PATH`
- [ ] Tests pass (client unit tests against a stubbed Flask server)

#### US-013: Research run lifecycle from Slack
**Description:** As an operator, I want "research X" in a thread to start a run and see hypotheses come back for approval.

**Acceptance Criteria:**
- [ ] `start_research` tool: launches a run seeded with the directive as `user_instruction`; SQLite maps thread ↔ run/session path
- [ ] Background poller (async task in the orchestrator) watches `/receive`; pending hypotheses post to the owning thread as Block Kit **Approve / Edit / Reject** buttons
- [ ] Approve submits unchanged; Edit round-trips operator text into the hypothesis dict; Reject asks the loop for a new proposal
- [ ] Loop completion posts metrics summary (IC/ICIR/ARR/IR/MDD/Sharpe) + a return chart rendered from `ret.pkl`
- [ ] `stop_run` / `resume_run` tools work from the thread
- [ ] Tests pass (poller + button-action handlers with stubbed client)

#### US-014: Per-run custom universes from ideas
**Description:** As an operator, I want "look into semiconductor names" to research a confirmed peer universe, since RD-Agent(Q) is cross-sectional.

**Acceptance Criteria:**
- [ ] `set_universe` tool: resolves idea → ticker list (operator-supplied or Claude-proposed), posts it for in-thread confirmation before any data work
- [ ] On confirm: validates tickers against the store (gaps reported), calls `make_universe.py`, renders the run's template copy with `market: <custom>`
- [ ] Warns below a minimum universe size (default 30; configurable) and suggests padding with peers
- [ ] Refuses "all US tickers" as a custom universe (points at `us_liquid`)
- [ ] Universe name + tickers recorded with the run (SQLite now, Notion in US-016)

### Phase 4 — Notion reporting

#### US-015: Notion databases + client
**Description:** As a developer, I want the five-database Notion schema and a thin client (raw HTTPS through OneCLI).

**Acceptance Criteria:**
- [ ] `orchestrator/notion_client.py`: create page / query DB / update page; always sends `Notion-Version: 2022-06-28`; no auth header (proxy injects); retries eventual-consistency reads
- [ ] Bootstrap script creates: Research Ideas, Hypothesis Log, Backtest Results, Decision Log, Trade Ledger under one parent page; IDs written to `orchestrator/config.yaml`
- [ ] One-writer-per-DB convention documented in `docs/reference/notion-schema.md` (schema copied/adapted from nanoclaw's thesis-trading docs)
- [ ] Tests pass (client unit tests with mocked HTTP)

#### US-016: Full research run recorded in Notion
**Description:** As an operator, I want to reconstruct any research run from Notion alone.

**Acceptance Criteria:**
- [ ] New directive → Research Ideas page (raw idea, refined directive, universe, status, thread link)
- [ ] Each hypothesis + operator action → Hypothesis Log row linked to the idea
- [ ] Each completed experiment → Backtest Results row (metrics, SOTA flag, workspace path)
- [ ] Manual audit: pick a finished run, verify idea → hypotheses → results chain is complete and linked in Notion with no reference to Slack or SQLite needed

### Phase 5 — Signal → paper trades

#### US-017: Strategy promotion
**Description:** As an operator, I want to promote a strategy explicitly so only deliberate choices trade.

**Acceptance Criteria:**
- [ ] "Promote" button on a completed run's summary; confirmation step restates universe, topk/n_drop, and backtest headline metrics before accepting
- [ ] Promotion pins the SOTA workspace path + config in SQLite as the single *production strategy* (new promotion replaces old, with Slack notice)
- [ ] Decision Log row written (who, when, run link, metrics snapshot)
- [ ] Rebalancer refuses to run with no promoted strategy

#### US-018: Signal extraction + target portfolio (deterministic core)
**Description:** As a developer, I want tested code that turns a promoted workspace into target weights.

**Acceptance Criteria:**
- [ ] `execution/signal.py`: refreshes the promoted workspace's predictions to the latest close (`qrun` rerun with `test_end=today`, or `params.pkl` + `model.predict`), reads latest `pred.pkl` cross-section
- [ ] Replicates `TopkDropoutStrategy` (topk/n_drop from the promoted config) → target holdings → weights (equal-weight v1)
- [ ] Unit tests with fixture `pred.pkl`: known scores → exact expected holdings, including the drop rule and ties
- [ ] Handles missing/stale predictions by **aborting loudly** (no trade), never by trading a partial book

#### US-019: Order diff + gate + Alpaca client
**Description:** As a developer, I want the money-touching path small, gated, and unit-tested.

**Acceptance Criteria:**
- [ ] `execution/alpaca_client.py`: bare HTTPS to `paper-api.alpaca.markets` (proxy injects `APCA-*`); `get_account`, `get_positions`, `list_orders`, `place_order`, `cancel_order`
- [ ] `execution/order_gate.py` enforces `limits.paper.json` (max order notional, max position % of equity, max day orders, max total positions — semantics ported from nanoclaw's alpaca gate) against a **fresh** account/positions read
- [ ] `execution/breaker.py`: halt-file check + max daily notional + drawdown kill-switch vs stored high-water mark; breaker tripped → no orders + Slack alert
- [ ] Order diff (current positions vs target weights → marketable-limit orders) unit-tested: fixture positions + targets → exact expected order list
- [ ] Every gate rejection is logged with the specific limit violated

#### US-020: [gate] Nightly rebalance end-to-end (paper)
**Description:** As an operator, I want the promoted strategy trading paper money unattended.

**Acceptance Criteria:**
- [ ] `execution/rebalance.py` chains: data refresh (`data/refresh.py`, FMP incremental pull) → signal → diff → gate/breaker → submit → poll fills
- [ ] `ops/rdq-rebalance.timer` + `rdq-data-refresh.timer` scheduled pre-open (US/Eastern aware), running as `rdq-exec-paper`
- [ ] Fills written to Notion Trade Ledger; daily summary (orders, fills, rejections, account equity) posted to Slack
- [ ] Failure at any step posts the error to Slack and exits nonzero (systemd `OnFailure` alert as backstop)
- [ ] Dry-run mode (`--dry-run`) prints the order list without submitting; used in the first supervised runs
- [ ] **Milestone check:** 10 consecutive trading days unattended, Ledger reconciles against Alpaca order history (scripted: `ops/reconcile.py` exits 0)

### Phase 6 — Guardrails (paper scope)

#### US-021: Operator halt + breaker controls
**Description:** As an operator, I want to stop trading from Slack instantly.

**Acceptance Criteria:**
- [ ] `halt_trading` / `resume_trading` tools write/remove the breaker halt file; state change confirmed in-thread and logged to Decision Log
- [ ] While halted: rebalancer exits 0 with "halted" notice (not an error), no orders placed
- [ ] Breaker state shown in daily summary

#### US-022: OneCLI approvals bridge (generic, future-proofing for live)
**Description:** As a developer, I want pending OneCLI approvals surfaced in Slack, so the live-trading PRD can gate on it later.

**Acceptance Criteria:**
- [ ] Poller long-polls `GET {ONECLI_URL}/api/approvals/pending`; each pending item posts Approve/Deny buttons to the channel; response submitted back to OneCLI
- [ ] If the API surface differs from expectations, documented fallback: Slack notification only + manual approval in the OneCLI web UI (already tailnet-served on 443) — verified and written up in `docs/decisions.md`
- [ ] No approval rules are created for paper hosts (paper stays friction-free)

#### US-023: Runbook
**Description:** As an operator, I want a tested procedure for when things go wrong.

**Acceptance Criteria:**
- [ ] `ops/runbook.md`: pause research loop, halt rebalancer, flatten all paper positions (scripted: `ops/flatten.py`), rotate keys via OneCLI, audit `tailscale serve status`
- [ ] Flatten script tested against the paper account (positions → zero, confirmed via `GET /v2/positions`)

### Phase 7 — Operations

#### US-024: Monitoring, dashboards, retention
**Description:** As an operator, I want to see what the system is doing and not run out of disk.

**Acceptance Criteria:**
- [ ] `rdagent ui` trace viewer on 127.0.0.1:19900, exposed tailnet-only: `tailscale serve --bg --https=19900 http://127.0.0.1:19900`; reachable from another tailnet device, NOT from public internet
- [ ] Retention sweep (`ops/sweep.py`, weekly timer): deletes non-SOTA, non-promoted workspaces + `mlruns` older than N days; never touches the promoted strategy's workspace
- [ ] All four services/timers healthy after a full host reboot (scripted check `ops/health.sh` exits 0)
- [ ] `ops/health.sh` also audits: no repo service listening on non-loopback interfaces; `tailscale serve status` matches the documented port plan

## 4. Functional Requirements

**Research**
- FR-1: The system must accept free-text investment ideas in Slack threads and refine them into structured directives (objective, universe, constraints) via the Claude layer.
- FR-2: The system must drive RD-Agent(Q) exclusively through its `server_ui` HTTP API with `interaction: true`; the pinned RD-Agent tree must never be edited (all customization via env vars, `APP_TPL`, and template copies under `research/`).
- FR-3: Every hypothesis produced by a run must be posted to the owning Slack thread and must not proceed to coding until the operator approves (possibly with edits) or rejects it.
- FR-4: Runs must support per-run universes; a run must fail fast (before launching) if its universe references tickers absent from the data store.
- FR-5: Completed loops must post IC, ICIR, Rank IC, ARR, IR, MDD, and Sharpe to the thread, with the return chart.

**Data**
- FR-6: The US data store must be built solely from FMP via OneCLI-proxied bare HTTPS; prices must be split/dividend-adjusted (validated against known split events).
- FR-7: Data backfill and refresh must be idempotent and resumable; a partial failure must never leave the store in a state that silently yields wrong backtests (write to temp, validate, swap).

**Trading (paper)**
- FR-8: Only one strategy may be "promoted" at a time; promotion requires an explicit operator confirmation in Slack and writes a Decision Log entry.
- FR-9: Order generation must be fully deterministic given (pred.pkl, positions, config); no LLM call may occur anywhere in `execution/`.
- FR-10: Every order must pass the order gate (limits.paper.json) and breaker check against fresh account state in the same run; rejections must name the violated limit.
- FR-11: The rebalancer must abort without trading if predictions are stale/missing, the market calendar says closed, or the halt file exists.
- FR-12: All orders and fills must be recorded in the Notion Trade Ledger; daily reconciliation against Alpaca order history must be scriptable.

**Platform**
- FR-13: No component may hold a raw credential for Alpaca, Notion, Anthropic, or FMP; all such calls go bare through the OneCLI proxy under the correct per-component identity (`rdq-orchestrator`, `rdq-research`, `rdq-exec-paper`). Slack tokens may fall back to repo-local `.env` only if OneCLI injection is verified unworkable, and this must be documented.
- FR-14: All listeners bind to `127.0.0.1`; human-facing UIs are exposed only via tailnet-only `tailscale serve`; `tailscale funnel` is prohibited.
- FR-15: All long-running components run as systemd user units with `Restart=always`; RD-Agent sessions must be resumable after crash via stored session paths.
- FR-16: Notion is the durable user-facing record (one writer per database); Slack is ephemeral; SQLite is orchestrator-internal state only.
- FR-17: Claude model selection must follow the tier policy: `claude-fable-5` for the orchestrator's judgment layer (with `claude-opus-4-8` server-side fallback and refusal handling), `claude-sonnet-5` for RD-Agent's internal loop, `claude-haiku-4-5` for utility calls. Model choice is centralized in one config/router module, not scattered across call sites.

## 5. Non-Goals (Out of Scope)

- **Live trading.** No `rdq-exec-live` identity, no live secrets, no live approval rules, no code path that targets `api.alpaca.markets`. Live enablement is a separate future PRD; this system must make going live *possible* (identity split, approvals bridge) but not *implemented*.
- Intraday/high-frequency trading — daily rebalance only, matching RD-Agent(Q)'s daily long-only top-k output.
- Options, crypto, shorting, margin management.
- Multi-user or multi-tenant support — one channel, one operator.
- Forking RD-Agent — upstream stays pinned and pristine.
- A custom web dashboard (the Streamlit trace viewer + Notion + Slack are the v1 surfaces).
- Vendor redundancy for market data (FMP only in v1).

## 6. Design Considerations

- Slack UX: Block Kit buttons for all approvals (hypothesis approve/edit/reject, universe confirm, promote, halt) — never free-text parsing for consequential actions. One thread = one research topic/run.
- Charts posted to Slack (return curves) should be rendered server-side (matplotlib) as PNG uploads.
- Notion schema conventions copied into `docs/reference/notion-schema.md` so the design survives nanoclaw's removal.

## 7. Technical Considerations

- **Stack:** Python ≥3.10 throughout (matches RD-Agent). Bolt for Python (Socket Mode), Anthropic SDK, no ORM (sqlite3/`aiosqlite`), pytest.
- **Claude model usage:** exact IDs only — `claude-fable-5` (orchestrator judgment layer; $10/$50 per MTok; requires ≥30-day org data retention; thinking always on — omit the `thinking` param; safety classifiers can return `stop_reason: "refusal"`, so ship the server-side fallback to `claude-opus-4-8` from day one), `claude-sonnet-5` (RD-Agent `CHAT_MODEL` via LiteLLM, `anthropic/claude-sonnet-5`; $3/$15), `claude-haiku-4-5` (utility; $1/$5). Don't downgrade the judgment layer for cost; don't upgrade bulk code-gen or formatting to Fable.
- **RD-Agent specifics:** loops checkpoint per step under `__session__/` (resume via `LoopBase.load`); artifacts per experiment: `combined_factors_df.parquet`, `model.py`, `qlib_res.csv`, `ret.pkl`, `mlruns/**/pred.pkl`. `server_ui` has known flask-cors advisories — keep it loopback-only.
- **OneCLI gotchas:** new identities have zero secrets (assign explicitly); `NO_PROXY` must cover `127.0.0.1,localhost` and the tailnet range so control-plane and dashboard traffic bypass the proxy.
- **FMP specifics:** `/stable/historical-price-eod/full` returns raw closes — adjustment is our job (US-008); respect 429 + `retry-after`; verify the current tier's endpoint access before the backfill design hardens.
- **GPU:** `fin_model`/`fin_quant` model iterations prefer CUDA; factor-only iterations are CPU-fine. If no GPU on this box, constrain early runs to factor loops and note model-loop enablement as environment-dependent.
- **Timezones:** rebalance scheduling is US/Eastern (market calendar), host may be UTC — timers must encode this explicitly.

## 8. Success Metrics

- Idea → first hypothesis in Slack in under 15 minutes (excluding first-time data/image builds).
- A full research run reconstructable from Notion alone (US-016 audit passes).
- 10 consecutive unattended paper trading days with zero manual intervention and clean ledger reconciliation (US-020).
- Zero raw credentials discoverable in the repo (`grep` audit for key patterns passes; only placeholders present).
- All operator actions that change money-touching state (promote, halt, resume) require an explicit Slack confirmation and appear in the Decision Log.

## 9. Open Questions

1. **FMP tier coverage:** does the current FMP plan include the adjusted-EOD variant and enough request volume for a ~1000-ticker, 10-year backfill? (Verify in US-008 before hardening the design; fallback is computing adjustment factors from splits/dividends endpoints.)
2. **Claude JSON-mode via LiteLLM:** US-003 spike outcome — if RD-Agent's stricter JSON-mode paths misbehave with Claude, do we fall back to another LiteLLM-supported chat provider, or patch prompts via `APP_TPL`?
3. **Slack tokens through OneCLI:** injection for `slack.com` Web API + `apps.connections.open` is plausible but unverified (US-005).
4. **OneCLI approvals API shape** for a non-nanoclaw client (US-022) — endpoint semantics for submitting an approval decision need confirmation against the running gateway version.
5. **Universe minimum size:** default warning threshold is 30 names — tune after the first few sector runs?
6. **Benchmark for custom universes:** S&P500 benchmark is wrong for narrow sector universes — accept for v1, or compute an equal-weight universe benchmark?

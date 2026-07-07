# RD-Agent(Q) → Slack-Driven Research & Trading System — Implementation Plan

**Goal:** Adapt [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)'s quant scenario (RD-Agent(Q)) so a user can steer it from a dedicated Slack channel — feeding it investment ideas and research directions — and have validated research turned into real orders executed on Alpaca. User-facing results are written to Notion.

**Standalone constraint:** This repo is **independent of nanoclaw**. It ships its own Slack bot, Alpaca client, Notion client, approval flow, and guardrails. The only external dependency assumed is the **box-wide OneCLI credential gateway** (`http://127.0.0.1:10254`). nanoclaw (`/home/nanoclaw/nanoclaw`) is used strictly as *reference code* — patterns worth copying are noted below and should be **copied into this repo** (not imported), since nanoclaw may not stay around as-is.

---

## 0. Verified assumptions & corrections

Findings from inspecting nanoclaw and RD-Agent before writing this plan:

1. **"openCLI" is OneCLI.** Your mental model is **confirmed with one nuance**: you make **bare HTTP calls to the real service URL** (e.g. `https://api.notion.com/v1/...`, `https://paper-api.alpaca.markets/v2/orders`) with `HTTPS_PROXY` pointed at the OneCLI gateway and its CA cert trusted (`SSL_CERT_FILE`). The proxy matches the destination **host pattern** and injects credentials (headers, bearer, or query params) on the wire. It is *not* URL rewriting — you never address OneCLI directly for API calls, only for management (`/api/agents`, `/api/container-config`, `/api/approvals/pending`). For a host process, either export the proxy env + CA yourself or wrap the command: `onecli run --agent <id> -- <cmd>`.
2. **Gotcha:** newly registered OneCLI agents start in `selective` secret mode with **zero secrets assigned** — every new agent identity needs `onecli agents set-secrets --id <id> --secret-ids ...` (or `set-secret-mode --mode all`) or every API call 401s.
3. **RD-Agent(Q) ships a ready-made control plane** — `rdagent server_ui` (Flask, port 19899, `rdagent/log/server/app.py`) runs research loops as subprocesses with human-in-the-loop interaction queues, exposed over HTTP: `POST /trace` (start run, `interaction: true`, seed with `user_instruction`), `GET /receive` (pending hypothesis/feedback dicts), `POST /user_interaction/submit` (apply user edits), `POST /control` (stop/resume). **Our Slack orchestrator is a thin client of this API — no RD-Agent fork needed for the interaction layer.**
4. **RD-Agent(Q) does NOT support US equities out of the box.** Templates hardcode Qlib China A-share data (`csi300`, `SH000300`, A-share trading rules). Qlib itself supports `region: us`; the port is confined to data prep + two template directories (Phase 2). This is the single most invasive change in the plan.
5. **RD-Agent's internal LLM is LiteLLM-based** (`CHAT_MODEL`, `OPENAI_API_BASE`, `EMBEDDING_MODEL`) and **requires JSON mode + an embedding model**. Anthropic works for chat via LiteLLM, but has no embeddings API — an OpenAI(-compatible) embedding endpoint is required regardless. Both keys go in the OneCLI vault and are injected via the proxy.
6. RD-Agent runs experiments in **Docker** (auto-built `local_qlib:latest`, `qrun` backtests inside), so everything here runs as **host processes** — no container orchestration layer at all in v1. MIT licensed; upstream activity has slowed in 2026 but the quant scenario is stable — **pin a commit**.
7. **Reference patterns to copy out of nanoclaw** (into `docs/reference/` or directly into our code, while the repo is still available):
   - Alpaca raw-fetch client with paper/live split + deterministic order gate: `container/agent-runner/src/mcp-servers/alpaca/` (`server.ts`, `gate.ts`, `policy/limits.{paper,live}.json`)
   - Trading circuit-breaker limits file: `src/modules/trading-breaker/`
   - Notion raw-HTTP usage + DB schema conventions: `plans/thesis-trading/01-notion-schema.md`, `groups/trading-chat-agent-paper/CLAUDE.local.md`
   - OneCLI approval long-poll bridge: `src/modules/approvals/onecli-approvals.ts`

---

## 1. Target architecture

Everything below lives in **this repo**, runs on the host, and is Python (matching RD-Agent, one toolchain).

```
Slack channel #quant-research
        │  Socket Mode (Bolt for Python, WebSocket — no inbound port)
        ▼
┌─ orchestrator/ ────────────────────────────────────────────────┐
│  Slack bot + conversational layer (Claude via Anthropic API,   │
│  tool-use loop; creds injected by OneCLI proxy)                │
│                                                                │
│  Tools: start/steer/stop research runs · read run results ·    │
│  write Notion · read Alpaca account/positions · promote        │
│  strategy · request trade sign-off (Block Kit buttons)         │
│                                                                │
│  Local state: SQLite (thread ↔ run mapping, pending            │
│  interactions, promoted strategy pointer)                      │
└───────┬───────────────────────┬────────────────────────────────┘
        │ HTTP :19899           │ HTTPS via OneCLI proxy
        ▼                       ▼
RD-Agent(Q) service        Notion API · Alpaca API
(rdagent server_ui;
 fin_quant loops: propose → Co-STEER code → qlib backtest
 (Docker) → feedback; outputs per experiment:
 combined_factors_df.parquet, model.py, qlib_res.csv,
 ret.pkl, mlruns/**/pred.pkl ← the signal)

┌─ execution/ ───────────────────────────────────────────────────┐
│  rebalance.py (nightly systemd timer) — deterministic:         │
│  refresh US qlib data → re-run promoted SOTA workspace with    │
│  test_end=today → latest pred.pkl cross-section → top-k        │
│  target weights → diff vs Alpaca positions → order gate +      │
│  breaker check → POST /v2/orders → fills to Notion Ledger +    │
│  Slack summary                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Division of labor:** the orchestrator's Claude layer is *conversation and judgment* (interpret ideas, relay/approve hypotheses, narrate results, collect trade sign-off). RD-Agent(Q) is the *research engine*. The rebalancer is *deterministic plumbing* — deliberately not an LLM.

**Proposed repo layout:**

```
rd-agent-q/
  orchestrator/        # Slack bot (Bolt, Socket Mode), Claude tool-use loop,
                       #   rdagent_client.py, notion_client.py, state.sqlite
  execution/           # alpaca_client.py, order_gate.py, breaker.py,
                       #   limits.paper.json, limits.live.json, rebalance.py
  research/            # RD-Agent config: .env, APP_TPL prompt overrides,
                       #   us_templates/ (patched factor/model YAMLs), base_factors/
  data/                # US qlib data-store build + refresh scripts
  ops/                 # systemd user units, runbook.md
  docs/reference/      # patterns copied from nanoclaw (schema, gate semantics)
  vendor/ or pip pin   # RD-Agent at a pinned commit
```

**OneCLI identities (box-wide gateway, per-identity secret scoping):**

| Identity | Secrets assigned | Used by |
|---|---|---|
| `rdq-orchestrator` | Anthropic, Notion, Alpaca **paper** (read-mostly) | Slack bot / Claude layer |
| `rdq-research` | Anthropic (chat, via LiteLLM) + OpenAI-compatible embedding key + FMP (data builds) | RD-Agent service + data pipeline |
| `rdq-exec-paper` | Alpaca **paper** | rebalancer (default) |
| `rdq-exec-live` | Alpaca **live** | rebalancer, only after Phase 6 sign-off |

Base URL choice = credential choice (nanoclaw's design, kept): `paper-api.alpaca.markets` and `api.alpaca.markets` are separate host patterns with separate secrets, so a paper-scoped identity *cannot* hit live even if the code tries.

**Slack tokens:** Socket Mode needs `xoxb-` + `xapp-` tokens. Both are used in HTTPS calls (`apps.connections.open`, Web API), so OneCLI header injection *should* work — verify in Phase 1; fallback is this repo's own `.env` (they're chat tokens, not brokerage keys — acceptable risk, decide then).

**Networking / Tailscale (verified on this box):** the host is `nanoclaw-prod.tail05c9bf.ts.net` (Tailscale 1.98.3) and already uses **tailnet-only `tailscale serve`** to front local services over HTTPS (`:443 → 127.0.0.1:10254` OneCLI web UI; `:3100 → 127.0.0.1:3001`). This repo follows the same rule: **every service binds to `127.0.0.1` only; nothing listens on a public interface.** Anything a human needs in a browser gets a tailnet-only `tailscale serve` mapping (never `tailscale funnel` — no service here should ever be public). Port plan:

| Service | Binds | Exposure |
|---|---|---|
| Slack Socket Mode | outbound WebSocket only | **none needed** — no inbound port at all |
| `rdagent server_ui` (control API :19899) | 127.0.0.1 | **not exposed** — orchestrator talks to it over localhost; it has known flask-cors advisories, keep it dark |
| `rdagent ui` (Streamlit trace viewer, run on :19900 to avoid the 19899 collision) | 127.0.0.1 | `tailscale serve --bg --https=19900 http://127.0.0.1:19900` when research monitoring is wanted |
| OneCLI web UI (:10254) | 127.0.0.1 | already served at `https://nanoclaw-prod.tail05c9bf.ts.net/` — used for approval rules (Phase 6) and manual approvals fallback |
| Any future dashboard (e.g. P&L page) | 127.0.0.1 | new tailnet-only serve mapping on a dedicated HTTPS port; pick ports that don't collide with the existing 443/3100 mappings |

`NO_PROXY` for all our processes must include `127.0.0.1,localhost` and the tailnet hostname/`100.64.0.0/10` range, so localhost control traffic and tailnet dashboard traffic never route through the OneCLI proxy.

---

## 2. Phases

### Phase 0 — Environment bring-up (RD-Agent vanilla)

*Prove RD-Agent(Q) runs at all before changing anything.*

1. Scaffold this repo (layout above); vendor or pip-pin RD-Agent at the current stable commit; Python ≥3.10 venv.
2. Register OneCLI identities (table above); vault the LLM keys; assign secrets (remember the selective-mode gotcha).
3. RD-Agent `.env`: `CHAT_MODEL`, `EMBEDDING_MODEL`, `OPENAI_API_BASE`, placeholder `OPENAI_API_KEY` (proxy injects the real one), `WORKSPACE_PATH`, `LOG_TRACE_PATH`. Launch pattern: `onecli run --agent rdq-research -- rdagent ...`.
   - *Decided:* RD-Agent's internal chat model is **Claude via LiteLLM**. Phase 0 includes a spike verifying LiteLLM JSON-mode behavior with Claude on RD-Agent's hypothesis prompts; an OpenAI-compatible **embedding** endpoint is still required (Anthropic has none). If the spike fails, fall back to an OpenAI chat model internally.
   - **Model-tier policy (match model to stakes):**
     | Workload | Model | Why |
     |---|---|---|
     | Orchestrator conversational layer — directive refinement, hypothesis relay/edit synthesis, promotion & trade-approval context | `claude-fable-5` | High-stakes judgment; ship with server-side fallback to `claude-opus-4-8` (`fallbacks` param + `server-side-fallback-2026-06-01` beta) and `stop_reason: "refusal"` handling. Note: Fable 5 requires the org to have ≥30-day data retention. |
     | RD-Agent internal loop (`CHAT_MODEL=anthropic/claude-sonnet-5`) — hypothesis gen + Co-STEER code evolution | `claude-sonnet-5` | High call volume, mostly code generation; near-Opus coding quality at ~⅓ the cost of Fable. RD-Agent has one global `CHAT_MODEL`; a LiteLLM-router split (Fable for hypothesis prompts, Sonnet for coding) is a possible later refinement, not v1. |
     | Orchestrator utility calls — Slack formatting, loop-log summarization for thread updates, ticker-list extraction | `claude-haiku-4-5` | Cheap, fast, no judgment required. |
     | Embeddings (RD-Agent requirement) | OpenAI-compatible endpoint | Anthropic has no embeddings API. |
4. `rdagent health_check` (Docker sudo-less, ports free). First run downloads cn_data + builds `local_qlib:latest` — slow; budget for it.
5. **Milestone:** `rdagent fin_factor --loop_n 1` completes on default China data; trace visible in `rdagent ui`.

### Phase 1 — Slack bot + conversational orchestrator (this repo's own)

*The user-facing surface, built from scratch — no nanoclaw runtime.*

1. Create a Slack app (Socket Mode; scopes: `chat:write`, `channels:history`, `reactions:write`, `commands` if we add slash commands) and a `#quant-research` channel. Bolt for Python listener: messages in the channel/threads reach the orchestrator; no inbound port needed.
2. Orchestrator core: a **Claude tool-use loop** (Anthropic Python SDK through the OneCLI proxy). System prompt: quant research desk PM — take user ideas, structure them into research directives, drive RD-Agent, report honestly (including failed hypotheses), never trade without explicit approval. Tools are thin wrappers over `rdagent_client.py`, `notion_client.py`, `alpaca_client.py` (read-only here).
3. Threading model: one Slack thread per research topic/run; SQLite maps `thread_ts ↔ run/session path ↔ Notion page`. Approvals (hypothesis edits, trade sign-off) rendered as **Block Kit buttons**, not free-text parsing; button actions land in the same Bolt app.
4. Run as a systemd user unit (`rdq-orchestrator.service`).
5. **Milestone:** user converses with the bot in the channel; it echoes structured "research directive" summaries and persists them.

### Phase 2 — US market port (the invasive change)

*Patches on the pinned RD-Agent commit; keep the diff small, hold it in `research/us_templates/` + `APP_TPL` overrides rather than in-tree edits.*

1. Build a US Qlib data store at `~/.qlib/qlib_data/us_data` from **FMP (Financial Modeling Prep)** — already vaulted in OneCLI box-wide (bare HTTPS to `financialmodelingprep.com`, proxy injects `?apikey=`; usage pattern in nanoclaw's `container/agent-runner/src/mcp-servers/fmp/server.ts`). `data/build_store.py` pulls daily EOD bars via `/stable/historical-price-eod/full?symbol=&from=&to=`, converts to Qlib format (`dump_bin`), builds the trading calendar. **Caveats:** FMP's EOD close is raw (not split/dividend-adjusted) — Qlib needs adjusted prices or a `factor` column, so use FMP's adjusted variant if the plan tier has it, else compute adjustment factors from splits/dividends endpoints; respect FMP rate limits when backfilling ~1000+ tickers (batch, checkpoint, resume). **Pull history for a broad set of US-listed tickers, not just an index** — the store is the superset everything else selects from; a ticker absent here is unreachable at research time.
2. **Universes are instrument lists inside the store** (`instruments/*.txt`, referenced by the template's `market:` field). Ship two from the start:
   - `us_liquid` (default) — broad but liquidity/price-filtered (roughly Russell-1000-ish: min ADV and min price), so the top-k strategy can't rank into illiquid microcaps;
   - `sp500` — for benchmarking/sanity checks.
   The universe is fixed per run at data-generation time; `data/make_universe.py` builds an instruments file + regenerates the matching factor source data.
3. Patch copies of `rdagent/scenarios/qlib/experiment/factor_template/` and `model_template/` YAMLs: `provider_uri` → us_data, `region: us`, `market: us_liquid`, S&P500 benchmark, remove A-share `limit_threshold`/costs, set realistic US costs.
4. Regenerate factor source data (`factor_data_template/generate.py` → `daily_pv_all.h5`) from the US store for the default universe; set `FACTOR_CoSTEER_DATA_FOLDER`.
5. Set date splits (`QLIB_QUANT_TRAIN_START` … `QLIB_QUANT_TEST_END`) to sensible US ranges.
6. Skim `prompts.yaml` for A-share-specific language; override via `APP_TPL` directory.
7. **Milestone:** `rdagent fin_quant --loop_n 2` completes end-to-end on US data (default `us_liquid` universe) with plausible metrics in `qlib_res.csv`.

### Phase 3 — Orchestrator ↔ RD-Agent control loop

1. Run `rdagent server_ui` as a systemd unit (`rdq-research.service`, port 19899, **bound to 127.0.0.1**, via `onecli run`). Localhost-to-localhost — ensure `NO_PROXY` covers localhost so control traffic skips the OneCLI proxy. Do **not** put this behind `tailscale serve` (see port plan in §1); the trace viewer `rdagent ui` on :19900 is the thing to expose for humans, tailnet-only.
2. `orchestrator/rdagent_client.py`: start run (`POST /trace` with `interaction: true`, user idea as `user_instruction`, optional `base_factors.json` seed), poll `GET /receive`, submit edits (`POST /user_interaction/submit`), stop/resume (`POST /control`), locate finished-loop artifacts (`qlib_res.csv`, `ret.pkl`, workspace paths) from `LOG_TRACE_PATH`.
3. Idea-injection paths, best-first: (a) `user_instruction` + hypothesis-edit via interaction queues (zero fork); (b) `--base_features_path` factor seeds; (c) only if needed, a custom `HypothesisGen` plugged in via `QLIB_QUANT_QUANT_HYPOTHESIS_GEN=<dotted.path>` env var — still not a fork.
4. **Per-run custom universes** — how ticker/sector-specific ideas map onto a cross-sectional framework. RD-Agent(Q) ranks *within* a universe, so "look into NVDA" becomes "research a universe where NVDA and its peers live." Orchestrator tool `set_universe(tickers | sector_query)`:
   - resolve the idea to a ticker list (user-supplied list, or the Claude layer proposes one for confirmation in-thread);
   - validate every ticker exists in the us_data store (report gaps — the store is the ceiling, per Phase 2);
   - call `data/make_universe.py` to write the instruments file + regenerate that universe's `daily_pv` h5 (cheap for a sector list; refuse "all US" here and point at the default instead);
   - render that run's template copy with `market: <custom>` and launch. Record the universe (name + tickers) in the run's Notion Research Ideas entry so results are interpretable later.
   Runs with no universe specified use the default `us_liquid`.
5. Async UX: a background poller watches `/receive`; pending hypotheses post to the owning Slack thread with approve/edit buttons; loop-complete events post the metrics summary (IC/ICIR/ARR/IR/MDD/Sharpe) + return chart from `ret.pkl`.
6. **Milestone:** from Slack — "research momentum factors on semiconductor names" → bot proposes a semiconductor universe for confirmation, run starts on it, hypothesis appears in-thread, user edits it, loop completes, bot posts the backtest summary.

### Phase 4 — Notion reporting

*Notion is the durable user-facing record; Slack is the ephemeral console.*

1. `orchestrator/notion_client.py`: raw HTTPS to `https://api.notion.com/v1/...` with `Notion-Version: 2022-06-28` (OneCLI injects auth). Databases under one parent page, **one writer per DB** (convention copied from nanoclaw's thesis-trading schema):
   - **Research Ideas** — raw idea, refined directive, status, run links *(writer: orchestrator)*
   - **Hypothesis Log** — each hypothesis, action (factor/model), user edits, accept/reject *(orchestrator)*
   - **Backtest Results** — per-experiment metrics, SOTA flag, workspace path *(orchestrator)*
   - **Decision Log** — human decisions: approvals, strategy promotions, live sign-offs *(orchestrator)*
   - **Trade Ledger** — every order + fill *(writer: rebalancer)*
2. DB IDs in `orchestrator/config.yaml` (committed IDs are fine; they're not secrets).
3. **Milestone:** a full research run is reconstructable from Notion alone.

### Phase 5 — Signal → trades (paper first)

*Deterministic rebalancer; the LLM's authority ends at approvals.*

1. **Strategy promotion:** user clicks "promote" in Slack → orchestrator records the promotion in Decision Log and pins the SOTA workspace path in SQLite as the *production strategy*.
2. **`execution/rebalance.py`** (nightly pre-open systemd timer, `onecli run --agent rdq-exec-paper`):
   - refresh the US qlib store to latest close (`data/refresh.py`);
   - re-run the promoted workspace (`qrun conf.yaml` with `test_end=today`, or load `params.pkl` + `model.predict`) → latest `pred.pkl` cross-section;
   - replicate `TopkDropoutStrategy` (topk/n_drop from the promoted config) → target holdings → weights;
   - `GET /v2/positions` (bare fetch; proxy injects `APCA-*` headers) → compute order diff;
   - `order_gate.py`: max order value, max position %, day-order counts (semantics ported from nanoclaw's `limits.{paper,live}.json`), plus `breaker.py` halt-file check;
   - submit marketable-limit orders (`POST /v2/orders`), write fills to Trade Ledger, post a fill summary to Slack.
   - RD-Agent's pipeline is **daily-frequency, long-only top-k** — a once-a-day rebalance matches it exactly; no intraday plumbing.
3. Order-diff and gate logic get unit tests with fixture `pred.pkl`/positions data — this is the code that spends money.
4. **Milestone:** ≥2 weeks unattended paper trading; Ledger reconciles with Alpaca account history; daily Slack summaries.

### Phase 6 — Guardrails & live gating

1. **Identity isolation (primary control):** the rebalancer runs as `rdq-exec-paper` by default; live keys exist only on `rdq-exec-live`. Going live = a deliberate config change *and* secret assignment — hitting `api.alpaca.markets` without the live identity just 401s.
2. **Human approval for live orders:** configure a OneCLI approval rule on the live Alpaca host pattern (web UI at `:10254`; the CLI can't create approval rules as of onecli 1.3.0). Build our own small **approvals bridge** in the orchestrator: long-poll `GET {ONECLI_URL}/api/approvals/pending` and post approve/deny buttons to Slack (pattern from nanoclaw's `onecli-approvals.ts`, re-implemented in Python). Every live order then requires a human tap even if all our software gates fail. If the pending-approvals API surface differs from expectations, fall back to approving via the OneCLI web UI (still a hard human gate) and keep Slack notification only.
3. `breaker.py`: max daily notional, max position count, drawdown kill-switch (halt + Slack alert if live equity drops X% below high-water mark). Breaker state is a file; the orchestrator gets a "halt trading" tool that writes it.
4. `ops/runbook.md`: pause research loop, halt rebalancer, flatten positions, rotate keys.
5. **Milestone:** live enablement is a multi-step deliberate act (Decision Log entry + identity switch + approval rule), documented and reversible.

### Phase 7 — Operations

- systemd user units: `rdq-orchestrator.service`, `rdq-research.service`, `rdq-rebalance.timer`, `rdq-data-refresh.timer`.
- Tailscale exposure (tailnet-only, per the §1 port plan): `tailscale serve --bg --https=19900 http://127.0.0.1:19900` for the research trace viewer; add mappings for any future dashboards. `tailscale serve status` is the source of truth for what's exposed; audit it in the runbook. Never use `funnel`.
- RD-Agent loops checkpoint after every step (`__session__/` pickles); crashes resume via `LoopBase.load(path)` — the orchestrator stores session paths and exposes a resume tool.
- Monitoring: `rdagent ui` for research traces; orchestrator/rebalancer failures page the Slack channel; disk retention sweep for workspaces + `mlruns/` (keep SOTA/promoted only).
- Pin everything: RD-Agent commit, qlib Docker image, limits files, Slack/Anthropic SDK versions. Document the procedure for rebasing our template/prompt patches onto an upstream update.

---

## 3. Key risks

| Risk | Mitigation |
|---|---|
| US data quality (FMP raw closes need split/dividend adjustment; survivorship in the small-cap tail) | Adjustment handling in `build_store.py` (adjusted endpoint or computed factors) validated against known split events (e.g. NVDA 2024 10:1); liquidity-filtered `us_liquid` default universe; sanity-check factors against known benchmarks |
| FMP rate limits / plan-tier gaps during ~1000-ticker backfill | Batched, checkpointed, resumable backfill; verify adjusted-EOD endpoint availability on the current tier in Phase 2 step 1 before committing |
| Tiny custom universes → statistically weak cross-sectional results (top-k over 20 names ≈ noise) | Orchestrator warns below a minimum universe size and pads with a confirmed peer group; results in Notion always record the universe |
| Backtest ≠ live (overfit factors, cn-tuned defaults) | Long paper period; promote only SOTA strategies with out-of-sample windows; small initial live sizing |
| RD-Agent upstream drift / slowed maintenance | Pin commit; customization confined to env-var plugins + `APP_TPL` + template copies, so the effective diff stays ~2 directories |
| JSON-mode/embedding constraints on the internal LLM | Validate the backend in Phase 0 step 3 before anything depends on it |
| Slack tokens through OneCLI unverified | Test header injection for `slack.com` in Phase 1; fallback to repo-local `.env` for chat tokens only |
| OneCLI approvals API shape unverified for a non-nanoclaw client | Phase 6 step 2 has a web-UI fallback that preserves the human gate |
| GPU: `fin_model` loop wants CUDA | Start factor-only loops (LGBM, CPU-fine); enable model evolution only with a GPU |
| Long backtests hanging (known upstream issue) | `--all_duration` wall-clock budgets on every run; poller reports stalls to Slack |
| Accidental public exposure of control/trading surfaces | Everything binds 127.0.0.1; human access only via tailnet-only `tailscale serve` (never funnel); `server_ui` (flask-cors advisories) stays localhost-only; runbook audits `tailscale serve status` |

## 4. Explicit non-goals (v1)

- No intraday/HFT — daily rebalance only, matching RD-Agent(Q)'s output.
- No options/crypto/shorting — long-only US equities top-k.
- No LLM computing order quantities — the rebalancer is deterministic.
- No RD-Agent fork beyond US-market template copies + prompt overrides.
- No multi-user/multi-tenant handling — one channel, one operator.

## 5. Suggested order of attack

Phases 0 and 1 are independent — run them in parallel. Then 2 (the long pole — start early), 3 and 4 interleave, then 5 → 6 → 7 strictly in order. First user-visible win is the Phase 3 milestone: idea in Slack → hypothesis approval in-thread → backtest summary back in the thread.

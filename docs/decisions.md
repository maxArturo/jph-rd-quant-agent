# Decision Log

Running log of technical decisions with rationale. Newest entries at the bottom.

## 2026-07-07 — RD-Agent pinned commit (US-002)

**Decision:** Pin microsoft/RD-Agent to commit
`4f9ecb005881cddc08df0124a2e894c018007679` (main HEAD as of 2026-05-06,
"Document FT-Agent ICML release (#1406)").

**Why:**
- Upstream activity slowed in 2026 (PLAN.md finding #6); this commit has been
  the unmoved tip of `main` for ~2 months at pin time — the de facto stable.
- It is 28 commits ahead of the last tagged release `v0.8.0` (2025-11-03) and
  includes the post-release fixes; setuptools-scm reports it as `0.8.1.dev28`.
- Pinning a full SHA (not a branch or tag) makes installs reproducible and
  makes our template/prompt patch surface (US-016) diffable against a fixed
  upstream tree.

**Install path:** `research/PINNED_COMMIT` holds the SHA;
`research/install.sh` installs `rdagent @ git+https://github.com/microsoft/RD-Agent@<SHA>`
into the project venv (`.venv`) and verifies the import. RD-Agent is
deliberately NOT a `pyproject.toml` dependency: its heavy dependency tree
(litellm, streamlit, docker, ...) is only needed by the research engine, and a
direct-URL git pin in `pyproject` would slow every editable reinstall.

**Verification:** `pip check` clean; `python -c 'import rdagent'` exits 0;
pip's `direct_url.json` records `commit_id = 4f9ecb00...` (exact pin).

**Rebase procedure (future upstream update):** update `research/PINNED_COMMIT`,
re-run `research/install.sh`, re-run the upstream-pristine test (US-016) and
re-diff `research/us_templates/` + `research/app_tpl/` against the new tree.

## 2026-07-07 — Embeddings via Voyage AI, no OpenAI anywhere

**Decision:** RD-Agent's required embedding model is **Voyage AI**
(`EMBEDDING_MODEL=voyage/voyage-3.5-lite`, host `api.voyageai.com`,
key `VOYAGE_API_KEY`). The OpenAI embedding dependency assumed at planning
time is dropped; the stack now uses no OpenAI APIs at all.

**Why:**
- Anthropic has no embeddings API, so a second provider is unavoidable for
  RD-Agent's knowledge base (cross-iteration retrieval/dedup memory) — but it
  does not have to be OpenAI. User wants an Anthropic-first, OpenAI-free stack.
- Voyage is Anthropic's recommended embedding partner and a first-class
  LiteLLM provider (`voyage/` prefix) — no proxy shim or OPENAI_API_BASE
  override needed.
- Hosted beats self-hosted here: the 8GB droplet also runs qlib backtests
  (multi-GB spikes); a local embedder (Ollama/TEI, ~0.5–1.2GB resident) would
  reserve RAM we need and add a service to babysit. RD-Agent's embedding
  volume (short hypothesis/error texts) sits comfortably inside Voyage's
  200M-token free tier.
- Escape hatch: any OpenAI-compatible local server (e.g. `infinity_emb` with
  `bge-small-en-v1.5`, ~400MB) is a two-env-var swap if we ever want zero
  external vendors.

**Changes:** PLAN.md (findings #5, identity table, Phase 0 step 3 + model-tier
table), scripts/ralph/prd.json US-004 acceptance criteria,
ops/setup_onecli.sh (`api.openai.com` → `api.voyageai.com` for rdq-research),
ops/check_onecli.sh (Voyage auth probe is a minimal 1-token POST to
`/v1/embeddings` — Voyage has no GET endpoint; probe() gained an optional
POST-body field).

**Action required (web UI):** vault the Voyage key for host
`api.voyageai.com`, then rerun `ops/setup_onecli.sh` — replaces the
previously planned OpenAI key. US-004's smoke test depends on it.

**Update 2026-07-08:** DONE. The Voyage key is now vaulted for
`api.voyageai.com` (confirmed via `onecli secrets list`). Rerun
`ops/setup_onecli.sh` so `rdq-research` gets it assigned (it was skipped
while the host had no vaulted secret), then US-004 can proceed.

## 2026-07-08 — Slack tokens via repo-local .env (not OneCLI)

**Decision:** Slack Socket Mode auth uses a repo-local, gitignored `.env`
rather than the OneCLI proxy. Two tokens:
- `SLACK_OAUTH_TOKEN` — the `xoxb-` bot token (Web API: `chat:write`,
  `channels:history`, `reactions:write`; `files:write` needed later for
  US-022 chart uploads).
- `SLACK_SOCKET_TOKEN` — the `xapp-` app-level token for
  `apps.connections.open` (Socket Mode websocket).

**Why:** These are chat tokens, not brokerage/market credentials — PLAN.md §1
pre-authorized this as an acceptable narrow exception (FR-13). Bolt's Socket
Mode client opens its own outbound websocket and does not go through an HTTPS
proxy cleanly, so routing it via OneCLI adds friction with no security gain
for chat-scoped tokens. Every money- or data-touching credential (Anthropic,
Voyage, FMP, Alpaca paper, Notion) still flows exclusively through OneCLI.

**Guardrail:** `.env` (and `research/.env`) are gitignored; the repo's
zero-raw-credentials success metric applies only to committed files. Do not
route Slack through the proxy; do not add these tokens to the OneCLI vault.

## 2026-07-08 — Notion parent page

**Decision:** All Notion databases (Research Ideas, Hypothesis Log, Backtest
Results, Decision Log, Trade Ledger) live under the operator's page
**"Automated AI Quant Investment"**, page id
`3979b1a4-36cf-8046-baa5-cc14c1ca7665`
(https://app.notion.com/p/Automated-AI-Quant-Investment-3979b1a436cf8046baa5cc14c1ca7665).
`ops/bootstrap_notion.py` (US-026) creates the five DBs under this page and
writes their ids into `orchestrator/config.yaml`.

**Auth: no Notion token in our vault or code.** OneCLI injects Notion auth for
`api.notion.com` via a connector-style integration ("JPH NanoClaw Connection",
integration id `36dd872b-594c-81c8-a8bc-00377693d395`), NOT a standard vaulted
secret — so it does NOT appear in `onecli secrets list` and
`ops/setup_onecli.sh` will print a harmless WARN about no vault secret for
`api.notion.com`. Ignore that WARN; do not try to vault a Notion key. Verified
2026-07-08: a bare proxied `GET /v1/users/me` as `rdq-orchestrator` returns
HTTP 200 (`ops/check_onecli.sh`'s Notion check now passes).

**Page sharing: DONE (verified 2026-07-08).** The parent page has been shared
with the "JPH NanoClaw Connection" integration; a bare proxied
`GET /v1/pages/3979b1a4-36cf-8046-baa5-cc14c1ca7665` as `rdq-orchestrator`
returns HTTP 200 with the page object (title "Automated AI Quant Investment").
US-026's live bootstrap and all real Notion writes are unblocked. No remaining
human action for Notion.

## 2026-07-08 — LLM backend probe outcome: Claude via LiteLLM CONFIRMED (US-004)

**Decision:** RD-Agent's LLM backend is confirmed as
`CHAT_MODEL=anthropic/claude-sonnet-5` + `EMBEDDING_MODEL=voyage/voyage-3.5-lite`,
both through LiteLLM 1.91.0 (the version rdagent's pin installs) and the
OneCLI proxy under the `rdq-research` identity. No fallback provider needed.

**Evidence** (`onecli run --agent rdq-research -- .venv/bin/python
research/probe_llm.py`, exit 0):
- Chat: JSON-mode (`response_format={"type": "json_object"}`) hypothesis
  prompt returned a JSON object that validates against the probe's
  hypothesis schema (`hypothesis`/`rationale`/`confidence`).
- Embeddings: one `voyage/voyage-3.5-lite` call returned a 1024-dim float
  vector end-to-end through the proxy.

**Behaviors found (encode these in later stories):**
- LiteLLM rejects `temperature` != 1 for `claude-sonnet-5`
  (`UnsupportedParamsError` — it treats it as a reasoning model). Omit the
  parameter (or set `litellm.drop_params = True`) anywhere RD-Agent configs
  let us control it.
- Anthropic JSON mode via LiteLLM may still wrap the object in ```json
  fences; parse tolerantly (see `extract_json_object` in
  `research/probe_llm.py`).
- Placeholder API keys work: the proxy overrides auth headers, so
  `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY` only need to be non-empty client-side
  (`research/.env.example` documents this).
- `onecli run` injects `HTTPS_PROXY` + `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`
  etc.; LiteLLM's httpx stack honors them with no extra wiring.

## 2026-07-08 — pydantic-ai-slim pinned to 1.107.0 (US-005)

**Problem:** the `rdagent` CLI crashed on import
(`ImportError: cannot import name 'MCPServerStreamableHTTP' from
'pydantic_ai.mcp'`). rdagent's pin leaves `pydantic-ai-slim[mcp,openai,prefect]`
unpinned; pip resolved it to 2.5.1, and the pydantic-ai 2.x line renamed the
MCP server classes (`MCPServerStreamableHTTP` → `MCPToolset` family) that
rdagent 4f9ecb00 imports.

**Decision:** `research/install.sh` now pins
`pydantic-ai-slim[mcp,openai,prefect]==1.107.0` — the last 1.x release, API-
compatible with the rdagent pin — immediately after installing rdagent, and
verifies `from rdagent.app.cli import app` (the full CLI import graph) rather
than just `import rdagent`. `pip check` is clean with this combination.

**When to revisit:** whenever `research/PINNED_COMMIT` is rebased; if the new
upstream commit supports pydantic-ai 2.x, drop the extra pin.

## 2026-07-08 — health_check env leg skipped in run_vanilla_factor.sh --check (US-005)

`rdagent health_check`'s env leg (`env_check()`) only understands
DeepSeek/OpenAI env layouts: with our Anthropic+Voyage variables it takes the
"no valid configuration" branch and then crashes with `UnboundLocalError`.
`ops/run_vanilla_factor.sh --check` therefore runs
`rdagent health_check --no-check-env` (docker + ports, the two things the
story cares about) and hard-asserts what upstream only logs as warnings
(sudo-less docker, port 19899 free, onecli gateway + rdq-research identity).
The LLM leg is covered separately and better by `research/probe_llm.py`
through the proxy (US-004).

## 2026-07-08 — US templates + APP_TPL prompt overrides (US-016)

**Goal:** point RD-Agent's qlib scenario at US data without editing the pinned
upstream tree. Two mechanisms: (1) full copies of the two workspace template
folders under `research/us_templates/` with US values patched in, (2) partial
prompt overrides under `research/app_tpl/` loaded via RD-Agent's `APP_TPL`
setting (env var `APP_TPL`, `RDAgentSettings.app_tpl` — no env prefix).

**Grep audit for A-share-specific language** (pinned rdagent 4f9ecb00;
`grep -riE 'csi|a-share|china|chinese|SH000300|cn_data|yuan|RMB' --include='*.yaml'`
over `rdagent/scenarios/qlib`, `rdagent/components/coder`,
`rdagent/app/qlib_rd_loop`):

| location | finding | handling |
|---|---|---|
| `experiment/factor_template/*.yaml`, `experiment/model_template/*.yaml` (5 files) | `cn_data`, `csi300`, `SH000300`, `limit_threshold: 0.095`, CN costs | patched copies in `research/us_templates/` |
| `experiment/prompts.yaml` → `qlib_factor_experiment_setting`, `qlib_model_experiment_setting` | "CSI300" dataset row in the experiment-setting tables | overridden in `research/app_tpl/scenarios/qlib/experiment/prompts.yaml` ("US stocks (us_liquid universe)") |
| `factor_experiment_loader/prompts.yaml` → `factor_viability_system`, `factor_relevance_system`, `factor_duplicate_system` | "daily frequency strategy in China A-share market" | overridden in `research/app_tpl/scenarios/qlib/factor_experiment_loader/prompts.yaml` ("the US equity market") |
| `factor_experiment_loader/prompts.yaml` → `classify_system_chinese` | Chinese-language classifier prompt | NOT overridden — dead code; only `classify_system` is referenced (`pdf_loader.py`) |
| everything else (incl. top-level `scenarios/qlib/prompts.yaml`, factor/model coder prompts) | no matches | — |

**Template YAML decisions:**
- **Benchmark `SPY`** (S&P 500 ETF), not the `^GSPC` index: qlib resolves the
  benchmark as an instrument in the store, and `SPY` flows through the existing
  FMP bars/splits/dividends pipeline like any stock symbol. Consequence: the
  us_data store build must include `SPY` (US-017 checks this).
- **Costs:** `open_cost: 0.0005`, `close_cost: 0.0005` (5 bps/side as a
  spread+slippage proxy; Alpaca is commission-free so a symmetric estimate
  replaces CN's asymmetric commission+stamp-duty 5/15 bps), `min_cost: 0`
  (no per-order minimum). `deal_price: close` and `account` unchanged.
- **`limit_threshold` removed** (A-share ±10 % daily price limit); qlib's
  `region: us` defaults it to `None`, and `trade_unit` to 1 (no board lots).

**APP_TPL mechanics (verified live):** `load_content()` prepends
`<app_tpl>/scenarios/qlib/.../prompts.yaml` to the search list; an absolute
`APP_TPL` path works (`Path / absolute-str` yields the absolute path). A
missing key in the override file raises `KeyError` internally and falls
through to upstream — so override files hold ONLY the overridden keys.
Regenerate overrides after a re-pin by re-extracting the keys from upstream
and re-applying the phrase substitutions above (tests re-audit the text).

**Consumption note for US-017/US-023:** `APP_TPL` does NOT cover the workspace
template folders — `QlibFactorExperiment`/`QlibModelExperiment` hardcode
`Path(__file__).parent / "factor_template"`. The supported hook is the
`QLIB_QUANT_*` env-configurable class paths in `rdagent/app/qlib_rd_loop/conf.py`
(e.g. point `scen`/`*_hypothesis2experiment` at small subclasses in our repo
that construct `QlibFBWorkspace` with `research/us_templates/...`).

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

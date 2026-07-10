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

## 2026-07-08 — server_ui control protocol: real endpoints vs PRD sketch (US-019)

**Decision:** `orchestrator/rdagent_client.py` speaks the REAL protocol of the
pinned upstream `rdagent/log/server/app.py`, not the endpoint names sketched
in the PRD acceptance criteria. The PRD assumed `POST /trace` starts a run,
`GET /receive` lists pending interactions, and `POST /control` supports
stop **and** resume. Upstream reality (verified by reading the pinned source
and probing the live service):

| Operation | PRD sketch | Actual upstream endpoint |
|---|---|---|
| start run | `POST /trace` | `POST /upload` (form: `scenario="Finance Whole Pipeline"`, `loops`, `all_duration`) → `{"id": "<scenario>/<trace_name>"}` |
| pending interactions | `GET /receive` | `POST /trace` `{"id", "all", "reset"}` — the message poll; each call drains ≤1 pending user-interaction request into the stream as a `tag="user_interaction.request"` message. (`POST /receive` is the *ingestion* endpoint the rdagent subprocess logger pushes messages to.) |
| answer interaction | `POST /user_interaction/submit` | same (payload = `{"id", "payload"}`) |
| stop | `POST /control` | same, `action="stop"` only |
| resume | `POST /control` | **not supported** — upstream 400s `"Only 'stop' action is supported"`. Client `resume()` sends `{"id", "action": "resume", "path"}` and maps that 400 to `UnsupportedActionError`. US-024 must add a resume extension to `research/server_ui.py` (e.g. a `before_request` hook that builds an `RDAgentTask(target_name="fin_quant", kwargs={"path": <session>})`), since the pinned tree cannot be patched. |

**Interaction handshake:** a server-started `fin_quant` run always gets IPC
queues (`fin_quant` is not in `_TARGETS_WITHOUT_USER_INTERACTION`) and blocks,
in order, on (1) init params — the response dict updates `plan`, and its
`user_instruction` key is where the operator directive lands; (2) base
features — expects a `{name: qlib_expression}` dict back; then on every
hypothesis and feedback. Requests and responses travel on independent FIFO
queues, so `start_run()` pre-seeds (1) the directive (+universe constraint)
and (2) rdagent's default ALPHA20 features immediately after `/upload` —
the run reaches hypothesis generation without an operator, and hypotheses/
feedbacks surface via `pending()` for the US-021 poller. `interaction=False`
cannot disable the queues server-side; the flag is recorded on `RunHandle`
so pollers know to auto-approve instead of waiting for the operator.

**Artifact locator:** a finished loop's backtest lands in the experiment
workspace (`qlib_res.csv` + `ret.pkl`, written by the workspace's
`read_exp_res.py`); the trace→workspace link only exists inside the pickled
`runner result` FileStorage objects (`experiment_workspace.workspace_path`)
— the /trace JSON stream does NOT carry workspace paths. `locate_artifacts()`
therefore unpickles `**/runner result/**/*.pkl` newest-first and returns the
first workspace containing `qlib_res.csv`.

**WARNING for US-020:** `ops/rdq-research.service` does NOT carry the US env
wiring from `ops/run_us_quant.sh` (QLIB_QUANT_/QLIB_FACTOR_/QLIB_MODEL_ dates,
`FACTOR_CoSTEER_DATA_FOLDER(_DEBUG)`, `APP_TPL`, `QLIB_QUANT_*` hook-class
paths). Server-started runs inherit the *server's* environment, so a
`start_run()` today would launch a CN-defaults `fin_quant`. US-020 must add
the env wiring to the unit (Environment=/EnvironmentFile=) before wiring
`start_research` to this client.

## 2026-07-08 — US-021: Slack hypothesis steering maps onto the upstream queue protocol

The poller (`orchestrator/poller.py`) posts every pending hypothesis as Block
Kit Approve/Edit/Reject buttons in the run's Slack thread. Mapping the three
operator actions onto the pinned upstream protocol
(`RDLoop._interact_hypo`: the answer must be the full hypothesis constructor
dict — the loop rebuilds via `type(hypo)(**res_dict)`):

- **Approve** submits the request `content` unchanged.
- **Edit** is an in-thread text round-trip (no modal): the button flips the
  SQLite row to `editing` and prompts for a reply; the operator's next
  message in that thread is submitted as `{**content, "hypothesis": <text>}`.
  Chosen over a Slack modal because the state persists in
  `pending_interactions` — a restart mid-edit keeps the round-trip alive,
  whereas modal view state would be lost.
- **Reject:** the queue protocol has NO regenerate/skip action — every
  hypothesis request must be answered with a constructible dict, and the loop
  always proceeds to convert/code/run it. Reject therefore submits the dict
  with the `hypothesis` text replaced by an explicit operator-rejection
  instruction ("do not implement it ... propose a materially different
  hypothesis in the next iteration"). The iteration itself still runs (likely
  failing fast into `skip_loop_error`), and the rejection lands in the trace,
  steering the next proposal. A true skip would need a server/loop extension
  (candidate for the US-024 `research/server_ui.py` work).

**Feedback interactions are auto-acknowledged** (submitted back unchanged,
row status `auto_approved`): the run blocks on feedback exactly like on
hypotheses, so leaving them unanswered would deadlock every run after its
first experiment. Operator steering of feedback is out of US-021 scope.

**FIFO safety:** responses answer the run's *oldest* unanswered request, so
the poller processes a run's interactions oldest-first and stops at the first
hypothesis still awaiting the operator — nothing may jump the queue (e.g. a
later feedback must not be auto-acked while a hypothesis is pending).

## 2026-07-08 — US-022: Sharpe is derived from ret.pkl (qlib does not log it)

The PRD asks the completion summary to post "IC, ICIR, Rank IC, ARR, IR, MDD,
Sharpe parsed from qlib_res.csv". The first six exist in qlib_res.csv under
qlib's real recorder keys (`IC`/`ICIR`/`Rank IC` from SigAnaRecord;
ARR/IR/MDD as `1day.excess_return_with_cost.{annualized_return,
information_ratio,max_drawdown}` from risk_analysis) — but **qlib never logs
a Sharpe metric**, so it cannot be parsed from the csv.

Decision (`orchestrator/summary.py`): Sharpe = annualized Sharpe of the
strategy's daily *net* return from ret.pkl (`(return - cost).mean() /
std() * sqrt(252)`), i.e. the absolute-return Sharpe complementing the
excess-return-based IR. If a sharpe-named key ever appears in the csv it
wins; with neither source the summary honestly shows `n/a` (as it does for
any missing metric) rather than substituting IR.

Related rendering choice: the equity curve plots **cumulative sums** of the
daily strategy net return and benchmark return, matching qlib's own
summation-accumulation convention (see `qlib.contrib.evaluate.risk_analysis`
docstring), not compounded products.

## 2026-07-08 — US-023: custom universes materialize artifacts; per-run env wiring deferred

`set_universe`/`confirm_universe` (orchestrator/universe.py) materialize a
confirmed custom universe as three artifacts, mirroring the us_liquid layout
from US-017:

- instruments file: `<store>/instruments/<name>.txt` (data/make_universe)
- factor source: `~/rdq-data/factor_source/<name>/{data_folder,data_folder_debug}`
  (data/make_factor_source)
- template copy: `~/rdq-data/templates/<name>/{factor_template,model_template}` —
  the US-016 templates with the `market: &market us_liquid` anchor line
  rewritten to `market: &market <name>` (benchmark stays SPY, which lives in
  the store but never inside a universe).

**Known gap (deliberate):** ops/rdq-research.service pins
`FACTOR_CoSTEER_DATA_FOLDER(_DEBUG)` and the us_quant hook templates to
us_liquid, and server_ui spawns every run from that single environment — so a
server-started run cannot yet consume a custom universe's factor source or
template copy (it gets the custom instruments universe only via the seeded
user_instruction). Per-run environment plumbing needs a server-side change
(research/server_ui.py, same seam US-024 extends for resume) and is deferred;
the artifacts are rendered now so that change is pure wiring. Manual runs can
already point ops/run_us_quant.sh's env overrides at the rendered paths.

Refusal policy: proposals whose ticker set covers `us_liquid` (or the whole
store) are refused as all-US universes — the built-in us_liquid exists for
that; built-in/reserved names (us_liquid, sp500, all) cannot be reused.
Below `min_size` (default 30) tickers the proposal warns and suggests padding
with liquid sector peers, because RD-Agent(Q) ranks cross-sectionally and a
thin cross-section makes top-k selection noise.

## 2026-07-09 — US-024: server-side resume extension + stop/resume tools

Upstream server_ui's `/control` implements only `stop`, but the run targets
themselves support session resume (`fin_quant` main(path=...) ->
`RDLoop.load(path, checkout=True)` picks the latest dumped step under
`<trace_dir>/__session__/`). research/server_ui.py therefore installs a
**runtime view wrapper** over the registered Flask `control_process` view
(`install_resume_control()`): `action: "resume"` relaunches the target from
the checkpointed session **under the same trace id** (messages, polling and
artifact resolution continue), everything else delegates to upstream
unchanged. The pinned tree on disk stays byte-identical (RECORD-hash test).

Resume semantics and guards:

- default session path = the trace dir itself; an explicit `path` must live
  under UI_TRACE_FOLDER (containment check) and contain dumped loop steps.
- refuses while the process is still alive; only the three fin scenarios are
  resumable (their mains take `path=`); `loops`/`all_duration` pass through.
- the old task's message history is carried onto the new task **minus END
  markers** — a stale END would make `status()` report the resumed run as
  instantly finished and the poller would close it out on the first tick.
- a resumed interactive run re-runs `_interact_init_params()`, i.e. it blocks
  on init-params + base-features again exactly like a fresh start. The
  poller deliberately never answers those kinds, so
  `RdAgentClient.resume(..., directive=, universe=)` re-seeds both answers
  the way `start_run` does — resume without a directive deadlocks the run.

Slack tools (conversation.py): `stop_run` = /control stop -> cancel the
thread's unanswered hypothesis rows (status 'cancelled'; a stopped run's IPC
queues are gone, and the resumed run re-proposes under fresh interaction
keys) -> run row 'stopped'. `resume_run` = /control resume from the stored
session_path with the saved directive re-seeded -> run row back to 'running',
which is what re-activates the poller (it polls `status='running'` rows).
If the poller's END(-1) handler wins the race with stop_run's status update,
both write 'stopped' — harmless either order.

## 2026-07-09 — US-035: Trade Ledger under rdq-exec-paper via a per-agent Notion app-connection grant

**Decision:** The nightly rebalancer (identity `rdq-exec-paper`) writes the
Notion Trade Ledger directly (`execution/ledger.py`, the database's sole
writer per docs/reference/notion-schema.md) — it does NOT relay rows through
the orchestrator.

**Finding: OneCLI app connections are granted per agent, and the grant has no
CLI.** Notion auth is the "JPH NanoClaw Connection" app connection (see the
2026-07-08 Notion entry), and the proxy injects it only for agents that hold
a row in the gateway's `agent_app_connections` table. That is why
`rdq-orchestrator` (granted 2026-07-07, presumably via the web UI) got
HTTP 200 while `rdq-exec-paper` got 401 until 2026-07-09, when the same grant
was added for `rdq-exec-paper`
(`agent 3c2ae9c1-1856-4c4c-b777-c09fc1cc5ab5 <- connection
af37465d-5f08-4382-afba-586c31f3dbbf`, inserted directly into the OneCLI
Postgres `agent_app_connections` table because neither `onecli` nor the
management REST API exposes the operation; the web UI equivalent is the
agent's app-connections editor). If the grant ever disappears (e.g. OneCLI
migration), re-add it in the web UI — do NOT vault a Notion key.

**check_onecli.sh consequence:** hosts with no vault secret are now probed
bare instead of failing the vault lookup — a 2xx proves app-connection
injection ("via app connection" in the PASS line), a 401/403 names the
missing grant. `ops/setup_onecli.sh` still prints its harmless WARN for
`api.notion.com` (app connections never appear in `onecli secrets list`).

**Ledger write policy:** best-effort (never aborts a run — by ledger time the
orders are already live at Alpaca) but never silent: failures accumulate in
`TradeLedger.failures` and are appended to the daily Slack summary as
WARNING lines, and `ops/reconcile.py` (US-037) exists to catch any rows that
were lost anyway.

## 2026-07-09 — US-039: OneCLI approvals bridge — API verified, gateway-URL nuance, no paper approval rules

**Finding: the pending-approvals API matches PLAN.md's expectations with one
nuance — it lives on the GATEWAY url, not the management API.** Probing
`GET http://127.0.0.1:10254/api/approvals/pending` (the PLAN sketch) returns
404. The real surface (confirmed both live against onecli 2.2.0 and in
`@onecli-sh/sdk`'s ApprovalClient, nanoclaw's reference implementation):

1. `GET {ONECLI_URL}/api/gateway-url` → `{"url": "http://localhost:10255"}`
   (resolve once, cache — verified live).
2. `GET {gateway}/api/approvals/pending[?exclude=id,id]` long-polls: the
   server holds the connection up to ~30s, returning
   `{"requests": [...], "timeoutSeconds": 180}` (verified live: empty list,
   ~6s hold; client timeout must exceed the hold — we use 35s like the SDK).
   Request objects are camelCase: `id, method, url, host, path, headers,
   bodyPreview, agent{id,name,externalId}, createdAt, expiresAt`.
3. `POST {gateway}/api/approvals/{id}/decision` with
   `{"decision": "approve"|"deny"}`. 410 = already expired server-side
   (denied by timeout) — tolerated, reported to the operator as expiry.
   Unknown id returns 404 (verified live).

Since the API works as expected, the primary path (Slack Approve/Deny
buttons in `orchestrator/approvals.py`) is the implementation; the web-UI
fallback survives as graceful degradation — any decision-submit failure
posts the error in Slack and points the operator at the OneCLI web UI at
:10254, which can always decide manually (notification-only mode).

**No approval rules exist for paper hosts, and none may ever be created.**
Approval rules are the future LIVE-trading gate (PLAN.md Phase 6): a rule on
`api.alpaca.markets` will hold every live order until a human taps Approve.
Paper trading (`paper-api.alpaca.markets`) must stay autonomous — the nightly
rebalancer runs unattended pre-open; an approval rule on a paper host would
deadlock it against a 3-minute approval timeout. With no rules configured the
pending list stays empty and the bridge idles. (Approval rules are created in
the OneCLI web UI only; the CLI has no command for them.)

**Restart/persistence choice:** posted-approval state is in-memory only.
Pending approvals expire in ~3 minutes (`timeoutSeconds: 180`), so SQLite
persistence buys nothing; after a restart the next poll re-lists anything
still pending and it is simply re-posted. Decisions are submitted by request
id alone (carried in the button value), so clicks on messages posted before
a restart still land.

**Proxy hygiene:** the bridge's calls to :10254/:10255 are local management
traffic and must not transit the credential proxy that `onecli run` injects —
the client sets `session.trust_env = False`, and rdq-orchestrator.service's
NO_PROXY now also covers `127.0.0.1,localhost` (which the RdAgentClient calls
to :19899 benefit from too).

## 2026-07-09 — Slack clients force proxy=None (slack_sdk ignores NO_PROXY)

**Problem found in production:** the orchestrator connected to Slack, then
went silently deaf within ~2 minutes — service up, no errors logged, no
replies. The Socket Mode websocket was tunneling through the OneCLI proxy
despite the unit's `NO_PROXY=slack.com`: slack_sdk's proxy discovery
(`proxy_env_variable_loader.load_http_proxy_from_env`) reads only
`HTTPS_PROXY`/`https_proxy` and never consults NO_PROXY, and
`SocketModeHandler` inherits the same env-loaded proxy via
`app.client.proxy`. The proxy drops long-lived tunnels; the builtin client
left a CLOSE-WAIT socket and never reconnected. Missed events are not
replayed by Slack.

**Decision:** enforce the existing "Slack never transits the proxy" rule
(2026-07-08 entry) in code, not env: `orchestrator/app.py main()` sets
`handler.client.proxy = None` and `web_client.proxy = None` after
construction. Passing `proxy=""`/`None` at construction does NOT work —
both clients fall back to env loading on falsy values, which is why the
override must happen post-construction.

**Diagnosis pattern (runbook §6):** a deaf bot shows no `:443` connection
for the orchestrator PID in `ss -tnp` (and typically a CLOSE-WAIT to
127.0.0.1:10254). The execution-side notifier (`execution/rebalance.py
slack_notifier`) constructs a plain WebClient under `onecli run` too — its
posts are short-lived HTTPS (not a persistent tunnel) so it has worked, but
it inherits the same env-loading behavior if this ever needs revisiting.

## 2026-07-09 — On-demand store backfill for custom universes (any-US-equity)

**Problem:** the qlib store held only the 31-ticker bootstrap set, so the
first real custom-universe proposal (30 AI-infrastructure names) died in
`materialize()` with "absent from the store — backfill with
data/build_store.py first". The operator's mental model — any US equity is
reachable — matched PLAN.md's intent (the store as a broad superset) but not
the store's actual contents, and no tool could close the gap from Slack.

**Decision:** universes are no longer store-bounded. `data/refresh.py` gains
`extend_store(store, client, symbols)` — full-history FMP backfill for new
tickers, aligned to the store's own calendar (never "today"), merged through
build_store's atomic swap with existing bars recovered from the bins and
custom universes preserved; plus a `--add-tickers` CLI. `UniverseService`
uses it: `propose()` warns which tickers will need backfill (and that
confirm will take minutes), `materialize()` backfills gaps after operator
confirmation; only symbols FMP has no bars for remain a hard
`UniverseGapError` (listing them, suggesting re-proposal). Partial backfill
before that error is deliberate: added bars are harmless, and the retry
without the bad symbols finds them already present.

**Identity change:** `rdq-orchestrator` now holds the
`financialmodelingprep.com` secret (setup_onecli.sh + check_onecli.sh
probe) — materialize runs inside the orchestrator process, which previously
could not reach FMP. Market data is read-only/low-stakes; the
paper/live-brokerage scoping is untouched.

**Cost note:** extend refetches splits/dividends for every EXISTING ticker
too (the rebuild recomputes all adjustment factors — same trade-off as
refresh_store). Fine at current store size; if the store grows to ~1000
names, revisit with a factor-preserving bundle path before making extends
routine.

## 2026-07-09 — US-043: fin_quant runs execute without conda (docker backtests + venv shims)

**Problem:** the first UI-launched fin_quant run stalled forever at startup —
`rd_loop._interact_init_params` re-asked for a base-feature config in an
infinite loop because upstream's `validate_qlib_features` can never pass on
this box: it hardcodes `QlibCondaEnv` (no conda here → the probe's PATH
collapses to /bin:/usr/bin with no `python`), and its probe queries SH600000
from the CN store (absent — only us_data exists). Two more conda assumptions
were waiting behind it: `QlibFBWorkspace.execute` defaults to
`env_type="conda"` for the real backtests, and `get_factor_env` builds
`CondaConf(conda_env_name=$CONDA_DEFAULT_ENV)` → pydantic error when unset.
Installing conda would NOT have fixed the gate (the CN-data probe still
fails) and would duplicate what docker already provides.

**Decision:** split execution by workload, pinned tree untouched.
(1) Backtests/model training opt into docker: `MODEL_CoSTEER_ENV_TYPE=docker`
(QTDockerEnv, local_qlib:latest built from the pinned dockerfile, mounts
~/.qlib rw) in rdq-research.service and run_us_quant.sh wire_env.
(2) Generated factor code runs with the repo venv's interpreter:
`FACTOR_CoSTEER_PYTHON_BIN=.venv/bin/python` (same two places).
(3) research/us_validation.py runtime-patches (US-024 pattern, by assignment
over every from-import binding) `validate_qlib_features` → a subprocess of
`sys.executable` probing the US store with SPY (knobs: RDQ_QLIB_STORE,
RDQ_VALIDATION_*), and `get_factor_env` → a LocalEnv on the venv's bin/.
The install hook is `research.us_quant` import — resolving the QLIB_QUANT_*
class paths imports it in every fin_quant process during loop construction,
before the interaction gate — plus a belt-and-braces call in
server_ui.main() (children are forked and inherit the patched modules).

**Caveat:** the patch targets mirror upstream call sites at the pinned
commit (rd_loop, utils.qlib, factor_coder.config, factor_experiment,
quant_experiment) — re-audit `grep -rn "validate_qlib_features\|get_factor_env"`
when bumping research/PINNED_COMMIT.

**Addendum (same day):** `QLIB_DOCKER_BUILD_FROM_DOCKERFILE=false` joined the
env set — the first probe run revealed upstream `QTDockerEnv.prepare()`
rebuilds the image on EVERY call via docker-py's legacy builder, which shares
no cache with the BuildKit-built local_qlib:latest (a silent hour-long
rebuild inside each run). Ops owns the image instead; rebuild it manually
after a research/PINNED_COMMIT bump if the upstream Dockerfile changed.

**Addendum 2 (same day):** the probe run surfaced a third conda/CN hardcode:
`QTDockerEnv.prepare` raises StopIteration when a caller replaces
`extra_volumes` with the empty default (`get_model_env(extra_volumes={})`,
via the quant scenario's `get_runtime_environment`), and otherwise
auto-downloads the CN dataset whenever `~/.qlib/qlib_data/cn_data` is
missing — it always is here. install_us_validation() now also class-patches
`QTDockerEnv.prepare` to image build/pull only (`DockerEnv.prepare`),
covering QlibFBWorkspace.execute backtests, get_model_env, and
generate_data_folder_from_qlib.

**Addendum 3 (same day):** the first chat completion of a fin_quant run has
never succeeded on this box — rdagent sends `temperature=0.5` and LiteLLM
raises UnsupportedParamsError for the Claude models (the known
research/CLAUDE.md gotcha). `LITELLM_DROP_PARAMS=true` (honored by litellm's
module init) joins the unit + wire_env so unsupported params are dropped
instead of erroring.

**Addendum 4 (same day):** every qlib backtest inside `local_qlib:latest`
died in `qlib/workflow/expm.py create_exp` — the MLflow shipped in the image
(>= 3.6) put the filesystem tracking backend in maintenance mode and raises
MlflowException for the `./mlruns` store qlib's recorder is hardwired to,
unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Whole loops burned LLM budget
"failing purely due to MLflow infrastructure" without ever testing factor
logic. Fix: `QLIB_DOCKER_ENV_DICT={"MLFLOW_ALLOW_FILE_STORE":"true"}` joins
the unit + wire_env — `QlibDockerConf.env_dict` (pydantic-settings, JSON
parse) is merged into every container env at the top of `Env.run`, covering
both the cached and retry docker paths. Migrating qlib to a DB backend is
not an option at the pinned commit; the env opt-out is upstream MLflow's
sanctioned escape hatch and stays correct across image rebuilds.

**Addendum 5 (same day):** with MLflow unblocked, every backtest then died in
qlib instrument loading — `ValueError: instrument ... us_data does not
contain data for day` — while the host store was provably intact (all three
trial loops of the miner_ai_conversion run failed identically; a manual
container with the documented `~/.qlib` mount read the store fine). Root
cause: `QTDockerEnv.__init__` declares `conf: DockerConf = QlibDockerConf()`
— a mutable default evaluated once at class definition, shared by every
`QTDockerEnv()` in the process. Upstream `get_model_env()` (called with no
volumes from both quant/model `get_runtime_environment` while rendering the
scenario prompt) assigns `env.conf.extra_volumes = {}` and
`running_timeout_period = 600` onto that shared object, silently deleting
the `~/.qlib -> /root/.qlib` mount (and shrinking the 3600s budget) for
every subsequent backtest container. The containers saw an empty
`/root/.qlib`, so qlib's calendar-glob-based `support_freq` came up empty
before any factor code ran. Upstream never hits this because its model env
defaults to conda; `MODEL_CoSTEER_ENV_TYPE=docker` (this box) activates the
docker path. Fix: `get_us_model_env` in research/us_validation.py (shim #4,
same US-024 assignment pattern, bindings in model_coder.conf +
model_experiment + quant_experiment) builds a FRESH `QlibDockerConf` per
call and merges caller volumes over the defaults, never touching the shared
class default. The forensic tell for a poisoned process: the qrun entry in
the docker execution log shows `timeout ... 600` instead of 3600.

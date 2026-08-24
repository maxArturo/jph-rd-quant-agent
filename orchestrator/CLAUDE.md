# orchestrator/ — module notes

- Config pattern: all env/config loading goes through `orchestrator/config.py`
  (process environment overrides the repo-root `.env`; raise `ConfigError`
  naming the missing variable and where to set it). Extend that module for new
  settings instead of reading `os.environ` elsewhere.
- Slack tokens come from the repo-root `.env` (SLACK_OAUTH_TOKEN xoxb-,
  SLACK_SOCKET_TOKEN xapp-, SLACK_CHANNEL_ID). Never route Slack through the
  OneCLI proxy and never vault these (docs/decisions.md 2026-07-08).
- Persistent state goes through `orchestrator/state.py` (`StateStore`), not ad
  hoc sqlite3 calls. It opens a short-lived connection per method, so one
  instance is safe to share between Bolt handlers and background threads —
  never cache a `sqlite3.Connection` across threads. Extend the schema by
  adding `CREATE ... IF NOT EXISTS` statements to `_SCHEMA` (migration reruns
  on every startup). Dedup/uniqueness lives in the schema (runs.thread_ts PK
  → `DuplicateRunError`), so restarts can't double-post. The
  `pending_interactions` table is legacy (the poller that wrote it was
  removed in US-027): read-only via `list_interactions`, never add a write
  path or a destructive migration for it. The DB runs in WAL mode
  with a 30s busy timeout (both set in `_connect`, WAL persists in the file)
  so cross-process writers (GPU pipeline unit, CLI promotes) queue instead of
  raising 'database is locked' — keep any new sqlite connection in this repo
  on the same settings.
- Promotions are audited (US-005): `set_promoted_strategy` appends an
  append-only `promotion_history` row (source `auto_gate`/`conversation`/
  `cli`, optional gate_verdict JSON, replaced_workspace) in the same
  transaction as the pointer flip — every new promotion path MUST go through
  `set_promoted_strategy` with an honest `source`, never write the
  promoted_strategy table directly. Read history with
  `list_promotion_history` (newest first); rows are never updated or
  deleted (rollback = a NEW row re-promoting an old workspace).

- All orchestrator LLM calls go through `orchestrator/llm.py` (`ModelRouter`):
  `judgment()` = claude-fable-5 (streamed, server-side refusal fallback to
  opus-4-8), `utility()` = claude-haiku-4-5, `judgment_tool_loop()` for tool
  use. Model IDs must not appear anywhere else — tests/test_llm.py greps for
  them. Never pass a `thinking` parameter (fable-5 400s on any explicit
  config). Refusals surface as `RefusalError`; check `stop_reason` before
  reading `content` on any hand-rolled call. `ModelRouter(client=...)` accepts
  a fake client for tests (see FakeClient in tests/test_llm.py — stub
  `client.beta.messages.stream` as a context manager with
  `get_final_message()`, and `client.messages.create`).

- Conversational behavior lives in `orchestrator/conversation.py`
  (`ConversationCore`) with prompt text in `orchestrator/prompts.py` — add new
  Slack-facing tools (start_research, set_universe, ...) as `ToolSpec`s built
  inside ConversationCore (handlers close over `thread_ts` + `say`), not as
  new Bolt listeners. `app.py` depends only on the `MessageResponder`
  protocol, so tests stub the core with a plain class (see FakeConversation in
  tests/test_slack_app.py) and core tests reuse FakeClient from
  tests/test_llm.py — no MagicMock of Anthropic needed. Durable per-thread
  context must reload from SQLite into the system prompt (in-memory history
  is lost on restart by design).

- The legacy control-plane client (`orchestrator/rdagent_client.py`) and the
  `resume_run` tool were removed in US-028 (the service itself went in
  US-026). The artifact helpers promotion still needs — `RunArtifacts`,
  `ArtifactNotFoundError`, `locate_artifacts` — live in
  `orchestrator/gpu_backend.py`; `locate_run_artifacts` there is the single
  promotion locate for both fetched GPU traces and legacy on-box trace dirs
  (`~/rdq-runs/server_ui/traces/...` paths in old rows still resolve).
- Legacy `runs` rows keep `backend='server_ui'` (no destructive migration):
  every code path iterating runs must tolerate that value —
  check_research_status reports them plainly, stop_run refuses them with an
  explanation, the run reaper filters on `backend == 'gpu'`.
- Run lifecycle via `runs.status`: GPU stop_run sends cancel and leaves the
  row 'running' — the PIPELINE flips it to 'stopped' when it finalizes. Runs
  are never resumable; never flip a row to 'running' without a live process
  behind it.

- Notion writes go through `orchestrator/notion_client.py` (`NotionClient`):
  bare HTTPS with only `Notion-Version: 2022-06-28` — NEVER add an
  Authorization header (a source-grep test enforces it; the OneCLI proxy
  injects the token via the connector integration when running under
  `onecli run --agent rdq-orchestrator`). Read-after-write goes through
  `query_db_until(db_id, predicate)` — Notion queries lag writes; a plain
  `query_db` right after `create_page` can miss the row. Retries: 429
  honors Retry-After; 409/5xx are transient (Notion returns 409
  conflict_error on concurrent saves — retry, don't fail). Tests inject
  `NotionClient(session=FakeSession, sleep=list.append)` (FakeSession here
  records `.request(method, url, json=, headers=)` — a superset of the
  test_fmp.py GET-only fake).

- Run memory (US-014/US-015) lives in `orchestrator/run_memory.py`:
  `build_digest(db_path, client)` composes the prior-run digest (Notion
  Strategy Notes `run_summary` JSON per row + incumbent section from local
  state/workspace artifacts); `build_digest_details` additionally returns
  the included-entry count (`Digest.runs`, reported in the run-start Slack
  message). US-015 injects it into RDQ_USER_INSTRUCTION at start_research:
  `compose_instruction(directive, digest)` = directive + `MEMORY_DELIMITER`
  + digest — the directive always comes first and only the digest is ever
  trimmed to fit. Anything recording the instruction durably MUST strip the
  digest first with `split_instruction` (gpu_pipeline.build_notion_context
  does — otherwise digest text compounds into every future digest).
  ConversationCore takes `digest_builder=` (None skips injection; app.py
  wires `lambda: build_digest_details(store.db_path)`), and the
  start_research tool's `include_memory` arg (default true) is the operator
  clean-slate switch.
  It NEVER raises and never stalls a launch: every failure degrades (Notion
  down / a row without parseable JSON falls back to local runs/directives,
  directive + status only), total Notion time is budgeted (15s, checked
  before each request), output is deterministic and truncates oldest-first
  at max_chars (4000). It opens state.sqlite READ-ONLY (guards `is_file()` —
  StateStore(path) would create the db) and runs with real Notion access
  only inside the orchestrator process / under `onecli run --agent
  rdq-orchestrator`; anywhere else it just degrades to local data. Gotcha
  discovered live: the Notion row's Directive property is the FULL run
  instruction while the local directive objective may be a shorter prefix
  (and Notion clips rich_text at 2000 chars), so run matching uses mutual
  whitespace-normalized prefix, not equality. Tests stub the client at the
  method boundary (query_db / list_block_children) and serve blocks produced
  by the REAL US-013 writer so reader and writer can't drift.

- Data-menu prompt injection (US-061/US-002): ConversationCore takes
  `menu_builder=` (None skips injection; app.py wires
  `prompts.data_menu_context`, which rebuilds the menu from the store on
  every turn — it tracks daily refreshes — and NEVER raises: any store-read
  failure degrades to `prompts.MENU_UNAVAILABLE_LINE`; `_system_prompt`
  additionally guards a raising builder). Never hand-copy store field lists
  into prompt text — render them from data/menu.py, the single source of
  truth.

- Directive data pre-flight (US-062/US-003): save_directive's `data_required`
  list is verified in ConversationCore via the injected `field_lister`
  (default `_store_field_lister` reads the REAL store through
  `data.menu.build_menu().field_names()`; tests inject a fixture-store
  lister). Any missing entry PARKS the directive — persisted in the
  `directives.missing_data` JSON column, so start_research's refusal is
  state-enforced across restarts, and `parked_directive_message()` is the
  single source of the "parked — needs ingestion: ..." wording (save summary
  + start refusal). Parking is computed AT SAVE TIME: after ingesting a
  series the directive must be saved again to unpark (start_research does
  not re-verify). A field_lister failure fails the save loud (error
  tool_result, nothing persisted) — never record a silent 'all present'.

- Lifecycle recording into Notion goes through `orchestrator/notion_recorder.py`
  (`NotionRecorder`, US-027) — the single write funnel for Research Ideas /
  Hypothesis Log / Backtest Results. It is best-effort BY DESIGN: every
  `record_*` method logs-and-swallows its own failures (a Notion outage must
  never break Slack flows), so call sites never wrap it in
  try/except — but also never rely on its return value for control flow.
  Page-id mappings live in StateStore's `notion_pages` table (kind `idea`
  keyed by thread_ts, kind `hypothesis` keyed by interaction_key); use
  `get_notion_page`/`set_notion_page`, never re-query Notion to find a page.
  Since US-027 only the idea/decision methods have live callers —
  `record_hypothesis`/`record_hypothesis_action`/`record_backtest` were
  poller-driven and are prod-dead (per-hypothesis rows don't exist on the
  GPU path; run history lives in Strategy Notes' run_summary JSON instead).
  Recorder property names must match
  docs/reference/notion-schema.md; metric properties reuse
  summary.METRIC_SPECS labels. Tests: real recorder + NotionClient over
  FakeSession (tests/test_notion_recorder.py) — recorder failures are
  invisible to callers, so assert on `session.calls` payloads, not behavior.

- Notion database ids live in `orchestrator/config.yaml` under
  `notion.databases.{research_ideas,hypothesis_log,backtest_results,
  decision_log,trade_ledger,account_snapshots,strategy_notes}` — written
  (and rewritten) by
  `ops/bootstrap_notion.py`; never hand-edit or hardcode the ids. The
  property schemas are documented in docs/reference/notion-schema.md; each
  database has exactly ONE writing component (one-writer-per-DB convention) —
  check the table there before adding a Notion write path. Relations are
  `single_property` (no synced back-reference on Research Ideas).

## Testing Bolt apps (see tests/test_slack_app.py)

- Bolt >=1.15 constructs a NEW real `WebClient` per request in
  `App._init_context` — injecting a mocked client into `App(client=...)` is
  NOT enough; `say()`/`context.client` would hit the network. Also
  monkeypatch `slack_bolt.app.app.WebClient` to return the mock, but only
  AFTER `App()` is constructed (its constructor isinstance-checks that same
  symbol).
- `MagicMock(spec=WebClient)` misses instance attributes `_init_context`
  reads (`base_url`, `timeout`, `ssl`, `proxy`, `headers`, `logger`,
  `retry_handlers`) — set them explicitly on the mock.
- Pass `process_before_response=True` in tests so listeners run synchronously
  inside `App.dispatch()`; otherwise assertions race Bolt's worker threads.
  Do NOT enable it in production once handlers are slow (Claude calls):
  Slack retries events not acked within ~3s.
- Dispatch events as
  `BoltRequest(body=json.dumps({"type": "event_callback", "event": {...}, ...}), mode="socket_mode")`.
- Handlers must ignore `subtype` messages (message_changed, channel_join, ...)
  and anything with `bot_id`, or the bot replies to its own replies (loop).
  ONE exception: bot ids listed in `RDQ_TRUSTED_BOT_IDS` (e.g. Claude in
  Slack) pass when the text @mentions our bot user (`bot_user_id` from
  auth.test, threaded through create_app). The mention gate is the loop
  brake — trusted bots post digests all day; only messages addressed to us
  are directives. Trusted-bot posts come in two shapes (app user + bot_id,
  or subtype `bot_message` + username override); both pass, other subtypes
  never. In-thread reply target: `event.get("thread_ts") or event["ts"]`.
  `RDQ_TRUSTED_ONLY=1` (2026-08-19, set in prod .env) additionally drops the
  plain-human path: trusted-bot @mentions become the ONLY input — the
  channel carries operator<->Claude conversation the bot must not act on.
  main() raises ConfigError when it's set with an empty allowlist.

- Custom universes live in `orchestrator/universe.py` (`UniverseService`):
  `propose()` is validation-only (refusals: built-in/reserved names, all-US
  ticker sets covering us_liquid or the whole store; warning below
  `min_size`); `materialize()` does the data work (gap check → instruments
  file → factor source → template copy with `market: <name>`) and is only
  called AFTER the operator confirms in-thread. The two-step state lives in
  the `universes` table (`propose_thread_universe` upserts back to
  'proposed'; `confirm_thread_universe` flips it), and start_research
  refuses while a proposal is unconfirmed, then copies name + tickers onto
  the run row (`runs.universe_tickers`, JSON). Artifact layout mirrors
  us_liquid: `~/rdq-data/factor_source/<name>` + `~/rdq-data/templates/<name>`
  — consumed by the GPU pipeline's `--universe` flag (GpuBackend.launch
  passes the run row's universe through; the worker launch wiring
  hard-refuses when the artifacts are missing). Keep `MARKET_LINE` in sync
  with research/us_templates conf yamls — the render hard-fails if the
  anchor line drifts.
- Schema changes to an EXISTING table cannot ride `CREATE TABLE IF NOT
  EXISTS` (it skips existing DBs): add the column to `_SCHEMA` for fresh DBs
  AND a guarded `ALTER TABLE` in `migrate()` (check `PRAGMA table_info`),
  like `runs.universe_tickers`.

- GPU burst workers are the ONLY research backend: `GpuBackend.launch`
  starts `ops/gpu_pipeline` as a transient user unit (`rdq-gpu-run-<ts>`),
  which drives the whole run end-to-end (provision → loops with per-loop
  Slack digests → fetch → completion summary → gate → destroy). The
  hypothesis poller and ALL Block Kit flows were removed in US-027
  (`orchestrator/poller.py`, the `hypo_approve`/`hypo_edit`/`hypo_reject`
  and `run_promote`/`promote_confirm`/`promote_cancel` action listeners,
  and the approve/reject hypothesis ToolSpecs); the only Bolt listeners
  left are the message handler and the OneCLI approvals buttons. Don't add
  new Block Kit action flows — operator decisions are conversational tools
  (US-044), and buttons proved unclickable via the Slack API anyway.
- The poller-era brakes (RDQ_MAX_HYPOTHESES cap, identical-error abort,
  per-hypothesis approval) do NOT exist on the GPU path (accepted loss,
  docs/decisions.md 2026-08-17). What bounds a GPU run instead: the
  `--loop_n` hypothesis budget, the pipeline's `--max-hours` wall-clock
  teardown (default 24h), the hourly `rdq-gpu-watchdog` billing reaper, the
  global run lock (US-020, one run at a time), and the stale run-row reaper
  (US-021).
- Gate/auto-promotion flow (US-007..011): at run end `gpu_pipeline`
  evaluates `ops/promotion_gate.evaluate_gate` — parity-enforced (window,
  market, instrument hash, topk/n_drop, costs) IR/MDD/IC criteria plus the
  reserved confirmation-window comparison (`ops/confirm_window`) — and on
  pass promotes via `ops.promote_fetched.promote_workspace` with
  source='auto_gate' and the verdict JSON in promotion_history.
  `promotion_gate.auto_promote: false` in config.yaml is the report-only
  kill-switch. Conversational (two-yes thread) and CLI promotion paths stay
  available and write the same history/Decision Log records (US-012).

- Global GPU run lock (US-020): `GpuBackend.active_run_lock()` reads
  `ops/run_lock.py`'s lock file (`(active, broken_stale)`; a stale lock —
  owning unit inactive, pid gone — is removed and the caller posts the broom
  note). start_research refuses while ANY thread holds the lock (runs are
  never queued — don't reintroduce that fiction in status text); stop_run's
  GPU branch refuses unless the requesting thread owns it, because cancel()
  kills the single shared worker's tmux session. The unit-active probe rides
  GpuBackend's injected runner, so tests stay hermetic (StubGpu in
  tests/test_conversation.py carries `lock`/`broken_lock` fields).
- Stale GPU run-row reaper (US-021, `orchestrator/run_reaper.py`): a
  background `GpuRunReaper` thread in app.py (ApprovalsBridge pattern) marks
  `running`/`gpu` run rows 'failed' once their `rdq-gpu-run-<ts>` unit has
  been inactive for one grace period (15 min default), posting a note in the
  run's thread FIRST and flipping the status LAST (a Slack failure retries
  the whole reap next tick — notify-then-commit convention). Grace
  tracking is in-memory (a restart only delays a reap). A 'failed' run row no
  longer blocks start_research: it passes `replace_failed=True` to
  `create_run`, which atomically deletes exactly-a-failed row before the
  insert (any live row still raises DuplicateRunError). Completed/stopped
  rows still block — promotion reads their session_path; never widen
  replace_failed beyond 'failed'.
- Broker visibility (US-046): check_account/check_orders/check_pnl are
  READ-ONLY ToolSpecs in ConversationCore behind the `BrokerReader` protocol
  (default: the real `execution.alpaca_client.AlpacaClient` — the
  rdq-orchestrator identity holds the paper secret, so proxy injection just
  works). They return formatted text straight to the model (no say() post —
  the model's reply carries the numbers); broker errors surface as error
  tool_results, never crash the turn. Never give the core a write-capable
  broker method: trading stays with the rebalancer, and the only trading
  control here remains the breaker halt. "Day P/L" caveat for prompts/tools:
  pre-open, equity-vs-last_equity is ~0 — check_pnl's history covers
  completed days. Tests: tests/test_broker_tools.py (StubBroker + FakeClient
  scripts; assert the tool_result fed back in stream_calls[1]).
- Trading halt/resume (US-038): halt_trading/resume_trading are ToolSpecs in
  ConversationCore like the run-lifecycle tools, but they flip the
  REBALANCER's kill switch (execution/breaker.py halt file on the default
  `~/rdq-data/breaker/` paths), not research runs. The core depends on the
  `TradingBreaker` protocol (halt/clear_halt/halted/halt_note/halt_file) and
  defaults to the real `Breaker(load_breaker_config())` — tests inject one
  over tmp paths. Both tools refuse redundant calls (already halted / not
  halted) so the model relays state instead of clobbering the existing halt
  note, and both write a Decision Log row via
  `NotionRecorder.record_decision` (types `halt`/`resume`).

- Run-completion output lives in `orchestrator/summary.py`: `load_metrics`
  (qlib_res.csv is a pandas Series csv — metric name index, one value
  column), `format_summary` (the metric-label -> qlib-key mapping lives in
  METRIC_SPECS; qlib logs NO Sharpe — it is derived from ret.pkl net daily
  returns, see docs/decisions.md US-022), `render_equity_curve` (matplotlib
  with the Agg backend selected BEFORE importing pyplot, lazy imports so
  offline tests stay fast; returns PNG bytes for `files_upload_v2`).
  ret.pkl is qlib's report_normal_1day DataFrame (columns account/return/
  turnover/cost/bench/..., trading-day index); treat `cost`/`bench` as
  optional when consuming it.
- Strategy promotion (US-033, conversational since US-027/US-044) lives in
  `orchestrator/promotion.py` (`PromotionFlow`): ConversationCore's
  `promote_run`/`confirm_promotion` ToolSpecs call
  `request_promotion`/`confirm_promotion` directly (protocol
  `PromotionManager` in conversation.py); the tools capture what the flow
  posted via a recording `say` wrapper and return it verbatim so the model
  relays the actual confirmation/refusal. Two-step promotion is preserved
  conversationally: the model is prompted to require an explicit second yes
  before confirm_promotion. Statuses in `PROMOTABLE_STATUSES` = completed
  AND operator-stopped ('stopped' is a normal successful ending; US-044).
  The candidate (workspace, topk/n_drop AND universe from the workspace's
  own conf via `execution.signal.load_strategy_params`/`load_market`,
  tickers from the store instruments file, headline metrics) is re-derived
  from SQLite + run artifacts on every call, so the flow survives restarts
  with no pending-promotion state. The universe is NEVER taken from the
  run-row label — the conf's market line bounds pred.pkl (2026-08-05
  incident); a disagreeing label is called out in every promotion message.
  Promotion refuses when the run is running/failed or topk/n_drop/market
  can't be read (the rebalancer couldn't reproduce the strategy); metrics
  merely degrade to n/a, and a missing instruments file degrades
  universe_tickers to None (the rebalancer's divergence check skips).
  Confirm pins workspace + config into the single `promoted_strategy` row
  (replacement is announced in-thread), writes a Decision Log row
  (`NotionRecorder.record_decision`), and moves the idea page's Status to
  `promoted`. US-012: the confirmation text shows the INCUMBENT's own
  IC/ARR/MDD + test window (read from its workspace via ops.gpu_trace,
  degrades to "unavailable" when swept), and the Decision Log details end
  with a gate-standing line — this path never runs the gate, so it appends
  `ops.promotion_gate.GATE_NOT_EVALUATED` (CLI/auto rows render theirs with
  `gate_summary_line`). The rebalancer-side check is `execution/promoted.py` —
  keep the pinned config keys (universe/universe_tickers/topk/n_drop/
  thread_ts/session_path) in sync with what US-034 consumes.

- OneCLI approvals bridge (US-039) lives in `orchestrator/approvals.py`
  (`ApprovalsBridge` + `OneCliApprovalsClient`): a background thread
  (started in app.py main) long-polls the gateway's
  pending credential approvals and posts them to the CHANNEL (not a thread)
  as `onecli_approve`/`onecli_deny` buttons, routed via the
  `ApprovalsHandler` protocol. The approvals endpoints live on the GATEWAY
  url (:10255), resolved once via `GET {ONECLI_URL}/api/gateway-url` — never
  hardcode the gateway port (docs/decisions.md 2026-07-09). Decisions are
  submitted by request id alone (the button value), so posted-approval state
  stays in-memory: approvals expire in ~3 min and a restart just re-posts
  still-pending ones. 410 on decision submit = expired (report, don't
  raise); any other submit failure posts the OneCLI web-UI fallback pointer.
  Calls to :10254/:10255 are local management traffic: the client sets
  `session.trust_env = False` so `onecli run`'s injected HTTP(S)_PROXY never
  captures them. NEVER create an approval rule for a paper host — rules are
  the future live-trading gate; one on paper would deadlock the unattended
  nightly rebalancer.

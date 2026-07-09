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
  instance is safe to share between Bolt handlers and background pollers —
  never cache a `sqlite3.Connection` across threads. Extend the schema by
  adding `CREATE ... IF NOT EXISTS` statements to `_SCHEMA` (migration reruns
  on every startup). Dedup/uniqueness lives in the schema (runs.thread_ts PK
  → `DuplicateRunError`; pending_interactions.interaction_key UNIQUE → insert
  returns `None`), so restarts can't double-post.

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

- All talk to rdagent server_ui goes through `orchestrator/rdagent_client.py`
  (`RdAgentClient`, default `http://127.0.0.1:19899`). It speaks the REAL
  upstream protocol, which differs from the PRD sketch — see the endpoint
  mapping table in docs/decisions.md (US-019 entry). Key semantics: runs
  start via POST /upload; `pending()` piggybacks on the POST /trace message
  poll (each poll drains ≤1 pending interaction server-side, and answered
  requests stay in the stream — dedup with `PendingInteraction.key`, and skip
  kinds `init_params`/`base_features`, which `start_run()` auto-answers);
  `submit()` answers the OLDEST unanswered interaction (FIFO queue, not
  addressed to a specific request); `resume()` needs the research/server_ui.py
  resume extension (a bare upstream server raises `UnsupportedActionError`)
  and MUST be passed `directive=`/`universe=` — a resumed run re-blocks on
  the init interactions like a fresh start, and the poller never answers
  those kinds, so resume re-seeds them the way `start_run` does.
  `locate_artifacts(trace_dir)` unpickles `runner result` pkls — trace dirs
  for server-started runs live under `~/rdq-runs/server_ui/traces/<trace_id>`,
  NOT under the LOG_TRACE_PATH convention of the CLI wrappers. Tests: stub
  the server with a real threaded Flask app (StubServerUi in
  tests/test_rdagent_client.py — reuse it for poller/tool tests) and pass
  `base_features={...}` so the client never imports rdagent.
- Session-path convention (US-020): `runs.session_path` stores
  `str(client.trace_dir(handle.trace_id))`; recover the trace id for API
  calls with `client.trace_id_of(session_path)`. Run-lifecycle tools should
  depend on the `ResearchLauncher` protocol in conversation.py
  (start_run/trace_dir/trace_id_of/stop/resume; stub-friendly — see
  StubLauncher in tests/test_conversation.py) rather than the concrete client.
- Run lifecycle via `runs.status` (US-024): stop_run flips
  running -> 'stopped' AND cancels the thread's unanswered
  pending/editing interaction rows ('cancelled' — a stopped run's IPC queues
  are dead and the resumed run re-proposes under fresh keys); resume_run
  flips back to 'running', which is what re-activates the poller (it only
  polls `status='running'` rows). Never flip a row to 'running' without
  actually resuming the server-side process, or the poller will poll a
  corpse forever.

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

- Lifecycle recording into Notion goes through `orchestrator/notion_recorder.py`
  (`NotionRecorder`, US-027) — the single write funnel for Research Ideas /
  Hypothesis Log / Backtest Results. It is best-effort BY DESIGN: every
  `record_*` method logs-and-swallows its own failures (a Notion outage must
  never break Slack flows or the poller), so call sites never wrap it in
  try/except — but also never rely on its return value for control flow.
  Page-id mappings live in StateStore's `notion_pages` table (kind `idea`
  keyed by thread_ts, kind `hypothesis` keyed by interaction_key); use
  `get_notion_page`/`set_notion_page`, never re-query Notion to find a page.
  Backtest Results rows are written at FEEDBACK auto-ack time (one feedback =
  one completed experiment; its `decision` field is the SOTA flag), not at
  run END. Recorder property names must match
  docs/reference/notion-schema.md; metric properties reuse
  summary.METRIC_SPECS labels. Tests: real recorder + NotionClient over
  FakeSession (tests/test_notion_recorder.py) — recorder failures are
  invisible to callers, so assert on `session.calls` payloads, not behavior.

- Notion database ids live in `orchestrator/config.yaml` under
  `notion.databases.{research_ideas,hypothesis_log,backtest_results,
  decision_log,trade_ledger}` — written (and rewritten) by
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
  In-thread reply target: `event.get("thread_ts") or event["ts"]`.

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
  — but rdq-research.service still points the run env at us_liquid, so
  server-spawned runs don't consume them yet (docs/decisions.md US-023).
  Keep `MARKET_LINE` in sync with research/us_templates conf yamls — the
  render hard-fails if the anchor line drifts.
- Schema changes to an EXISTING table cannot ride `CREATE TABLE IF NOT
  EXISTS` (it skips existing DBs): add the column to `_SCHEMA` for fresh DBs
  AND a guarded `ALTER TABLE` in `migrate()` (check `PRAGMA table_info`),
  like `runs.universe_tickers`.

- Hypothesis steering lives in `orchestrator/poller.py` (`HypothesisPoller`):
  one instance per process polls all `running` runs and also owns the button
  handlers (`approve`/`reject`/`request_edit`/`consume_edit_reply`). app.py
  depends on it only via the `InteractionHandler` protocol and registers the
  Block Kit `hypo_approve`/`hypo_edit`/`hypo_reject` action listeners plus the
  edit-reply interception (checked BEFORE the conversational core sees a
  thread message). Lifecycle lives in `pending_interactions.status`:
  `pending → editing → approved|edited|rejected` (feedback: `auto_approved`);
  dedup is the schema UNIQUE key, so restarts never repost. Answer FIFO rule:
  never submit anything for a run while an earlier hypothesis row is still
  `pending`/`editing` — responses answer the oldest blocked request. If a
  Slack post or submit fails, free/keep the row so the next poll or click
  retries (never resolve a row whose submit didn't go through).
- Reject has no upstream regenerate action — `rejection_payload()` rides the
  instruction in the hypothesis text (see docs/decisions.md US-021 entry)
  and MUST keep the exact constructor key set (`type(hypo)(**dict)`).
- Testing Bolt block actions: dispatch a `{"type": "block_actions", ...}`
  payload as `BoltRequest(body=json.dumps(payload), mode="socket_mode")`
  (no event_callback envelope — interactive payloads ARE the body); Bolt
  injects `ack`/`action`/`say`, and `process_before_response=True` keeps it
  synchronous. See dispatch_action in tests/test_poller.py.

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
- Poller completion order (US-022): render/parse artifacts FIRST (so
  deterministically-bad artifacts degrade to an honest message instead of a
  retry loop), then post summary, upload chart, and update the run row to its
  terminal status LAST — the status flip is what removes the run from the
  `running` set, so a transient Slack failure retries the whole completion on
  the next poll. Terminal mapping from the upstream END message:
  end_code 0/None -> `completed`, -1 (operator stop) -> `stopped`,
  else -> `failed` (`terminal_status()` in poller.py).

- Strategy promotion (US-033) lives in `orchestrator/promotion.py`
  (`PromotionFlow`): the poller adds the Promote button to a completed run's
  summary (`promotion_offer_blocks`), and app.py routes the three
  `run_promote`/`promote_confirm`/`promote_cancel` actions via the
  `PromotionHandler` protocol. Button values carry ONLY the thread_ts — the
  candidate (workspace, universe, topk/n_drop from the workspace's own conf
  via `execution.signal.load_strategy_params`, headline metrics) is
  re-derived from SQLite + run artifacts on every click, so buttons survive
  restarts with no pending-promotion state. Promotion refuses when the run
  isn't `completed` or topk/n_drop can't be read (the rebalancer couldn't
  reproduce the strategy); metrics merely degrade to n/a. Confirm pins
  workspace + config into the single `promoted_strategy` row (replacement is
  announced in-thread), writes a Decision Log row
  (`NotionRecorder.record_decision`), and moves the idea page's Status to
  `promoted`. The rebalancer-side check is `execution/promoted.py` —
  keep the pinned config keys (universe/universe_tickers/topk/n_drop/
  thread_ts/session_path) in sync with what US-034 consumes.

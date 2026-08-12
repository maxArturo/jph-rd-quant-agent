# orchestrator/ — module notes

- Config pattern: all env/config loading goes through `orchestrator/config.py`
  (process environment overrides the repo-root `.env`; raise `ConfigError`
  naming the missing variable and where to set it). Extend that module for new
  settings instead of reading `os.environ` elsewhere.
- Slack tokens come from the repo-root `.env` (SLACK_OAUTH_TOKEN xoxb-,
  SLACK_SOCKET_TOKEN xapp-, SLACK_CHANNEL_ID; optional SLACK_LIVE_CHANNEL_ID
  — setting it ARMS live-trading features and must differ from
  SLACK_CHANNEL_ID; absent/empty means `live_channel_id=None` and paper-only
  behavior). Never route Slack through the OneCLI proxy and never vault
  these (docs/decisions.md 2026-07-08).
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
  decision_log,trade_ledger,account_snapshots,strategy_notes}` — written
  (and rewritten) by
  `ops/bootstrap_notion.py`; never hand-edit or hardcode the ids. The
  property schemas are documented in docs/reference/notion-schema.md; each
  database has exactly ONE writing component (one-writer-per-DB convention) —
  check the table there before adding a Notion write path. Relations are
  `single_property` (no synced back-reference on Research Ideas).
  Live keys (US-011): `trade_ledger_live` / `account_snapshots_live` (+
  `notion.live_parent_page_id`) are written only by `bootstrap_notion --live`
  and are OPTIONAL in `NotionDatabases` (`str | None = None` — a paper-only
  config keeps loading; `load_notion_databases` requires only the
  defaultless fields). `write_config` MERGES into `notion.databases`, so a
  paper-only rerun never drops live ids and vice versa. Decision Log stays
  SHARED between paper and live decisions (orchestrator sole writer); the
  live rebalancer is the sole writer of both (Live) databases.

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
  — consumed since 2026-08-05 via the upload/resume `universe` field
  (research/server_ui.py wires the run env per-universe and 400s when the
  artifacts are missing; rdagent_client sends the field on start_run AND
  resume). Keep `MARKET_LINE` in sync with research/us_templates conf yamls —
  the render hard-fails if the anchor line drifts.
- Schema changes to an EXISTING table cannot ride `CREATE TABLE IF NOT
  EXISTS` (it skips existing DBs): add the column to `_SCHEMA` for fresh DBs
  AND a guarded `ALTER TABLE` in `migrate()` (check `PRAGMA table_info`),
  like `runs.universe_tickers`.
- Run notifications route home via `runs.channel_id` (US-005): the channel
  that started a run is stamped at create_run time (ConversationCore passes
  its wired `channel_id`; app.py wires `config.channel_id`). NULL — legacy
  rows and unwired cores — always means the paper channel. Anything posting
  about a run (poller digests, completion, buttons) should resolve the
  destination with `store.run_channel(thread_ts, paper_channel_id)`, never
  assume the global config channel. The poller does this everywhere via
  `_run_home(run)` (US-007) — hypothesis buttons, loop narration, completion
  summary, chart upload; button/edit replies ride Bolt's channel-bound
  `say`. Completion rule: only paper-channel runs get the Block Kit Promote
  button; a live-channel run's summary appends `LIVE_PROMOTABLE_NOTE`
  ("say *promote to live*") instead — a paper-promotion button must never
  appear in the real-money channel (the spoken flow is US-010). GPU runs
  route the same way out-of-process (US-017): `_start_research_tool` passes
  the run's home channel to `GpuBackend.launch`, which forwards it as the
  pipeline's `--channel` — the pipeline itself never reads the runs table.
- Dual-channel routing (US-006): with `live_channel_id` set, app.py's
  actionable check accepts BOTH channels; all other filtering is identical.
  The per-message channel rides Bolt's `Say` (`say.channel` — Bolt binds it
  to the event's channel), read via the `_say_channel` helpers in
  conversation.py and poller.py: a non-string/missing attribute (test fakes,
  MagicMock) means "unknown" and falls back to today's single-channel
  behavior, so never rely on a truthy Mock. The message channel overrides
  the core's wired default when stamping `runs.channel_id`, and thread-keyed
  lookups reachable from a message (edit-reply consumption, spoken
  approve/reject) refuse when the message's channel is not the channel that
  owns the run — a live-channel thread_ts can never act on a paper-channel
  record. New Slack-facing tools that act on thread-keyed state should take
  the resolved `channel` the same way. Replies must always go through
  `say`, never `chat_postMessage(config.channel_id, ...)`, or a thread
  migrates channels.

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
- Autonomous runs (US-045, the DEFAULT — `runs.supervised=0`): the poller
  auto-approves each hypothesis (submit unchanged → resolve `auto_approved` →
  narrate to the thread, no buttons; narration is best-effort and never
  unwinds a submit) and posts a one-line verdict per auto-acked feedback.
  Brakes: after `max_hypotheses` submitted hypothesis rows (statuses in
  `SUBMITTED_STATUSES`; RDQ_MAX_HYPOTHESES via config.py, default 10) the
  next proposal is recorded 'cancelled' and the run stopped via /control —
  the run row stays 'running' ON PURPOSE so the US-022 completion path posts
  the summary + Promote offer; never flip it in the halt. 3 consecutive
  failed feedbacks with identical `reason` text abort the run early
  (infrastructure failure signature). All budget/streak state derives from
  `StateStore.list_interactions` each poll — no in-memory counters, restarts
  resume cleanly. Supervised runs come from `start_research supervised=true`
  and keep the button flow above; `editing` rows block autonomous approval
  too (operator owns the FIFO slot).
- Reject has no upstream regenerate action — `rejection_payload()` rides the
  instruction in the hypothesis text (see docs/decisions.md US-021 entry)
  and MUST keep the exact constructor key set (`type(hypo)(**dict)`).
- Testing Bolt block actions: dispatch a `{"type": "block_actions", ...}`
  payload as `BoltRequest(body=json.dumps(payload), mode="socket_mode")`
  (no event_callback envelope — interactive payloads ARE the body); Bolt
  injects `ack`/`action`/`say`, and `process_before_response=True` keeps it
  synchronous. See dispatch_action in tests/test_poller.py.

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
- Live halt/resume (US-008): `halt_live_trading`/`resume_live_trading` are
  registered ONLY when the core's `live_channel_id` is set (the model never
  sees them in a paper-only deployment) and flip a SECOND breaker on the
  live paths (`breaker.LIVE_HALT_FILE` etc.; the core default-constructs it
  only when the live channel is armed — tests inject one over tmp paths via
  `live_breaker=`). Channel gating is asymmetric BY DESIGN: the live tools
  require positive identification (`channel == live_channel_id`; an unknown
  channel from a fake/mock refuses too), while paper
  halt_trading/resume_trading refuse only a POSITIVELY-identified live
  channel (unknown stays permissive — the paper-freeze rule). Decision Log
  types are `halt_live`/`resume_live`; formatters must say LIVE
  unmistakably. Follow this shape (registration gate +
  `_require_live_channel` + `_refuse_paper_tool_in_live_channel`) for every
  future live-only tool (US-010's promote_to_live/demote_live).

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
  (`PromotionFlow`): the poller adds the Promote button to a finished run's
  summary (`promotion_offer_blocks`; statuses in `PROMOTABLE_STATUSES` =
  completed AND operator-stopped — unbounded orchestrator runs only ever end
  by an operator stop, so 'stopped' is their normal successful ending;
  US-044), and app.py routes the three
  `run_promote`/`promote_confirm`/`promote_cancel` actions via the
  `PromotionHandler` protocol. Button values carry ONLY the thread_ts — the
  candidate (workspace, topk/n_drop AND universe from the workspace's own
  conf via `execution.signal.load_strategy_params`/`load_market`, tickers
  from the store instruments file, headline metrics) is re-derived from
  SQLite + run artifacts on every click, so buttons survive restarts with no
  pending-promotion state. The universe is NEVER taken from the run-row
  label — the conf's market line bounds pred.pkl (2026-08-05 incident); a
  disagreeing label is called out in every promotion message. Promotion
  refuses when the run is running/failed or topk/n_drop/market can't be read
  (the rebalancer couldn't reproduce the strategy); metrics merely degrade
  to n/a, and a missing instruments file degrades universe_tickers to None
  (the rebalancer's divergence check skips). Confirm pins
  workspace + config into the single `promoted_strategy` row (replacement is
  announced in-thread), writes a Decision Log row
  (`NotionRecorder.record_decision`), and moves the idea page's Status to
  `promoted`. The rebalancer-side check is `execution/promoted.py` —
  keep the pinned config keys (universe/universe_tickers/topk/n_drop/
  thread_ts/session_path) in sync with what US-034 consumes.
- Live promotion slot (US-003): `promoted_strategy_live` is a second
  single-row table with the same shape, accessed ONLY via
  `set_promoted_strategy_live`/`get_promoted_strategy_live`/
  `clear_promoted_strategy_live` and read by
  `execution.promoted.load_promoted_strategy_live`. Its pinned config carries
  the paper keys plus `live_equity_allocation_pct` (captured at promote time
  for audit). No paper-slot code path may read or write the live table, and
  vice versa — tests assert both directions.
- Live promotion backend (US-009): `orchestrator/promotion.py`'s
  `LivePromotion` (over a `PromotionFlow` for provenance) is the ONLY writer
  of the live slot; US-010's promote_to_live/demote_live tools must stay thin
  wrappers over `promote()`/`demote()`. Resolution order: explicit run
  reference (thread_ts exact, or >=6-char session-path fragment; ambiguous
  and no-match both refuse with a listing) → the current thread's run → the
  paper slot copied IN FULL (never re-derived). Every refusal raises
  `LivePromotionError` before anything is written. Run provenance goes
  through `PromotionFlow.candidate_from_run` — never fork the conf-market/
  tickers derivation. Pred-refresh snapshot rules differ from paper: a
  COMPLETE snapshot is never touched (it may carry an operator-pinned
  market), an incomplete one is regenerated with a warning in
  `result.warnings`, and a snapshot FAILURE refuses (paper warns-and-
  promotes). Decision Log types are `promote_live`/`demote_live`; the
  triggering-message permalink is passed in by the caller
  (`trigger_permalink=` — the recorder's own permalink fn is paper-channel
  only).
- Live promotion tools (US-010): `promote_to_live`/`demote_live` are ONE
  message with NO confirmation step (2026-08-10 decision — the posted armed
  summary IS the confirmation; never add a confirm round-trip). Registered
  only when BOTH `live_channel_id` and `live_promotions=` (a LivePromotion)
  are wired; gated by `_require_live_channel` like the halt tools. Tool-side
  refusals that the backend deliberately does NOT own: live breaker halted,
  and live guardrail config unusable (`_live_guardrails` loads
  limits.live.json + breaker.live.json BEFORE promoting — the armed summary
  quotes their numbers, so an unusable file refuses with nothing written).
  The (channel, ts) permalink resolver rides the core's `permalink=` ctor
  param (app.py wires `chat_getPermalink` in the message's own channel);
  resolution is best-effort — a Slack failure degrades to None, never blocks
  arming. `LIVE_REBALANCE_SCHEDULE` in conversation.py names when the live
  rebalance fires — keep it in sync with ops/rdq-rebalance-live.timer
  (US-019).
- Live status reads (US-012): `check_live_account`/`check_live_orders`/
  `check_live_pnl` answer from `orchestrator/live_status.py`'s
  `LiveStatusReader` — the latest Account Snapshots (Live) / Trade Ledger
  (Live) rows plus orchestrator state (live slot, live breaker). NEVER give
  the orchestrator a live broker client: the US-053 read-path decision
  (docs/decisions.md 2026-08-13) is Notion/state reads only, and
  tests/test_live_status_tools.py greps orchestrator/*.py for the live host
  and `allow_live` — neither may ever appear. Registered like the other
  live tools (live_channel_id + `live_status=` wired; `_require_live_channel`
  gating), and the paper check_* twins refuse a positively-identified live
  channel via `_refuse_paper_tool_in_live_channel`. Empty/unbootstrapped
  live databases (ids still None in config.yaml) read as "no live data yet"
  — a graceful answer, not an error; only a real Notion outage surfaces as
  an error tool result. `NotionClient.query_db(max_results=)` exists for
  exactly these "latest N rows" reads — pass sorts + max_results instead of
  paginating a growing database.

- Spoken hypothesis decisions + conversational promotion (US-044):
  ConversationCore optionally takes `interactions=` (the HypothesisPoller) and
  `promotions=` (the PromotionFlow) and, ONLY when wired, offers
  `approve_hypothesis`/`reject_hypothesis` and `promote_run`/
  `confirm_promotion` ToolSpecs — a spoken "approve" rides the exact same
  poller/flow handlers as the buttons (protocols `HypothesisSteering` /
  `PromotionManager` in conversation.py). The hypothesis tools act on the
  thread's OLDEST pending row (the FIFO rule) and re-check the row's status
  after the handler runs: a failed submit reports back as a normal tool
  result (the handler already posted the failure in-thread), never resolves
  the row. The promotion tools capture what the flow posted via a recording
  `say` wrapper and return it verbatim so the model relays the actual
  confirmation/refusal. Edit stays button-driven (the edit-reply interception
  in app.py consumes the NEXT thread message raw — a conversational edit tool
  would race it). Two-step promotion is preserved conversationally: the model
  is prompted to require an explicit second yes before confirm_promotion.

- OneCLI approvals bridge (US-039) lives in `orchestrator/approvals.py`
  (`ApprovalsBridge` + `OneCliApprovalsClient`): a second background thread
  (started alongside the poller in app.py main) long-polls the gateway's
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

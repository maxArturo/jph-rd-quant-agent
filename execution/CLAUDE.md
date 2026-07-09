# execution/ — module notes

- PAPER ONLY. All broker traffic goes through `execution/alpaca_client.py`
  (`AlpacaClient`, default base `https://paper-api.alpaca.markets`) as bare
  HTTPS — no APCA headers anywhere in code (a source-grep test enforces it;
  the OneCLI proxy injects BOTH paper secrets when running under
  `onecli run --agent rdq-exec-paper`). The constructor hard-refuses the live
  host; never add a code path that targets it.
- Retry policy is deliberate: only 429 is retried (Retry-After honored,
  exponential fallback). 5xx is NEVER retried — a 500/timeout on
  POST /v2/orders is ambiguous (the order may have been accepted) and a blind
  retry could double-submit. Don't "improve" this with generic transient
  retries; idempotent replay needs `client_order_id` dedup, not client loops.
- Alpaca v2 wire quirks: numeric fields arrive as STRINGS ("equity":
  "100000.25"); position `qty` is signed (negative = short) alongside a
  `side` field; notional orders have `qty: null`; DELETE /v2/orders/{id}
  returns 204 with no body. `place_order` sends qty/limit_price as strings
  (API canonical form). GET /v2/orders returns newest-first with no cursor —
  `list_orders(after=, until=)` takes RFC3339 submitted_at bounds and
  ops/reconcile.py pages backwards by tightening `until` to the oldest stamp
  of each full page (dedupe by order id; a full page sharing one timestamp
  cannot be paged past and raises).
- Client testability pattern (same as data/fmp.py / orchestrator/
  notion_client.py): `AlpacaClient(session=..., sleep=...)` — FakeSession
  records `.request(method, url, params=, json=, timeout=)` calls and returns
  queued FakeResponses; sleeps captured in a list. Reuse from
  tests/test_alpaca_client.py.
- Live tests: `@pytest.mark.live` + self-skip unless `RDQ_LIVE_TESTS=1`; run
  via `RDQ_LIVE_TESTS=1 onecli run --agent rdq-exec-paper -- .venv/bin/pytest
  tests/test_alpaca_client.py -m live`.
- `execution/signal.py` replicates qlib's TopkDropoutStrategy selection
  line-for-line (a parity test in tests/test_signal.py transcribes the
  upstream lines and diffs against it) with two documented deviations: ties
  rank alphabetically (upstream is unstable-sort luck) and degenerate slices
  (`n_drop=0`, over-held book) are clamped instead of nonsense-trading. Don't
  "fix" the algorithm without re-reading
  qlib/contrib/strategy/signal_strategy.py.
- Signal-extraction failure policy: EVERYTHING raises `SignalError` before
  any `TargetBook` exists (stale/missing pred, empty cross-section, dup
  holdings). US-034 must treat SignalError as abort-without-trading, never
  catch-and-continue.
- Workspace qlib confs keep their jinja placeholders (qrun renders at run
  time) — parse them by rendering with `jinja2.Undefined` first
  (`load_strategy_params` is the template). topk/n_drop live at
  `port_analysis_config.strategy.kwargs`; all conf*.yaml in a workspace must
  agree or the loader refuses.
- `execution/order_gate.py` is PURE — no HTTP, no state. The caller passes a
  fresh `Account`/`Position` snapshot plus today's order count and gets a
  `GateResult` back; US-034 should call `result.raise_for_rejections()` to
  abort-without-trading on any rejection. Boundary semantics: exactly AT a
  limit passes, strictly over fails. Rejection messages start with the
  violated JSON key from `execution/limits.paper.json` (all four keys
  required; unknown keys refused — edit the file and `load_limits` in sync).
- Gate projection details: batches evaluate sequentially and cumulatively
  (approved orders update the projected book; rejected ones don't), position
  exposure is marked at the order's limit price, and only orders that GROW
  |position| are pct-checked so an oversized position can always be trimmed.
- `execution/diff.py` is PURE like the gate and emits the gate's
  `ProposedOrder` type directly. US-034 must pass a `prices` map covering
  every symbol that is held OR targeted (exits need a price too) and treat
  `DiffError` as abort-without-trading. Rules that must not drift: target
  shares = floor(weight*equity/ref_price); full exits are exact-qty and
  NEVER skipped (a short exit is a buy); the min-notional skip applies ONLY
  to held+targeted rebalance deltas (at-threshold trades); buys round limit
  prices UP to the cent, sells DOWN; output is sells-then-buys, alphabetical
  within each side.
- `execution/breaker.py` mirrors the gate's conventions: thresholds in
  `execution/breaker.paper.json` (both keys required, unknown refused,
  at-limit passes / strictly-over trips), trip messages prefixed with the
  violated key. State files default under `~/rdq-data/breaker/`: `halt`
  (operator kill switch — US-038 tools call `Breaker.halt()`/`clear_halt()`)
  and `high_water_mark.json`. Check order: halt → daily notional → drawdown.
  US-034 must branch on `trip.reason`: `HALT_FILE` exits 0 ("halted" notice);
  the other trips exit nonzero.
- The high-water mark only moves UP, and only on a CLEAN pass (a trip never
  touches it). A corrupt/unreadable HWM file raises `BreakerStateError`
  (refuse to trade) — never "fix" it by silently re-seeding; that disarms
  the drawdown kill switch.
- pred.pkl = mlflow artifact at `mlruns/<exp>/<run>/artifacts/pred.pkl`,
  MultiIndex (datetime, instrument), first column is the score (upstream uses
  `.iloc[:, 0]` too). Newest mtime wins when a workspace holds several runs.
- Freshness rule: latest pred cross-section date must be >= the last store
  calendar entry on/before as_of (`~/.qlib/qlib_data/us_data/calendars/
  day.txt`). Predictions are made FROM day T FOR T+1, so pred dated the last
  completed trading day is fresh for a pre-open rebalance.
- `execution/rebalance.py` is the pipeline assembly (US-034): market calendar
  -> promoted load -> signal -> diff -> gate -> breaker -> submit -> poll
  fills. `run_rebalance()` returns the process exit code — 0 for traded /
  dry-run / nothing-to-trade / operator halt, 1 for every
  abort-without-trading (the reason is posted via the injected `notify`
  callable AND printed). Known aborts are the `_ABORT_ERRORS` tuple; anything
  else notifies then re-raises (bugs must crash loudly). Keep new failure
  modes inside that contract.
- Rebalance conventions downstream stories rely on: the trading-day check is
  Alpaca `GET /v2/calendar` (the qlib store calendar ends at the last built
  bar and cannot say whether *today* trades); "today" for the day-order count
  and traded notional means `submitted_at` converted to America/New_York;
  `client_order_id` is `rdq-<YYYY-MM-DD>-<side>-<symbol>` so a same-day rerun
  is rejected by Alpaca's uniqueness check instead of doubling the book;
  reference prices come from the store's latest close/factor with
  `Position.current_price` as the held-name fallback. A dry run still runs
  the gate and breaker (and can seed/raise the HWM) — it only skips
  submission.
- Slack from the rebalancer is a plain `slack_sdk.WebClient` notifier
  (`slack_notifier()`, lazy import — never Bolt in execution/); tests inject
  a list-append notify. The future service unit needs the same
  NO_PROXY=slack.com trick as rdq-orchestrator.service. `--no-slack` swaps in
  a stderr notifier for supervised local runs; without it, missing Slack
  config refuses to run at all.
- Testing the pipeline: tests/test_rebalance.py's `FakeBroker`/`RoutedSession`
  route (method, path) -> handler through the REAL `AlpacaClient` parsing
  (integration per the AC) — reuse them for US-035/036 instead of stubbing
  the client. Fixture helpers `write_bins` (store price bins) there and
  `write_pred`/`write_calendar`/`write_conf` in tests/test_signal.py compose
  into a full promoted-strategy environment in a tmp dir.
- The rebalancer's entrypoint check is `execution/promoted.py`
  (`load_promoted_strategy()`): it refuses with `NoPromotedStrategyError`
  when the orchestrator state DB is absent (it must NEVER create it from the
  execution side), when no `promoted_strategy` row exists, or when the pinned
  workspace directory is gone. US-034 must call it FIRST and treat the error
  as abort-without-trading. The pinned config dict carries
  universe/universe_tickers/topk/n_drop/thread_ts/session_path — pass topk/
  n_drop into `signal.StrategyParams` rather than re-deriving them from the
  workspace conf (the operator confirmed those exact values when promoting).
- `execution/ledger.py` (`TradeLedger`) is the Notion Trade Ledger's SOLE
  writer (one-writer-per-DB; the orchestrator's NotionRecorder must never
  touch that database). Row lifecycle: `record_submitted` right after each
  POST /v2/orders succeeds (so a mid-batch submit failure still leaves rows
  for the live orders), `record_final` with the post-poll snapshot; if the
  submit-time create failed, `record_final` creates the full row instead of
  updating. Writes are best-effort — never raise, never abort a run — but
  failures accumulate in `TradeLedger.failures` and the rebalancer appends
  them to the daily summary as WARNING lines. Property names must match the
  Trade Ledger schema in docs/reference/notion-schema.md; Alpaca statuses map
  through `ledger_status()` (canceled -> cancelled; non-terminal ->
  submitted/partially_filled).
- The daily Slack digest is `rebalance.format_daily_summary()` (equity,
  orders placed, fills via fill_summary, gate/breaker rejections, ledger
  warnings). It is posted on every day the pipeline reaches the gate:
  traded and no-trade days exit 0; gate-rejection and breaker-trip days post
  it WITH the rejection lines and exit 1 (those paths no longer go through
  the generic "rebalance aborted" message — earlier failures still do).
  US-038 adds the breaker state line here.
- Live Notion access for `rdq-exec-paper` is an app-connection GRANT on the
  agent, not a vault secret — see docs/decisions.md 2026-07-09 (US-035). If
  ledger writes start 401ing, re-grant the Notion connection to the agent in
  the OneCLI web UI; do not vault a key.

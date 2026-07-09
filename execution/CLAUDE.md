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
  (API canonical form).
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
- pred.pkl = mlflow artifact at `mlruns/<exp>/<run>/artifacts/pred.pkl`,
  MultiIndex (datetime, instrument), first column is the score (upstream uses
  `.iloc[:, 0]` too). Newest mtime wins when a workspace holds several runs.
- Freshness rule: latest pred cross-section date must be >= the last store
  calendar entry on/before as_of (`~/.qlib/qlib_data/us_data/calendars/
  day.txt`). Predictions are made FROM day T FOR T+1, so pred dated the last
  completed trading day is fresh for a pre-open rebalance.

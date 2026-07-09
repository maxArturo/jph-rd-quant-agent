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

# data/ module notes

- HTTP clients here make **bare HTTPS calls** — never put an apikey in code, params, or env.
  The OneCLI proxy injects credentials when the process runs under
  `onecli run --agent rdq-research` (FMP secret is assigned to that identity).
- `data/fmp.py` returns **raw (unadjusted) closes** sorted ascending by date. Splits and
  dividends come from `get_splits`/`get_dividends`; adjustment factors are computed
  downstream (data/adjust.py), never inside the fetch layer.
- Client testability pattern: `FmpClient(session=..., sleep=...)` — inject a fake session
  (queued responses, recorded calls) and capture backoff sleeps in a list. No
  monkeypatching needed. Reuse `FakeSession`/`FakeResponse` from tests/test_fmp.py.
- Live tests must be `@pytest.mark.live` AND self-skip unless `RDQ_LIVE_TESTS=1`
  (`make check` runs all markers). Run them via:
  `RDQ_LIVE_TESTS=1 onecli run --agent rdq-research -- .venv/bin/pytest -m live`
- FMP /stable quirks: list endpoints return newest-first (sort before use); errors can be
  a JSON *object* (`{"Error Message": ...}`) with HTTP 200-family semantics broken — the
  client raises FmpError on non-list payloads. `Retry-After` may be an HTTP-date; the
  client falls back to exponential backoff.

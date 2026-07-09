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
- `data/build_store.py` owns the Qlib bin store. Format facts (verified against qlib
  0.9.7 `FileFeatureStorage`): each `features/<sym_lower>/<field>.day.bin` is a
  little-endian float32 array whose FIRST element is the calendar index of the ticker's
  first bar; `instruments/<market>.txt` is tab-separated `SYMBOL\tstart\tend`;
  `calendars/day.txt` is one ISO date per line. Stored prices are ADJUSTED
  (raw * factor), volume is raw / factor, and a `factor` field is kept so raw values are
  recoverable (close / factor) — incremental refresh (US-036) needs that.
- Store builds write to `<target>.tmp`, validate, then swap via `<target>.old`; a failed
  build/validation must never leave a partial store. Validation hard-fails on NaN
  close/factor inside a ticker's own span (mid-series source gap) — don't relax silently.
- Backfill checkpoints live at `<output>.checkpoint/<SYM>.json` keyed by (start, end)
  window; a window change or corrupt file triggers refetch, same window resumes.
- `pyqlib>=0.9.7` is a declared project dep (installs clean on py3.12, coexists with
  rdagent). `qlib.init(provider_uri=..., region="us")` + `D.features` is the read path;
  import qlib lazily (multi-second import).
- Universes: `data/make_universe.py` writes `instruments/<name>.txt` into an EXISTING store
  (rows are `SYMBOL\tstart\tend` with spans copied from `all.txt`; name `all` is reserved).
  Built-in universe configs live in `data/config.yaml` (`us_liquid` = min ADV + min price
  filters, defaults to every store ticker; `sp500` = committed snapshot
  `data/sp500_tickers.txt`, refresh command in the yaml comment). Qlib resolves a universe
  via `D.instruments(market="<name>")` — the market string IS the instruments filename.
- Liquidity math exploits the store's field conventions: stored close * stored volume ==
  RAW daily dollar volume (factors cancel), and raw price on the last day = close / factor.
  Don't "fix" filters to de-adjust first.
- Factor source h5 (`data/make_factor_source.py`): RD-Agent's factor coder consumes a
  FOLDER (env `FACTOR_CoSTEER_DATA_FOLDER` / `..._DEBUG`) whose files are ALL linked into
  each factor workspace; the LLM prompt describes the DEBUG folder's files by name. Both
  folders must therefore hold the SAME filename `daily_pv.h5` + a README.md explaining
  `pd.read_hdf(..., key="data")`. Our generator writes `daily_pv_all.h5`/`daily_pv_debug.h5`
  at the output root (upstream generate.py naming) plus ready-to-point `data_folder/` and
  `data_folder_debug/` subfolders — US-017 sets the env vars to those subfolders.
- The daily_pv frame contract (upstream parity, tested against qlib `D.features`):
  MultiIndex `(datetime, instrument)`, float32 columns
  `$open/$close/$high/$low/$volume/$factor`, rows only inside each instrument's own span.
  Reading the store bins directly with numpy reproduces `D.features(...).swaplevel()
  .sort_index()` exactly and avoids the multi-second qlib import — but note
  `pd.DataFrame.to_hdf` APPENDS to an existing file; unlink first when regenerating.
- `data/adjust.py` is the ONLY place adjustment math lives: backward adjustment, factor
  1.0 on the window's last bar, events strictly-before-ex-date get the multiplier
  (split: 1/ratio; dividend: (prev_close - D)/prev_close using the last bar close before
  the ex-date). Events dated on/before the first bar or after the last bar are IGNORED —
  FMP's /dividends can list announced *future* ex-dates, which must not adjust today's
  store. Adjusted close = raw close * factor; for Qlib volume, divide raw volume by the
  factor (US-013).
- Incremental refresh (`data/refresh.py`, US-036): recovers RAW bars from the
  store bins (raw price = stored/factor, raw volume = stored*factor), pulls only
  bars after each ticker's own last date, refetches full split/dividend history,
  and rebuilds through `build_store` — so a split landing between refreshes
  re-scales the whole history correctly. `build_store(..., extra_instruments=)`
  carries make_universe files across the rebuild (spans refreshed) inside the
  same atomic swap; without it a rebuild DELETES `instruments/<universe>.txt`.
  Default `--end` is *yesterday* in America/New_York, never today — FMP's EOD
  endpoint returns a partial bar for an in-progress session. When nothing is
  new, the store is left byte-for-byte untouched (safe to run any time).

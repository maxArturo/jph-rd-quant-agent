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
- Market series (US-065): `get_commodity_eod` (/historical-price-eod/light — price+volume
  only, no OHLC; commodities trade some NYSE holidays, DXUSD has weekend rows) and
  `get_treasury_rates` (chunks <=`TREASURY_CHUNK_DAYS`=90 calendar days per request and
  dedups boundary dates — the endpoint silently truncates wider windows with HTTP 200).
  Ingestion (US-066/067) must use these, never re-implement the chunking.
- Market broadcast fields (US-066): `build_store(..., market_series={"mkt_x": [(date, value), ...]})`
  writes each series RAW into EVERY instrument's feature dir (identical value per date,
  implicit factor 1 — the adjustment math never touches them). Names carry no `$` at this
  layer (bins never do). Per-series rules in `_market_matrix`: forward-fill equity days
  with no observation (off-calendar weekend prints count as observations), NaN before the
  series' first observation, and direct coverage >= `MARKET_COVERAGE_MIN` (99%) of trading
  days since first observation or the build fails naming the series. The canonical set is
  `MARKET_FIELDS` (= `COMMODITY_SYMBOLS` keys + `TREASURY_FIELD`); `fetch_market_series`
  fetches them all (skipping None year10 rows). CLI: `--market-start` (canonical
  `MARKET_SERIES_START` 2025-01-02).
- Market fields in the refresh (US-067): `data/refresh.py` reads any `mkt_*` bins back as
  observations (`read_market_series` — ffilled values read back as direct observations,
  NaN heads dropped, all symbols overlaid since no single ticker need span the calendar)
  and carries them through every rebuild, including `extend_store`. Stored series advance
  incrementally under the same yesterday-ET `--end`; a per-series FMP failure degrades to
  forward-fill + a warning in `RefreshResult.warnings` (the CLI prints them and posts one
  Slack line via `execution.rebalance.slack_notifier`; `--no-slack` skips) and NEVER fails
  the run. `--market-start` is the one-time introduction path: it backfills canonical
  MARKET_FIELDS the store lacks and forces a rebuild even with no new equity bars.
  GOTCHA: this only covers `mkt_*`-prefixed broadcast bins — any future per-ticker raw
  field (e.g. US-014's news_ct_1d) needs its own carry-through or a refresh drops it.
- Stock news (US-071/072): `FmpClient.iter_stock_news_pages`/`get_stock_news` walk
  /stable/news/stock (newest-first). TERMINATION RULE: the endpoint has page-size
  jitter — mid-stream pages come back short of `limit` with more history behind;
  ONLY an empty page ends the walk, never a short page. `publishedDate` is
  US/Eastern wall-clock at second resolution — keep it naive, never tz-convert.
- News counts (`data/build_news.py`): PIT cutoff is strictly-after 16:00:00 ET ->
  next trading day (16:00:00 exactly = same day); bucketing is pure date/time
  arithmetic on the store calendar, DST-safe because timestamps are already ET.
  Layout under the news root (default ~/rdq-data/news): `archive/<SYM>/<date>.json`
  (raw ts/headline/url/publisher keyed by PUBLISHED date — sentiment scoring later
  reads this, never refetches; date files rewritten wholesale per fetch) and
  `checkpoints/<SYM>.json` keyed by (start, end) window. Counts ALWAYS derive from
  the archive (`daily_counts`: gapless, 0 on no-news trading days); an article
  published after the window's last close stays archived and counts once a later
  window covers the next trading day. GOTCHA (same trap as mkt_*): US-014's
  news_ct_1d bins will be DROPPED by refresh rebuilds unless US-015 adds a
  read-back+carry. The full us_liquid backfill (US-013) lives on-box at
  `~/rdq-data/news/` (outside the repo tree): 589 tickers, window
  2025-01-02..2026-08-23 in every checkpoint — reruns with the same window
  resume for free; a different --end refetches per ticker.
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
  (rows are `SYMBOL\tstart\tend`; name `all` is reserved). Built-in universe configs live
  in `data/config.yaml` (`us_liquid` = min ADV + min price filters, defaults to every
  store ticker; `sp500` = committed snapshot `data/sp500_tickers.txt`, refresh command in
  the yaml comment). Qlib resolves a universe via `D.instruments(market="<name>")` — the
  market string IS the instruments filename.
- Universe membership modes (US-023): `mode: pit` (us_liquid's default) re-evaluates
  ADV/price on each month's FIRST trading day with one-period entry/exit hysteresis (a
  flip needs the opposite signal on 2 consecutive evaluations) and emits MULTIPLE
  span rows per symbol; `mode: last_window` (default elsewhere, `--mode` overrides) is the
  legacy full-span filter used by frozen `*_promoted_*` snapshots. Anything parsing a
  universe file must dedup symbols (`set` of column 0) — multi-row symbols are legal.
  PIT span ends are clamped to the ticker's own `all.txt` end, and evaluations tolerate
  a shorter-than-window early history (legacy parity).
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
  Both folders also get `market_series.h5` (US-068: plain DatetimeIndex x `$mkt_*`
  columns, key="data"; debug variant windowed to the debug frame's trading days) plus a
  README section on market-level semantics — but ONLY when the store carries mkt_* bins;
  a pre-introduction store yields no market file, no README section, and stale copies
  are removed on regeneration. Adding a new companion file means BOTH folders get the
  same filename and the README describes it, or the coder LLM misuses/misses it.
- The daily_pv frame contract (upstream parity, tested against qlib `D.features`):
  MultiIndex `(datetime, instrument)`, float32 columns
  `$open/$close/$high/$low/$volume/$factor`, rows only inside each instrument's own span.
  Reading the store bins directly with numpy reproduces `D.features(...).swaplevel()
  .sort_index()` exactly and avoids the multi-second qlib import — but note
  `pd.DataFrame.to_hdf` APPENDS to an existing file; unlink first when regenerating.
- `data/menu.py` is the single source of truth for "what data exists" (US-061):
  store introspection + `CURATED_FIELDS`. Adding a store field requires a curated
  entry AND regenerating the doc (`python -m data.menu --write-doc`) or the drift
  test in tests/test_menu.py fails. The doc is deliberately schema-only — never
  embed calendar dates/universe counts in it (they change with each daily refresh).
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
  carries make_universe files across the rebuild inside the same atomic swap
  (without it a rebuild DELETES `instruments/<universe>.txt`). It takes FULL
  (symbol, start, end) rows written verbatim; span semantics live in
  `refresh.refresh_universe_spans`: an open span (end == ticker's pre-refresh
  last bar) follows the ticker's new end, a closed span (PIT exit) is carried
  byte-for-byte — never re-derive universe spans from all.txt, that flattens
  PIT membership back to full-history (US-024).
  Default `--end` is *yesterday* in America/New_York, never today — FMP's EOD
  endpoint returns a partial bar for an in-progress session. When nothing is
  new, the store is left byte-for-byte untouched (safe to run any time).

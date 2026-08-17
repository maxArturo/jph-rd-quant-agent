# ops/ — OneCLI + shell script conventions

- Proxied requests: `onecli run --agent <identifier> -- curl ...`. The
  "gateway connected" banner goes to **stderr**; the wrapped command's stdout
  is clean, so `-w '%{http_code}' 2>/dev/null` capture works.
- `onecli agents create/run` take the **identifier** (e.g. `rdq-research`);
  `onecli agents secrets` / `set-secrets` take the agent **UUID** — resolve it
  from `onecli agents list` by identifier first.
- `onecli agents set-secrets` replaces the full assignment list (not additive);
  always pass the complete computed set.
- All onecli list commands output JSON (`.data[]`); parse with jq (installed).
- Bare probe endpoints that work through the proxy: Anthropic
  `GET /v1/models` (requires `anthropic-version: 2023-06-01` header or it
  400s), Alpaca `GET /v2/account`, FMP `/stable/search-symbol?query=AAPL`
  (proxy appends `apikey` even when the URL already has query params).
- Paper-only rule: never register `rdq-exec-live` or assign a secret with
  host pattern `api.alpaca.markets`; check_onecli.sh treats a 2xx from the
  live host as a hard failure.
- Scripts must pass shellcheck; use `set -euo pipefail` for setup-style
  scripts, `set -uo pipefail` (no `-e`) for check-style scripts that collect
  failures.

## GPU burst worker (ops/gpu_worker/)

- `gpu_worker.sh` drives the whole lifecycle FROM this box (provision →
  bootstrap → tunnel → check → run → fetch → destroy) against a disposable
  DO GPU droplet. The worker holds no credentials: LLM traffic rides an
  `ssh -R` reverse tunnel (transient user unit `rdq-gpu-tunnel`) to the local
  OneCLI proxy (:10255), with the per-agent proxy token + MITM CA copied only
  to the worker (`/root/rdq-proxy.env`, mode 600). Never write the token to
  the control-box state dir or echo it — print only host:port.
- `RDQ_LAUNCHER=direct` in ops/run_us_quant.sh is the worker-side launch path
  (no onecli there): it requires HTTPS_PROXY + REQUESTS_CA_BUNDLE pre-exported
  and TCP-probes the proxy before exec. Keep its contract in sync with what
  `gpu_worker.sh tunnel` writes into /root/rdq-proxy.env.
- The droplet BILLS UNTIL DESTROYED (default RTX 4000 Ada ~$0.76/hr) —
  `destroy` is part of finishing a run; `status` shows what's still alive.
  Runbook: ops/gpu_worker/README.md (promotion happens outside this flow —
  see its "Promotion caveat").
- `python -m ops.gpu_pipeline` is the one-command wrapper (provision→run→
  per-loop Slack digests→fetch→destroy; teardown also on failure/--max-hours).
  Slack-started runs launch it as a transient unit `rdq-gpu-run-<thread>`
  (orchestrator/gpu_backend.GpuBackend.launch via systemd-run — clean env, so
  the bot's onecli proxy never leaks into doctl/ssh). It posts into the
  START THREAD (--thread-ts), writes live status JSON (--status-file, read by
  the bot's check_research_status tool), seeds the directive
  (--instruction → research/quant_runner.py — plain `rdagent fin_quant`
  ignores directives entirely), wires custom universes (--universe; worker
  hard-refuses missing artifacts), uploads the candidate-vs-promoted chart
  (orchestrator/summary.render_comparison_curve), runs ops/notion_summary
  under `onecli run --agent rdq-orchestrator` and posts the page URL, and
  finalizes the run row (session_path → fetched trace dir, status).
  SAFETY: ConversationCore takes gpu=None by default and REFUSES to start
  runs — only app.py wires the real GpuBackend. Never give it a live default:
  the test suite once auto-launched a real droplet through one (2026-08-06).
  It reads loop progress with `ops.gpu_trace` (offline FileStorage parsing,
  sweep.py-style: hypothesis/feedback/runner-result pkls per Loop_<n>) run ON
  the worker over `gpu_worker.sh ssh`; fetched copies need the --remap flag
  because runner-result pkls store the worker's ABSOLUTE workspace paths.
  Slack posting mirrors app.py: WebClient with `proxy = None`, chat_postMessage
  with thread_ts (root post first). Stop a run with the stop_run tool from
  the OWNING thread (it cancels via GpuBackend; other threads are refused —
  US-020), or manually with `gpu_worker.sh ssh tmux kill-session -t rdq-run`.
- Rolling TEST_END (US-008): `gpu_pipeline` computes RDQ_TEST_END at launch —
  store calendar end minus `--confirm-days` (default 42, env RDQ_CONFIRM_DAYS)
  TRADING days — BEFORE provisioning (a broken store must cost $0), and ships
  it via `gpu_worker.sh run --test-end` into the worker launch script. The
  reserved slice (TEST_END, store end] is the gate's confirmation window; the
  status file (pipeline_status.json) records `test_end`,
  `confirmation_window`, `confirm_days`, and `instrument_hash`
  (promotion_gate.hash_instruments over the resolved store instrument list) —
  US-010/US-011 must read parity inputs from there, not re-derive them.
  run_us_quant.sh keeps a hardcoded fallback TEST_END for manual runs but
  fails both modes when the resulting test end trails the store calendar end
  by >90 calendar days (`check_test_end_lag`) — refresh the fallback when it
  trips.
- Run memory contract (US-013): every Strategy Notes page body carries a
  machine-readable `run_summary` JSON — a fenced `json` code block after the
  prose, chunked to Notion's 2000-char rich-text element cap (and its
  100-elements-per-block cap), written by `ops/notion_summary.py`
  (`build_run_summary` → `run_summary_blocks`) and read back with
  `parse_run_summary` (handles both write- and API-read rich_text shapes;
  returns None on anything unparseable — callers degrade, never raise). Bump
  `RUN_SUMMARY_SCHEMA_VERSION` when the shape changes. The summary is built
  FROM the pipeline's context JSON (`gpu_pipeline.build_notion_context`) —
  the write-up runs as a subprocess and sees only that file, so any new field
  the summary needs must be added to the context builder, not read from disk.
  The reader side (US-014) is `orchestrator/run_memory.py build_digest`
  (NotionClient.list_block_children → parse_run_summary per row) — when the
  run_summary shape changes, update build_run_summary, the digest's
  `_summary_entry`, and bump the schema version together. Since US-015 the
  launch instruction may be directive + run-memory digest — the context's
  `directive` field strips the digest via `run_memory.split_instruction`
  (keep it that way, or digest text compounds into every future digest).
- `python -m ops.promote_fetched --workspace <dir> [--yes]` promotes a fetched
  GPU workspace: same records as orchestrator/promotion.py confirm_promotion
  (snapshot_pred_refresh + promoted_strategy row, conf-derived market per
  US-023) plus, since US-012, the Notion Decision Log row (via
  ops.promotion_decision.record_via_onecli — falls back to a manual reminder
  when the write fails; `--no-notion` skips). It runs the promotion gate in
  ADVISORY mode first (candidate hash from the CURRENT store list,
  confirmation window re-derived via gpu_pipeline.compute_run_dates — both
  approximations of the launch-recorded values a bare workspace can't reach);
  a non-PASS verdict blocks `--yes` unless `--force`, and the override is
  recorded in promotion_history.gate_verdict as `"forced": true`.
  Never creates state.sqlite; dry-run by default. Its
  `promote_workspace(ws, source=..., gate_verdict=...)` is THE promotion
  write path (validate → tickers → snapshot → pointer flip + history row,
  atomically last): the snapshot runs BEFORE the pointer flip so a snapshot
  failure raises with NOTHING written — any new promotion route must call it,
  never re-implement (and never copy orchestrator/promotion.py's
  warn-and-promote-anyway behavior).
- Auto-promotion (US-011): `gpu_pipeline.gate_and_promote` runs LAST in the
  pipeline (after chart + Notion write-up, so those describe the
  pre-promotion world) and NEVER raises — gate fail, gate error, and
  snapshot failure all finalize the run unpromoted with the verdict/error
  posted; on pass it promotes via promote_workspace with source='auto_gate'
  and the verdict JSON in promotion_history. Kill-switch:
  `promotion_gate.auto_promote: false` in config.yaml = report-only (verdict
  still posted). Parity hashes: candidate's from launch
  (pipeline_status.json), incumbent's from its promotion record's
  `universe_tickers` via hash_instruments (both canonicalize identically —
  verified matching on the real pair 2026-08-15). Status file gains
  `gate` (verdict dict or {"error": ...}) + `auto_promoted` fields.
- Base snapshots (US-022, `ops/gpu_snapshot.py`): worker boots are keyed on a
  short worker-inputs hash (research/PINNED_COMMIT + gpu_worker.sh + the
  Makefile venv targets + a STRUCTURAL store marker — field/calendar names
  only; store CONTENT rolls daily and must never churn the hash) embedded in
  the image name `rdq-gpu-base-<hash>-<ts>`. Selection is hash+REGION matched
  (snapshots are regional — the size-plan fallback must never boot an image
  that isn't in its region); no match = full bootstrap + auto-rebake at
  teardown (before destroy), pruning to the newest 2 images. All
  selection/prune logic lives in gpu_snapshot.py; gpu_worker.sh DELEGATES
  (`python -m ops.gpu_snapshot select|hash|prune`) so there is exactly one
  offline-testable implementation — never re-grow doctl parsing in the shell.
  Boot-mode facts a run needs later must be captured at launch
  (booted_from_snapshot rides the status file). Bake/selection failures are
  ALWAYS degrade-paths (full bootstrap / Slack warning), never run failures.
  The hash does NOT cover `local_qlib:latest` — after rebuilding it, force
  `--snapshot bake` (bootstrap only ships docker images to workers that lack
  one). Kill-switches: `--snapshot bake` / `--no-snapshot`.
- Global run mutual exclusion (US-020, `ops/run_lock.py`): exactly ONE GPU
  run may exist at a time — gpu_worker.sh keys everything off a single
  worker.env, so a second pipeline's teardown would DESTROY the first run's
  droplet. `gpu_pipeline` acquires `~/rdq-runs/gpu_worker/run.lock` before
  doing anything (`acquire_pipeline_lock`) and releases it in its finally; a
  REFUSED start exits 1 without running teardown (that's the whole point —
  never add worker_sh calls to the refusal path). The lock records
  unit + thread_ts + pid; stale = unit inactive AND pid gone (manual CLI runs
  have no real unit — the pid keeps their lock live), broken automatically
  with a Slack note. `run_lock.unit_name` is THE transient-unit naming
  convention (gpu_backend imports it — keep them shared). Orchestrator side:
  `GpuBackend.active_run_lock()` → start_research refuses naming the owning
  thread; stop_run only cancels when the requesting thread owns the lock
  (cancel kills THE worker's tmux session, whoever's it is).
- `ops/promotion_gate.py` (US-007) codifies "beats the incumbent":
  `evaluate_gate(candidate, incumbent, config)` is PURE — callers do IO via
  `load_metric_bundle(workspace, instrument_hash=...)` (every field degrades
  independently; a missing parity input FAILS parity against an incumbent,
  never guesses). Thresholds live in orchestrator/config.yaml
  `promotion_gate:` (`load_gate_config`; missing section = defaults;
  bootstrap_notion rewrites of config.yaml preserve the section but drop its
  comments). Boundary semantics: IR margin and min_ic are strict `>`, the
  MDD tolerance passes at-limit; MDD compares MAGNITUDES (abs) so qlib's
  negative sign convention can't flip a verdict. `hash_instruments` is the
  canonical universe hash (sorted/deduped/sha256[:16]) — US-008's
  pipeline_status hash and anything else comparing universes MUST use it.
  The verdict's `to_json()` is the promotion_history.gate_verdict payload;
  `slack_text()` is the operator block. `SURVIVORSHIP_CAVEAT` (US-025) is
  the single source of the standing delisted-names caveat — `slack_text()`
  appends it unconditionally and `ops/notion_summary.create_summary_page`
  imports it into every page body; if the wording changes, keep the
  docs/decisions.md US-025 entry it references in sync. The instrument hash cannot be
  derived from a workspace (conf names the market, not the resolved list) —
  it must be recorded at launch and passed in.
  Confirmation criterion (US-010): with an incumbent, `evaluate_gate`
  REQUIRES `ConfirmationEvidence` — omitting it (or any evaluation error)
  fails the criterion as `confirmation_unavailable`; there is no silent
  skip. Candidate confirmation-window IR must be strictly > incumbent's ×
  `confirm_ir_margin` (config, default 1.0 = no margin).
  `load_confirmation_evidence(cand, inc, start, end, **confirm_kwargs)` is
  the IO half: runs the incumbent FIRST (its docker re-predict is usually
  skipped, so failures are cheap before the candidate's ~minutes-long run)
  and never raises for evaluation problems — ConfirmWindowError lands in
  `evidence.error` naming the side.
- `ops/confirm_window.py` (US-009) turns any workspace WITH pred-refresh
  snapshot files into confirmation-window portfolio daily returns:
  `confirmation_returns(ws, start, end)` re-predicts via the US-049 docker
  machinery (test_end overridden to the window end) ONLY when the newest
  pred.pkl doesn't already cover every needed signal day — the incumbent's
  daily refresh usually spares its docker run; a fresh candidate always needs
  one (and needs `snapshot_pred_refresh` run on it first — this module never
  snapshots). Simulation: day d's return goes to the book selected from pred
  dated d-1 (live-rebalancer timing, one day AHEAD of qlib's own backtest),
  close-to-close on ADJUSTED store closes, equal-weight, open/close_cost on
  traded fractions, min_cost ignored — only cross-strategy comparability
  matters, so both sides of a gate comparison MUST come from this module.
  Every gap raises `ConfirmWindowError` (US-010 maps it to
  confirmation_unavailable); `annualized_ir` here is the confirmation
  criterion's metric (benchmark-free mean/std·√252, compute_sharpe
  convention). Reproduction check (US-010, from the 2026-08-15 c9587797
  incident): whenever the pred being used is NOT the workspace's original
  backtested pred (oldest mlruns pred.pkl) — fresh re-predict OR cached
  daily-refresh pred — it must mean-spearman >= `min_reproduction` (0.98)
  against the original on sampled overlap days, else ConfirmWindowError. A
  degenerate refresh (the incident: ~0.13) can therefore never feed the
  gate; the live promoted c9587797 currently FAILS this check by design
  until its parquet-conf snapshot is fixed or rolled back.
- `ops/promotion_decision.py` (US-012) is how ops-side code writes Notion
  Decision Log rows: the bearer only injects under `onecli run --agent
  rdq-orchestrator`, so promote_fetched and the pipeline's auto-promotion
  build a JSON payload (`build_payload`) and hop identities with
  `record_via_onecli` (subprocess re-entering `python -m
  ops.promotion_decision --payload <file>` — same pattern as notion_writeup).
  The write itself still goes through `NotionRecorder.record_decision`
  (one-writer-per-DB holds). Every promotion source's row carries a
  `promotion_gate.gate_summary_line` gate-standing line (conversational rows
  use the `GATE_NOT_EVALUATED` constant). Tests must monkeypatch
  `record_via_onecli` on any path that promotes — on a box with onecli
  installed the real thing writes actual Notion rows.
- `python -m ops.rollback_promotion [--to <ws>] [--yes]` (US-006) re-promotes
  a prior promotion_history entry using that entry's RECORDED config (never
  re-derives from the workspace conf) and appends a new history row (source
  'cli', gate_verdict carries the rollback marker). Snapshot re-run happens
  BEFORE the pointer flip so a failure leaves the current promotion intact;
  `--keep-snapshot` preserves an operator-pinned conf_pred_refresh.yaml (a
  frozen *_promoted_* universe would otherwise be regenerated away). Rollback
  targets stay restorable because sweep.py protects the workspaces of the
  last `RECENT_PROMOTIONS_KEPT` (3) history entries — if you raise/lower
  that, remember it bounds how far back rollback can reach on disk.
- `rdq-gpu-watchdog.{service,timer}` (hourly oneshot, in install_services
  UNITS + health.sh lists): warns on idle workers, fetch+destroys workers
  older than --max-hours 24. Stateless per tick; deletes stale worker.env
  when the droplet is already gone.
- `rdq-divergence.{service,timer}` (US-017): weekday 16:30 America/New_York
  post-close run of `execution.divergence` as rdq-exec-paper (portfolio
  history read needs the Alpaca secrets) with NO_PROXY=slack.com;
  After=rdq-rebalance.service. Persistent=true ON PURPOSE — it's a safety
  monitor, so a missed check fires on boot (unlike the rebalance timer).
  Quiet days are silent exit 0 (warmup/ok/nothing-promoted), so the journal,
  not Slack, is where a healthy run shows up.

## Python in ops/

- `ops/` is a real package (`ops/__init__.py`, listed in pyproject
  `packages.find` include) so tests can `from ops.foo import ...` — new
  Python entrypoints here are run as `python -m ops.<module>` (usually under
  `onecli run --agent <identity>`), not as loose scripts.
- Notion database bootstrap: `ops/bootstrap_notion.py` owns the seven DB
  schemas — its `database_properties()` must stay in sync with
  docs/reference/notion-schema.md (a test cross-checks property names against
  the doc's tables). DB ids land in `orchestrator/config.yaml` under
  `notion:`; rerunning is idempotent (matches child databases by title under
  the parent page).

## systemd user units

- Units live in `ops/` and are symlinked into `~/.config/systemd/user/` by
  `ops/install_services.sh` — append new units/timers to its `UNITS` array
  (tests/test_services.py asserts every listed unit file exists).
- Use `%h` for home paths; `WantedBy=default.target` (user manager has no
  multi-user.target); put `StartLimitIntervalSec` in `[Unit]`, not `[Service]`.
- The user manager's default PATH is minimal (`/usr/bin:/bin`) — any unit (or
  systemd-run transient) that shells out to doctl/onecli must set
  `Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin` (see
  rdq-gpu-watchdog.service and gpu_backend.py's --setenv). A subprocess call
  to a missing binary raises FileNotFoundError BEFORE any Slack notify —
  guard entrypoints with `shutil.which` and fail loud (gpu_watchdog.py
  pattern).
- `onecli run` injects HTTP(S)_PROXY process-wide but PRESERVES a pre-set
  NO_PROXY — any service wrapping `onecli run` must `Environment=` a NO_PROXY
  exemption for hosts that may not transit the proxy (Slack: `slack.com`;
  urllib suffix-matches, covering wss-primary/files subdomains).
  CAVEAT: NO_PROXY only helps clients that honor it — slack_sdk loads
  HTTPS_PROXY and ignores NO_PROXY entirely, so its websocket + WebClient
  need `proxy = None` forced in code (orchestrator/app.py main(), 2026-07-09;
  see runbook §6 "Slack-bot deafness check").
- From non-login shells (agents, cron) set
  `XDG_RUNTIME_DIR=/run/user/$(id -u)` before `systemctl --user` /
  `systemd-analyze --user verify`, or they can't reach the user manager.
- Failure notification (US-018): every `rdq-*.service` carries
  `OnFailure=rdq-notify-failure@%n.service` — a NEW service must add the line
  (tests/test_services.py globs ops/rdq-*.service and asserts it). The
  template (`rdq-notify-failure@.service` → `python -m ops.notify_failure %i`)
  posts "unit <name> failed" + journal tail to Slack; it deliberately has NO
  OnFailure of its own (no recursion) and no [Install]. Template units with
  `@` are invisible to the health.sh/install cross-check regexes
  (`rdq-[a-z-]+\.(service|timer)`), so they don't need a health.sh list entry.
  Verify one on-box with `systemd-run --user --unit=<name> -p Type=oneshot
  -p OnFailure='rdq-notify-failure@<name>.service.service' /bin/false`, then
  `systemctl --user reset-failed <name>.service`.
- `rdq-research.service` (the server_ui control plane) was DECOMMISSIONED in
  US-026: unit deleted from the repo, symlink + out-of-repo
  `rdq-research.service.d/resources.conf` drop-in removed on-box, port 19899
  dark (health.sh keeps 19899 in REPO_PORTS/FORBIDDEN_SERVE_PORT so any
  reappearance fails the audit). The `rdq-research` OneCLI IDENTITY still
  exists (setup/check_onecli.sh, run_us_quant.sh local mode) — that's
  separate from the service. After editing any unit: `daemon-reload` +
  restart, then check `/proc/<MainPID>/environ` — `systemctl
  show-environment` does NOT reflect per-unit Environment= lines.
- OneCLI has TWO injection mechanisms: vaulted secrets (host-pattern matched,
  managed by `onecli secrets`/`agents set-secrets`, what setup_onecli.sh
  assigns) and app connections (OAuth connectors, e.g. Notion). App
  connections are granted PER AGENT and the grant has no CLI or REST
  endpoint — it lives in the gateway's `agent_app_connections` table and is
  normally edited in the web UI. check_onecli.sh probes no-vault-secret
  hosts bare and reports "via app connection" on 2xx; setup_onecli.sh's
  "no vault secret for host api.notion.com" WARN is expected and harmless
  (docs/decisions.md 2026-07-08 + 2026-07-09).
- Timer-driven jobs (US-036 pattern): `Type=oneshot` service with NO `[Install]`
  section + a matching `.timer` with `WantedBy=timers.target` — enable the
  TIMER, never the service. Schedule market-relative jobs with an explicit
  timezone in the calendar spec (`OnCalendar=Mon..Fri 06:30 America/New_York`;
  sanity-check with `systemd-analyze calendar "<spec>"`). Persistent=true only
  when a missed run is harmless to catch up (data refresh: idempotent);
  trading jobs use Persistent=false so a boot mid-day doesn't fire a stale
  pre-open rebalance.
- `ops/flatten.py` (US-040) is the emergency go-to-zero script (cancel all →
  close all → poll /v2/positions empty), run as rdq-exec-paper. Exit codes:
  0 = confirmed flat, 1 = liquidations submitted but not confirmed (usually
  closed market — rerun after the open), 2 = operational failure. Its
  liquidation orders have NO Trade Ledger rows, so ops/reconcile.py will
  flag them for that date — expected; note the flatten in the Decision Log.
  Never run it (or its live test) casually: it liquidates whatever the paper
  book holds. Operator procedures live in `ops/runbook.md` — keep it current
  when halt/rotate/exposure mechanics change (tests/test_flatten.py asserts
  its required sections).
- `ops/reconcile.py` (US-037) is read-only by default and runs as
  rdq-exec-paper (Alpaca vault secrets + Notion app connection both inject
  for that identity). Exit codes: 0 = ledger matches broker history exactly
  (or every mismatch was repaired), 1 = unresolved mismatches (printed with
  order id + differing fields), 2 = the comparison itself failed
  (config/auth/HTTP). `--update` (US-019) repairs ONLY pure fill-state
  mismatches (Status / Filled Qty / Filled Avg Price all-that-differ — the
  fill-poll-timeout case) to broker truth via the same ledger_status
  vocabulary; identity mismatches, orphans, duplicates, and missing rows are
  never touched and keep exit 1. `--notify` posts a ONE-LINE Slack summary
  only when mismatches were found; `--lookback N` widens the range for the
  timer's Persistent catch-up. Any smoke test that writes a
  Trade Ledger row MUST archive it afterwards, or reconcile flags it as an
  orphan forever (archived pages are invisible to Notion queries — that is
  the sanctioned cleanup mechanism, not deletion).
- `rdq-reconcile.{service,timer}` (US-019): weekday 16:15 America/New_York —
  post-close so day orders are terminal, before the 16:30 divergence check —
  runs `ops.reconcile --update --notify --lookback 4` as rdq-exec-paper with
  NO_PROXY=slack.com; Persistent=true (idempotent repair + lookback covers
  missed days). `rdq-health.{service,timer}` (US-019): daily 09:00
  America/New_York run of ops/health.sh with NO direct Slack path — findings
  exit non-zero and reach Slack via the US-018 OnFailure notifier, so an
  unresolved health finding pages EVERY DAY until fixed (that's the point;
  don't allowlist things just to quiet it).
- `ops/health.sh` (US-042) is the box audit: rdq unit states + loopback audit +
  tailscale exposure vs the PLAN.md §1 allowlist. When adding a unit, add it to
  BOTH install_services.sh UNITS and the matching health.sh list
  (LONG_RUNNING / TIMERS / ONESHOTS) — tests/test_health.py cross-checks them.
  Gotchas baked in: oneshot units are healthy when "inactive" (only `is-failed`
  == failed is a failure), and `tailscale serve` terminates TLS on the TAILNET
  interface (100.64.0.0/10 / fd7a:115c:a1e0::), so an allowed serve port bound
  there is sanctioned, not a leak — 19899 has no allowed mapping and fails
  everywhere. Scripts calling systemctl/ss/tailscale by bare name are testable
  end-to-end with PATH-shimmed stub binaries (tests/test_health.py pattern) —
  both exit paths get real coverage without touching box state. It also audits
  env-file modes (.env, research/.env must be 600 or stricter); tests point it
  at temp files via `RDQ_ENV_FILES` (colon-separated) so they never depend on
  real repo file modes. Keep the real env files 600 — anything that recreates
  them (editors, `cp`, deploy scripts) can silently reset to 644.
- `ops/sweep.py` (US-041) derives SOTA **offline from the trace logs**: the
  FileStorage layout is `<trace>/Loop_<n>/<step>/<tag>/<pid>/<ts>.pkl`, and a
  loop's `feedback` pkl (`.decision` attr) pairs with its `runner result` pkl
  (workspace paths) via the shared `Loop_<n>` ancestor dir — reuse this if
  anything else needs run outcomes without the orchestrator DB. The sweep is
  conservative on unknowns (unreadable feedback = SOTA, uncorrelatable runner
  result = protected) and its "age" is the NEWEST lstat mtime in a tree, so
  actively-written workspaces never look old. It reads the promoted row via
  StateStore only when state.sqlite already EXISTS (StateStore(path) CREATES
  the db on init — always guard with `is_file()` from read-only callers, same
  as execution/promoted.py). It also prunes rdagent's CWD-relative repo
  droppings (US-030, `repo_prune_actions`): `log/<ts>/` trace trees (aged by
  newest mtime) and `pickle_cache/<fn>/` cache FILES (the function dirs stay)
  under `--repo-root` (default: this checkout — tests calling `main()` must
  pass a tmp `--repo-root` or they'll plan deletions in the real repo).
  Nothing else in the checkout is ever touched; both dirs are gitignored
  rdagent output, recreated on any local rdagent invocation.

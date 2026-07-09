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

## Python in ops/

- `ops/` is a real package (`ops/__init__.py`, listed in pyproject
  `packages.find` include) so tests can `from ops.foo import ...` — new
  Python entrypoints here are run as `python -m ops.<module>` (usually under
  `onecli run --agent <identity>`), not as loose scripts.
- Notion database bootstrap: `ops/bootstrap_notion.py` owns the five DB
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
- `onecli run` injects HTTP(S)_PROXY process-wide but PRESERVES a pre-set
  NO_PROXY — any service wrapping `onecli run` must `Environment=` a NO_PROXY
  exemption for hosts that may not transit the proxy (Slack: `slack.com`;
  urllib suffix-matches, covering wss-primary/files subdomains).
- From non-login shells (agents, cron) set
  `XDG_RUNTIME_DIR=/run/user/$(id -u)` before `systemctl --user` /
  `systemd-analyze --user verify`, or they can't reach the user manager.
- `rdq-research.service` duplicates the US run environment from
  `ops/run_us_quant.sh` `wire_env` (dates under all three QLIB_* prefixes,
  factor-source folders, APP_TPL, hook-class paths) — fin_quant runs spawned
  via server_ui `/upload` inherit the SERVICE environment, not the wrapper's.
  When changing wire_env, change the unit too;
  `tests/test_services.py::test_unit_dates_match_run_us_quant_defaults`
  enforces the date sync. After editing any unit: `daemon-reload` + restart,
  then check `/proc/<MainPID>/environ` — `systemctl show-environment` does
  NOT reflect per-unit Environment= lines.
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
- `ops/reconcile.py` (US-037) is READ-ONLY on both sides and runs as
  rdq-exec-paper (Alpaca vault secrets + Notion app connection both inject
  for that identity). Exit codes: 0 = ledger matches broker history exactly,
  1 = mismatches (printed with order id + differing fields), 2 = the
  comparison itself failed (config/auth/HTTP). Any smoke test that writes a
  Trade Ledger row MUST archive it afterwards, or reconcile flags it as an
  orphan forever (archived pages are invisible to Notion queries — that is
  the sanctioned cleanup mechanism, not deletion).
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
  both exit paths get real coverage without touching box state.
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
  as execution/promoted.py).

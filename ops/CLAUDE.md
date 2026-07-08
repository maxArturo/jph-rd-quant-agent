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

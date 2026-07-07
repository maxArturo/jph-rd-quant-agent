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

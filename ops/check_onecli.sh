#!/usr/bin/env bash
# check_onecli.sh — prove OneCLI credential scoping for rd-agent-q identities.
#
# For each identity, makes one BARE proxied HTTPS request per assigned service
# (this script contains no credentials; the OneCLI gateway injects them on the
# wire via `onecli run --agent <identity> -- curl ...`) and reports pass/fail
# per check. A check fails when the vault secret is missing, unassigned to the
# identity, or the service still answers 401/403 after injection.
#
# Live-host isolation proof: rdq-exec-paper MUST get 401/403 from
# https://api.alpaca.markets (live host) — it has no live secret, so a 2xx
# there would mean credential scoping is broken.
#
# Exit: 0 when every check passes; nonzero otherwise, listing the missing
# secret assignments that need fixing (vault + assign, or rerun setup_onecli.sh).
set -uo pipefail

ONECLI_URL="${ONECLI_URL:-http://127.0.0.1:10254}"
LIVE_ALPACA_URL="https://api.alpaca.markets/v2/account"
CURL_TIMEOUT=30

# identity|host pattern|probe URL|optional extra header
CHECKS=(
  "rdq-orchestrator|api.anthropic.com|https://api.anthropic.com/v1/models|anthropic-version: 2023-06-01"
  "rdq-orchestrator|api.notion.com|https://api.notion.com/v1/users/me|Notion-Version: 2022-06-28"
  "rdq-orchestrator|paper-api.alpaca.markets|https://paper-api.alpaca.markets/v2/account|"
  "rdq-research|api.anthropic.com|https://api.anthropic.com/v1/models|anthropic-version: 2023-06-01"
  "rdq-research|api.openai.com|https://api.openai.com/v1/models|"
  "rdq-research|financialmodelingprep.com|https://financialmodelingprep.com/stable/search-symbol?query=AAPL|"
  "rdq-exec-paper|paper-api.alpaca.markets|https://paper-api.alpaca.markets/v2/account|"
)

die() { echo "ERROR: $*" >&2; exit 1; }

command -v onecli >/dev/null || die "onecli not found on PATH"
command -v jq >/dev/null || die "jq not found on PATH"
curl -fsS -o /dev/null --max-time 5 "$ONECLI_URL" \
  || die "OneCLI gateway not reachable at $ONECLI_URL"

agents_json=$(onecli agents list) || die "onecli agents list failed"
secrets_json=$(onecli secrets list) || die "onecli secrets list failed"

# Proxied request via the identity; prints the HTTP status code only.
probe() { # probe <identity> <url> [extra-header]
  local identity=$1 url=$2 header=${3:-}
  local args=(-s -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT")
  [[ -n "$header" ]] && args+=(-H "$header")
  onecli run --agent "$identity" -- curl "${args[@]}" "$url" 2>/dev/null
}

fails=0
missing=()

fail() { # fail <label> <reason>
  echo "FAIL  $1  ($2)"
  fails=$((fails + 1))
}

declare -A assigned_cache
assigned_for() { # assigned_for <identity> -> newline-separated secret ids
  local identity=$1 uuid
  if [[ -z "${assigned_cache[$identity]+x}" ]]; then
    uuid=$(jq -r --arg id "$identity" \
      '.data[] | select(.identifier == $id) | .id' <<<"$agents_json")
    if [[ -z "$uuid" ]]; then
      assigned_cache[$identity]="__NO_AGENT__"
    else
      assigned_cache[$identity]=$(onecli agents secrets --id "$uuid" | jq -r '.data[]')
    fi
  fi
  echo "${assigned_cache[$identity]}"
}

for check in "${CHECKS[@]}"; do
  IFS='|' read -r identity host url header <<<"$check"
  label="$identity -> $host"

  assigned=$(assigned_for "$identity")
  if [[ "$assigned" == "__NO_AGENT__" ]]; then
    fail "$label" "identity not registered — run ops/setup_onecli.sh"
    missing+=("$identity <- $host (identity missing)")
    continue
  fi

  mapfile -t vault_ids < <(jq -r --arg h "$host" \
    '.data[] | select(.hostPattern == $h) | .id' <<<"$secrets_json")
  if [[ ${#vault_ids[@]} -eq 0 ]]; then
    fail "$label" "no vault secret for host"
    missing+=("$identity <- $host (secret not in vault)")
    continue
  fi

  unassigned=0
  for sid in "${vault_ids[@]}"; do
    grep -qx "$sid" <<<"$assigned" || unassigned=1
  done
  if [[ $unassigned -eq 1 ]]; then
    fail "$label" "vault secret not assigned to identity"
    missing+=("$identity <- $host (assignment missing)")
    continue
  fi

  code=$(probe "$identity" "$url" "$header")
  case "$code" in
    2??) echo "PASS  $label  (HTTP $code)" ;;
    401 | 403)
      fail "$label" "HTTP $code — credential not injected or invalid"
      missing+=("$identity <- $host (injected credential rejected)")
      ;;
    *) fail "$label" "HTTP $code" ;;
  esac
done

# --- Live-host isolation proof ------------------------------------------------
label="rdq-exec-paper -> api.alpaca.markets (live) isolation"
if [[ "$(assigned_for rdq-exec-paper)" == "__NO_AGENT__" ]]; then
  fail "$label" "identity not registered — run ops/setup_onecli.sh"
else
  code=$(probe rdq-exec-paper "$LIVE_ALPACA_URL")
  case "$code" in
    401 | 403) echo "PASS  $label  (HTTP $code as required)" ;;
    2??) fail "$label" "HTTP $code — LIVE CREDENTIALS REACHABLE FROM PAPER IDENTITY" ;;
    *) fail "$label" "HTTP $code — expected 401/403" ;;
  esac
fi

echo
if [[ $fails -eq 0 ]]; then
  echo "All checks passed."
  exit 0
fi
echo "$fails check(s) failed."
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing secret assignments:"
  printf '  %s\n' "${missing[@]}"
  echo "Vault the secret in the OneCLI web UI ($ONECLI_URL), then rerun ops/setup_onecli.sh and this check."
fi
exit 1

#!/usr/bin/env bash
# check_onecli.sh — prove OneCLI credential scoping for rd-agent-q identities.
#
# For each identity, makes one BARE proxied HTTPS request per assigned service
# (this script contains no credentials; the OneCLI gateway injects them on the
# wire via `onecli run --agent <identity> -- curl ...`) and reports pass/fail
# per check. A check fails when the vault secret is missing, unassigned to the
# identity, or the service still answers 401/403 after injection.
#
# Isolation proofs, four directions (US-018):
#   rdq-exec-live    -> live host  MUST authenticate (2xx)
#   rdq-exec-paper   -> live host  MUST fail (401/403)
#   rdq-exec-live    -> paper host MUST fail (401/403)
#   rdq-orchestrator -> live host  MUST fail (401/403)
# A 2xx in any MUST-fail direction means credential scoping is broken. The
# live-auth probe is SKIPPED with a warning (not failed) while the live Alpaca
# secret is not yet vaulted/assigned — vaulting it is the operator's FINAL
# go-live step, so this script stays runnable (and green) before it.
#
# Exit: 0 when every check passes; nonzero otherwise, listing the missing
# secret assignments that need fixing (vault + assign, or rerun setup_onecli.sh).
set -uo pipefail

ONECLI_URL="${ONECLI_URL:-http://127.0.0.1:10254}"
LIVE_ALPACA_HOST="api.alpaca.markets"
LIVE_ALPACA_URL="https://api.alpaca.markets/v2/account"
PAPER_ALPACA_URL="https://paper-api.alpaca.markets/v2/account"
CURL_TIMEOUT=30

# identity|host pattern|probe URL|optional extra header|optional POST body
# (Voyage has no GET endpoint, so its auth check is a minimal 1-token
# embedding POST — free-tier cost is negligible.)
CHECKS=(
  "rdq-orchestrator|api.anthropic.com|https://api.anthropic.com/v1/models|anthropic-version: 2023-06-01|"
  "rdq-orchestrator|api.notion.com|https://api.notion.com/v1/users/me|Notion-Version: 2022-06-28|"
  "rdq-orchestrator|paper-api.alpaca.markets|https://paper-api.alpaca.markets/v2/account||"
  "rdq-orchestrator|financialmodelingprep.com|https://financialmodelingprep.com/stable/search-symbol?query=AAPL||"
  "rdq-research|api.anthropic.com|https://api.anthropic.com/v1/models|anthropic-version: 2023-06-01|"
  "rdq-research|api.voyageai.com|https://api.voyageai.com/v1/embeddings|Content-Type: application/json|{\"model\":\"voyage-3.5-lite\",\"input\":\"ping\"}"
  "rdq-research|financialmodelingprep.com|https://financialmodelingprep.com/stable/search-symbol?query=AAPL||"
  "rdq-exec-paper|paper-api.alpaca.markets|https://paper-api.alpaca.markets/v2/account||"
  "rdq-exec-paper|api.notion.com|https://api.notion.com/v1/users/me|Notion-Version: 2022-06-28|"
  "rdq-exec-paper|financialmodelingprep.com|https://financialmodelingprep.com/stable/search-symbol?query=AAPL||"
)

die() { echo "ERROR: $*" >&2; exit 1; }

command -v onecli >/dev/null || die "onecli not found on PATH"
command -v jq >/dev/null || die "jq not found on PATH"
curl -fsS -o /dev/null --max-time 5 "$ONECLI_URL" \
  || die "OneCLI gateway not reachable at $ONECLI_URL"

agents_json=$(onecli agents list) || die "onecli agents list failed"
secrets_json=$(onecli secrets list) || die "onecli secrets list failed"

# Proxied request via the identity; prints the HTTP status code only.
# A non-empty <post-body> switches the request to POST with that JSON body.
probe() { # probe <identity> <url> [extra-header] [post-body]
  local identity=$1 url=$2 header=${3:-} body=${4:-}
  local args=(-s -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT")
  [[ -n "$header" ]] && args+=(-H "$header")
  [[ -n "$body" ]] && args+=(-X POST -d "$body")
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
  IFS='|' read -r identity host url header body <<<"$check"
  label="$identity -> $host"

  assigned=$(assigned_for "$identity")
  if [[ "$assigned" == "__NO_AGENT__" ]]; then
    fail "$label" "identity not registered — run ops/setup_onecli.sh"
    missing+=("$identity <- $host (identity missing)")
    continue
  fi

  # Hosts with no vault secret may still be injected via an app connection
  # (e.g. api.notion.com — docs/decisions.md 2026-07-08); probe those bare
  # and let the wire result decide instead of failing on the vault lookup.
  mapfile -t vault_ids < <(jq -r --arg h "$host" \
    '.data[] | select(.hostPattern == $h) | .id' <<<"$secrets_json")
  connector=""
  if [[ ${#vault_ids[@]} -eq 0 ]]; then
    connector=" via app connection"
  else
    unassigned=0
    for sid in "${vault_ids[@]}"; do
      grep -qx "$sid" <<<"$assigned" || unassigned=1
    done
    if [[ $unassigned -eq 1 ]]; then
      fail "$label" "vault secret not assigned to identity"
      missing+=("$identity <- $host (assignment missing)")
      continue
    fi
  fi

  code=$(probe "$identity" "$url" "$header" "$body")
  case "$code" in
    2??) echo "PASS  $label  (HTTP $code$connector)" ;;
    401 | 403)
      if [[ -n "$connector" ]]; then
        fail "$label" "HTTP $code — no vault secret and no app connection grants this identity access"
        missing+=("$identity <- $host (grant the app connection to this agent in the OneCLI web UI)")
      else
        fail "$label" "HTTP $code — credential not injected or invalid"
        missing+=("$identity <- $host (injected credential rejected)")
      fi
      ;;
    *) fail "$label" "HTTP $code" ;;
  esac
done

# --- Isolation proofs: four directions (US-018) --------------------------------
# MUST-fail directions first: identity|probe URL|host label. A 2xx anywhere
# here means credential scoping is broken (real-money blast radius).
ISOLATION_MUST_FAIL=(
  "rdq-exec-paper|$LIVE_ALPACA_URL|api.alpaca.markets (live)"
  "rdq-orchestrator|$LIVE_ALPACA_URL|api.alpaca.markets (live)"
  "rdq-exec-live|$PAPER_ALPACA_URL|paper-api.alpaca.markets"
)
for check in "${ISOLATION_MUST_FAIL[@]}"; do
  IFS='|' read -r identity url hostlabel <<<"$check"
  label="$identity -> $hostlabel isolation"
  if [[ "$(assigned_for "$identity")" == "__NO_AGENT__" ]]; then
    fail "$label" "identity not registered — run ops/setup_onecli.sh"
    continue
  fi
  code=$(probe "$identity" "$url")
  case "$code" in
    401 | 403) echo "PASS  $label  (HTTP $code as required)" ;;
    2??) fail "$label" "HTTP $code — CREDENTIALS FOR $hostlabel REACHABLE FROM $identity" ;;
    *) fail "$label" "HTTP $code — expected 401/403" ;;
  esac
done

# MUST-pass direction: rdq-exec-live authenticates to the live host. Skipped
# with a warning until the live Alpaca secret is vaulted AND assigned to
# rdq-exec-live (the operator's final go-live step).
label="rdq-exec-live -> api.alpaca.markets (live) auth"
live_assigned=$(assigned_for rdq-exec-live)
if [[ "$live_assigned" == "__NO_AGENT__" ]]; then
  fail "$label" "identity not registered — run ops/setup_onecli.sh"
else
  mapfile -t live_secret_ids < <(jq -r --arg h "$LIVE_ALPACA_HOST" \
    '.data[] | select(.hostPattern == $h) | .id' <<<"$secrets_json")
  live_ready=1
  if [[ ${#live_secret_ids[@]} -eq 0 ]]; then
    live_ready=0
  else
    for sid in "${live_secret_ids[@]}"; do
      grep -qx "$sid" <<<"$live_assigned" || live_ready=0
    done
  fi
  if [[ $live_ready -eq 0 ]]; then
    echo "SKIP  $label  (WARN: live Alpaca secret not vaulted/assigned yet —" \
      "expected before the operator's go-live step; vault it, rerun" \
      "ops/setup_onecli.sh, then this check)"
  else
    code=$(probe rdq-exec-live "$LIVE_ALPACA_URL")
    case "$code" in
      2??) echo "PASS  $label  (HTTP $code)" ;;
      401 | 403)
        fail "$label" "HTTP $code — live credential not injected or invalid"
        missing+=("rdq-exec-live <- $LIVE_ALPACA_HOST (injected credential rejected)")
        ;;
      *) fail "$label" "HTTP $code" ;;
    esac
  fi
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

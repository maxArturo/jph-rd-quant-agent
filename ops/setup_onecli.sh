#!/usr/bin/env bash
# setup_onecli.sh — register rd-agent-q identities with the box-wide OneCLI
# gateway (management API at http://127.0.0.1:10254) and assign vaulted
# secrets per the PLAN.md identity table. Idempotent: safe to rerun.
#
# Identities created:
#   rdq-orchestrator  Anthropic + Notion + Alpaca paper + FMP  (Slack bot / Claude layer;
#                     FMP backs on-demand universe backfill, orchestrator/universe.py)
#   rdq-research      Anthropic + Voyage embeddings + FMP (RD-Agent + data pipeline)
#   rdq-exec-paper    Alpaca paper + Notion + FMP        (nightly rebalancer + Trade Ledger + store refresh)
#
# Deliberately NO rdq-exec-live: live trading is out of scope for this repo.
# As defense in depth, this script refuses to assign any secret whose host
# pattern is the live Alpaca host (api.alpaca.markets) to any identity.
#
# Gotcha this script handles: new OneCLI agents start in 'selective' secret
# mode with ZERO secrets assigned (every proxied call 401s). We therefore
# always (re)compute and set the assignment list, so a rerun also repairs
# drifted or missing assignments. Secrets expected by the PLAN table but not
# yet present in the vault (e.g. Notion, embeddings key) are warned about,
# never fabricated — vault them via the OneCLI web UI, then rerun.
set -euo pipefail

ONECLI_URL="${ONECLI_URL:-http://127.0.0.1:10254}"
LIVE_ALPACA_HOST="api.alpaca.markets"

# Ordered so output is stable; bash assoc arrays iterate unordered.
IDENTITIES=(rdq-orchestrator rdq-research rdq-exec-paper)
declare -A IDENTITY_NAMES=(
  [rdq-orchestrator]="RDQ Orchestrator"
  [rdq-research]="RDQ Research"
  [rdq-exec-paper]="RDQ Exec (paper)"
)
# identity -> space-separated host patterns whose vault secrets it gets
declare -A IDENTITY_HOSTS=(
  [rdq-orchestrator]="api.anthropic.com api.notion.com paper-api.alpaca.markets financialmodelingprep.com"
  [rdq-research]="api.anthropic.com api.voyageai.com financialmodelingprep.com"
  [rdq-exec-paper]="paper-api.alpaca.markets api.notion.com financialmodelingprep.com"
)

die() { echo "ERROR: $*" >&2; exit 1; }

command -v onecli >/dev/null || die "onecli not found on PATH"
command -v jq >/dev/null || die "jq not found on PATH"
curl -fsS -o /dev/null --max-time 5 "$ONECLI_URL" \
  || die "OneCLI gateway not reachable at $ONECLI_URL"

agents_json=$(onecli agents list)

# --- 1. Create missing identities (idempotent) -------------------------------
for identity in "${IDENTITIES[@]}"; do
  if jq -e --arg id "$identity" '.data[] | select(.identifier == $id)' \
      <<<"$agents_json" >/dev/null; then
    echo "exists:  $identity"
  else
    onecli agents create --name "${IDENTITY_NAMES[$identity]}" \
      --identifier "$identity" >/dev/null
    echo "created: $identity"
  fi
done

# Refresh so newly created agents have known UUIDs.
agents_json=$(onecli agents list)
secrets_json=$(onecli secrets list)

# --- 2. Assign vault secrets by host pattern ---------------------------------
missing_hosts=()
for identity in "${IDENTITIES[@]}"; do
  uuid=$(jq -r --arg id "$identity" \
    '.data[] | select(.identifier == $id) | .id' <<<"$agents_json")
  [[ -n "$uuid" ]] || die "agent $identity not found after create"

  secret_ids=()
  for host in ${IDENTITY_HOSTS[$identity]}; do
    [[ "$host" == "$LIVE_ALPACA_HOST" ]] \
      && die "refusing to assign live Alpaca host to $identity"
    mapfile -t ids < <(jq -r --arg h "$host" \
      '.data[] | select(.hostPattern == $h) | .id' <<<"$secrets_json")
    if [[ ${#ids[@]} -eq 0 ]]; then
      echo "WARN: no vault secret for host '$host' (wanted by $identity) — vault it, then rerun" >&2
      missing_hosts+=("$identity <- $host")
    else
      secret_ids+=("${ids[@]}")
    fi
  done

  if [[ ${#secret_ids[@]} -gt 0 ]]; then
    joined=$(IFS=,; echo "${secret_ids[*]}")
    onecli agents set-secrets --id "$uuid" --secret-ids "$joined" >/dev/null
    echo "secrets: $identity <- ${#secret_ids[@]} assigned ($joined)"
  else
    echo "secrets: $identity <- none available to assign" >&2
  fi
done

if [[ ${#missing_hosts[@]} -gt 0 ]]; then
  echo
  echo "Setup finished with ${#missing_hosts[@]} host(s) lacking vault secrets:"
  printf '  %s\n' "${missing_hosts[@]}"
  echo "Identities are registered; add the missing secrets in the OneCLI web UI ($ONECLI_URL) and rerun."
fi
echo "Done. Verify with ops/check_onecli.sh"

#!/usr/bin/env bash
# probe_market_series.sh — FMP market-series availability probe (PRD US-064).
#
# Verifies, over the store window (2025-01-02 -> yesterday America/New_York),
# that every market series the data-substrate expansion ingests is served by
# the current FMP plan:
#   - Commodity/index EOD via /stable/historical-price-eod/light:
#     BZUSD (Brent), CLUSD (WTI), HOUSD (heating oil), NGUSD (natgas),
#     RBUSD (gasoline), GCUSD (gold), DXUSD (US dollar index)
#   - Treasury curve via /stable/treasury-rates, chunked <=90 calendar days
#     per request (the endpoint silently truncates wider windows).
#
# Every request rides the OneCLI proxy (`onecli run --agent rdq-research --
# curl ...`; the gateway injects the apikey — no credentials here). Non-200
# or empty responses print FAILURE and the script exits nonzero; nothing is
# silently skipped. The treasury cap is asserted, not assumed: a single
# full-window request MUST return fewer rows than the chunked fetch, proving
# chunking is required and working.
#
# Coverage is reported against NYSE trading days from the Qlib store calendar
# (commodities can exceed 100%: they trade some NYSE holidays).
#
# Probe outcome recorded in docs/decisions.md (2026-08-24 US-064 entry).
#
# Env overrides (tests): RDQ_PROBE_AGENT, RDQ_PROBE_START, RDQ_PROBE_END,
# RDQ_PROBE_CALENDAR.
set -uo pipefail

AGENT="${RDQ_PROBE_AGENT:-rdq-research}"
BASE_URL="https://financialmodelingprep.com/stable"
START="${RDQ_PROBE_START:-2025-01-02}"
END="${RDQ_PROBE_END:-$(TZ=America/New_York date -d yesterday +%F)}"
CALENDAR="${RDQ_PROBE_CALENDAR:-$HOME/.qlib/qlib_data/us_data/calendars/day.txt}"
TREASURY_CHUNK_DAYS=90
CURL_TIMEOUT=60

SYMBOLS=(BZUSD CLUSD HOUSD NGUSD RBUSD GCUSD DXUSD)

die() { echo "ERROR: $*" >&2; exit 1; }

command -v onecli >/dev/null || die "onecli not found on PATH"
command -v jq >/dev/null || die "jq not found on PATH"
[[ -f "$CALENDAR" ]] || die "store calendar not found at $CALENDAR"

fails=0
fail() { echo "FAILURE  $*"; fails=$((fails + 1)); }

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

fetch() { # fetch <url> <body-out-file> -> prints HTTP status code
  onecli run --agent "$AGENT" -- curl -s -o "$2" -w '%{http_code}' \
    --max-time "$CURL_TIMEOUT" "$1" 2>/dev/null
}

trading_days_in() { # trading_days_in <start> <end> -> count from store calendar
  awk -v s="$1" -v e="$2" '$0 >= s && $0 <= e' "$CALENDAR" | wc -l
}

trading_days=$(trading_days_in "$START" "$END")
[[ "$trading_days" -gt 0 ]] || die "no trading days in [$START, $END] per $CALENDAR"

coverage_pct() { # coverage_pct <rows>
  awk -v r="$1" -v t="$trading_days" 'BEGIN { printf "%.1f", 100 * r / t }'
}

echo "FMP market-series probe: $START -> $END ($trading_days NYSE trading days per store calendar)"
echo

# --- commodity / index EOD -------------------------------------------------
for sym in "${SYMBOLS[@]}"; do
  body="$WORKDIR/$sym.json"
  url="$BASE_URL/historical-price-eod/light?symbol=$sym&from=$START&to=$END"
  code=$(fetch "$url" "$body")
  if [[ "$code" != 2?? ]]; then
    fail "$sym  HTTP $code from /stable/historical-price-eod/light"
    continue
  fi
  rows=$(jq 'if type == "array" then length else -1 end' "$body")
  if [[ "$rows" -lt 0 ]]; then
    fail "$sym  non-array response: $(head -c 200 "$body")"
    continue
  fi
  if [[ "$rows" -eq 0 ]]; then
    fail "$sym  empty response (no rows in window)"
    continue
  fi
  first=$(jq -r 'map(.date) | min' "$body")
  last=$(jq -r 'map(.date) | max' "$body")
  echo "PASS  $sym  rows=$rows first=$first last=$last coverage=$(coverage_pct "$rows")% of $trading_days NYSE trading days"
done

# --- treasury curve, chunked <=90 calendar days per request -----------------
echo
chunk_start="$START"
requests=0
dates_file="$WORKDIR/treasury_dates.txt"
: > "$dates_file"
treasury_ok=1
year10_ok=1
while [[ ! "$chunk_start" > "$END" ]]; do
  chunk_end=$(date -d "$chunk_start + $((TREASURY_CHUNK_DAYS - 1)) days" +%F)
  [[ "$chunk_end" > "$END" ]] && chunk_end="$END"
  body="$WORKDIR/treasury_$chunk_start.json"
  url="$BASE_URL/treasury-rates?from=$chunk_start&to=$chunk_end"
  code=$(fetch "$url" "$body")
  requests=$((requests + 1))
  if [[ "$code" != 2?? ]]; then
    fail "treasury-rates  HTTP $code for chunk $chunk_start -> $chunk_end"
    treasury_ok=0
  else
    rows=$(jq 'if type == "array" then length else -1 end' "$body")
    if [[ "$rows" -le 0 ]]; then
      # A short final chunk can legitimately hold zero trading days.
      if [[ "$rows" -eq 0 && "$(trading_days_in "$chunk_start" "$chunk_end")" -eq 0 ]]; then
        :
      else
        fail "treasury-rates  empty/non-array response for chunk $chunk_start -> $chunk_end"
        treasury_ok=0
      fi
    else
      jq -r '.[].date' "$body" >> "$dates_file"
      jq -e '.[0] | has("year10")' "$body" >/dev/null || year10_ok=0
    fi
  fi
  chunk_start=$(date -d "$chunk_end + 1 day" +%F)
done

if [[ "$year10_ok" -eq 0 ]]; then
  fail "treasury-rates  rows missing 'year10' (needed for \$mkt_y10)"
  treasury_ok=0
fi

if [[ "$treasury_ok" -eq 1 ]]; then
  t_rows=$(sort -u "$dates_file" | wc -l)
  t_first=$(sort "$dates_file" | head -n 1)
  t_last=$(sort "$dates_file" | tail -n 1)
  echo "PASS  treasury-rates  rows=$t_rows first=$t_first last=$t_last coverage=$(coverage_pct "$t_rows")% of $trading_days NYSE trading days ($requests chunked requests, <=$TREASURY_CHUNK_DAYS days each)"

  # Cap assertion: an unbounded window silently returns only the last ~3
  # months, so a single full-window request MUST come back smaller than the
  # chunked fetch — otherwise chunking is unnecessary or broken.
  window_days=$((($(date -d "$END" +%s) - $(date -d "$START" +%s)) / 86400 + 1))
  if [[ "$window_days" -le "$TREASURY_CHUNK_DAYS" ]]; then
    fail "treasury cap  window too short ($window_days days) to demonstrate the ~$TREASURY_CHUNK_DAYS-day cap"
  else
    body="$WORKDIR/treasury_full.json"
    code=$(fetch "$BASE_URL/treasury-rates?from=$START&to=$END" "$body")
    if [[ "$code" != 2?? ]]; then
      fail "treasury cap  HTTP $code for the single full-window request"
    else
      single_rows=$(jq 'if type == "array" then length else 0 end' "$body")
      if [[ "$single_rows" -lt "$t_rows" ]]; then
        echo "PASS  treasury cap  single full-window request returned $single_rows rows < $t_rows chunked — endpoint truncates at ~$TREASURY_CHUNK_DAYS calendar days; chunking required and working"
      else
        fail "treasury cap  single full-window request returned $single_rows rows vs $t_rows chunked — cap not demonstrated"
      fi
    fi
  fi
fi

echo
if [[ "$fails" -eq 0 ]]; then
  echo "All series probed clean."
  exit 0
fi
echo "$fails FAILURE(s) — unavailable series need a documented substitute or an explicit drop (docs/decisions.md US-064)."
exit 1

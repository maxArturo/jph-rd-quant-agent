#!/usr/bin/env bash
# probe_news.sh — FMP stock-news probe + backfill budget (PRD US-071).
#
# Re-verifies, as a scripted check rather than a one-off terminal session,
# the 2026-08-24 ad-hoc findings the news backfill (US-072/073) depends on:
#   - /stable/news/stock?symbols=<T>&from=&to=&limit=250&page=N serves
#     per-ticker history back to the store start (2025-01-02) for 5 sample
#     tickers spanning mega-cap / mid / thin coverage
#   - publishedDate carries SECOND resolution (the 16:00 ET PIT cutoff
#     needs it)
#   - pagination works at limit=250 (a non-empty page 1 continues page 0
#     with no ordering overlap between pages)
#   - timestamps are US/Eastern: the newest article on the live wire
#     (/stable/news/stock-latest) must sit within minutes of now-ET; a
#     newest timestamp minutes in the FUTURE is the UTC smoking gun (UTC
#     runs 4-5h ahead of ET) and fails loud
#
# It also MEASURES the backfill budget: full-window pagination walk per
# sample ticker -> articles/ticker/day distribution, requests per ticker,
# and an extrapolated request count + runtime for the full universe
# backfill at the chosen throttle. Outcome recorded in docs/decisions.md
# (2026-08-24 US-071 entry).
#
# TERMINATION RULE (live finding, 2026-08-24): the endpoint has page-size
# JITTER — a mid-stream page can return fewer than `limit` rows (e.g. 249)
# with more history behind it. Only an EMPTY page marks end-of-history;
# the walk here (and the US-072 backfill) must never stop on a short page,
# or history silently truncates. Short non-final pages are counted and
# reported as a NOTE.
#
# Every request rides the OneCLI proxy (`onecli run --agent rdq-research --
# curl ...`; the gateway injects the apikey — no credentials here). Non-200
# or non-array responses print FAILURE and the script exits nonzero;
# nothing is silently skipped. Depth is asserted strictly only for mega-cap
# samples (a mega name has news every day, so a late oldest article = plan
# cap; a thin name's late oldest is genuine sparsity and is only reported).
#
# Env overrides (tests): RDQ_PROBE_AGENT, RDQ_PROBE_START, RDQ_PROBE_END,
# RDQ_PROBE_NOW_ET, RDQ_PROBE_UNIVERSE, RDQ_PROBE_TICKERS ("SYM:role ...",
# role mega|mid|thin), RDQ_PROBE_LIMIT, RDQ_PROBE_MAX_PAGES,
# RDQ_PROBE_THROTTLE, RDQ_PROBE_DEPTH_DAYS, RDQ_PROBE_TZ_MAX_STALE_MIN.
set -uo pipefail

AGENT="${RDQ_PROBE_AGENT:-rdq-research}"
BASE_URL="https://financialmodelingprep.com/stable"
START="${RDQ_PROBE_START:-2025-01-02}"
END="${RDQ_PROBE_END:-$(TZ=America/New_York date -d yesterday +%F)}"
NOW_ET="${RDQ_PROBE_NOW_ET:-$(TZ=America/New_York date '+%F %T')}"
UNIVERSE="${RDQ_PROBE_UNIVERSE:-$HOME/.qlib/qlib_data/us_data/instruments/us_liquid.txt}"
TICKERS="${RDQ_PROBE_TICKERS:-AAPL:mega NVDA:mega EXPE:mid DECK:mid ASAN:thin}"
LIMIT="${RDQ_PROBE_LIMIT:-250}"
MAX_PAGES="${RDQ_PROBE_MAX_PAGES:-100}"
THROTTLE="${RDQ_PROBE_THROTTLE:-3}" # req/s the backfill (US-072/073) will use
DEPTH_DAYS="${RDQ_PROBE_DEPTH_DAYS:-7}" # mega oldest-article slack vs START
TZ_MAX_STALE_MIN="${RDQ_PROBE_TZ_MAX_STALE_MIN:-180}"
CURL_TIMEOUT=60
TS_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'

die() { echo "ERROR: $*" >&2; exit 1; }

command -v onecli >/dev/null || die "onecli not found on PATH"
command -v jq >/dev/null || die "jq not found on PATH"
[[ -f "$UNIVERSE" ]] || die "universe file not found at $UNIVERSE"

fails=0
fail() { echo "FAILURE  $*"; fails=$((fails + 1)); }

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

fetch() { # fetch <url> <body-out-file> -> prints HTTP status code
  onecli run --agent "$AGENT" -- curl -s -o "$2" -w '%{http_code}' \
    --max-time "$CURL_TIMEOUT" "$1" 2>/dev/null
}

read -r -a SAMPLES <<< "$TICKERS"
n_samples=${#SAMPLES[@]}
[[ "$n_samples" -gt 0 ]] || die "no sample tickers configured"

n_universe=$(awk '{print $1}' "$UNIVERSE" | sort -u | wc -l)
window_days=$((($(date -d "$END" +%s) - $(date -d "$START" +%s)) / 86400 + 1))
[[ "$window_days" -gt 0 ]] || die "empty window [$START, $END]"

echo "FMP stock-news probe: $START -> $END ($window_days calendar days), samples: $TICKERS"
echo

# --- per-ticker full-window pagination walk ---------------------------------
total_requests=0
total_jitter=0
pagination_proven=0
for entry in "${SAMPLES[@]}"; do
  sym=${entry%%:*}
  role=${entry##*:}
  ts_file="$WORKDIR/$sym.ts"
  : > "$ts_file"
  page=0
  pages=0
  ticker_fail=0
  prev_short=0
  page0_min=""
  page1_max=""
  while [[ "$page" -lt "$MAX_PAGES" ]]; do
    body="$WORKDIR/$sym.p$page.json"
    url="$BASE_URL/news/stock?symbols=$sym&from=$START&to=$END&limit=$LIMIT&page=$page"
    code=$(fetch "$url" "$body")
    pages=$((pages + 1))
    total_requests=$((total_requests + 1))
    if [[ "$code" != 2?? ]]; then
      fail "$sym  HTTP $code from /stable/news/stock (page=$page)"
      ticker_fail=1
      break
    fi
    rows=$(jq 'if type == "array" then length else -1 end' "$body")
    if [[ "$rows" -lt 0 ]]; then
      fail "$sym  non-array response (page=$page): $(head -c 200 "$body")"
      ticker_fail=1
      break
    fi
    # Only an EMPTY page is end-of-history; a short page can be jitter.
    [[ "$rows" -eq 0 ]] && break
    if [[ "$rows" -gt "$LIMIT" ]]; then
      fail "$sym  page $page returned $rows rows > limit=$LIMIT — limit not respected"
      ticker_fail=1
      break
    fi
    [[ "$prev_short" -eq 1 ]] && total_jitter=$((total_jitter + 1))
    prev_short=$((rows < LIMIT ? 1 : 0))
    jq -r '.[].publishedDate' "$body" >> "$ts_file"
    if [[ "$page" -eq 0 ]]; then
      page0_min=$(jq -r 'map(.publishedDate) | min // empty' "$body")
    elif [[ "$page" -eq 1 ]]; then
      page1_max=$(jq -r 'map(.publishedDate) | max // empty' "$body")
    fi
    page=$((page + 1))
    if [[ "$page" -ge "$MAX_PAGES" ]]; then
      fail "$sym  still returning articles after $MAX_PAGES pages — raise RDQ_PROBE_MAX_PAGES"
      ticker_fail=1
    fi
  done
  [[ "$ticker_fail" -eq 1 ]] && continue

  total=$(wc -l < "$ts_file")
  if [[ "$total" -eq 0 ]]; then
    fail "$sym ($role)  zero articles over $START -> $END — endpoint serves nothing for this ticker"
    continue
  fi

  bad_ts=$(grep -cvE "$TS_RE" "$ts_file" || true)
  if [[ "$bad_ts" -gt 0 ]]; then
    fail "$sym  $bad_ts publishedDate value(s) not at second resolution, e.g. '$(grep -vE "$TS_RE" "$ts_file" | head -n 1)'"
    continue
  fi

  oldest=$(cut -c1-10 "$ts_file" | sort | head -n 1)
  newest=$(cut -c1-10 "$ts_file" | sort | tail -n 1)
  depth_limit=$(date -d "$START + $DEPTH_DAYS days" +%F)
  if [[ "$role" == "mega" && "$oldest" > "$depth_limit" ]]; then
    fail "$sym (mega)  history looks plan-capped: oldest article $oldest, expected <= $depth_limit (a mega-cap has news every day back to $START)"
    continue
  fi

  # Pagination proof: a non-empty page 1 whose newest timestamp does not
  # overtake page 0's oldest (rows are served newest-first; page 0 may be
  # one short of `limit` — jitter — and still have history behind it).
  if [[ -n "$page1_max" ]]; then
    if [[ "$page1_max" > "$page0_min" ]]; then
      fail "$sym  pagination ordering overlap: page 1 newest '$page1_max' > page 0 oldest '$page0_min'"
      continue
    fi
    pagination_proven=1
  fi

  # Articles/day distribution: mean over calendar days, percentiles over
  # days that had at least one article.
  read -r active_days p50 p90 dmax <<< "$(cut -c1-10 "$ts_file" | sort | uniq -c \
    | awk '{print $1}' | sort -n \
    | awk '{ a[NR] = $1 } END {
        p50 = a[int((NR + 1) / 2)]
        i90 = int(NR * 0.9); if (i90 < 1) i90 = 1
        printf "%d %d %d %d\n", NR, p50, a[i90], a[NR]
      }')"
  mean=$(awk -v t="$total" -v d="$window_days" 'BEGIN { printf "%.1f", t / d }')
  echo "PASS  $sym ($role)  articles=$total requests=$pages oldest=$oldest newest=$newest avg/day=$mean active_days=$active_days/$window_days p50/day=$p50 p90/day=$p90 max/day=$dmax"
done

echo
if [[ "$total_jitter" -gt 0 ]]; then
  echo "NOTE  $total_jitter short non-final page(s) observed (page-size jitter) — the backfill must treat only an EMPTY page as end-of-history, never a short page"
fi
if [[ "$pagination_proven" -eq 1 ]]; then
  echo "PASS  pagination verified at limit=$LIMIT (ordered non-empty page 1 continues page 0)"
else
  fail "pagination not demonstrated: no sample ticker needed a second page at limit=$LIMIT — add a denser sample"
fi

# --- wire timezone check -----------------------------------------------------
echo
body="$WORKDIR/latest.json"
code=$(fetch "$BASE_URL/news/stock-latest?page=0&limit=20" "$body")
if [[ "$code" != 2?? ]]; then
  fail "wire  HTTP $code from /stable/news/stock-latest"
else
  rows=$(jq 'if type == "array" then length else -1 end' "$body")
  if [[ "$rows" -le 0 ]]; then
    fail "wire  empty/non-array response from /stable/news/stock-latest"
  else
    wire_newest=$(jq -r 'map(.publishedDate) | max' "$body")
    if ! grep -qE "$TS_RE" <<< "$wire_newest"; then
      fail "wire  newest publishedDate '$wire_newest' not at second resolution"
    else
      gap_min=$((($(TZ=America/New_York date -d "$NOW_ET" +%s) - $(TZ=America/New_York date -d "$wire_newest" +%s)) / 60))
      if [[ "$gap_min" -lt -5 ]]; then
        fail "wire  newest article '$wire_newest' is $((-gap_min)) min in the FUTURE vs now-ET ($NOW_ET) — publishedDate looks like UTC, the 16:00 ET cutoff would be wrong"
      elif [[ "$gap_min" -gt "$TZ_MAX_STALE_MIN" ]]; then
        fail "wire  newest article '$wire_newest' is $gap_min min stale vs now-ET ($NOW_ET) — wire dead or timestamps not ET"
      else
        echo "PASS  wire  newest='$wire_newest' now-ET='$NOW_ET' gap=${gap_min}m — publishedDate is US/Eastern"
      fi
    fi
  fi
fi

# --- backfill budget ---------------------------------------------------------
echo
today=$(TZ=America/New_York date +%F)
backfill_days=$((($(date -d "$today" +%s) - $(date -d "$START" +%s)) / 86400 + 1))
read -r mean_pages est_requests est_minutes <<< "$(awk \
  -v r="$total_requests" -v n="$n_samples" -v u="$n_universe" -v t="$THROTTLE" \
  'BEGIN {
    m = r / n
    e = int((r * u + n - 1) / n)
    printf "%.1f %d %.0f\n", m, e, e / t / 60
  }')"
echo "BUDGET  sampled $total_requests page-requests over $n_samples tickers (mean $mean_pages requests/ticker, incl. the empty end-of-history page each)"
echo "BUDGET  full backfill ($n_universe tickers, $START -> $today, $backfill_days days) ~= $est_requests requests; at $THROTTLE req/s ~= $est_minutes min (sample is mega-skewed — treat as an upper bound)"

echo
if [[ "$fails" -eq 0 ]]; then
  echo "News endpoint probed clean."
  exit 0
fi
echo "$fails FAILURE(s) — resolve before committing to the 589-ticker backfill (docs/decisions.md US-071)."
exit 1

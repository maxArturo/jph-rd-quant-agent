#!/usr/bin/env bash
# expose_traces.sh — expose the rdagent trace viewer over the tailnet (US-042).
#
# Adds the PLAN.md §1 port-table mapping (tailnet-only, NEVER funnel):
#   tailscale serve --bg --https=19900 http://127.0.0.1:19900
#
# The trace viewer (`rdagent ui` on 127.0.0.1:19900) does not need to be
# running yet — tailscale proxies to it whenever it comes up. Idempotent:
# rerunning with the mapping present is a no-op.
#
# Remove with: tailscale serve --https=19900 off
# Audit with:  ops/health.sh (or tailscale serve status)
set -euo pipefail

PORT=19900
TARGET="http://127.0.0.1:${PORT}"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v tailscale >/dev/null || die "tailscale not found on PATH"
command -v ss >/dev/null || die "ss not found on PATH"

if ! ss -tlnH "( sport = :${PORT} )" 2>/dev/null | grep -q .; then
  echo "note: nothing is listening on 127.0.0.1:${PORT} yet — start the trace" \
    "viewer with \`rdagent ui --port ${PORT}\`; the mapping works once it is up."
fi

if tailscale serve status 2>/dev/null | grep -q "proxy ${TARGET}$"; then
  echo "already mapped: https=${PORT} -> ${TARGET} (nothing to do)"
else
  tailscale serve --bg --https="${PORT}" "${TARGET}"
fi

# Verify: mapping present, tailnet-only, and no funnel anywhere.
status=$(tailscale serve status 2>&1) || die "tailscale serve status failed: $status"
grep -q "proxy ${TARGET}$" <<<"$status" || die "mapping missing after serve command"
grep -E ":${PORT} \(" <<<"$status" | grep -q "(tailnet only)" \
  || die "mapping for :${PORT} is not tailnet-only — remove it: tailscale serve --https=${PORT} off"
if grep -qi "funnel" <<<"$status"; then
  die "funnel output present — this box must never funnel (tailscale funnel reset)"
fi

echo "OK: trace viewer served tailnet-only at https=${PORT} -> ${TARGET}"
echo "remove with: tailscale serve --https=${PORT} off"

#!/usr/bin/env bash
# Burst-compute runbook for broad research runs (runbook §7): resize the
# droplet UP to dedicated CPU before a big run, back DOWN after. DO bills
# per-second and a CPU/RAM-only resize is reversible, so a 24h run on c-16
# (16 dedicated vCPU / 32G, $0.50/hr) costs ~$12 on top of the $48/mo base.
#
# RUN THIS FROM YOUR LAPTOP (or any doctl-authenticated host with ssh access
# to the box over tailscale) — NEVER from the droplet itself: the resize
# powers the droplet off, which would kill the script mid-flight. The script
# refuses to run on the target droplet.
#
# THE ONE HARD RULE: never pass --resize-disk to the resize action. A disk
# resize is PERMANENT and would block ever resizing back down to the cheap
# base size. This script only ever does CPU/RAM-only resizes.
#
# Usage:
#   ops/resize_research.sh up [size]   # default c-16; shutdown -> resize ->
#                                      #   power-on -> re-derive caps (remote
#                                      #   ops/research_caps.sh) -> restart
#   ops/resize_research.sh down        # back to the base size (s-4vcpu-8gb)
#   ops/resize_research.sh status      # size, state, $/hr, active-run check
#   --force                            # proceed even if a run looks active
#
# Config (env overrides):
#   RDQ_DROPLET_ID   default 573294655            (nanoclaw-prod, nyc3)
#   RDQ_SSH_TARGET   default nanoclaw@nanoclaw-prod.tail05c9bf.ts.net
#   RDQ_BURST_SIZE   default c-16                 (16 vCPU / 32G, $0.50/hr)
#   RDQ_BASE_SIZE    default s-4vcpu-8gb          (4 vCPU / 8G, $48/mo)
#   RDQ_POLL_SECS    default 5                    (state-poll interval)
#
# Requires: doctl (authenticated: `doctl auth init`), jq, ssh.
set -euo pipefail

DROPLET_ID="${RDQ_DROPLET_ID:-573294655}"
SSH_TARGET="${RDQ_SSH_TARGET:-nanoclaw@nanoclaw-prod.tail05c9bf.ts.net}"
BURST_SIZE="${RDQ_BURST_SIZE:-c-16}"
BASE_SIZE="${RDQ_BASE_SIZE:-s-4vcpu-8gb}"
POLL_SECS="${RDQ_POLL_SECS:-5}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)
# The remote pgrep pattern self-matches the ssh-spawned shell unless the
# bracket breaks the literal string.
ACTIVE_RUN_CMD='docker ps -q --filter ancestor=local_qlib:latest 2>/dev/null; pgrep -f "fin[_]quant" 2>/dev/null'

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "ERROR: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required (see script header)"; }

# --- argument parsing --------------------------------------------------------
CMD="" TARGET_SIZE="" FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --help|-h) usage; exit 0 ;;
    up|down|status) [[ -n "$CMD" ]] && die "one command only"; CMD="$arg" ;;
    *)
      [[ "$CMD" == "up" && -z "$TARGET_SIZE" ]] || die "unexpected argument: $arg"
      TARGET_SIZE="$arg"
      ;;
  esac
done
[[ -n "$CMD" ]] || { usage; exit 2; }

# --- guards ------------------------------------------------------------------
# Refuse to run on the droplet being resized: the shutdown kills the script.
self_id="$(curl -s -m 2 http://169.254.169.254/metadata/v1/id 2>/dev/null || true)"
if [[ "$self_id" == "$DROPLET_ID" ]]; then
  die "refusing to run on the target droplet itself — run this from your laptop"
fi

need doctl; need jq; need ssh
doctl account get >/dev/null 2>&1 || die "doctl is not authenticated (run: doctl auth init)"

# --- doctl helpers -----------------------------------------------------------
# `doctl ... get -o json` wraps single results in an array in some versions.
jq_first() { jq -r "if type==\"array\" then .[0] else . end | $1"; }

droplet_field() { doctl compute droplet get "$DROPLET_ID" -o json | jq_first "$1"; }

size_row() { # size_row <slug> <jq-expr>
  doctl compute size list -o json | jq -r --arg slug "$1" \
    "[.[] | select(.slug == \$slug)] | if length == 0 then \"\" else .[0] | $2 end"
}

wait_for_status() { # wait_for_status <wanted> <tries>
  local wanted="$1" tries="$2" st
  for (( i = 0; i < tries; i++ )); do
    st="$(droplet_field .status)"
    [[ "$st" == "$wanted" ]] && return 0
    sleep "$POLL_SECS"
  done
  return 1
}

active_run() { # prints container ids / pids if a research run looks live
  # shellcheck disable=SC2029  # client-side expansion of the constant is intended
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$ACTIVE_RUN_CMD" 2>/dev/null || true
}

# --- commands ----------------------------------------------------------------
show_status() {
  local slug status
  slug="$(droplet_field .size_slug)"
  status="$(droplet_field .status)"
  echo "droplet:  $DROPLET_ID ($SSH_TARGET)"
  echo "size:     $slug ($(size_row "$slug" '"\(.vcpus) vCPU / \(.memory / 1024 | floor)G, $\(.price_hourly)/hr, $\(.price_monthly)/mo"'))"
  echo "status:   $status"
  if [[ "$status" == "active" ]]; then
    local run; run="$(active_run)"
    if [[ -n "$run" ]]; then echo "research: RUN ACTIVE — do not resize"; else echo "research: idle"; fi
  fi
}

resize_to() {
  local target="$1" cur cur_disk target_disk price
  cur="$(droplet_field .size_slug)"
  cur_disk="$(droplet_field .disk)"
  if [[ "$cur" == "$target" ]]; then
    echo "already at $target — nothing to do"
    return 0
  fi

  target_disk="$(size_row "$target" .disk)"
  [[ -n "$target_disk" ]] || die "unknown size slug: $target (see: doctl compute size list)"
  # DO only allows resizing to plans whose base disk covers the current disk,
  # and we NEVER grow the disk (permanent — see header), so this must hold in
  # both directions.
  if (( target_disk < cur_disk )); then
    die "$target has a ${target_disk}G disk < current ${cur_disk}G — DO cannot resize to it (this is why the disk must never be grown)"
  fi
  price="$(size_row "$target" '"$\(.price_hourly)/hr ($\(.price_monthly)/mo)"')"
  echo "resizing $cur -> $target ($price), CPU/RAM only"

  if [[ "$(droplet_field .status)" == "active" ]]; then
    local run; run="$(active_run)"
    if [[ -n "$run" ]] && (( ! FORCE )); then
      die "a research run looks active on the box (local_qlib container or fin_quant process) — the resize would kill it. Wait, or re-run with --force"
    fi
    echo "shutting down..."
    doctl compute droplet-action shutdown "$DROPLET_ID" --wait >/dev/null || true
    if ! wait_for_status off 24; then
      echo "graceful shutdown did not land; forcing power-off..."
      doctl compute droplet-action power-off "$DROPLET_ID" --wait >/dev/null
      wait_for_status off 24 || die "droplet did not reach 'off'"
    fi
  fi

  # CPU/RAM-only resize: intentionally NO --resize-disk (permanent, would
  # block resizing back down — see header).
  echo "resizing..."
  doctl compute droplet-action resize "$DROPLET_ID" --size "$target" --wait >/dev/null

  echo "powering on..."
  doctl compute droplet-action power-on "$DROPLET_ID" --wait >/dev/null

  echo "waiting for ssh..."
  local up=0
  for (( i = 0; i < 60; i++ )); do
    if ssh "${SSH_OPTS[@]}" "$SSH_TARGET" true 2>/dev/null; then up=1; break; fi
    sleep "$POLL_SECS"
  done
  (( up )) || die "droplet resized to $target but ssh never came back — check it manually, then run: ssh $SSH_TARGET 'rd-agent-q/ops/research_caps.sh'"

  echo "re-deriving research caps for the new size..."
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" 'rd-agent-q/ops/research_caps.sh'

  echo "done: $cur -> $target"
  show_status
}

case "$CMD" in
  status) show_status ;;
  up)     resize_to "${TARGET_SIZE:-$BURST_SIZE}" ;;
  down)   resize_to "$BASE_SIZE" ;;
esac

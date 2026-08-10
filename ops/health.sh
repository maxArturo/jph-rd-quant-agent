#!/usr/bin/env bash
# health.sh — one-shot health check + exposure audit for rd-agent-q (US-042).
#
# Proves three things about this box:
#   1. Every rdq-* systemd user unit is in its expected state: long-running
#      services active, timers active/waiting, timer-driven oneshot services
#      not in the "failed" state (inactive/dead is their healthy resting state).
#   2. No repo-owned process listens on a non-loopback interface (ss audit of
#      the PIDs inside each rdq-* service cgroup), and the repo's reserved
#      ports (19899 server_ui, 19900 trace viewer) are loopback-only no matter
#      who owns them.
#   3. `tailscale serve status` matches the PLAN.md §1 port table: every
#      mapping tailnet-only, never funnel, :19899 never exposed, :19900 (when
#      enabled via ops/expose_traces.sh) proxying exactly http://127.0.0.1:19900,
#      and no mapping outside the known allowlist. The allowlist distinguishes
#      REPO_SERVE (ours to add/remove) from FOREIGN_SERVE (other projects
#      sharing this box — audited, but never turned off from here).
#
# Exit: 0 on a healthy box; nonzero otherwise, naming each failing check.
set -uo pipefail

# Long-running services (Restart=always) that must be active.
LONG_RUNNING=(
  rdq-orchestrator.service
  rdq-research.service
)
# Timers that must be active (waiting counts as active).
TIMERS=(
  rdq-data-refresh.timer
  rdq-pred-refresh.timer
  rdq-rebalance.timer
  rdq-sweep.timer
  rdq-gpu-watchdog.timer
)
# Timer-driven oneshots: healthy is "anything but failed" (dead between runs).
ONESHOTS=(
  rdq-data-refresh.service
  rdq-pred-refresh.service
  rdq-rebalance.service
  rdq-sweep.service
  rdq-gpu-watchdog.service
)

# PLAN.md §1 port table: tailscale serve allowlist, serve port -> proxy target.
# Split by OWNERSHIP, because the two halves have different remedies when the
# audit trips.
#
# REPO_SERVE — mappings this repo owns and may add/remove itself: currently
# just the rdagent trace viewer, added on demand by ops/expose_traces.sh.
declare -A REPO_SERVE=(
  [19900]="http://127.0.0.1:19900"
)
# FOREIGN_SERVE — mappings belonging to OTHER projects co-tenanted on this
# box. Still audited (tailnet-only, expected target, never a repo-reserved
# port), but NOT ours to change: drift here is a question for the owning
# project, never a `serve --https=<port> off` from this repo.
#   443  -> OneCLI web UI (secrets vault + Notion app-connection grants)
#   3100 -> Grafana, monitoring compose stack
#   8443 -> jph-master-tracker app server
declare -A FOREIGN_SERVE=(
  [443]="http://127.0.0.1:10254"
  [3100]="http://127.0.0.1:3001"
  [8443]="http://127.0.0.1:3200"
)
# Repo-reserved ports that must be loopback-only regardless of owning process.
REPO_PORTS=(19899 19900)
# server_ui must never be tailscale-served (flask-cors advisories; PLAN.md).
FORBIDDEN_SERVE_PORT=19899

# Union of both halves: what may appear in `tailscale serve status` at all.
declare -A ALLOWED_SERVE=()
for port in "${!REPO_SERVE[@]}"; do ALLOWED_SERVE[$port]=${REPO_SERVE[$port]}; done
for port in "${!FOREIGN_SERVE[@]}"; do
  # A co-tenant may never claim one of our reserved ports: that would let a
  # foreign entry silently whitelist the exposure the audit exists to catch.
  for repo_port in "${REPO_PORTS[@]}"; do
    if [[ "$port" == "$repo_port" ]]; then
      echo "ERROR: FOREIGN_SERVE claims repo-reserved port $port" >&2
      exit 2
    fi
  done
  ALLOWED_SERVE[$port]=${FOREIGN_SERVE[$port]}
done

# systemctl --user needs the user manager socket from non-login shells.
if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  XDG_RUNTIME_DIR="/run/user/$(id -u)"
  export XDG_RUNTIME_DIR
fi

die() { echo "ERROR: $*" >&2; exit 2; }
command -v systemctl >/dev/null || die "systemctl not found on PATH"
command -v ss >/dev/null || die "ss not found on PATH"
command -v tailscale >/dev/null || die "tailscale not found on PATH"

fails=0
failed_checks=()
pass() { echo "PASS  $1  ($2)"; }
fail() {
  echo "FAIL  $1  ($2)"
  fails=$((fails + 1))
  failed_checks+=("$1")
}

# --- 1. Unit states -----------------------------------------------------------
for unit in "${LONG_RUNNING[@]}"; do
  state=$(systemctl --user is-active "$unit" 2>/dev/null) || true
  if [[ "$state" == "active" ]]; then
    pass "service $unit" "active"
  else
    fail "service $unit" "expected active, is ${state:-unknown}"
  fi
done

for timer in "${TIMERS[@]}"; do
  state=$(systemctl --user is-active "$timer" 2>/dev/null) || true
  if [[ "$state" == "active" ]]; then
    pass "timer $timer" "active"
  else
    fail "timer $timer" "expected active, is ${state:-unknown} — systemctl --user enable --now $timer"
  fi
done

for unit in "${ONESHOTS[@]}"; do
  state=$(systemctl --user is-failed "$unit" 2>/dev/null) || true
  if [[ "$state" == "failed" ]]; then
    fail "oneshot $unit" "last run failed — journalctl --user -u $unit"
  else
    pass "oneshot $unit" "${state:-unknown}"
  fi
done

# --- 2. Loopback audit ---------------------------------------------------------
# Collect every PID inside each rdq service's cgroup (children included);
# fall back to MainPID when the cgroup path is not readable.
declare -A rdq_pid_unit=()
for unit in "${LONG_RUNNING[@]}" "${ONESHOTS[@]}"; do
  cg=$(systemctl --user show "$unit" -p ControlGroup --value 2>/dev/null) || cg=""
  if [[ -n "$cg" && -r "/sys/fs/cgroup${cg}/cgroup.procs" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && rdq_pid_unit[$pid]=$unit
    done <"/sys/fs/cgroup${cg}/cgroup.procs"
  else
    pid=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null) || pid=""
    [[ -n "$pid" && "$pid" != "0" ]] && rdq_pid_unit[$pid]=$unit
  fi
done

is_loopback_addr() { # <addr without :port>
  [[ "$1" == 127.* || "$1" == "[::1]" ]]
}

# Tailnet interface addresses: CGNAT 100.64.0.0/10 or the Tailscale ULA
# prefix. tailscale serve terminates TLS here for its mappings (the
# pre-existing :443/:3100 mappings bind the same way), so an allowed serve
# port bound on the tailnet address is the sanctioned mechanism, not a leak.
is_tailnet_addr() { # <addr without :port>
  [[ "$1" =~ ^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\. || "$1" == "[fd7a:115c:a1e0:"* ]]
}

loopback_violations=0
ss_out=$(ss -tlnpH 2>/dev/null) || die "ss -tlnp failed"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  local_field=$(awk '{print $4}' <<<"$line")
  addr=${local_field%:*}
  port=${local_field##*:}
  is_loopback_addr "$addr" && continue

  # Repo-reserved ports must never listen beyond loopback, whoever owns them —
  # except tailscaled terminating an allowed serve mapping on the tailnet
  # interface (audited by the tailscale section below). 19899 has no allowed
  # mapping, so even a tailnet bind of it fails here.
  for repo_port in "${REPO_PORTS[@]}"; do
    if [[ "$port" == "$repo_port" ]]; then
      if is_tailnet_addr "$addr" && [[ -n "${ALLOWED_SERVE[$port]:-}" ]]; then
        continue
      fi
      fail "loopback port $port" "listens on $local_field (must be 127.0.0.1 only)"
      loopback_violations=$((loopback_violations + 1))
    fi
  done

  # Any rdq-owned socket beyond loopback is a violation.
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if [[ -n "${rdq_pid_unit[$pid]+x}" ]]; then
      fail "loopback ${rdq_pid_unit[$pid]}" "pid $pid listens on $local_field (must be loopback only)"
      loopback_violations=$((loopback_violations + 1))
    fi
  done < <(grep -o 'pid=[0-9]*' <<<"$line" | cut -d= -f2 | sort -u)
done <<<"$ss_out"
if [[ $loopback_violations -eq 0 ]]; then
  pass "loopback audit" "${#rdq_pid_unit[@]} rdq pid(s), no non-loopback listeners"
fi

# --- 3. Tailscale exposure audit ------------------------------------------------
serve_out=$(tailscale serve status 2>&1) || die "tailscale serve status failed: $serve_out"

serve_violations=0
if grep -qi "funnel" <<<"$serve_out"; then
  fail "tailscale funnel" "funnel output present — never funnel (tailscale funnel reset)"
  serve_violations=$((serve_violations + 1))
fi
if grep -q "$FORBIDDEN_SERVE_PORT" <<<"$serve_out"; then
  fail "tailscale serve :$FORBIDDEN_SERVE_PORT" "server_ui must never be exposed (tailscale serve --https=$FORBIDDEN_SERVE_PORT off)"
  serve_violations=$((serve_violations + 1))
fi

cur_port=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" == "No serve config"* ]] && continue
  if [[ "$line" =~ ^https://[^:/[:space:]]+(:([0-9]+))?[[:space:]]+\((.+)\)[[:space:]]*$ ]]; then
    cur_port=${BASH_REMATCH[2]:-443}
    scope=${BASH_REMATCH[3]}
    if [[ "$scope" != "tailnet only" ]]; then
      fail "tailscale serve :$cur_port" "scope is '$scope', must be tailnet only"
      serve_violations=$((serve_violations + 1))
    fi
  elif [[ "$line" =~ proxy[[:space:]]+(.+)$ ]]; then
    target=${BASH_REMATCH[1]}
    if [[ -z "$cur_port" ]]; then
      fail "tailscale serve" "proxy line outside a mapping block: $line"
      serve_violations=$((serve_violations + 1))
    elif [[ -z "${ALLOWED_SERVE[$cur_port]:-}" ]]; then
      fail "tailscale serve :$cur_port" "-> $target not in the PLAN.md port table (tailscale serve --https=$cur_port off; if another project on this box owns it, add it to FOREIGN_SERVE in ops/health.sh instead)"
      serve_violations=$((serve_violations + 1))
    elif [[ "${ALLOWED_SERVE[$cur_port]}" != "$target" ]]; then
      if [[ -n "${FOREIGN_SERVE[$cur_port]:-}" ]]; then
        fail "tailscale serve :$cur_port" "-> $target, expected ${FOREIGN_SERVE[$cur_port]} — co-tenant mapping retargeted; ask its owner, do not turn it off from this repo"
      else
        fail "tailscale serve :$cur_port" "-> $target, expected ${REPO_SERVE[$cur_port]} (tailscale serve --https=$cur_port off, then ops/expose_traces.sh)"
      fi
      serve_violations=$((serve_violations + 1))
    fi
  else
    fail "tailscale serve" "unrecognized serve status line: $line"
    serve_violations=$((serve_violations + 1))
  fi
done <<<"$serve_out"
if [[ $serve_violations -eq 0 ]]; then
  pass "tailscale exposure" "all mappings tailnet-only and in the PLAN.md port table (${#REPO_SERVE[@]} repo-owned, ${#FOREIGN_SERVE[@]} co-tenant)"
fi

# --- Summary --------------------------------------------------------------------
echo
if [[ $fails -eq 0 ]]; then
  echo "HEALTHY — all checks passed."
  exit 0
fi
echo "$fails check(s) failed:"
printf '  %s\n' "${failed_checks[@]}"
exit 1

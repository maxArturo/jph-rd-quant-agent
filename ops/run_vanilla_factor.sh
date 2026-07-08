#!/usr/bin/env bash
# ops/run_vanilla_factor.sh — vanilla RD-Agent fin_factor run wrapper (Phase 0).
#
# Usage:
#   ops/run_vanilla_factor.sh --check   # fast environment check, no launch
#   ops/run_vanilla_factor.sh           # one loop: rdagent fin_factor --loop_n 1
#
# Default mode wraps:
#   onecli run --agent rdq-research -- rdagent fin_factor --loop_n 1
# with LOG_TRACE_PATH and WORKSPACE_PATH exported (defaults under
# ~/rdq-runs/vanilla_factor/; override by exporting either var beforehand).
#
# !! FIRST FULL RUN IS SLOW — BUDGET HOURS !!
# The first full run downloads the default China Qlib store (cn_data, several
# GB into ~/.qlib/qlib_data/cn_data) and builds the local_qlib:latest docker
# image. Both are reused by later runs.
#
# Full-run completion criterion (the Phase 0 milestone): the run workspace
# ($WORKSPACE_PATH/<workspace-id>/) contains qlib_res.csv with a non-null IC
# value. Inspect the trace with: rdagent ui --port 19899 (point it at
# $LOG_TRACE_PATH).
#
# --check mode runs `rdagent health_check --no-check-env` and then hard-asserts
# what upstream only logs as warnings: docker usable without sudo, port 19899
# free, onecli gateway up with the rdq-research identity registered. The env/LLM
# leg is intentionally skipped: upstream env_check only understands
# OpenAI/DeepSeek env layouts and crashes (UnboundLocalError) on our
# Anthropic+Voyage layout. The LLM path is instead verified by
#   onecli run --agent rdq-research -- .venv/bin/python research/probe_llm.py
# (US-004; see docs/decisions.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
RDAGENT="${VENV}/bin/rdagent"
ONECLI_URL="${ONECLI_URL:-http://127.0.0.1:10254}"

RUN_ROOT="${RDQ_RUN_ROOT:-${HOME}/rdq-runs/vanilla_factor}"

usage() {
  sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

check_mode() {
  [[ -x "${RDAGENT}" ]] || fail "rdagent not installed at ${RDAGENT} (run research/install.sh)"

  echo "== rdagent health_check (docker + ports; env leg skipped, see header) =="
  "${RDAGENT}" health_check --no-check-env || fail "rdagent health_check exited nonzero"

  # health_check logs docker/port problems but still exits 0 — assert them hard.
  echo "== hard assertions =="
  docker info >/dev/null 2>&1 || fail "docker is not usable without sudo (add user to the docker group)"
  echo "PASS: docker reachable sudo-less"

  "${VENV}/bin/python" - <<'PY' || fail "port 19899 (rdagent ui/server_ui) is already in use"
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sys.exit(1 if s.connect_ex(("127.0.0.1", 19899)) == 0 else 0)
PY
  echo "PASS: port 19899 free"

  command -v onecli >/dev/null 2>&1 || fail "onecli CLI not found on PATH"
  curl -fsS "${ONECLI_URL}/api/health" >/dev/null 2>&1 \
    || curl -fsS "${ONECLI_URL}/" >/dev/null 2>&1 \
    || fail "OneCLI gateway not reachable at ${ONECLI_URL}"
  onecli agents list 2>/dev/null | jq -e \
    '.data[] | select(.identifier == "rdq-research")' >/dev/null \
    || fail "identity rdq-research not registered (run ops/setup_onecli.sh)"
  echo "PASS: onecli gateway up, rdq-research registered"

  echo "OK: environment ready for a vanilla fin_factor run"
}

run_mode() {
  [[ -x "${RDAGENT}" ]] || fail "rdagent not installed at ${RDAGENT} (run research/install.sh)"

  local env_file="${REPO_ROOT}/research/.env"
  [[ -f "${env_file}" ]] || fail "missing ${env_file} (copy research/.env.example and keep placeholder keys)"
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a

  local ts
  ts="$(date -u +%Y-%m-%d_%H-%M-%S)"
  export LOG_TRACE_PATH="${LOG_TRACE_PATH:-${RUN_ROOT}/log/${ts}}"
  export WORKSPACE_PATH="${WORKSPACE_PATH:-${RUN_ROOT}/workspace}"
  mkdir -p "${LOG_TRACE_PATH}" "${WORKSPACE_PATH}"

  echo "LOG_TRACE_PATH=${LOG_TRACE_PATH}"
  echo "WORKSPACE_PATH=${WORKSPACE_PATH}"
  echo "Launching: onecli run --agent rdq-research -- rdagent fin_factor --loop_n 1"
  exec onecli run --agent rdq-research -- "${RDAGENT}" fin_factor --loop_n 1
}

case "${1:-}" in
  --check) check_mode ;;
  -h|--help) usage ;;
  "") run_mode ;;
  *)
    usage >&2
    fail "unknown argument: $1"
    ;;
esac

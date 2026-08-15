#!/usr/bin/env bash
# ops/gpu_worker/gpu_worker.sh — burst GPU droplet for hypothesis research runs.
#
# Runs FROM the control box (this droplet). The worker is a disposable
# DigitalOcean GPU droplet that runs the full `rdagent fin_quant` loop with
# model training/backtests on the GPU; it holds NO credentials — LLM/embedding
# traffic rides an SSH reverse tunnel back to the local OneCLI proxy (:10255),
# authenticated by the per-agent proxy token and the proxy's MITM CA bundle.
#
# Lifecycle (see ops/gpu_worker/README.md for the full runbook):
#   gpu_worker.sh sizes                 # list GPU sizes/regions/images (needs doctl auth)
#   gpu_worker.sh provision             # create the droplet, wait for SSH
#   gpu_worker.sh bootstrap             # packages, docker image, data, venv (idempotent)
#   gpu_worker.sh tunnel                # (re)start reverse tunnel + write proxy env
#   gpu_worker.sh check                 # remote run_us_quant.sh --check (RDQ_LAUNCHER=direct)
#   gpu_worker.sh run [--loop_n N] [--all_duration DUR] [--test-end YYYY-MM-DD]  # launch in remote tmux
#   gpu_worker.sh snapshot              # bake worker into rdq-gpu-base-* image (fast boots)
#   gpu_worker.sh ssh <cmd...>          # arbitrary remote command (worker SSH opts)
#   gpu_worker.sh status                # droplet / tunnel / run / GPU utilization
#   gpu_worker.sh fetch                 # rsync results back under the state dir
#   gpu_worker.sh destroy [--force]     # delete the droplet (BILLING STOPS HERE)
#
# Config (env overrides):
#   RDQ_GPU_SIZE      droplet size slug        (default gpu-4000adax1-20gb, ~$0.76/hr)
#   RDQ_GPU_REGION    region slug              (default tor1; GPUs: nyc2/tor1/atl1/...)
#   RDQ_GPU_IMAGE     image slug               (default gpu-h100x1-base — DO's AI/ML-ready
#                     image, used for ALL single-GPU sizes: NVIDIA driver, docker,
#                     nvidia-container-toolkit preinstalled)
#   RDQ_GPU_NAME      droplet name             (default rdq-gpu-worker)
#   RDQ_GPU_STATE_DIR state/results dir        (default ~/rdq-runs/gpu_worker)
#   RDQ_GPU_SSH_KEY   ssh private key path     (default ~/.ssh/rdq_gpu_worker; created)
#
# Requires: doctl authenticated (doctl auth init, or DIGITALOCEAN_ACCESS_TOKEN).
# Minimum token scopes: droplet create/read/delete + ssh_key create/read (the
# AI/ML image disables root passwords, so the API demands a registered key).
# The droplet bills until `destroy` — always destroy when the run is fetched.
set -euo pipefail
shopt -s inherit_errexit  # errexit must survive $(...) (bash default: it doesn't)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RDQ_GPU_SIZE="${RDQ_GPU_SIZE:-gpu-4000adax1-20gb}"
RDQ_GPU_REGION="${RDQ_GPU_REGION:-tor1}"
# Empty = auto: newest rdq-gpu-base-* snapshot if one exists (fast boot),
# else DO's AI/ML-ready base image. Set explicitly to force either.
RDQ_GPU_IMAGE="${RDQ_GPU_IMAGE:-}"
DEFAULT_GPU_IMAGE="gpu-h100x1-base"
RDQ_GPU_NAME="${RDQ_GPU_NAME:-rdq-gpu-worker}"
STATE_DIR="${RDQ_GPU_STATE_DIR:-${HOME}/rdq-runs/gpu_worker}"
SSH_KEY="${RDQ_GPU_SSH_KEY:-${HOME}/.ssh/rdq_gpu_worker}"

STATE_FILE="${STATE_DIR}/worker.env"
TUNNEL_UNIT="rdq-gpu-tunnel"
REMOTE_REPO="/root/rd-agent-q"
PROXY_ENV_FILE="/root/rdq-proxy.env"
RUN_SESSION="rdq-run"
RUN_LOG="/root/rdq-runs/gpu-run.log"

# systemctl --user needs the user manager socket from non-login shells.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

usage() {
  sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

note() {
  echo "==> $*"
}

ssh_opts() {
  # Printed one-per-line so callers can build both ssh arrays and rsync -e.
  echo "-i ${SSH_KEY}"
  echo "-o IdentitiesOnly=yes"
  echo "-o BatchMode=yes"
  echo "-o StrictHostKeyChecking=accept-new"
  echo "-o ConnectTimeout=15"
  echo "-o UserKnownHostsFile=${STATE_DIR}/known_hosts"
  echo "-o ServerAliveInterval=30"
  echo "-o ServerAliveCountMax=4"
}

# shellcheck disable=SC2207
SSH_OPTS=($(ssh_opts))

remote() {
  # shellcheck disable=SC2029  # client-side expansion is intended
  ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" "$@"
}

rsync_remote() {
  rsync -az -e "ssh $(ssh_opts | tr '\n' ' ')" "$@"
}

ensure_doctl() {
  command -v doctl >/dev/null 2>&1 \
    || fail "doctl not installed — snap install doctl, or download from https://github.com/digitalocean/doctl/releases into ~/.local/bin"
  # Scope-friendly auth probe: custom-scoped tokens (droplet + ssh_key only)
  # 403 on /v2/account, so probe the droplet scope we actually need.
  doctl compute droplet list --format ID >/dev/null 2>&1 \
    || fail "doctl not authenticated — run 'doctl auth init' with a DO API token (droplet + ssh_key write scopes), or export DIGITALOCEAN_ACCESS_TOKEN"
}

ensure_ssh_key() {
  if [[ ! -f "${SSH_KEY}" ]]; then
    note "generating worker SSH key at ${SSH_KEY}"
    ssh-keygen -t ed25519 -N "" -C "rdq-gpu-worker" -f "${SSH_KEY}" >/dev/null
  fi
  chmod 600 "${SSH_KEY}"
}

register_ssh_key() {
  # stdout is the fingerprint (captured by the caller) — notes go to stderr.
  # An account-registered key is unavoidable: the AI/ML image disables root
  # passwords and the droplets API rejects create without --ssh-keys (422),
  # user-data injection notwithstanding.
  local fp
  fp="$(ssh-keygen -E md5 -lf "${SSH_KEY}.pub" | awk '{print $2}' | sed 's/^MD5://')"
  if ! doctl compute ssh-key list --format FingerPrint --no-header | grep -qF "${fp}"; then
    note "registering worker SSH key with DigitalOcean" >&2
    doctl compute ssh-key import "${RDQ_GPU_NAME}" --public-key-file "${SSH_KEY}.pub" >/dev/null \
      || fail "could not register the SSH key — token needs the ssh_key create/read scopes"
  fi
  echo "${fp}"
}

require_size_in_region() {
  # Pre-flight the 422: GPU stock comes and goes; fail with the live table.
  local regions
  regions="$(doctl compute size list -o json \
    | jq -r --arg s "${RDQ_GPU_SIZE}" '.[] | select(.slug == $s) | (.regions // []) | join(",")')"
  [[ ",${regions}," == *",${RDQ_GPU_REGION},"* ]] \
    || fail "size ${RDQ_GPU_SIZE} not currently available in ${RDQ_GPU_REGION} (available: ${regions:-nowhere}) — override RDQ_GPU_SIZE/RDQ_GPU_REGION; see '$(basename "$0") sizes'"
}

load_state() {
  [[ -f "${STATE_FILE}" ]] \
    || fail "no worker state at ${STATE_FILE} — run '$(basename "$0") provision' first"
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  [[ -n "${DROPLET_ID:-}" && -n "${DROPLET_IP:-}" ]] \
    || fail "corrupt state file ${STATE_FILE} (missing DROPLET_ID/DROPLET_IP)"
}

tunnel_active() {
  systemctl --user is-active --quiet "${TUNNEL_UNIT}" 2>/dev/null
}

cmd_sizes() {
  ensure_doctl
  echo "--- GPU droplet sizes (slug / regions with stock NOW / \$-hr) ---"
  doctl compute size list -o json \
    | jq -r '.[] | select(.slug | startswith("gpu-"))
        | [.slug, ((.regions // []) | join(",") | if . == "" then "SOLD OUT" else . end), "\(.price_hourly)/hr"]
        | @tsv'
  echo
  echo "--- AI/ML-ready images (single-GPU sizes all use gpu-h100x1-base) ---"
  doctl compute image list --public --format Slug,Name --no-header | grep -i 'gpu' || true
}

cmd_provision() {
  ensure_doctl
  ensure_ssh_key
  mkdir -p "${STATE_DIR}"

  if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
    if [[ -n "${DROPLET_ID:-}" ]] && doctl compute droplet get "${DROPLET_ID}" >/dev/null 2>&1; then
      fail "worker droplet ${DROPLET_ID} (${DROPLET_IP:-?}) already exists — 'destroy' it before provisioning another"
    fi
    note "stale state file (droplet gone) — overwriting"
  fi

  require_size_in_region

  local image="${RDQ_GPU_IMAGE}"
  if [[ -z "${image}" ]]; then
    image="$(latest_snapshot_id)"
    if [[ -n "${image}" ]]; then
      note "booting from base snapshot image ${image} (bootstrap will be a quick delta sync)"
    else
      image="${DEFAULT_GPU_IMAGE}"
    fi
  fi

  local fp
  fp="$(register_ssh_key)"

  note "creating ${RDQ_GPU_SIZE} in ${RDQ_GPU_REGION} (image ${image}) — billing starts now"
  local droplet_id
  droplet_id="$(doctl compute droplet create "${RDQ_GPU_NAME}" \
    --size "${RDQ_GPU_SIZE}" --region "${RDQ_GPU_REGION}" --image "${image}" \
    --ssh-keys "${fp}" --tag-name rdq-gpu-worker --wait \
    --format ID --no-header)"
  [[ -n "${droplet_id}" ]] || fail "droplet create returned no ID"

  local ip=""
  for _ in $(seq 1 30); do
    ip="$(doctl compute droplet get "${droplet_id}" --format PublicIPv4 --no-header)"
    [[ -n "${ip}" ]] && break
    sleep 5
  done
  [[ -n "${ip}" ]] || fail "droplet ${droplet_id} never reported a public IP"

  umask 077
  cat > "${STATE_FILE}" <<EOF
DROPLET_ID=${droplet_id}
DROPLET_IP=${ip}
SIZE=${RDQ_GPU_SIZE}
REGION=${RDQ_GPU_REGION}
CREATED_AT=$(date -u +%FT%TZ)
EOF
  note "droplet ${droplet_id} at ${ip} — waiting for SSH"

  DROPLET_IP="${ip}"
  for _ in $(seq 1 60); do
    if remote true 2>/dev/null; then
      note "SSH is up — next: '$(basename "$0") bootstrap'"
      return 0
    fi
    sleep 5
  done
  fail "SSH never came up on ${ip} (droplet left running — check the DO console, then retry or 'destroy')"
}

cmd_bootstrap() {
  load_state
  require_local_layout

  note "installing packages on the worker"
  # A fresh droplet races first-boot apt activity (cloud-init, unattended
  # upgrades) for the dpkg lock — the 2026-08-11 run died on exactly that.
  # Wait out first boot, then have apt wait for the lock instead of failing.
  remote "command -v cloud-init >/dev/null && timeout 600 cloud-init status --wait >/dev/null 2>&1 || true"
  remote "DEBIAN_FRONTEND=noninteractive apt-get -yq -o DPkg::Lock::Timeout=600 update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get -yq -o DPkg::Lock::Timeout=600 install rsync tmux git make python3-venv python3-pip zstd jq >/dev/null"

  note "verifying GPU + docker runtime"
  remote "nvidia-smi -L" || fail "nvidia-smi failed — wrong image? (need the AI/ML-ready ${RDQ_GPU_IMAGE})"
  remote "docker info >/dev/null" || fail "docker not running on the worker"

  if remote "docker image inspect local_qlib:latest >/dev/null 2>&1"; then
    note "local_qlib:latest already on the worker — skipping transfer"
  else
    note "shipping local_qlib:latest (~16.5GB raw, zstd-compressed on the wire; budget 10-30 min)"
    if command -v zstd >/dev/null 2>&1; then
      docker save local_qlib:latest | zstd -T0 -3 -q \
        | ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" "zstd -d -q | docker load"
    else
      docker save local_qlib:latest \
        | ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" "docker load"
    fi
  fi

  note "verifying CUDA is visible to torch inside the image"
  remote "docker run --rm --gpus all local_qlib:latest python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA device\"; print(\"GPU:\", torch.cuda.get_device_name(0))'"

  note "syncing repo, qlib store, factor source, CA bundle"
  # rsync only creates the final path component — make the parents first.
  remote "mkdir -p /root/.qlib/qlib_data /root/rdq-data/factor_source /root/.onecli"
  rsync_remote --delete \
    --exclude '.git' --exclude '.venv' --exclude 'git_ignore_folder' \
    --exclude 'pickle_cache' --exclude '/log' --exclude '__pycache__' \
    --exclude '.pytest_cache' --exclude '.ruff_cache' \
    "${REPO_ROOT}/" "root@${DROPLET_IP}:${REMOTE_REPO}/"
  rsync_remote "${HOME}/.qlib/qlib_data/us_data/" "root@${DROPLET_IP}:/root/.qlib/qlib_data/us_data/"
  # ALL universes ship (factor sources + rendered per-universe templates), so
  # runs against operator-created universes need no extra sync step.
  rsync_remote "${HOME}/rdq-data/factor_source/" "root@${DROPLET_IP}:/root/rdq-data/factor_source/"
  if [[ -d "${HOME}/rdq-data/templates" ]]; then
    rsync_remote "${HOME}/rdq-data/templates/" "root@${DROPLET_IP}:/root/rdq-data/templates/"
  fi
  rsync_remote "${HOME}/.onecli/ca-bundle.pem" "root@${DROPLET_IP}:/root/.onecli/ca-bundle.pem"
  # research/.env may be a symlink (worktree checkouts) — ship the content.
  rsync_remote --copy-links "${REPO_ROOT}/research/.env" "root@${DROPLET_IP}:${REMOTE_REPO}/research/.env"

  note "building the worker venv (make venv + pinned rdagent — first time is slow)"
  remote "cd ${REMOTE_REPO} && make venv >/dev/null && research/install.sh"

  note "bootstrap complete — next: '$(basename "$0") tunnel'"
}

require_local_layout() {
  [[ -f "${REPO_ROOT}/research/.env" ]] \
    || fail "missing ${REPO_ROOT}/research/.env (copy research/.env.example) — the worker run sources it"
  [[ -d "${HOME}/.qlib/qlib_data/us_data" ]] || fail "missing ~/.qlib/qlib_data/us_data on the control box"
  [[ -d "${HOME}/rdq-data/factor_source/us_liquid" ]] || fail "missing ~/rdq-data/factor_source/us_liquid"
  [[ -f "${HOME}/.onecli/ca-bundle.pem" ]] || fail "missing ~/.onecli/ca-bundle.pem (OneCLI proxy CA)"
  docker image inspect local_qlib:latest >/dev/null 2>&1 \
    || fail "local_qlib:latest not present locally — build it first (see rdq-research.service comments)"
}

cmd_tunnel() {
  load_state
  command -v onecli >/dev/null 2>&1 || fail "onecli CLI not found on PATH"

  # The per-agent proxy URL (scheme://x:TOKEN@127.0.0.1:10255). Fetched fresh
  # every time; never echoed or stored on the control box.
  local proxy_url
  proxy_url="$(onecli run --agent rdq-research -- printenv HTTPS_PROXY 2>/dev/null | tail -1)"
  [[ "${proxy_url}" == http://* ]] \
    || fail "could not obtain HTTPS_PROXY via 'onecli run --agent rdq-research' (is the OneCLI gateway up?)"
  local hostport="${proxy_url##*@}"
  local port="${hostport##*:}"

  if tunnel_active; then
    note "restarting tunnel unit ${TUNNEL_UNIT}"
    systemctl --user stop "${TUNNEL_UNIT}" 2>/dev/null || true
  fi
  # Reverse-forward the OneCLI proxy onto the WORKER's loopback at the same
  # port, so the proxy URL is valid verbatim on both ends. Transient user
  # unit with Restart=always survives network blips for multi-day runs.
  note "starting reverse tunnel (worker 127.0.0.1:${port} -> control box OneCLI proxy)"
  systemd-run --user --collect --unit "${TUNNEL_UNIT}" \
    --property=Restart=always --property=RestartSec=5 \
    ssh "${SSH_OPTS[@]}" -N -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:${port}:127.0.0.1:${port}" "root@${DROPLET_IP}" >/dev/null

  note "writing ${PROXY_ENV_FILE} on the worker"
  remote "umask 077 && cat > ${PROXY_ENV_FILE}" <<EOF
# Written by ops/gpu_worker/gpu_worker.sh tunnel — OneCLI proxy env, reached
# over the SSH reverse tunnel from the control box. Source before running.
export HTTPS_PROXY='${proxy_url}'
export HTTP_PROXY='${proxy_url}'
export https_proxy='${proxy_url}'
export http_proxy='${proxy_url}'
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
export SSL_CERT_FILE=/root/.onecli/ca-bundle.pem
export REQUESTS_CA_BUNDLE=/root/.onecli/ca-bundle.pem
export CURL_CA_BUNDLE=/root/.onecli/ca-bundle.pem
EOF

  # End-to-end probe: TLS through the tunnel, MITM CA trusted, key injected
  # by the proxy (Anthropic 400s without the version header; 200 = healthy).
  local code
  code="$(remote ". ${PROXY_ENV_FILE} && curl -sS --max-time 20 -o /dev/null -w '%{http_code}' -H 'anthropic-version: 2023-06-01' https://api.anthropic.com/v1/models")" \
    || fail "probe through the tunnel failed outright (tunnel or CA problem)"
  [[ "${code}" == "200" ]] \
    || fail "Anthropic probe returned HTTP ${code} through the proxy (expected 200 — token/injection problem?)"
  note "tunnel healthy — Anthropic probe 200 via worker. Next: '$(basename "$0") check'"
}

cmd_check() {
  load_state
  tunnel_active || fail "tunnel unit ${TUNNEL_UNIT} not active — run '$(basename "$0") tunnel' first"
  remote "set -a && . ${PROXY_ENV_FILE} && set +a && RDQ_LAUNCHER=direct ${REMOTE_REPO}/ops/run_us_quant.sh --check"
}

cmd_run() {
  local loop_n="1" all_duration="" instruction="" universe="" test_end=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --loop_n)
        [[ $# -ge 2 ]] || fail "--loop_n needs a value"
        loop_n="$2"; shift 2 ;;
      --test-end)
        # Launch-computed rolling TEST_END (US-008, ops/gpu_pipeline.py) —
        # exported as RDQ_TEST_END so run_us_quant.sh skips its stale fallback.
        [[ $# -ge 2 ]] || fail "--test-end needs a value"
        [[ "$2" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
          || fail "--test-end must be YYYY-MM-DD (got '$2')"
        test_end="$2"; shift 2 ;;
      --all_duration)
        [[ $# -ge 2 ]] || fail "--all_duration needs a value"
        all_duration="$2"; shift 2 ;;
      --instruction)
        [[ $# -ge 2 ]] || fail "--instruction needs a value"
        instruction="$2"; shift 2 ;;
      --universe)
        [[ $# -ge 2 ]] || fail "--universe needs a value"
        universe="$2"; shift 2 ;;
      *) fail "unknown run argument: $1" ;;
    esac
  done

  load_state

  # Custom universe: hard-refuse when its artifacts are missing on the worker
  # (a silent us_liquid fallback is exactly the 2026-08-05 mislabeling bug the
  # server_ui path 400s on). us_liquid/empty means the pinned defaults.
  local universe_exports=""
  if [[ -n "${universe}" && "${universe}" != "us_liquid" ]]; then
    remote "test -d /root/rdq-data/factor_source/${universe}/data_folder \
      && test -d /root/rdq-data/templates/${universe} \
      && test -f /root/.qlib/qlib_data/us_data/instruments/${universe}.txt" \
      || fail "universe '${universe}' artifacts missing on the worker — re-run bootstrap after materializing it"
    universe_exports="export RDQ_UNIVERSE='${universe}'
export RDQ_FACTOR_SOURCE='/root/rdq-data/factor_source/${universe}'
export RDQ_UNIVERSE_TEMPLATES='/root/rdq-data/templates/${universe}'"
  fi
  tunnel_active || fail "tunnel unit ${TUNNEL_UNIT} not active — run '$(basename "$0") tunnel' first"
  if remote "tmux has-session -t ${RUN_SESSION} 2>/dev/null"; then
    fail "a run is already active in tmux session '${RUN_SESSION}' (see 'status'; attach with: ssh root@${DROPLET_IP} tmux attach -t ${RUN_SESSION})"
  fi

  # The directive can hold any operator text (quotes, newlines) — ship it
  # base64'd so the generated script never re-parses it.
  local instr_line=""
  if [[ -n "${instruction}" ]]; then
    local instr_b64
    instr_b64="$(printf '%s' "${instruction}" | base64 -w0)"
    instr_line="export RDQ_USER_INSTRUCTION=\"\$(printf '%s' '${instr_b64}' | base64 -d)\""
  fi

  local test_end_line=""
  if [[ -n "${test_end}" ]]; then
    test_end_line="export RDQ_TEST_END='${test_end}'"
  fi

  note "writing launch script (loop_n=${loop_n}${all_duration:+, all_duration=${all_duration}}${universe:+, universe=${universe}}${instruction:+, directive-seeded}${test_end:+, test_end=${test_end}})"
  remote "umask 077 && cat > /root/rdq-launch.sh && chmod +x /root/rdq-launch.sh" <<EOF
#!/usr/bin/env bash
# Written by gpu_worker.sh run — executed inside tmux on the worker.
set -uo pipefail
set -a; . ${PROXY_ENV_FILE}; set +a
export RDQ_LAUNCHER=direct
${universe_exports}
${instr_line}
${test_end_line}
mkdir -p /root/rdq-runs
{
  echo "=== run start \$(date -u +%FT%TZ) loop_n=${loop_n}${all_duration:+ all_duration=${all_duration}} ==="
  ${REMOTE_REPO}/ops/run_us_quant.sh --loop_n ${loop_n}${all_duration:+ --all_duration ${all_duration}} 2>&1
  echo "=== run exit=\$? \$(date -u +%FT%TZ) ==="
} | tee -a ${RUN_LOG}
EOF

  remote "tmux new-session -d -s ${RUN_SESSION} /root/rdq-launch.sh"
  note "run launched in tmux '${RUN_SESSION}' on ${DROPLET_IP}"
  note "follow with:  $(basename "$0") status   |   ssh root@${DROPLET_IP} tail -f ${RUN_LOG}"
}

SNAPSHOT_PREFIX="rdq-gpu-base"

latest_snapshot_id() {
  # Newest private image named ${SNAPSHOT_PREFIX}-* (name sorts by date suffix).
  doctl compute image list --format ID,Name --no-header 2>/dev/null \
    | awk -v p="${SNAPSHOT_PREFIX}-" 'index($2, p) == 1 { print $1, $2 }' \
    | sort -k2 | tail -1 | awk '{print $1}'
}

cmd_snapshot() {
  # Bake the bootstrapped worker into a DO image so future provisions skip
  # the 16.5GB docker-image ship + venv build (~20 min -> ~3 min boot).
  # Regional: usable for provisions in the SAME region the worker runs in.
  load_state
  ensure_doctl
  local name
  name="${SNAPSHOT_PREFIX}-$(date -u +%Y%m%d-%H%M)"
  note "powering off ${DROPLET_ID} for a consistent snapshot"
  doctl compute droplet-action power-off "${DROPLET_ID}" --wait >/dev/null
  note "taking snapshot ${name} (several minutes)"
  doctl compute droplet-action snapshot "${DROPLET_ID}" --snapshot-name "${name}" --wait >/dev/null
  note "powering back on"
  doctl compute droplet-action power-on "${DROPLET_ID}" --wait >/dev/null
  for _ in $(seq 1 60); do
    remote true 2>/dev/null && break
    sleep 5
  done
  remote true 2>/dev/null || fail "worker did not come back after snapshot (droplet still exists)"
  # Keep only the newest base image — older ones just accrue storage cost.
  local keep old_ids
  keep="$(latest_snapshot_id)"
  old_ids="$(doctl compute image list --format ID,Name --no-header \
    | awk -v p="${SNAPSHOT_PREFIX}-" -v keep="${keep}" \
        'index($2, p) == 1 && $1 != keep { print $1 }')"
  local image_id
  for image_id in ${old_ids}; do
    note "pruning old base image ${image_id}"
    doctl compute image delete -f "${image_id}" || true
  done
  note "snapshot ${name} ready — future provisions in this region boot from it"
}

cmd_ssh() {
  # Arbitrary remote command with the worker's SSH options (used by
  # ops/gpu_pipeline.py so the SSH config lives in exactly one place).
  load_state
  [[ $# -ge 1 ]] || fail "ssh needs a command"
  remote "$@"
}

cmd_status() {
  load_state
  echo "droplet:  ${DROPLET_ID} @ ${DROPLET_IP} (${SIZE:-?} in ${REGION:-?}, created ${CREATED_AT:-?})"
  if command -v doctl >/dev/null 2>&1 && doctl compute droplet list --format ID >/dev/null 2>&1; then
    echo "DO state: $(doctl compute droplet get "${DROPLET_ID}" --format Status --no-header 2>/dev/null || echo 'NOT FOUND (already destroyed?)')"
  fi
  if tunnel_active; then echo "tunnel:   active (${TUNNEL_UNIT})"; else echo "tunnel:   DOWN"; fi
  if ! remote true 2>/dev/null; then
    echo "ssh:      UNREACHABLE"
    return 0
  fi
  if remote "tmux has-session -t ${RUN_SESSION} 2>/dev/null"; then
    echo "run:      ACTIVE (tmux ${RUN_SESSION})"
  else
    echo "run:      no active tmux session"
  fi
  echo "gpu:      $(remote "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo 'n/a')"
  echo "--- last log lines (${RUN_LOG}) ---"
  remote "tail -n 8 ${RUN_LOG} 2>/dev/null" || echo "(no run log yet)"
}

cmd_fetch() {
  load_state
  local dest="${STATE_DIR}/results"
  mkdir -p "${dest}"
  note "fetching /root/rdq-runs/ -> ${dest}/ (traces, workspaces, run log)"
  rsync_remote "root@${DROPLET_IP}:/root/rdq-runs/" "${dest}/"
  note "results in ${dest} (trace logs under us_quant/log/, workspaces under us_quant/workspace/)"
}

cmd_destroy() {
  local force=0
  [[ "${1:-}" == "--force" ]] && force=1
  load_state

  if [[ ${force} -eq 0 ]] && remote "tmux has-session -t ${RUN_SESSION} 2>/dev/null"; then
    fail "a run is still active — 'fetch' what you need, then destroy with --force (or wait for it to finish)"
  fi

  if tunnel_active; then
    note "stopping tunnel unit"
    systemctl --user stop "${TUNNEL_UNIT}" 2>/dev/null || true
  fi

  ensure_doctl
  note "deleting droplet ${DROPLET_ID} (billing stops)"
  doctl compute droplet delete -f "${DROPLET_ID}"
  rm -f "${STATE_FILE}" "${STATE_DIR}/known_hosts"
  note "destroyed. Results (if fetched) remain under ${STATE_DIR}/results"
}

[[ $# -ge 1 ]] || { usage >&2; fail "missing subcommand"; }
SUBCOMMAND="$1"
shift

case "${SUBCOMMAND}" in
  sizes)     cmd_sizes ;;
  provision) cmd_provision ;;
  bootstrap) cmd_bootstrap ;;
  tunnel)    cmd_tunnel ;;
  check)     cmd_check ;;
  run)       cmd_run "$@" ;;
  snapshot)  cmd_snapshot ;;
  ssh)       cmd_ssh "$@" ;;
  status)    cmd_status ;;
  fetch)     cmd_fetch ;;
  destroy)   cmd_destroy "$@" ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    fail "unknown subcommand: ${SUBCOMMAND}"
    ;;
esac

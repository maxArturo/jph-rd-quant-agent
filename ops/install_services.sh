#!/usr/bin/env bash
# Link this repo's systemd user units into ~/.config/systemd/user/ and
# daemon-reload (US-010). Idempotent: relinking an already-linked unit is a
# no-op. Later stories append their units/timers to UNITS below.
#
# Usage: ops/install_services.sh
# Then:  systemctl --user enable --now <unit>
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNITS=(
  rdq-orchestrator.service
  rdq-research.service
)

mkdir -p "$UNIT_DIR"
for unit in "${UNITS[@]}"; do
  src="$REPO_DIR/ops/$unit"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: unit file missing: $src" >&2
    exit 1
  fi
  ln -sfn "$src" "$UNIT_DIR/$unit"
  echo "linked $UNIT_DIR/$unit -> $src"
done

systemctl --user daemon-reload
echo "daemon-reloaded; enable with: systemctl --user enable --now <unit>"

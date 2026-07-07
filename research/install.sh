#!/usr/bin/env bash
# Install microsoft/RD-Agent at the pinned commit into the project venv.
#
# The pin lives in research/PINNED_COMMIT (single line, full 40-char SHA).
# Rationale for the chosen commit: docs/decisions.md.
#
# Usage: research/install.sh
# Idempotent: re-running reinstalls the same pinned commit (pip no-ops if
# already satisfied at that exact revision is not guaranteed, so we use
# --force-reinstall only when the installed version does not import).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PIN_FILE="${REPO_ROOT}/research/PINNED_COMMIT"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "error: project venv not found at ${VENV} (run 'make venv' first)" >&2
  exit 1
fi

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "error: ${PIN_FILE} missing" >&2
  exit 1
fi

SHA="$(tr -d '[:space:]' < "${PIN_FILE}")"
if [[ ! "${SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: ${PIN_FILE} does not contain a full 40-char commit SHA (got: '${SHA}')" >&2
  exit 1
fi

echo "Installing rdagent @ ${SHA} into ${VENV} ..."
"${VENV}/bin/pip" install --quiet "rdagent @ git+https://github.com/microsoft/RD-Agent@${SHA}"

echo "Verifying import ..."
"${VENV}/bin/python" -c "import rdagent; print('rdagent import OK, version:', getattr(rdagent, '__version__', 'unknown'))"

echo "Done."

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

# rdagent leaves pydantic-ai-slim unpinned; the 2.x line renamed the MCP
# server classes (MCPServerStreamableHTTP -> MCPToolset) and breaks the
# `rdagent` CLI import at our pinned commit. Pin the last 1.x release.
# See docs/decisions.md (2026-07-08 pydantic-ai-slim pin).
echo "Pinning pydantic-ai-slim to a 1.x release compatible with the rdagent pin ..."
"${VENV}/bin/pip" install --quiet "pydantic-ai-slim[mcp,openai,prefect]==1.107.0"

echo "Verifying import ..."
"${VENV}/bin/python" -c "import rdagent; print('rdagent import OK, version:', getattr(rdagent, '__version__', 'unknown'))"
# The CLI pulls in the full app graph (incl. pydantic_ai) — a stronger check.
"${VENV}/bin/python" -c "from rdagent.app.cli import app; print('rdagent CLI import OK')"

echo "Done."

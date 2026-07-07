"""US-002: RD-Agent is installed at the pinned commit."""

import importlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = REPO_ROOT / "research" / "PINNED_COMMIT"


def test_pinned_commit_file_is_full_sha() -> None:
    sha = PIN_FILE.read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"not a full 40-char SHA: {sha!r}"


def test_rdagent_imports() -> None:
    module = importlib.import_module("rdagent")
    assert module is not None


def test_installed_rdagent_matches_pin() -> None:
    sha = PIN_FILE.read_text().strip()
    dist_infos = list(
        (REPO_ROOT / ".venv" / "lib").glob("python*/site-packages/rdagent-*.dist-info")
    )
    assert dist_infos, "rdagent dist-info not found in .venv (run research/install.sh)"
    direct_url = json.loads((dist_infos[0] / "direct_url.json").read_text())
    assert direct_url["vcs_info"]["commit_id"] == sha

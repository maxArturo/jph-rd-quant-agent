"""Offline tests for ops/setup_onecli.sh + ops/check_onecli.sh (US-018).

Both scripts shell out to `onecli` / `jq` / `curl` by bare name, so the
end-to-end tests run the REAL scripts with a stub `onecli` (and a no-op
`curl` for the gateway reachability ping) on PATH, driven by JSON fixtures —
no gateway, no credentials, no network. jq is real.

What US-018 pins down:
- setup_onecli.sh registers rdq-exec-live with EXACTLY the live-host
  allowlist (api.alpaca.markets api.notion.com financialmodelingprep.com —
  no paper host) and dies if the live host appears in any OTHER identity's
  allowlist.
- check_onecli.sh proves all four isolation directions and SKIPS (never
  fails) the live-auth probe while the live secret is not vaulted/assigned.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP = REPO_ROOT / "ops" / "setup_onecli.sh"
CHECK = REPO_ROOT / "ops" / "check_onecli.sh"

LIVE_URL = "https://api.alpaca.markets/v2/account"
PAPER_URL = "https://paper-api.alpaca.markets/v2/account"
FMP_URL = "https://financialmodelingprep.com/stable/search-symbol?query=AAPL"

STUB_ONECLI = """#!/usr/bin/env bash
# stub onecli driven by files in $ONECLI_STUB_DIR
set -euo pipefail
case "$1 $2" in
  "agents list") cat "$ONECLI_STUB_DIR/agents.json" ;;
  "secrets list") cat "$ONECLI_STUB_DIR/secrets.json" ;;
  "agents secrets") cat "$ONECLI_STUB_DIR/assigned_$4.json" ;;
  "agents create")
    shift 2
    ident=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --identifier) ident=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    jq --arg id "$ident" '.data += [{"identifier": $id, "id": ("u-" + $id)}]' \\
      "$ONECLI_STUB_DIR/agents.json" > "$ONECLI_STUB_DIR/agents.json.tmp"
    mv "$ONECLI_STUB_DIR/agents.json.tmp" "$ONECLI_STUB_DIR/agents.json"
    echo "$ident" >> "$ONECLI_STUB_DIR/created.log"
    ;;
  "agents set-secrets")
    shift 2
    uuid="" ids=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --id) uuid=$2; shift 2 ;;
        --secret-ids) ids=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    echo "$uuid $ids" >> "$ONECLI_STUB_DIR/setsecrets.log"
    ;;
  "run --agent")
    identity=$3
    url=""
    for arg in "$@"; do url=$arg; done
    code=$(grep -F "$identity|$url|" "$ONECLI_STUB_DIR/codes.txt" \\
      | head -1 | cut -d'|' -f3)
    printf '%s' "${code:-000}"
    ;;
  *) echo "stub onecli: unhandled: $*" >&2; exit 64 ;;
esac
"""

# The gateway reachability ping is a direct `curl` — always "reachable".
STUB_CURL = "#!/usr/bin/env bash\nexit 0\n"

PAPER_AGENTS = [
    {"identifier": "rdq-orchestrator", "id": "u-rdq-orchestrator"},
    {"identifier": "rdq-research", "id": "u-rdq-research"},
    {"identifier": "rdq-exec-paper", "id": "u-rdq-exec-paper"},
]
LIVE_AGENT = {"identifier": "rdq-exec-live", "id": "u-rdq-exec-live"}

# Vault secrets by host pattern; Notion deliberately has NO vault secret
# (app connection — docs/decisions.md 2026-07-08).
BASE_SECRETS = [
    {"hostPattern": "api.anthropic.com", "id": "s-anth"},
    {"hostPattern": "api.voyageai.com", "id": "s-voy"},
    {"hostPattern": "paper-api.alpaca.markets", "id": "s-paper"},
    {"hostPattern": "financialmodelingprep.com", "id": "s-fmp"},
]
LIVE_SECRET = {"hostPattern": "api.alpaca.markets", "id": "s-live"}

ASSIGNED = {
    "u-rdq-orchestrator": ["s-anth", "s-paper", "s-fmp"],
    "u-rdq-research": ["s-anth", "s-voy", "s-fmp"],
    "u-rdq-exec-paper": ["s-paper", "s-fmp"],
}

# All ten standard per-service CHECKS answer 200 so the isolation logic is
# what decides each scenario's outcome.
HEALTHY_SERVICE_CODES = [
    ("rdq-orchestrator", "https://api.anthropic.com/v1/models", "200"),
    ("rdq-orchestrator", "https://api.notion.com/v1/users/me", "200"),
    ("rdq-orchestrator", PAPER_URL, "200"),
    ("rdq-orchestrator", FMP_URL, "200"),
    ("rdq-research", "https://api.anthropic.com/v1/models", "200"),
    ("rdq-research", "https://api.voyageai.com/v1/embeddings", "200"),
    ("rdq-research", FMP_URL, "200"),
    ("rdq-exec-paper", PAPER_URL, "200"),
    ("rdq-exec-paper", "https://api.notion.com/v1/users/me", "200"),
    ("rdq-exec-paper", FMP_URL, "200"),
]
HEALTHY_ISOLATION_CODES = [
    ("rdq-exec-paper", LIVE_URL, "401"),
    ("rdq-orchestrator", LIVE_URL, "401"),
    ("rdq-exec-live", PAPER_URL, "401"),
    ("rdq-exec-live", LIVE_URL, "200"),
]


def make_stub_box(
    tmp_path: Path,
    *,
    live_agent: bool = True,
    live_secret_vaulted: bool = True,
    live_secret_assigned: bool = True,
    code_overrides: dict[tuple[str, str], str] | None = None,
) -> dict[str, str]:
    """A PATH-shimmed OneCLI box; scenarios break individual pieces."""
    bin_dir = tmp_path / "bin"
    stub_dir = tmp_path / "stub"
    bin_dir.mkdir()
    stub_dir.mkdir()
    for name, body in [("onecli", STUB_ONECLI), ("curl", STUB_CURL)]:
        script = bin_dir / name
        script.write_text(body)
        script.chmod(0o755)

    agents = list(PAPER_AGENTS) + ([LIVE_AGENT] if live_agent else [])
    secrets = list(BASE_SECRETS) + ([LIVE_SECRET] if live_secret_vaulted else [])
    assigned = dict(ASSIGNED)
    live_ids = ["s-fmp"]
    if live_secret_vaulted and live_secret_assigned:
        live_ids = ["s-live", "s-fmp"]
    assigned["u-rdq-exec-live"] = live_ids

    (stub_dir / "agents.json").write_text(json.dumps({"data": agents}))
    (stub_dir / "secrets.json").write_text(json.dumps({"data": secrets}))
    for uuid, ids in assigned.items():
        (stub_dir / f"assigned_{uuid}.json").write_text(json.dumps({"data": ids}))

    codes = {(i, u): c for i, u, c in HEALTHY_SERVICE_CODES + HEALTHY_ISOLATION_CODES}
    codes.update(code_overrides or {})
    (stub_dir / "codes.txt").write_text(
        "".join(f"{identity}|{url}|{code}\n" for (identity, url), code in codes.items())
    )
    return {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "ONECLI_STUB_DIR": str(stub_dir),
        "HOME": str(tmp_path),
    }


def run_script(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, check=False
    )


class TestScripts:
    def test_exist_and_executable(self) -> None:
        for script in (SETUP, CHECK):
            assert script.is_file()
            assert script.stat().st_mode & 0o111, f"{script.name} not executable"

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    @pytest.mark.parametrize("script", [SETUP, CHECK], ids=lambda p: p.name)
    def test_shellcheck_clean(self, script: Path) -> None:
        result = subprocess.run(
            ["shellcheck", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_setup_live_allowlist_exact(self) -> None:
        """rdq-exec-live's allowlist is exactly live Alpaca + Notion + FMP."""
        text = SETUP.read_text()
        assert (
            '[rdq-exec-live]="api.alpaca.markets api.notion.com financialmodelingprep.com"'
            in text
        )

    def test_setup_live_host_guard_scoped_to_live_identity(self) -> None:
        """The old unconditional die is now scoped: only rdq-exec-live may
        hold the live host; the blanket never-register rule is gone."""
        text = SETUP.read_text()
        assert '"$identity" != "rdq-exec-live"' in text
        assert "refusing to assign live Alpaca host" in text
        assert "Deliberately NO rdq-exec-live" not in text

    def test_check_names_all_four_directions(self) -> None:
        text = CHECK.read_text()
        assert '"rdq-exec-paper|$LIVE_ALPACA_URL' in text
        assert '"rdq-orchestrator|$LIVE_ALPACA_URL' in text
        assert '"rdq-exec-live|$PAPER_ALPACA_URL' in text
        assert "rdq-exec-live -> api.alpaca.markets (live) auth" in text


class TestSetupOnecli:
    def test_creates_live_identity_and_scopes_live_secret_to_it(self, tmp_path: Path) -> None:
        env = make_stub_box(tmp_path, live_agent=False)
        result = run_script(SETUP, env)
        assert result.returncode == 0, result.stdout + result.stderr
        stub_dir = Path(env["ONECLI_STUB_DIR"])

        created = (stub_dir / "created.log").read_text().split()
        assert created == ["rdq-exec-live"]

        assignments = dict(
            line.split(" ", 1)
            for line in (stub_dir / "setsecrets.log").read_text().splitlines()
        )
        live = assignments["u-rdq-exec-live"].split(",")
        assert "s-live" in live
        assert "s-paper" not in live, "live identity must never hold the paper host"
        for uuid, joined in assignments.items():
            if uuid != "u-rdq-exec-live":
                assert "s-live" not in joined.split(","), (
                    f"live secret leaked to {uuid}"
                )

    def test_missing_live_secret_warns_but_succeeds(self, tmp_path: Path) -> None:
        """Before the operator vaults live keys the secret is just missing —
        the script must WARN and finish, exactly like other missing hosts."""
        env = make_stub_box(tmp_path, live_agent=False, live_secret_vaulted=False)
        result = run_script(SETUP, env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "no vault secret for host 'api.alpaca.markets'" in result.stderr
        stub_dir = Path(env["ONECLI_STUB_DIR"])
        assignments = dict(
            line.split(" ", 1)
            for line in (stub_dir / "setsecrets.log").read_text().splitlines()
        )
        assert assignments["u-rdq-exec-live"].split(",") == ["s-fmp"]

    def test_rerun_with_existing_live_identity_is_idempotent(self, tmp_path: Path) -> None:
        env = make_stub_box(tmp_path)
        result = run_script(SETUP, env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "exists:  rdq-exec-live" in result.stdout
        assert not (Path(env["ONECLI_STUB_DIR"]) / "created.log").exists()


class TestCheckOnecliIsolation:
    def test_all_four_directions_pass_when_armed(self, tmp_path: Path) -> None:
        env = make_stub_box(tmp_path)
        result = run_script(CHECK, env)
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert "PASS  rdq-exec-paper -> api.alpaca.markets (live) isolation  (HTTP 401" in out
        assert "PASS  rdq-orchestrator -> api.alpaca.markets (live) isolation  (HTTP 401" in out
        assert "PASS  rdq-exec-live -> paper-api.alpaca.markets isolation  (HTTP 401" in out
        assert "PASS  rdq-exec-live -> api.alpaca.markets (live) auth  (HTTP 200)" in out
        assert "SKIP" not in out

    @pytest.mark.parametrize("assigned", [False, True], ids=["unvaulted", "unassigned"])
    def test_live_auth_skipped_with_warning_before_operator_step(
        self, tmp_path: Path, assigned: bool
    ) -> None:
        """No live secret yet (or vaulted but not assigned): the auth probe
        SKIPs with a warning, the must-fail directions still run, exit 0."""
        env = make_stub_box(
            tmp_path, live_secret_vaulted=assigned, live_secret_assigned=False
        )
        result = run_script(CHECK, env)
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert "SKIP  rdq-exec-live -> api.alpaca.markets (live) auth" in out
        assert "WARN" in out
        assert "PASS  rdq-exec-paper -> api.alpaca.markets (live) isolation" in out
        assert "PASS  rdq-exec-live -> paper-api.alpaca.markets isolation" in out

    @pytest.mark.parametrize("identity", ["rdq-exec-paper", "rdq-orchestrator"])
    def test_live_host_reachable_from_other_identity_is_hard_failure(
        self, tmp_path: Path, identity: str
    ) -> None:
        env = make_stub_box(tmp_path, code_overrides={(identity, LIVE_URL): "200"})
        result = run_script(CHECK, env)
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"REACHABLE FROM {identity}" in result.stdout

    def test_paper_host_reachable_from_live_identity_is_hard_failure(
        self, tmp_path: Path
    ) -> None:
        env = make_stub_box(tmp_path, code_overrides={("rdq-exec-live", PAPER_URL): "200"})
        result = run_script(CHECK, env)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "REACHABLE FROM rdq-exec-live" in result.stdout

    def test_missing_live_identity_fails_pointing_at_setup(self, tmp_path: Path) -> None:
        env = make_stub_box(tmp_path, live_agent=False)
        result = run_script(CHECK, env)
        assert result.returncode == 1, result.stdout + result.stderr
        assert result.stdout.count("identity not registered") >= 2  # isolation + auth
        assert "ops/setup_onecli.sh" in result.stdout

    def test_rejected_live_credential_fails_not_skips(self, tmp_path: Path) -> None:
        """Secret vaulted + assigned but the live host still 401s: that is a
        real failure (bad key), never a skip."""
        env = make_stub_box(tmp_path, code_overrides={("rdq-exec-live", LIVE_URL): "401"})
        result = run_script(CHECK, env)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "live credential not injected or invalid" in result.stdout
        assert "SKIP" not in result.stdout

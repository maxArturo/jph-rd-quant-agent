"""Offline tests for ops/resize_research.sh (burst-compute resize runbook).

The script drives doctl/ssh/curl by bare name, so the end-to-end tests run
the REAL script with stateful stub binaries on PATH (tests/test_health.py
pattern): the doctl stub keeps droplet size/status in files so shutdown ->
resize -> power-on transitions behave like the DO API, without touching it.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "resize_research.sh"
RUNBOOK = REPO_ROOT / "ops" / "runbook.md"

DROPLET_ID = "573294655"

STUB_DOCTL = """#!/usr/bin/env bash
d="$RESIZE_STUB_DIR"
echo "doctl $*" >> "$d/calls"
slug="$(cat "$d/slug" 2>/dev/null || echo s-4vcpu-8gb)"
status="$(cat "$d/status" 2>/dev/null || echo active)"
# CPU/RAM-only resizes never change the droplet's disk — it stays at the
# base 160G even while on c-16 (whose PLAN disk is 200G). $d/disk simulates
# the aftermath of a (forbidden) permanent disk resize.
disk="$(cat "$d/disk" 2>/dev/null || echo 160)"
case "$*" in
  "account get")
    exit 0 ;;
  "compute droplet get "*)
    printf '[{"id": %s, "size_slug": "%s", "status": "%s", "disk": %s}]\\n' \
      "573294655" "$slug" "$status" "$disk" ;;
  "compute size list -o json")
    cat <<'EOF'
[{"slug":"s-4vcpu-8gb","vcpus":4,"memory":8192,"disk":160,"price_hourly":0.07143,"price_monthly":48.0},
 {"slug":"c-8","vcpus":8,"memory":16384,"disk":100,"price_hourly":0.25,"price_monthly":168.0},
 {"slug":"c-16","vcpus":16,"memory":32768,"disk":200,"price_hourly":0.5,"price_monthly":336.0}]
EOF
    ;;
  "compute droplet-action shutdown "*|"compute droplet-action power-off "*)
    echo off > "$d/status" ;;
  "compute droplet-action resize "*)
    prev=""
    for arg in "$@"; do
      if [[ "$prev" == "--size" ]]; then echo "$arg" > "$d/slug"; fi
      prev="$arg"
    done ;;
  "compute droplet-action power-on "*)
    echo active > "$d/status" ;;
esac
exit 0
"""

STUB_SSH = """#!/usr/bin/env bash
d="$RESIZE_STUB_DIR"
echo "ssh $*" >> "$d/calls"
if [[ "$*" == *local_qlib* && -f "$d/run_active" ]]; then echo "abc123"; fi
exit 0
"""

# The metadata self-guard: prints the droplet id when $d/self_id exists (we
# ARE the droplet), otherwise fails like curl does off-cloud.
STUB_CURL = """#!/usr/bin/env bash
d="$RESIZE_STUB_DIR"
if [[ -f "$d/self_id" ]]; then cat "$d/self_id"; exit 0; fi
exit 1
"""


def make_stubs(stub_dir: Path) -> None:
    for name, body in (("doctl", STUB_DOCTL), ("ssh", STUB_SSH), ("curl", STUB_CURL)):
        stub = stub_dir / name
        stub.write_text(body)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)


def run_script(*args: str, stub_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RESIZE_STUB_DIR"] = str(stub_dir)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["RDQ_POLL_SECS"] = "0"
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, check=False, env=env
    )


@pytest.fixture()
def stub_dir(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir()
    make_stubs(d)
    return d


def calls(stub_dir: Path) -> str:
    f = stub_dir / "calls"
    return f.read_text() if f.exists() else ""


class TestScriptContract:
    def test_exists_and_executable(self) -> None:
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), "script must be executable"

    def test_never_invokes_resize_disk(self) -> None:
        """The one hard rule: a disk resize is permanent and would block ever
        resizing back down. The flag may only appear in comments."""
        for line in SCRIPT.read_text().splitlines():
            code = line.split("#", 1)[0]
            assert "--resize-disk" not in code

    def test_usage_without_command(self, stub_dir: Path) -> None:
        result = run_script(stub_dir=stub_dir)
        assert result.returncode == 2

    def test_runbook_documents_burst_resize(self) -> None:
        runbook = RUNBOOK.read_text()
        assert "Burst compute" in runbook
        assert "resize_research.sh" in runbook
        assert "research_caps.sh" in runbook
        assert "--resize-disk" in runbook


class TestGuards:
    def test_refuses_to_run_on_the_droplet_itself(self, stub_dir: Path) -> None:
        (stub_dir / "self_id").write_text(DROPLET_ID)
        result = run_script("up", stub_dir=stub_dir)
        assert result.returncode != 0
        assert "refusing" in result.stderr
        assert "droplet-action" not in calls(stub_dir)

    def test_refuses_resize_over_active_run(self, stub_dir: Path) -> None:
        (stub_dir / "run_active").touch()
        result = run_script("up", stub_dir=stub_dir)
        assert result.returncode != 0
        assert "active" in result.stderr
        assert "shutdown" not in calls(stub_dir)

    def test_force_overrides_active_run_guard(self, stub_dir: Path) -> None:
        (stub_dir / "run_active").touch()
        result = run_script("up", "--force", stub_dir=stub_dir)
        assert result.returncode == 0
        assert "resize 573294655 --size c-16" in calls(stub_dir)

    def test_rejects_target_with_smaller_disk(self, stub_dir: Path) -> None:
        result = run_script("up", "c-8", stub_dir=stub_dir)
        assert result.returncode != 0
        assert "disk" in result.stderr
        assert "droplet-action" not in calls(stub_dir)

    def test_down_refused_after_a_permanent_disk_resize(self, stub_dir: Path) -> None:
        """If the disk was ever grown (the forbidden flavor), DO can no longer
        resize back to the base plan — the script must say so, not try."""
        (stub_dir / "slug").write_text("c-16\n")
        (stub_dir / "disk").write_text("200\n")
        result = run_script("down", stub_dir=stub_dir)
        assert result.returncode != 0
        assert "disk" in result.stderr
        assert "droplet-action" not in calls(stub_dir)

    def test_rejects_unknown_size_slug(self, stub_dir: Path) -> None:
        result = run_script("up", "c-9000", stub_dir=stub_dir)
        assert result.returncode != 0
        assert "unknown size" in result.stderr


class TestFlows:
    def test_up_shuts_down_resizes_powers_on_and_rederives_caps(
        self, stub_dir: Path
    ) -> None:
        result = run_script("up", stub_dir=stub_dir)
        assert result.returncode == 0, result.stderr
        log = calls(stub_dir)
        shutdown = log.index(f"droplet-action shutdown {DROPLET_ID} --wait")
        resize = log.index(f"droplet-action resize {DROPLET_ID} --size c-16 --wait")
        power_on = log.index(f"droplet-action power-on {DROPLET_ID} --wait")
        assert shutdown < resize < power_on
        assert "--resize-disk" not in log
        assert "research_caps.sh" in log, "must re-derive caps after the resize"
        assert (stub_dir / "slug").read_text().strip() == "c-16"

    def test_down_returns_to_base_size(self, stub_dir: Path) -> None:
        (stub_dir / "slug").write_text("c-16\n")
        result = run_script("down", stub_dir=stub_dir)
        assert result.returncode == 0, result.stderr
        assert f"droplet-action resize {DROPLET_ID} --size s-4vcpu-8gb --wait" in calls(
            stub_dir
        )
        assert (stub_dir / "slug").read_text().strip() == "s-4vcpu-8gb"

    def test_up_when_already_at_target_is_a_noop(self, stub_dir: Path) -> None:
        (stub_dir / "slug").write_text("c-16\n")
        result = run_script("up", stub_dir=stub_dir)
        assert result.returncode == 0
        assert "nothing to do" in result.stdout
        assert "droplet-action" not in calls(stub_dir)

    def test_status_reports_size_and_run_state(self, stub_dir: Path) -> None:
        result = run_script("status", stub_dir=stub_dir)
        assert result.returncode == 0, result.stderr
        assert "s-4vcpu-8gb" in result.stdout
        assert "idle" in result.stdout
        (stub_dir / "run_active").touch()
        result = run_script("status", stub_dir=stub_dir)
        assert "RUN ACTIVE" in result.stdout

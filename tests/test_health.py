"""Offline tests for ops/health.sh + ops/expose_traces.sh (US-042).

health.sh shells out to systemctl / ss / tailscale by bare name, so the
end-to-end tests run the REAL script with stub binaries on PATH simulating a
healthy box and each failure mode (including exit 0, which the live box
cannot show while operator blockers are open).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTH = REPO_ROOT / "ops" / "health.sh"
EXPOSE = REPO_ROOT / "ops" / "expose_traces.sh"
INSTALL = REPO_ROOT / "ops" / "install_services.sh"
RUNBOOK = REPO_ROOT / "ops" / "runbook.md"

LONG_RUNNING = ["rdq-orchestrator.service", "rdq-research.service"]
TIMERS = [
    "rdq-data-refresh.timer",
    "rdq-pred-refresh.timer",
    "rdq-rebalance.timer",
    "rdq-sweep.timer",
]
ONESHOTS = [
    "rdq-data-refresh.service",
    "rdq-pred-refresh.service",
    "rdq-rebalance.service",
    "rdq-sweep.service",
]

HEALTHY_SS = (
    "LISTEN 0 128 127.0.0.1:19899 0.0.0.0:* users:((\"python\",pid=4242,fd=3))\n"
    "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:* users:((\"sshd\",pid=1,fd=3))\n"
    "LISTEN 0 4096 100.68.23.10:443 0.0.0.0:*\n"
    "LISTEN 0 4096 100.68.23.10:3100 0.0.0.0:*\n"
)
HEALTHY_SERVE = (
    "https://box.tail.ts.net:3100 (tailnet only)\n"
    "|-- / proxy http://127.0.0.1:3001\n"
    "\n"
    "https://box.tail.ts.net (tailnet only)\n"
    "|-- / proxy http://127.0.0.1:10254\n"
)

STUB_SYSTEMCTL = """#!/usr/bin/env bash
# stub systemctl --user <verb> ... driven by files in $HEALTH_STUB_DIR
verb=$2
unit=$3
case "$verb" in
  is-active)
    f="$HEALTH_STUB_DIR/active_$unit"
    if [[ -f "$f" ]]; then cat "$f"; else echo active; fi ;;
  is-failed)
    f="$HEALTH_STUB_DIR/failed_$unit"
    if [[ -f "$f" ]]; then cat "$f"; exit 0; fi
    echo inactive; exit 1 ;;
  show)
    case "$5" in
      ControlGroup) echo "" ;;
      MainPID)
        f="$HEALTH_STUB_DIR/mainpid_$unit"
        if [[ -f "$f" ]]; then cat "$f"; else echo 0; fi ;;
    esac ;;
esac
"""

STUB_SS = """#!/usr/bin/env bash
cat "$HEALTH_STUB_DIR/ss.txt"
"""

STUB_TAILSCALE = """#!/usr/bin/env bash
cat "$HEALTH_STUB_DIR/serve.txt"
"""


def make_stub_box(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A PATH-shimmed healthy box; tests then break individual pieces."""
    bin_dir = tmp_path / "bin"
    stub_dir = tmp_path / "stub"
    bin_dir.mkdir()
    stub_dir.mkdir()
    for name, body in [
        ("systemctl", STUB_SYSTEMCTL),
        ("ss", STUB_SS),
        ("tailscale", STUB_TAILSCALE),
    ]:
        script = bin_dir / name
        script.write_text(body)
        script.chmod(0o755)
    (stub_dir / "ss.txt").write_text(HEALTHY_SS)
    (stub_dir / "serve.txt").write_text(HEALTHY_SERVE)
    (stub_dir / "mainpid_rdq-research.service").write_text("4242\n")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HEALTH_STUB_DIR": str(stub_dir),
        "XDG_RUNTIME_DIR": "/tmp",
        "HOME": str(tmp_path),
    }
    return stub_dir, env


def run_health(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HEALTH)], capture_output=True, text=True, env=env, check=False
    )


class TestScripts:
    def test_exist_and_executable(self) -> None:
        for script in (HEALTH, EXPOSE):
            assert script.is_file()
            assert script.stat().st_mode & 0o111, f"{script.name} not executable"

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    @pytest.mark.parametrize("script", [HEALTH, EXPOSE], ids=lambda p: p.name)
    def test_shellcheck_clean(self, script: Path) -> None:
        result = subprocess.run(
            ["shellcheck", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_health_covers_every_installed_unit(self) -> None:
        """The three unit lists in health.sh must cover install_services.sh UNITS."""
        install_units = set(
            re.findall(r"^\s+(rdq-[a-z-]+\.(?:service|timer))$", INSTALL.read_text(), re.MULTILINE)
        )
        assert install_units, "failed to parse UNITS from install_services.sh"
        health_units = set(
            re.findall(r"^\s+(rdq-[a-z-]+\.(?:service|timer))$", HEALTH.read_text(), re.MULTILINE)
        )
        assert install_units <= health_units, (
            f"health.sh misses units: {install_units - health_units}"
        )
        assert health_units == set(LONG_RUNNING + TIMERS + ONESHOTS)

    def test_expose_traces_exact_mapping_command(self) -> None:
        text = EXPOSE.read_text()
        assert 'tailscale serve --bg --https="${PORT}" "${TARGET}"' in text
        assert "PORT=19900" in text
        assert 'TARGET="http://127.0.0.1:${PORT}"' in text
        assert "funnel" in text  # verifies the never-funnel guard exists

    def test_runbook_references_scripts(self) -> None:
        runbook = RUNBOOK.read_text()
        assert "expose_traces.sh" in runbook
        assert "health.sh" in runbook


class TestHealthyBox:
    def test_exits_zero(self, tmp_path: Path) -> None:
        _, env = make_stub_box(tmp_path)
        result = run_health(env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "HEALTHY" in result.stdout
        assert "FAIL" not in result.stdout

    def test_allowed_19900_serve_and_tailnet_bind_stay_healthy(self, tmp_path: Path) -> None:
        """Regression: tailscaled terminating :19900 on the tailnet IP is sanctioned."""
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "serve.txt").write_text(
            HEALTHY_SERVE
            + "\nhttps://box.tail.ts.net:19900 (tailnet only)\n"
            + "|-- / proxy http://127.0.0.1:19900\n"
        )
        (stub_dir / "ss.txt").write_text(
            HEALTHY_SS
            + "LISTEN 0 4096 100.68.23.10:19900 0.0.0.0:*\n"
            + "LISTEN 0 4096 [fd7a:115c:a1e0::1:1]:19900 [::]:*\n"
        )
        result = run_health(env)
        assert result.returncode == 0, result.stdout + result.stderr


class TestFailureModes:
    def test_inactive_service_named(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "active_rdq-orchestrator.service").write_text("inactive\n")
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  service rdq-orchestrator.service" in result.stdout
        assert "service rdq-orchestrator.service" in result.stdout.split("check(s) failed")[-1]

    def test_inactive_timer_named(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "active_rdq-rebalance.timer").write_text("inactive\n")
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  timer rdq-rebalance.timer" in result.stdout

    def test_failed_oneshot_named_with_journal_hint(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "failed_rdq-rebalance.service").write_text("failed\n")
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  oneshot rdq-rebalance.service" in result.stdout
        assert "journalctl --user -u rdq-rebalance.service" in result.stdout

    def test_rdq_process_on_non_loopback_fails(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "ss.txt").write_text(
            'LISTEN 0 128 0.0.0.0:19899 0.0.0.0:* users:(("python",pid=4242,fd=3))\n'
        )
        result = run_health(env)
        assert result.returncode == 1
        assert "loopback" in result.stdout
        assert "0.0.0.0:19899" in result.stdout

    def test_repo_port_on_all_interfaces_fails_even_unowned(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "ss.txt").write_text(HEALTHY_SS + "LISTEN 0 128 0.0.0.0:19900 0.0.0.0:*\n")
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  loopback port 19900" in result.stdout

    def test_19899_tailnet_bind_fails(self, tmp_path: Path) -> None:
        """server_ui has no allowed serve mapping — even a tailnet bind fails."""
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "ss.txt").write_text(HEALTHY_SS + "LISTEN 0 128 100.68.23.10:19899 0.0.0.0:*\n")
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  loopback port 19899" in result.stdout

    def test_19899_in_serve_status_fails(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "serve.txt").write_text(
            HEALTHY_SERVE
            + "\nhttps://box.tail.ts.net:19899 (tailnet only)\n"
            + "|-- / proxy http://127.0.0.1:19899\n"
        )
        result = run_health(env)
        assert result.returncode == 1
        assert "19899" in result.stdout
        assert "never be exposed" in result.stdout

    def test_funnel_fails(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "serve.txt").write_text(
            "https://box.tail.ts.net:3100 (Funnel on)\n|-- / proxy http://127.0.0.1:3001\n"
        )
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  tailscale funnel" in result.stdout

    def test_unknown_serve_mapping_fails(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "serve.txt").write_text(
            HEALTHY_SERVE
            + "\nhttps://box.tail.ts.net:8443 (tailnet only)\n"
            + "|-- / proxy http://127.0.0.1:8443\n"
        )
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  tailscale serve :8443" in result.stdout
        assert "not in the PLAN.md port table" in result.stdout

    def test_wrong_target_for_allowed_port_fails(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "serve.txt").write_text(
            "https://box.tail.ts.net:3100 (tailnet only)\n|-- / proxy http://127.0.0.1:9999\n"
            "\nhttps://box.tail.ts.net (tailnet only)\n|-- / proxy http://127.0.0.1:10254\n"
        )
        result = run_health(env)
        assert result.returncode == 1
        assert "FAIL  tailscale serve :3100" in result.stdout

    def test_multiple_failures_all_listed(self, tmp_path: Path) -> None:
        stub_dir, env = make_stub_box(tmp_path)
        (stub_dir / "active_rdq-orchestrator.service").write_text("inactive\n")
        (stub_dir / "failed_rdq-rebalance.service").write_text("failed\n")
        result = run_health(env)
        assert result.returncode == 1
        summary = result.stdout.split("check(s) failed")[-1]
        assert "service rdq-orchestrator.service" in summary
        assert "oneshot rdq-rebalance.service" in summary

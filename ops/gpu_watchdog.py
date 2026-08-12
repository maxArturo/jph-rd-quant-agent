"""Orphaned GPU-worker guard: a crashed pipeline must not leak $/hr forever.

Runs hourly from rdq-gpu-watchdog.timer. Stateless decision per tick:

- no worker state file            -> exit 0 (nothing provisioned)
- droplet gone from DO            -> clean up the stale state file, exit 0
- droplet age > --max-hours       -> DESTROY (fetch results first, best-effort)
- research tmux session dead      -> warn to Slack (a healthy pipeline
  destroys the worker minutes after the run ends, so a lingering idle
  droplet means the driver died); destruction still waits for max-hours so
  a between-stages pipeline is never yanked.

Alerts post to the channel that owns the run — read from the pipeline's
status file (US-017) — falling back to the paper research channel when the
owner is unknown. All lifecycle mechanics go through
ops/gpu_worker/gpu_worker.sh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.gpu_backend import STATUS_FILE as DEFAULT_STATUS_FILE

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_WORKER = REPO_ROOT / "ops" / "gpu_worker" / "gpu_worker.sh"
DEFAULT_STATE_FILE = Path.home() / "rdq-runs" / "gpu_worker" / "worker.env"


def read_state(state_file: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    for line in state_file.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            state[key.strip()] = value.strip()
    return state


def droplet_age_hours(created_at: str, now: datetime) -> float | None:
    try:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (now - created).total_seconds() / 3600


def owning_channel(status_file: Path) -> str | None:
    """The Slack channel recorded in the pipeline's status file, else None."""
    try:
        data = json.loads(status_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    channel = data.get("channel")
    return channel if isinstance(channel, str) and channel else None


def notify(text: str, no_slack: bool, channel: str | None = None) -> None:
    print(text, file=sys.stderr)
    if no_slack:
        return
    try:
        if channel is not None:
            from slack_sdk import WebClient

            from orchestrator.config import load_slack_config

            client = WebClient(token=load_slack_config().bot_token)
            # slack_sdk loads HTTPS_PROXY and ignores NO_PROXY — never proxy Slack.
            client.proxy = None
            client.chat_postMessage(channel=channel, text=text)
            return
        # Owner unknown — the paper research channel is the fallback home.
        from execution.rebalance import slack_notifier

        slack_notifier()(text)
    except Exception as exc:  # noqa: BLE001 — the watchdog must never die on Slack
        print(f"slack notice failed ({exc})", file=sys.stderr)


def worker_sh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(GPU_WORKER), *args], capture_output=True, text=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATUS_FILE,
        help="pipeline status JSON naming the channel that owns the run",
    )
    parser.add_argument("--no-slack", action="store_true")
    args = parser.parse_args(argv)

    if not args.state_file.is_file():
        return 0
    state = read_state(args.state_file)
    droplet_id = state.get("DROPLET_ID", "?")
    channel = owning_channel(args.status_file)

    exists = subprocess.run(
        ["doctl", "compute", "droplet", "get", droplet_id, "--format", "ID", "--no-header"],
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode != 0:
        # Droplet already gone (destroyed elsewhere / API says 404): drop the
        # stale state so provision doesn't refuse. Auth errors also land here —
        # deleting state is safe either way (provision re-checks live droplets).
        args.state_file.unlink(missing_ok=True)
        print(f"stale state for droplet {droplet_id} removed", file=sys.stderr)
        return 0

    age = droplet_age_hours(state.get("CREATED_AT", ""), datetime.now(timezone.utc))
    size = state.get("SIZE", "?")
    if age is not None and age > args.max_hours:
        notify(
            f":rotating_light: GPU worker {droplet_id} ({size}) exceeded {args.max_hours:.0f}h — "
            "watchdog fetching results and destroying it",
            args.no_slack,
            channel,
        )
        worker_sh("fetch")  # best-effort; results may already be gone
        destroy = worker_sh("destroy", "--force")
        if destroy.returncode == 0:
            notify(
                ":wastebasket: watchdog destroyed the orphaned worker — billing stopped",
                args.no_slack,
                channel,
            )
            return 0
        notify(
            f":rotating_light: watchdog could NOT destroy droplet {droplet_id} — "
            "it is still billing; run ops/gpu_worker/gpu_worker.sh destroy --force",
            args.no_slack,
            channel,
        )
        return 1

    run_alive = worker_sh("ssh", "tmux has-session -t rdq-run 2>/dev/null").returncode == 0
    if not run_alive:
        age_text = f" (age {age:.1f}h)" if age is not None else ""
        notify(
            f":warning: GPU worker {droplet_id} ({size}) is up with NO active research "
            f"run{age_text} — a healthy pipeline tears down promptly; if nothing is "
            "using it: ops/gpu_worker/gpu_worker.sh destroy",
            args.no_slack,
            channel,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Slack configuration loading for the orchestrator.

Token source (docs/decisions.md, 2026-07-08): Slack chat tokens live in the
repo-root ``.env`` (gitignored) — a PLAN.md-sanctioned exception to the
everything-through-OneCLI rule. Process environment takes precedence over the
file so systemd/CI can override without editing ``.env``.

Required variables:
- ``SLACK_OAUTH_TOKEN``  — xoxb- bot token (Web API calls)
- ``SLACK_SOCKET_TOKEN`` — xapp- app-level token (Socket Mode websocket)
- ``SLACK_CHANNEL_ID``   — channel id of #quant-research (e.g. C0123456789)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# OneCLI management API (approvals bridge, US-039). Local, unauthenticated.
DEFAULT_ONECLI_URL = "http://127.0.0.1:10254"


class ConfigError(RuntimeError):
    """A required configuration value is missing or malformed."""


@dataclass(frozen=True)
class SlackConfig:
    bot_token: str  # xoxb- (SLACK_OAUTH_TOKEN)
    app_token: str  # xapp- (SLACK_SOCKET_TOKEN)
    channel_id: str  # SLACK_CHANNEL_ID


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file (no interpolation, no multiline).

    Blank lines and ``#`` comments are ignored; an optional ``export `` prefix
    and single/double quotes around the value are stripped.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _require(
    name: str,
    environ: Mapping[str, str],
    file_values: Mapping[str, str],
    env_file: Path,
    hint: str,
) -> str:
    value = environ.get(name) or file_values.get(name) or ""
    if not value:
        raise ConfigError(
            f"{name} is not set. Set it in the process environment or in "
            f"{env_file} ({hint})."
        )
    return value


def load_slack_config(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> SlackConfig:
    """Load Slack tokens + target channel; raise ConfigError naming what's missing."""
    env = os.environ if environ is None else environ
    file_values = parse_env_file(env_file)

    bot_token = _require(
        "SLACK_OAUTH_TOKEN", env, file_values, env_file, "xoxb- bot token from the Slack app"
    )
    app_token = _require(
        "SLACK_SOCKET_TOKEN", env, file_values, env_file, "xapp- app-level Socket Mode token"
    )
    channel_id = _require(
        "SLACK_CHANNEL_ID",
        env,
        file_values,
        env_file,
        "channel id of #quant-research, e.g. C0123456789 — copy it from the "
        "channel's 'View channel details' pane",
    )

    if not bot_token.startswith("xoxb-"):
        raise ConfigError("SLACK_OAUTH_TOKEN must be an xoxb- bot token (got a non-xoxb value).")
    if not app_token.startswith("xapp-"):
        raise ConfigError(
            "SLACK_SOCKET_TOKEN must be an xapp- app-level token (got a non-xapp value)."
        )
    return SlackConfig(bot_token=bot_token, app_token=app_token, channel_id=channel_id)


def load_onecli_url(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> str:
    """OneCLI management API base URL (ONECLI_URL; defaults to the local gateway)."""
    env = os.environ if environ is None else environ
    file_values = parse_env_file(env_file)
    return env.get("ONECLI_URL") or file_values.get("ONECLI_URL") or DEFAULT_ONECLI_URL

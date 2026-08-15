"""OnFailure Slack notifier for rdq systemd user units (US-018).

Instantiated as ``rdq-notify-failure@<failed-unit>.service`` by the
``OnFailure=rdq-notify-failure@%n.service`` line every rdq service carries.
Posts "unit <name> failed" plus the failed unit's last journal lines to the
ops Slack channel through the same repo-.env ``slack_notifier`` the
rebalancer uses (Slack never transits the OneCLI proxy —
docs/decisions.md 2026-07-08).

The template unit itself carries NO OnFailure line on purpose: a notifier
that cannot notify must die quietly in the journal, never recurse.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

JOURNAL_LINES = 10
# Slack renders long messages fine but truncates around 4k chars in-channel;
# keep the tail comfortably under that, dropping the OLDEST lines first.
MAX_TAIL_CHARS = 3500


def journal_tail(unit: str, lines: int = JOURNAL_LINES) -> str:
    """Last ``lines`` journal lines of ``unit``, or a placeholder explaining why not.

    Never raises: the notifier must still post the failure headline when the
    journal itself is unreadable.
    """
    try:
        result = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                unit,
                "-n",
                str(lines),
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(journal unavailable: {exc})"
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return f"(journalctl failed: {detail})"
    tail = result.stdout.strip()
    if not tail:
        return "(journal empty)"
    if len(tail) > MAX_TAIL_CHARS:
        tail = "…" + tail[-MAX_TAIL_CHARS:]
    return tail


def build_message(unit: str, tail: str) -> str:
    return (
        f":rotating_light: unit {unit} failed — last journal lines:\n"
        f"```\n{tail}\n```\n"
        f"journalctl --user -u {unit}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unit", help="failed unit name (systemd %%i instance)")
    parser.add_argument("--lines", type=int, default=JOURNAL_LINES)
    parser.add_argument("--no-slack", action="store_true", help="print to stderr only")
    args = parser.parse_args(argv)

    message = build_message(args.unit, journal_tail(args.unit, args.lines))
    print(message, file=sys.stderr)
    if args.no_slack:
        return 0
    try:
        from execution.rebalance import slack_notifier

        slack_notifier()(message)
    except Exception as exc:  # noqa: BLE001 — nothing left to notify with; journal is the record
        print(f"slack notice failed ({exc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

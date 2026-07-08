"""Loopback launcher for the rdagent server_ui control plane (US-018).

Upstream ``rdagent server_ui`` (``rdagent/log/server/app.py`` ``main()``)
binds Flask to host 0.0.0.0 — but PLAN.md requires every service on this box
to bind 127.0.0.1 (server_ui carries flask-cors advisories and must stay
dark), and the pinned upstream tree must remain unmodified
(tests/test_us_templates.py hashes it against pip's RECORD). This module
replicates upstream ``main()`` with the bind host forced to loopback.

Run by ops/rdq-research.service as:

    onecli run --agent rdq-research -- .venv/bin/python -m research.server_ui

Trace/static locations come from the ``UI_`` env vars (``UI_TRACE_FOLDER``,
``UI_STATIC_PATH``) read by ``rdagent.log.ui.conf`` at import time; the
systemd unit points them under ~/rdq-runs/server_ui/. Both directories are
created here so first boot on a fresh box works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19899


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Mirror of upstream ``rdagent.log.server.app.main`` bound to *host*."""
    # rdagent imports take seconds — keep them inside main() (repo convention)
    # so offline tests can monkeypatch module attributes before this binds.
    from rdagent.log.server.app import _load_existing_traces, app, log_folder_path

    log_folder_path.mkdir(parents=True, exist_ok=True)
    if app.static_folder:
        Path(app.static_folder).mkdir(parents=True, exist_ok=True)
    app.config["UI_SERVER_PORT"] = port
    _load_existing_traces(log_folder_path)
    app.run(debug=False, host=host, port=port)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rdagent server_ui bound to loopback (US-018)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind host (default %(default)s)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="bind port (default %(default)s)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(host=args.host, port=args.port)

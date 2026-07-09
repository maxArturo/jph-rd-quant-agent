"""Loopback launcher for the rdagent server_ui control plane (US-018/US-024).

Upstream ``rdagent server_ui`` (``rdagent/log/server/app.py`` ``main()``)
binds Flask to host 0.0.0.0 — but PLAN.md requires every service on this box
to bind 127.0.0.1 (server_ui carries flask-cors advisories and must stay
dark), and the pinned upstream tree must remain unmodified
(tests/test_us_templates.py hashes it against pip's RECORD). This module
replicates upstream ``main()`` with the bind host forced to loopback.

It also extends the upstream ``/control`` endpoint with a ``resume`` action
(US-024) — upstream only implements ``stop``. Resume relaunches a stopped
run's target from its dumped ``__session__`` checkpoints (``RDLoop.load``,
same mechanism as the upstream CLI's ``path=`` argument) under the SAME
trace id, so messages, polling, and artifact resolution continue seamlessly.
The extension wraps the registered Flask view at runtime; the pinned tree on
disk stays untouched.

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
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19899

# Scenarios whose target main() accepts a session ``path`` to resume from
# (fire CLI signature in rdagent/app/qlib_rd_loop/{quant,factor,model}.py).
RESUMABLE_TARGETS = {
    "Finance Whole Pipeline": "fin_quant",
    "Finance Data Building": "fin_factor",
    "Finance Model Implementation": "fin_model",
}

_RESUME_FLAG = "RDQ_RESUME_EXTENSION"


def _session_problem(session_path: Path) -> str | None:
    """Why *session_path* cannot be resumed from, or None when it can.

    Mirrors what ``RDLoop.load`` accepts: a trace dir containing
    ``__session__``, the ``__session__`` dir itself, or one dumped step file.
    """
    if session_path.is_file():
        return None
    if not session_path.is_dir():
        return f"no session to resume: {session_path} does not exist"
    folder = session_path if session_path.name == "__session__" else session_path / "__session__"
    if not folder.is_dir():
        return f"no session to resume: {folder} does not exist"
    if not any(folder.glob("*/*_*")):
        return f"no session to resume: {folder} contains no dumped loop steps"
    return None


def install_resume_control() -> None:
    """Wrap upstream /control with a ``resume`` action (idempotent).

    Runtime view wrapping only — the pinned rdagent tree on disk is never
    modified. Non-resume actions delegate to the upstream handler unchanged.
    """
    from flask import jsonify, request
    from rdagent.log.server import app as server_app

    app = server_app.app
    if app.config.get(_RESUME_FLAG):
        return
    upstream_control = app.view_functions["control_process"]

    def control_process_with_resume() -> Any:
        data = request.get_json(silent=True) or {}
        if data.get("action") != "resume":
            return upstream_control()
        trace_id = str(data.get("id") or "")
        if not trace_id:
            return jsonify({"error": "Missing 'id' or 'action' in request"}), 400
        scenario, _, trace_name = trace_id.partition("/")
        target_name = RESUMABLE_TARGETS.get(scenario)
        if target_name is None or not trace_name:
            return (
                jsonify(
                    {
                        "error": f"cannot resume trace id {trace_id!r}: scenario must be one"
                        f" of {sorted(RESUMABLE_TARGETS)}"
                    }
                ),
                400,
            )
        # log_folder_path/rdagent_processes/RDAgentTask are read through the
        # module at call time so tests can monkeypatch them.
        trace_root = Path(server_app.log_folder_path)
        full_id = str(trace_root / trace_id)
        existing = server_app.rdagent_processes.get(full_id)
        if existing is not None and existing.is_alive():
            return (
                jsonify({"error": "process is still running; stop it before resuming"}),
                400,
            )
        session_path = Path(str(data.get("path") or full_id)).expanduser()
        try:
            session_path.resolve().relative_to(trace_root.resolve())
        except ValueError:
            return (
                jsonify(
                    {"error": f"session path must live under the trace folder {trace_root}"}
                ),
                400,
            )
        problem = _session_problem(session_path)
        if problem is not None:
            return jsonify({"error": problem}), 400

        kwargs: dict[str, Any] = {"path": str(session_path)}
        if data.get("loops"):
            kwargs["loop_n"] = int(data["loops"])
        if data.get("all_duration"):
            kwargs["all_duration"] = f"{data['all_duration']}h"  # upstream appends "h" too
        task = server_app.RDAgentTask(
            target_name=target_name,
            kwargs=kwargs,
            stdout_path=str(trace_root / scenario / f"{trace_name}.log"),
            log_trace_path=full_id,
            scenario=scenario,
            trace_name=trace_name,
            ui_server_port=app.config.get("UI_SERVER_PORT"),
        )
        if existing is not None:
            # Continue the message history, but drop END markers — the run is
            # live again and a stale END would read as instantly finished.
            task.messages = [m for m in existing.messages if m.get("tag") != "END"]
        task.start()
        server_app.rdagent_processes[full_id] = task
        app.logger.warning(f"Task {full_id} resumed from {session_path}.")
        return jsonify({"status": "resumed", "id": trace_id}), 200

    app.view_functions["control_process"] = control_process_with_resume
    app.config[_RESUME_FLAG] = True


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Mirror of upstream ``rdagent.log.server.app.main`` bound to *host*."""
    # rdagent imports take seconds — keep them inside main() (repo convention)
    # so offline tests can monkeypatch module attributes before this binds.
    from rdagent.log.server.app import _load_existing_traces, app, log_folder_path

    log_folder_path.mkdir(parents=True, exist_ok=True)
    if app.static_folder:
        Path(app.static_folder).mkdir(parents=True, exist_ok=True)
    app.config["UI_SERVER_PORT"] = port
    install_resume_control()
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

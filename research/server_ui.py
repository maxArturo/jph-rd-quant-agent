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

Per-run custom universes (US-023 completion): ``/upload`` accepts a
``universe`` form field and resume a ``universe`` JSON key. A non-default
universe must have been materialized by the orchestrator
(``UniverseService.materialize``: factor source under
``~/rdq-data/factor_source/<name>``, rendered templates under
``~/rdq-data/templates/<name>``) — the run is refused with a 400 otherwise,
never silently started on the default env (that silence is exactly the
2026-08-05 mislabeled-universe incident). The per-universe env
(``FACTOR_CoSTEER_DATA_FOLDER(_DEBUG)`` + ``RDQ_UNIVERSE_TEMPLATES``, read by
research/us_quant.py) is patched into ``os.environ`` around the task fork
under a lock — ``RDAgentTask.start()`` forks, so the child inherits it while
concurrent requests never observe it. The default ``us_liquid`` (or an empty
universe) changes nothing: those runs keep the service-level env.

Run by ops/rdq-research.service as:

    onecli run --agent rdq-research -- .venv/bin/python -m research.server_ui

Trace/static locations come from the ``UI_`` env vars (``UI_TRACE_FOLDER``,
``UI_STATIC_PATH``) read by ``rdagent.log.ui.conf`` at import time; the
systemd unit points them under ~/rdq-runs/server_ui/. Both directories are
created here so first boot on a fresh box works.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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
_UNIVERSE_FLAG = "RDQ_UNIVERSE_EXTENSION"

# The universe the service-level env already points at (ops/rdq-research.service
# pins FACTOR_CoSTEER_DATA_FOLDER + templates to us_liquid); requesting it (or
# no universe) applies no overrides. Artifact roots mirror
# orchestrator/universe.py's materialize layout and are env-overridable so
# tests and relocations never patch this module.
DEFAULT_UNIVERSE = "us_liquid"
DEFAULT_FACTOR_SOURCE_ROOT = Path("~/rdq-data/factor_source")
DEFAULT_TEMPLATES_ROOT = Path("~/rdq-data/templates")

# Same shape data/make_universe.py enforces on instruments names; also keeps
# the name path-safe (it is joined into filesystem paths below).
_UNIVERSE_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")

# Serializes env patching around task forks: concurrent /upload or resume
# requests must never fork under each other's overrides.
_spawn_env_lock = threading.Lock()


class UniverseEnvError(ValueError):
    """The requested universe cannot be wired (bad name / missing artifacts)."""


def universe_run_env(universe: str | None) -> dict[str, str]:
    """Env overrides for a run on *universe*; {} for the default/empty.

    Refuses (rather than falling back to the default env) when the universe's
    materialized artifacts are missing — a run silently started on the wrong
    universe is the failure mode this extension exists to kill.
    """
    name = (universe or "").strip()
    if not name or name == DEFAULT_UNIVERSE:
        return {}
    if not _UNIVERSE_NAME_RE.fullmatch(name):
        raise UniverseEnvError(
            f"invalid universe name {name!r}: lowercase letters, digits, underscores"
        )
    factor_root = Path(
        os.environ.get("RDQ_FACTOR_SOURCE_ROOT") or DEFAULT_FACTOR_SOURCE_ROOT
    ).expanduser()
    templates_root = Path(
        os.environ.get("RDQ_TEMPLATES_ROOT") or DEFAULT_TEMPLATES_ROOT
    ).expanduser()
    data_folder = factor_root / name / "data_folder"
    data_folder_debug = factor_root / name / "data_folder_debug"
    templates = templates_root / name
    missing = [
        str(p)
        for p in (
            data_folder,
            data_folder_debug,
            templates / "factor_template",
            templates / "model_template",
        )
        if not p.is_dir()
    ]
    if missing:
        raise UniverseEnvError(
            f"universe '{name}' is not materialized (missing: {', '.join(missing)})"
            " — confirm it via the orchestrator's set_universe flow first"
        )
    return {
        "RDQ_UNIVERSE": name,
        "RDQ_UNIVERSE_TEMPLATES": str(templates),
        "FACTOR_CoSTEER_DATA_FOLDER": str(data_folder),
        "FACTOR_CoSTEER_DATA_FOLDER_DEBUG": str(data_folder_debug),
    }


@contextmanager
def _patched_environ(overrides: dict[str, str]) -> Iterator[None]:
    """Apply env *overrides* for the duration of a task fork, then restore.

    RDAgentTask.start() forks, so the child snapshots os.environ at start();
    the lock keeps concurrent spawns from forking under foreign overrides.
    No-op (and no lock) when there is nothing to override.
    """
    if not overrides:
        yield
        return
    with _spawn_env_lock:
        saved = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            yield
        finally:
            for key, old in saved.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old


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


def install_universe_env() -> None:
    """Wrap upstream /upload with per-run universe env wiring (idempotent).

    Same runtime view-wrapping pattern as install_resume_control — the pinned
    rdagent tree on disk is never modified. Requests without a ``universe``
    field (or with the default) delegate unchanged.
    """
    from flask import jsonify, request
    from rdagent.log.server import app as server_app

    app = server_app.app
    if app.config.get(_UNIVERSE_FLAG):
        return
    upstream_upload = app.view_functions["upload_file"]

    def upload_with_universe() -> Any:
        try:
            overrides = universe_run_env(request.form.get("universe"))
        except UniverseEnvError as exc:
            return jsonify({"error": str(exc)}), 400
        # The upstream view constructs AND forks the task, so the whole
        # delegate call sits inside the env patch.
        with _patched_environ(overrides):
            return upstream_upload()

    app.view_functions["upload_file"] = upload_with_universe
    app.config[_UNIVERSE_FLAG] = True


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
        try:
            universe_overrides = universe_run_env(data.get("universe"))
        except UniverseEnvError as exc:
            return jsonify({"error": str(exc)}), 400

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
        with _patched_environ(universe_overrides):
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

    from research.us_validation import install_us_validation

    log_folder_path.mkdir(parents=True, exist_ok=True)
    if app.static_folder:
        Path(app.static_folder).mkdir(parents=True, exist_ok=True)
    app.config["UI_SERVER_PORT"] = port
    install_resume_control()
    install_universe_env()
    # Patch the parent before Flask starts: run processes are forked, so they
    # inherit the US feature-validation / factor-env bindings even if the
    # QLIB_QUANT_* class-path env vars are ever unset (research/us_validation.py).
    install_us_validation()
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

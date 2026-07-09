"""Tests for research/server_ui.py — the loopback server_ui launcher (US-018).

Offline tests monkeypatch the upstream Flask app; the end-to-end test boots
the real server in a subprocess on an ephemeral port and asserts both the
HTTP response and the loopback-only bind.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from research.server_ui import DEFAULT_HOST, DEFAULT_PORT, main, parse_args

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestMainBindsLoopback:
    def test_run_called_with_loopback_host(self, monkeypatch, tmp_path) -> None:
        import rdagent.log.server.app as server_app

        run_calls: list[dict] = []
        monkeypatch.setattr(server_app, "log_folder_path", tmp_path / "traces")
        monkeypatch.setattr(
            server_app.app, "run", lambda **kw: run_calls.append(kw), raising=False
        )
        main(port=12345)
        assert run_calls == [{"debug": False, "host": "127.0.0.1", "port": 12345}]

    def test_creates_trace_and_static_dirs(self, monkeypatch, tmp_path) -> None:
        import rdagent.log.server.app as server_app

        traces = tmp_path / "traces"
        static = tmp_path / "static"
        monkeypatch.setattr(server_app, "log_folder_path", traces)
        monkeypatch.setattr(server_app.app, "static_folder", str(static))
        monkeypatch.setattr(server_app.app, "run", lambda **kw: None, raising=False)
        main()
        assert traces.is_dir()
        assert static.is_dir()

    def test_loads_existing_traces_before_serving(self, monkeypatch, tmp_path) -> None:
        import rdagent.log.server.app as server_app

        order: list[str] = []
        monkeypatch.setattr(server_app, "log_folder_path", tmp_path / "traces")
        monkeypatch.setattr(
            server_app, "_load_existing_traces", lambda p: order.append(f"load:{p}")
        )
        monkeypatch.setattr(
            server_app.app, "run", lambda **kw: order.append("run"), raising=False
        )
        main()
        assert order == [f"load:{tmp_path / 'traces'}", "run"]


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.host == DEFAULT_HOST == "127.0.0.1"
        assert args.port == DEFAULT_PORT == 19899

    def test_overrides(self) -> None:
        args = parse_args(["--host", "127.0.0.1", "--port", "12346"])
        assert (args.host, args.port) == ("127.0.0.1", 12346)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestEndToEnd:
    def test_serves_http_on_loopback_only(self, tmp_path) -> None:
        """Boot the real server on an ephemeral port; GET / must answer over
        127.0.0.1 and the listening socket must not be wildcard-bound."""
        port = _free_port()
        env = dict(os.environ)
        env["UI_TRACE_FOLDER"] = str(tmp_path / "traces")
        env["UI_STATIC_PATH"] = str(tmp_path / "static")
        proc = subprocess.Popen(
            [sys.executable, "-m", "research.server_ui", "--port", str(port)],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 60  # rdagent import takes seconds
            status: int | None = None
            while time.monotonic() < deadline:
                assert proc.poll() is None, "server exited before answering"
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                    conn.request("GET", "/test")
                    status = conn.getresponse().status
                    conn.close()
                    break
                except OSError:
                    time.sleep(0.5)
            assert status == 200, f"no HTTP response on 127.0.0.1:{port}"

            # the resume extension is installed (US-024): an unknown-scenario
            # resume gets OUR error, not upstream's "Only 'stop'" rejection
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST",
                "/control",
                body=json.dumps({"id": "nope/run", "action": "resume"}),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            body = json.loads(resp.read())
            conn.close()
            assert resp.status == 400
            assert "cannot resume" in body["error"], body

            ss = subprocess.run(
                ["ss", "-ltnH"], capture_output=True, text=True, check=True
            )
            listeners = [line for line in ss.stdout.splitlines() if f":{port} " in line]
            assert listeners, f"no listener found for port {port}"
            assert all(f"127.0.0.1:{port}" in line for line in listeners), (
                f"server not loopback-bound: {listeners}"
            )
        finally:
            proc.terminate()
            proc.wait(timeout=10)


# --- /control resume extension (US-024) --------------------------------------


class FakeTask:
    """Stands in for upstream RDAgentTask so no rdagent subprocess spawns."""

    def __init__(
        self,
        *,
        target_name: str,
        kwargs: dict,
        stdout_path: str,
        log_trace_path: str,
        scenario: str,
        trace_name: str,
        ui_server_port: int | None = None,
        create_process: bool = True,
    ) -> None:
        self.target_name = target_name
        self.kwargs = kwargs
        self.stdout_path = stdout_path
        self.log_trace_path = log_trace_path
        self.scenario = scenario
        self.trace_name = trace_name
        self.ui_server_port = ui_server_port
        self.create_process = create_process
        self.messages: list[dict] = []
        self.alive = False
        self.started = False

    def is_alive(self) -> bool:
        return self.alive

    def start(self) -> None:
        self.started = True


class TestResumeControl:
    TRACE_ID = "Finance Whole Pipeline/run1"

    def _setup(self, monkeypatch, tmp_path):
        """Install the extension and point the upstream app at a tmp world."""
        import rdagent.log.server.app as server_app

        from research.server_ui import install_resume_control

        install_resume_control()
        install_resume_control()  # idempotent — a double install must not re-wrap
        traces = tmp_path / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(server_app, "log_folder_path", traces)
        monkeypatch.setattr(server_app, "RDAgentTask", FakeTask)
        monkeypatch.setattr(server_app, "rdagent_processes", {})
        return server_app, traces

    def _make_session(self, traces: Path, trace_id: str) -> Path:
        trace_dir = traces / trace_id
        step = trace_dir / "__session__" / "0" / "0_propose"
        step.parent.mkdir(parents=True)
        step.write_bytes(b"pickled-session")
        return trace_dir

    def test_resume_relaunches_from_session_under_same_trace_id(
        self, monkeypatch, tmp_path
    ) -> None:
        server_app, traces = self._setup(monkeypatch, tmp_path)
        trace_dir = self._make_session(traces, self.TRACE_ID)

        old = FakeTask(
            target_name="fin_quant",
            kwargs={},
            stdout_path="",
            log_trace_path=str(trace_dir),
            scenario="Finance Whole Pipeline",
            trace_name="run1",
        )
        old.messages = [
            {"tag": "user_interaction.request", "content": {"hypothesis": "h"}},
            {"tag": "END", "content": {"end_code": -1}},
        ]
        server_app.rdagent_processes[str(trace_dir)] = old

        resp = server_app.app.test_client().post(
            "/control", json={"id": self.TRACE_ID, "action": "resume"}
        )
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json() == {"status": "resumed", "id": self.TRACE_ID}

        task = server_app.rdagent_processes[str(trace_dir)]
        assert task is not old and isinstance(task, FakeTask)
        assert task.started is True
        assert task.target_name == "fin_quant"
        assert task.kwargs == {"path": str(trace_dir)}  # default: the trace dir itself
        assert task.log_trace_path == str(trace_dir)
        # history carried over, END markers dropped (run is live again)
        assert task.messages == [
            {"tag": "user_interaction.request", "content": {"hypothesis": "h"}}
        ]

    def test_resume_honors_explicit_path_and_passthrough(
        self, monkeypatch, tmp_path
    ) -> None:
        server_app, traces = self._setup(monkeypatch, tmp_path)
        trace_dir = self._make_session(traces, self.TRACE_ID)
        session = trace_dir / "__session__" / "0" / "0_propose"

        resp = server_app.app.test_client().post(
            "/control",
            json={
                "id": self.TRACE_ID,
                "action": "resume",
                "path": str(session),
                "loops": "2",
                "all_duration": "4",
            },
        )
        assert resp.status_code == 200, resp.get_json()
        task = server_app.rdagent_processes[str(trace_dir)]
        assert task.kwargs == {"path": str(session), "loop_n": 2, "all_duration": "4h"}

    def test_resume_refuses_while_process_alive(self, monkeypatch, tmp_path) -> None:
        server_app, traces = self._setup(monkeypatch, tmp_path)
        trace_dir = self._make_session(traces, self.TRACE_ID)
        alive = FakeTask(
            target_name="fin_quant",
            kwargs={},
            stdout_path="",
            log_trace_path=str(trace_dir),
            scenario="Finance Whole Pipeline",
            trace_name="run1",
        )
        alive.alive = True
        server_app.rdagent_processes[str(trace_dir)] = alive

        resp = server_app.app.test_client().post(
            "/control", json={"id": self.TRACE_ID, "action": "resume"}
        )
        assert resp.status_code == 400
        assert "still running" in resp.get_json()["error"]
        assert server_app.rdagent_processes[str(trace_dir)] is alive

    def test_resume_refuses_missing_session(self, monkeypatch, tmp_path) -> None:
        server_app, traces = self._setup(monkeypatch, tmp_path)
        (traces / self.TRACE_ID).mkdir(parents=True)  # trace dir, no __session__

        resp = server_app.app.test_client().post(
            "/control", json={"id": self.TRACE_ID, "action": "resume"}
        )
        assert resp.status_code == 400
        assert "no session to resume" in resp.get_json()["error"]
        assert server_app.rdagent_processes == {}

    def test_resume_refuses_path_outside_trace_folder(self, monkeypatch, tmp_path) -> None:
        server_app, traces = self._setup(monkeypatch, tmp_path)
        self._make_session(traces, self.TRACE_ID)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        resp = server_app.app.test_client().post(
            "/control",
            json={"id": self.TRACE_ID, "action": "resume", "path": str(outside)},
        )
        assert resp.status_code == 400
        assert "trace folder" in resp.get_json()["error"]

    def test_resume_refuses_unknown_scenario(self, monkeypatch, tmp_path) -> None:
        server_app, _ = self._setup(monkeypatch, tmp_path)
        resp = server_app.app.test_client().post(
            "/control", json={"id": "Data Science/run1", "action": "resume"}
        )
        assert resp.status_code == 400
        assert "scenario" in resp.get_json()["error"]

    def test_non_resume_actions_delegate_to_upstream(self, monkeypatch, tmp_path) -> None:
        server_app, _ = self._setup(monkeypatch, tmp_path)
        client = server_app.app.test_client()
        # upstream stop semantics intact: unknown id -> its own error message
        resp = client.post("/control", json={"id": "x/y", "action": "stop"})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "No running process for given id"
        # upstream unknown-action guard intact
        resp = client.post("/control", json={"id": "x/y", "action": "explode"})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "Only 'stop' action is supported"

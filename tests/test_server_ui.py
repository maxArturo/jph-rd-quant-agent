"""Tests for research/server_ui.py — the loopback server_ui launcher (US-018).

Offline tests monkeypatch the upstream Flask app; the end-to-end test boots
the real server in a subprocess on an ephemeral port and asserts both the
HTTP response and the loopback-only bind.
"""

from __future__ import annotations

import http.client
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

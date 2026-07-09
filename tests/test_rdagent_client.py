"""Tests for orchestrator/rdagent_client.py against a stubbed Flask server_ui.

The stub replicates the pinned upstream protocol (rdagent/log/server/app.py):
POST /upload starts a run, POST /trace polls messages and drains AT MOST ONE
pending user-interaction request per call, POST /user_interaction/submit
queues an answer, POST /control supports only "stop" (a resume-capable
variant covers the planned US-024 server extension).
"""

from __future__ import annotations

import pickle
import socket
import threading
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from orchestrator.rdagent_client import (
    KIND_BASE_FEATURES,
    KIND_FEEDBACK,
    KIND_HYPOTHESIS,
    KIND_INIT_PARAMS,
    KIND_UNKNOWN,
    ArtifactNotFoundError,
    PendingInteraction,
    RdAgentClient,
    RdAgentClientError,
    RdAgentServerError,
    UnsupportedActionError,
    classify_interaction,
    locate_artifacts,
)

FEATURES = {"RESI5": "Resi($close, 5)/$close"}

HYPOTHESIS = {
    "hypothesis": "Short-term reversal predicts returns",
    "reason": "Liquidity provision",
    "concise_reason": "reversal",
    "concise_observation": "obs",
    "concise_justification": "just",
    "concise_knowledge": "know",
    "action": "factor",
}

FEEDBACK = {
    "observations": "IC improved",
    "hypothesis_evaluation": "good",
    "new_hypothesis": "try 10d window",
    "reason": "better IC",
    "decision": True,
}


class StubServerUi:
    """Threaded Flask stub with upstream-faithful semantics."""

    def __init__(self, resume_supported: bool = False) -> None:
        self.resume_supported = resume_supported
        self.uploads: list[dict[str, str]] = []
        self.submitted: dict[str, list[Any]] = defaultdict(list)
        self.controls: list[dict[str, Any]] = []
        self.messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.pending_requests: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.started_traces: set[str] = set()
        self._clock = 0

        app = Flask(__name__)

        @app.post("/upload")
        def upload() -> Any:
            form = dict(request.form)
            self.uploads.append(form)
            scenario = form.get("scenario")
            if not scenario:
                return jsonify({"error": "Unknown scenario"}), 400
            trace_id = f"{scenario}/stub-run"
            self.started_traces.add(trace_id)
            return jsonify({"id": trace_id}), 200

        @app.post("/trace")
        def trace() -> Any:
            data = request.get_json()
            trace_id = data.get("id")
            if not trace_id:
                return jsonify({"error": "Trace ID is required"}), 400
            # Upstream drains at most one pending user request per poll.
            queue = self.pending_requests[trace_id]
            if queue:
                self._clock += 1
                self.messages[trace_id].append(
                    {
                        "tag": "user_interaction.request",
                        "timestamp": f"2026-07-08T00:00:{self._clock:02d}+00:00",
                        "content": queue.popleft(),
                    }
                )
            msgs = self.messages[trace_id]
            if data.get("all"):
                return jsonify(msgs), 200
            return jsonify(msgs[:2]), 200

        @app.post("/user_interaction/submit")
        def submit() -> Any:
            data = request.get_json(silent=True) or {}
            trace_id = data.get("id")
            payload = data.get("payload")
            if not trace_id:
                return jsonify({"error": "Trace ID is required"}), 400
            if payload is None:
                return jsonify({"error": "Missing 'payload'"}), 400
            self.submitted[trace_id].append(payload)
            return jsonify({"status": "success"}), 200

        @app.post("/control")
        def control() -> Any:
            data = request.get_json()
            if not data or "id" not in data or "action" not in data:
                return jsonify({"error": "Missing 'id' or 'action' in request"}), 400
            self.controls.append(dict(data))
            action = data["action"]
            if action != "stop" and not self.resume_supported:
                return jsonify({"error": "Only 'stop' action is supported"}), 400
            if data["id"] not in self.started_traces:
                return jsonify({"error": "No running process for given id"}), 400
            if action == "stop":
                self.messages[data["id"]].append(
                    {
                        "tag": "END",
                        "timestamp": "2026-07-08T01:00:00+00:00",
                        "content": {
                            "error_msg": "RD-Agent process was stopped by user.",
                            "end_code": -1,
                        },
                    }
                )
                return jsonify({"status": "stopped"}), 200
            return jsonify({"status": "resumed"}), 200

        @app.get("/test")
        def test_route() -> Any:
            return jsonify({"msgs": {}, "pointers": {}}), 200

        self._server = make_server("127.0.0.1", 0, app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


@pytest.fixture()
def stub():
    server = StubServerUi()
    yield server
    server.shutdown()


@pytest.fixture()
def resume_stub():
    server = StubServerUi(resume_supported=True)
    yield server
    server.shutdown()


def make_client(server: StubServerUi, **kwargs: Any) -> RdAgentClient:
    kwargs.setdefault("base_features", FEATURES)
    return RdAgentClient(server.base_url, **kwargs)


def start(server: StubServerUi, client: RdAgentClient, **kwargs: Any):
    return client.start_run("Find momentum factors", "us_liquid", **kwargs)


# -- start_run ------------------------------------------------------------


def test_start_run_uploads_scenario_and_passthrough(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = client.start_run(
        "Find momentum factors", "us_liquid", loop_n=2, all_duration_hours=4
    )
    assert handle.trace_id == "Finance Whole Pipeline/stub-run"
    assert stub.uploads == [
        {"scenario": "Finance Whole Pipeline", "loops": "2", "all_duration": "4"}
    ]


def test_start_run_seeds_directive_then_base_features(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    seeded = stub.submitted[handle.trace_id]
    assert len(seeded) == 2
    assert "Find momentum factors" in seeded[0]["user_instruction"]
    assert "us_liquid" in seeded[0]["user_instruction"]
    assert seeded[1] == FEATURES


def test_start_run_records_interaction_flag(stub: StubServerUi) -> None:
    client = make_client(stub)
    assert start(stub, client).interaction is True
    assert start(stub, client, interaction=False).interaction is False


def test_start_run_rejects_empty_directive(stub: StubServerUi) -> None:
    client = make_client(stub)
    with pytest.raises(ValueError):
        client.start_run("   ", "us_liquid")
    assert stub.uploads == []


# -- pending / submit ------------------------------------------------------


def test_pending_classifies_kinds_and_keys_are_stable(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    stub.pending_requests[handle.trace_id].append({"user_instruction": None})
    first = client.pending(handle.trace_id)
    assert [p.kind for p in first] == [KIND_INIT_PARAMS]

    stub.pending_requests[handle.trace_id].append(HYPOTHESIS)
    second = client.pending(handle.trace_id)
    assert [p.kind for p in second] == [KIND_INIT_PARAMS, KIND_HYPOTHESIS]
    assert second[1].content == HYPOTHESIS
    # keys are stable across polls (dedup contract for US-021) and unique
    assert second[0].key == first[0].key
    assert len({p.key for p in second}) == 2


def test_pending_drains_one_request_per_poll(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    stub.pending_requests[handle.trace_id].extend([{"user_instruction": None}, FEEDBACK])
    assert [p.kind for p in client.pending(handle.trace_id)] == [KIND_INIT_PARAMS]
    assert [p.kind for p in client.pending(handle.trace_id)] == [
        KIND_INIT_PARAMS,
        KIND_FEEDBACK,
    ]


def test_submit_forwards_payload(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    client.submit(handle.trace_id, HYPOTHESIS)
    assert stub.submitted[handle.trace_id][-1] == HYPOTHESIS


def test_classify_interaction_shapes() -> None:
    assert classify_interaction({"user_instruction": None}) == KIND_INIT_PARAMS
    assert (
        classify_interaction({"features": FEATURES, "feature_validation_msg": ""})
        == KIND_BASE_FEATURES
    )
    assert classify_interaction(HYPOTHESIS) == KIND_HYPOTHESIS
    assert classify_interaction(FEEDBACK) == KIND_FEEDBACK
    assert classify_interaction({"mystery": 1}) == KIND_UNKNOWN


def test_pending_interaction_key_format() -> None:
    item = PendingInteraction(
        trace_id="s/t", timestamp="2026-07-08T00:00:01+00:00", kind="hypothesis", content={}
    )
    assert item.key == "s/t|2026-07-08T00:00:01+00:00|hypothesis"


# -- stop / resume / status ------------------------------------------------


def test_stop_then_status_finished(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    assert client.status(handle.trace_id).finished is False
    client.stop(handle.trace_id)
    assert stub.controls[-1] == {"id": handle.trace_id, "action": "stop"}
    status = client.status(handle.trace_id)
    assert status.finished is True
    assert status.end_code == -1
    assert status.error_msg is not None and "stopped" in status.error_msg


def test_stop_unknown_trace_raises(stub: StubServerUi) -> None:
    client = make_client(stub)
    with pytest.raises(RdAgentServerError, match="No running process"):
        client.stop("Finance Whole Pipeline/nope")


def test_resume_unsupported_upstream_raises_specific_error(stub: StubServerUi) -> None:
    client = make_client(stub)
    handle = start(stub, client)
    with pytest.raises(UnsupportedActionError, match="research/server_ui.py"):
        client.resume(handle.trace_id)


def test_resume_against_resume_capable_server(resume_stub: StubServerUi) -> None:
    client = make_client(resume_stub)
    handle = start(resume_stub, client)
    seeded_before = len(resume_stub.submitted[handle.trace_id])
    client.resume(handle.trace_id, session_path="/traces/x/__session__/3/0_propose")
    assert resume_stub.controls[-1] == {
        "id": handle.trace_id,
        "action": "resume",
        "path": "/traces/x/__session__/3/0_propose",
    }
    # without a directive nothing is re-seeded
    assert len(resume_stub.submitted[handle.trace_id]) == seeded_before


def test_resume_with_directive_reseeds_init_interactions(resume_stub: StubServerUi) -> None:
    """A resumed run re-blocks on init params + base features — resume must
    re-seed both, mirroring start_run's FIFO pre-seeding."""
    client = make_client(resume_stub)
    handle = start(resume_stub, client)
    client.resume(
        handle.trace_id,
        session_path="/traces/x",
        directive="Find momentum factors",
        universe="us_liquid",
    )
    seeded = resume_stub.submitted[handle.trace_id]
    assert len(seeded) == 4  # 2 from start_run + 2 re-seeded by resume
    assert "Find momentum factors" in seeded[2]["user_instruction"]
    assert "us_liquid" in seeded[2]["user_instruction"]
    assert seeded[3] == FEATURES


def test_resume_failure_does_not_seed(resume_stub: StubServerUi) -> None:
    client = make_client(resume_stub)
    with pytest.raises(RdAgentServerError, match="No running process"):
        client.resume(
            "Finance Whole Pipeline/nope", directive="Find momentum factors"
        )
    assert resume_stub.submitted["Finance Whole Pipeline/nope"] == []


# -- transport errors / health ---------------------------------------------


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_connection_error_names_the_service() -> None:
    client = RdAgentClient(f"http://127.0.0.1:{_closed_port()}", base_features=FEATURES)
    with pytest.raises(RdAgentServerError, match="rdq-research.service"):
        client.messages("x")


def test_health(stub: StubServerUi) -> None:
    assert make_client(stub).health() is True
    down = RdAgentClient(f"http://127.0.0.1:{_closed_port()}", base_features=FEATURES)
    assert down.health() is False


# -- artifact locator --------------------------------------------------------


def _write_runner_result(
    trace_dir: Path, workspace: Path, stamp: str, obj: Any = None
) -> Path:
    pkl_dir = trace_dir / "Loop_0" / "running" / "runner result" / "12345"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / f"{stamp}.pkl"
    if obj is None:
        obj = SimpleNamespace(
            experiment_workspace=SimpleNamespace(workspace_path=workspace)
        )
    pkl_path.write_bytes(pickle.dumps(obj))
    return pkl_path


def _make_workspace(root: Path, name: str, ret: bool = True) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True)
    (workspace / "qlib_res.csv").write_text("IC,0.05\n")
    if ret:
        (workspace / "ret.pkl").write_bytes(pickle.dumps({"ret": [0.01]}))
    return workspace


def test_locate_artifacts_resolves_workspace(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    workspace = _make_workspace(tmp_path, "ws1")
    pkl = _write_runner_result(trace, workspace, "2026-07-08_10-00-00-000000")
    result = locate_artifacts(trace)
    assert result.workspace_path == workspace
    assert result.qlib_res_csv == workspace / "qlib_res.csv"
    assert result.ret_pkl == workspace / "ret.pkl"
    assert result.source_pkl == pkl


def test_locate_artifacts_newest_result_wins(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    old_ws = _make_workspace(tmp_path, "ws_old")
    new_ws = _make_workspace(tmp_path, "ws_new")
    _write_runner_result(trace, old_ws, "2026-07-08_10-00-00-000000")
    _write_runner_result(trace, new_ws, "2026-07-08_11-00-00-000000")
    assert locate_artifacts(trace).workspace_path == new_ws


def test_locate_artifacts_skips_unreadable_and_incomplete(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    good_ws = _make_workspace(tmp_path, "ws_good")
    empty_ws = tmp_path / "ws_empty"  # no qlib_res.csv
    empty_ws.mkdir()
    _write_runner_result(trace, good_ws, "2026-07-08_10-00-00-000000")
    _write_runner_result(trace, empty_ws, "2026-07-08_11-00-00-000000")
    corrupt = trace / "Loop_0" / "running" / "runner result" / "12345"
    (corrupt / "2026-07-08_12-00-00-000000.pkl").write_bytes(b"not a pickle")
    assert locate_artifacts(trace).workspace_path == good_ws


def test_locate_artifacts_ret_pkl_optional(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    workspace = _make_workspace(tmp_path, "ws1", ret=False)
    _write_runner_result(trace, workspace, "2026-07-08_10-00-00-000000")
    assert locate_artifacts(trace).ret_pkl is None


def test_locate_artifacts_errors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError, match="does not exist"):
        locate_artifacts(tmp_path / "missing")
    empty_trace = tmp_path / "trace"
    empty_trace.mkdir()
    with pytest.raises(ArtifactNotFoundError, match="no 'runner result'"):
        locate_artifacts(empty_trace)
    ws = tmp_path / "ws_no_csv"
    ws.mkdir()
    _write_runner_result(empty_trace, ws, "2026-07-08_10-00-00-000000")
    with pytest.raises(ArtifactNotFoundError, match="no qlib_res.csv"):
        locate_artifacts(empty_trace)


def test_client_artifacts_resolves_under_trace_folder(
    tmp_path: Path, stub: StubServerUi
) -> None:
    client = make_client(stub, trace_folder=tmp_path)
    trace_id = "Finance Whole Pipeline/stub-run"
    assert client.trace_dir(trace_id) == tmp_path / trace_id
    workspace = _make_workspace(tmp_path, "ws1")
    _write_runner_result(tmp_path / trace_id, workspace, "2026-07-08_10-00-00-000000")
    assert client.artifacts(trace_id).workspace_path == workspace


def test_trace_id_of_round_trips_trace_dir(tmp_path: Path, stub: StubServerUi) -> None:
    """US-020 stores session_path = str(trace_dir); later stories recover the id."""
    client = make_client(stub, trace_folder=tmp_path)
    trace_id = "Finance Whole Pipeline/2026-07-08_22-00-00"
    session_path = str(client.trace_dir(trace_id))
    assert client.trace_id_of(session_path) == trace_id


def test_trace_id_of_rejects_foreign_paths(tmp_path: Path, stub: StubServerUi) -> None:
    client = make_client(stub, trace_folder=tmp_path / "traces")
    with pytest.raises(RdAgentClientError, match="not under the trace folder"):
        client.trace_id_of(tmp_path / "elsewhere" / "run")

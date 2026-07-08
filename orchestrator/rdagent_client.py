"""Typed client for the rdagent server_ui control API (127.0.0.1:19899).

Speaks the REAL upstream protocol (rdagent/log/server/app.py at the pinned
commit), which differs from the endpoint names sketched in the PRD — the
mapping is recorded in docs/decisions.md:

- start a run          -> POST /upload   (form: scenario/loops/all_duration)
- poll messages +      -> POST /trace    (one pending user-interaction request
  pending interactions               is drained into the message list per poll)
- answer interaction   -> POST /user_interaction/submit  ({"id", "payload"})
- stop a run           -> POST /control  ({"id", "action": "stop"})
- resume a run         -> POST /control  action "resume" — NOT supported by
  the pinned upstream (it 400s "Only 'stop' action is supported"); the client
  raises UnsupportedActionError until US-024 extends research/server_ui.py.

Interaction model (rdagent RDLoop._interact_*): a fin_quant run started via
the server always gets IPC queues and blocks, in order, on
(1) init params — expects a plan-update dict; this is where the directive
    lands as ``user_instruction``,
(2) base features — expects a feature-name -> qlib-expression dict,
then on every hypothesis and every feedback. Responses are queued FIFO and
independently of the requests, so ``start_run`` pre-seeds (1) and (2)
immediately; hypotheses/feedbacks are surfaced by ``pending()`` for the
US-021 poller.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:19899"
DEFAULT_TRACE_FOLDER = "~/rdq-runs/server_ui/traces"
DEFAULT_TIMEOUT_SECONDS = 30.0

# server_ui scenario name that launches `fin_quant` (see /upload upstream).
SCENARIO_FIN_QUANT = "Finance Whole Pipeline"

INTERACTION_TAG = "user_interaction.request"
END_TAG = "END"

# Pending-interaction kinds, classified from the request dict's keys
# (shapes defined by rdagent RDLoop._interact_init_params/_interact_hypo/
# _interact_feedback).
KIND_INIT_PARAMS = "init_params"
KIND_BASE_FEATURES = "base_features"
KIND_HYPOTHESIS = "hypothesis"
KIND_FEEDBACK = "feedback"
KIND_UNKNOWN = "unknown"


class RdAgentClientError(RuntimeError):
    """Base error for rdagent server_ui client failures."""


class RdAgentServerError(RdAgentClientError):
    """The server was unreachable or returned an HTTP error."""


class UnsupportedActionError(RdAgentServerError):
    """The pinned upstream /control endpoint rejected the action."""


class ArtifactNotFoundError(RdAgentClientError):
    """No finished loop with backtest artifacts could be resolved."""


@dataclass(frozen=True)
class RunHandle:
    """Identity of a started run.

    ``trace_id`` ("<scenario>/<trace_name>") is the id every other endpoint
    takes; ``interaction`` records the caller's intent so pollers know whether
    to wait for the operator or auto-approve pending interactions.
    """

    trace_id: str
    directive: str
    universe: str
    interaction: bool


@dataclass(frozen=True)
class PendingInteraction:
    """One user-interaction request drained from the run's message stream."""

    trace_id: str
    timestamp: str
    kind: str
    content: dict[str, Any]

    @property
    def key(self) -> str:
        """Stable dedup key (pending_interactions.interaction_key in SQLite)."""
        return f"{self.trace_id}|{self.timestamp}|{self.kind}"


@dataclass(frozen=True)
class RunStatus:
    finished: bool
    end_code: int | None = None
    error_msg: str | None = None


@dataclass(frozen=True)
class RunArtifacts:
    """A finished loop's backtest outputs, resolved from its trace dir."""

    workspace_path: Path
    qlib_res_csv: Path
    ret_pkl: Path | None  # equity-curve DataFrame; absent on some failures
    source_pkl: Path  # the trace pkl the workspace was resolved from


def classify_interaction(content: dict[str, Any]) -> str:
    """Map a raw interaction-request dict to a kind by its key shape."""
    keys = set(content)
    if keys == {"user_instruction"}:
        return KIND_INIT_PARAMS
    if {"features", "feature_validation_msg"} <= keys:
        return KIND_BASE_FEATURES
    if {"hypothesis", "action"} <= keys:
        return KIND_HYPOTHESIS
    if {"decision", "observations"} <= keys:
        return KIND_FEEDBACK
    return KIND_UNKNOWN


class RdAgentClient:
    """HTTP client for the supervised server_ui instance (US-018 unit).

    ``session`` and ``base_features`` are injectable for tests; by default
    base features are rdagent's ALPHA20 (imported lazily — rdagent import is
    slow and only needed when actually starting a run).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        trace_folder: str | Path = DEFAULT_TRACE_FOLDER,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_features: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._trace_folder = Path(trace_folder).expanduser()
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._base_features = base_features

    # -- run lifecycle ----------------------------------------------------

    def start_run(
        self,
        directive: str,
        universe: str,
        interaction: bool = True,
        *,
        loop_n: int | None = None,
        all_duration_hours: int | None = None,
    ) -> RunHandle:
        """Launch a fin_quant run seeded with the directive as user_instruction.

        POSTs /upload, then pre-seeds the two blocking init interactions
        (plan update carrying the directive, then default base features) so
        the run proceeds to hypothesis generation without an operator.
        """
        directive = directive.strip()
        if not directive:
            raise ValueError("directive must be a non-empty string")
        form: dict[str, str] = {"scenario": SCENARIO_FIN_QUANT}
        if loop_n is not None:
            form["loops"] = str(loop_n)
        if all_duration_hours is not None:
            # the server appends "h" itself
            form["all_duration"] = str(all_duration_hours)
        body = self._request("POST", "/upload", data=form)
        trace_id = body.get("id") if isinstance(body, dict) else None
        if not trace_id:
            raise RdAgentServerError(f"/upload returned no trace id: {body!r}")

        self.submit(trace_id, {"user_instruction": self._instruction_text(directive, universe)})
        self.submit(trace_id, self._default_base_features())
        return RunHandle(
            trace_id=trace_id, directive=directive, universe=universe, interaction=interaction
        )

    def stop(self, trace_id: str) -> None:
        """POST /control stop; terminates the run's subprocess."""
        self._request("POST", "/control", json={"id": trace_id, "action": "stop"})

    def resume(self, trace_id: str, session_path: str | Path | None = None) -> None:
        """POST /control resume (from ``session_path`` if given).

        The pinned upstream only supports "stop" — until research/server_ui.py
        grows a resume extension (US-024) this raises UnsupportedActionError
        against the real server.
        """
        payload: dict[str, Any] = {"id": trace_id, "action": "resume"}
        if session_path is not None:
            payload["path"] = str(session_path)
        try:
            self._request("POST", "/control", json=payload)
        except RdAgentServerError as exc:
            if "only 'stop' action is supported" in str(exc).lower():
                raise UnsupportedActionError(
                    "server_ui does not support resume yet (upstream /control only "
                    "implements 'stop'; US-024 adds a resume extension in "
                    f"research/server_ui.py). Server said: {exc}"
                ) from exc
            raise

    # -- inspection -------------------------------------------------------

    def messages(self, trace_id: str) -> list[dict[str, Any]]:
        """Full message stream for a run (POST /trace with all+reset).

        Side effect (upstream semantics): each call drains at most one pending
        user-interaction request into the stream, so polling this is what
        makes interactions visible.
        """
        body = self._request("POST", "/trace", json={"id": trace_id, "all": True, "reset": True})
        if not isinstance(body, list):
            raise RdAgentServerError(f"/trace returned non-list body: {body!r}")
        return body

    def pending(self, trace_id: str) -> list[PendingInteraction]:
        """All user-interaction requests seen so far, oldest first.

        Answered requests stay in the stream — callers dedup via ``.key``
        (US-021 persists keys in pending_interactions) and should ignore
        KIND_INIT_PARAMS / KIND_BASE_FEATURES, which start_run auto-answers.
        """
        out: list[PendingInteraction] = []
        for msg in self.messages(trace_id):
            if msg.get("tag") != INTERACTION_TAG:
                continue
            content = msg.get("content")
            if not isinstance(content, dict):
                continue
            out.append(
                PendingInteraction(
                    trace_id=trace_id,
                    timestamp=str(msg.get("timestamp", "")),
                    kind=classify_interaction(content),
                    content=content,
                )
            )
        return out

    def submit(self, trace_id: str, payload: Any) -> None:
        """Answer the run's oldest unanswered interaction (FIFO queue).

        Payload shape depends on the interaction kind: init_params -> plan
        update dict, base_features -> features dict, hypothesis -> full
        hypothesis-constructor dict, feedback -> HypothesisFeedback dict.
        """
        self._request(
            "POST", "/user_interaction/submit", json={"id": trace_id, "payload": payload}
        )

    def status(self, trace_id: str) -> RunStatus:
        """Finished iff the stream carries an END message (completed/stopped)."""
        for msg in reversed(self.messages(trace_id)):
            if msg.get("tag") == END_TAG:
                content = msg.get("content") or {}
                return RunStatus(
                    finished=True,
                    end_code=content.get("end_code"),
                    error_msg=content.get("error_msg"),
                )
        return RunStatus(finished=False)

    def health(self) -> bool:
        """True when GET /test answers 200 (cheap liveness probe)."""
        try:
            return self._raw_request("GET", "/test").status_code == 200
        except requests.RequestException:
            return False

    # -- artifacts --------------------------------------------------------

    def trace_dir(self, trace_id: str) -> Path:
        """On-disk trace directory for a server-started run."""
        return self._trace_folder / trace_id

    def artifacts(self, trace_id: str) -> RunArtifacts:
        return locate_artifacts(self.trace_dir(trace_id))

    # -- internals --------------------------------------------------------

    @staticmethod
    def _instruction_text(directive: str, universe: str) -> str:
        universe = universe.strip()
        if not universe:
            return directive
        return (
            f"{directive}\n\n"
            f"Constrain all ideas to the '{universe}' universe (the qlib market "
            "the backtest configuration uses)."
        )

    def _default_base_features(self) -> dict[str, str]:
        if self._base_features is not None:
            return dict(self._base_features)
        # Lazy: pulls in the (slow) rdagent import only on a real start_run.
        from rdagent.utils.qlib import ALPHA20

        return dict(ALPHA20)

    def _raw_request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        return self._session.request(
            method, f"{self._base_url}{path}", timeout=self._timeout, **kwargs
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._raw_request(method, path, **kwargs)
        except requests.RequestException as exc:
            raise RdAgentServerError(
                f"cannot reach rdagent server_ui at {self._base_url} ({exc}). "
                "Is the control-plane service running? Check: "
                "systemctl --user status rdq-research.service"
            ) from exc
        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise RdAgentServerError(
                f"{method} {path} failed (HTTP {response.status_code}): {detail}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RdAgentServerError(f"{method} {path} returned non-JSON body") from exc


def locate_artifacts(trace_path: str | Path) -> RunArtifacts:
    """Resolve a finished loop's workspace + qlib_res.csv + ret.pkl from a trace dir.

    rdagent logs each finished backtest experiment under a ``runner result``
    tag (FileStorage pkl whose object carries
    ``experiment_workspace.workspace_path``); qlib_res.csv / ret.pkl live in
    that workspace (written by the workspace's read_exp_res.py). Newest
    result wins; unreadable pkls and workspaces without qlib_res.csv are
    skipped.
    """
    trace_path = Path(trace_path).expanduser()
    if not trace_path.is_dir():
        raise ArtifactNotFoundError(f"trace directory does not exist: {trace_path}")

    candidates = sorted(
        trace_path.glob("**/runner result/**/*.pkl"),
        key=lambda p: p.name,
        reverse=True,
    )
    problems: list[str] = []
    for pkl_file in candidates:
        try:
            with pkl_file.open("rb") as handle:
                obj = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - any unpickle failure just skips this candidate
            problems.append(f"{pkl_file}: failed to unpickle ({exc})")
            continue
        workspace = getattr(getattr(obj, "experiment_workspace", None), "workspace_path", None)
        if workspace is None:
            problems.append(f"{pkl_file}: object has no experiment_workspace.workspace_path")
            continue
        workspace_path = Path(workspace)
        qlib_res_csv = workspace_path / "qlib_res.csv"
        if not qlib_res_csv.is_file():
            problems.append(f"{pkl_file}: no qlib_res.csv in workspace {workspace_path}")
            continue
        ret_pkl = workspace_path / "ret.pkl"
        return RunArtifacts(
            workspace_path=workspace_path,
            qlib_res_csv=qlib_res_csv,
            ret_pkl=ret_pkl if ret_pkl.is_file() else None,
            source_pkl=pkl_file,
        )

    detail = "; ".join(problems) if problems else "no 'runner result' pkl found"
    raise ArtifactNotFoundError(
        f"no finished loop with backtest artifacts under {trace_path}: {detail}"
    )

"""Hypothesis poller: drive run interactions, autonomously by default (US-021/US-045).

A background thread polls the rdagent server_ui message stream for every run
with status ``running``. For an UNSUPERVISED run (the default since US-045)
each new hypothesis is auto-approved — submitted back unchanged and narrated
to the owning thread without buttons — because the engine's own
propose→backtest→feedback loop already turns failures into better next
hypotheses; the operator's judgment points are the directive and the
promotion, not individual factor formulations. Two brakes bound the loop:

- Budget: after ``max_hypotheses`` submitted hypotheses (default 10) the next
  proposal stops the run via ``/control`` instead of being approved. The run
  row stays ``running`` so the normal completion path (below) posts the
  metrics summary and Promote offer once the END message lands.
- Identical-error abort: ``identical_error_limit`` consecutive failed
  feedbacks with the exact same reason text (the signature of a crashed
  experiment, e.g. a broken qlib mount — not a judged-and-rejected idea) stop
  the run early and tell the thread an operator needs to look.

Both are derived from the ``pending_interactions`` history on every poll, so
restarts resume with the same budget/streak state.

For a SUPERVISED run (``start_research`` with ``supervised=true``) each new
hypothesis is posted to its owning thread as Block Kit Approve / Edit /
Reject buttons, exactly the pre-US-045 flow. Interactions persist in
``pending_interactions`` (keyed by ``PendingInteraction.key``, UNIQUE in the
schema) so a restart neither drops nor double-posts them.

Upstream protocol constraints (rdagent RDLoop._interact_hypo, see US-019 notes
and docs/decisions.md):

- The run blocks until an answer is submitted; the answer must be the full
  hypothesis constructor dict (``type(hypo)(**res_dict)``). Approve therefore
  submits the dict unchanged, Edit submits it with the operator's text merged
  into the ``hypothesis`` field.
- There is NO regenerate/skip action in the queue protocol. Reject submits
  the dict with the hypothesis text replaced by an explicit operator-rejection
  instruction, which steers the loop away from the rejected idea (the
  iteration itself still runs).
- ``feedback`` interactions also block the run; the poller auto-acknowledges
  them (submits the loop's own feedback unchanged) so runs never deadlock
  waiting on a message nobody sees. ``init_params``/``base_features`` are
  pre-answered by start_run and skipped here.

Run completion (US-022): every poll also checks the run's END status. When a
run finishes, the poller posts the backtest metrics summary (qlib_res.csv)
and uploads the equity-curve chart (ret.pkl -> PNG) to the owning thread,
then moves the run row to its terminal status — which removes it from the
``running`` set, so completion is handled exactly once. A completed or
operator-stopped run's summary carries the Promote button (US-033; the click
handlers live in orchestrator/promotion.py).

Notion audit trail (US-027): when a NotionRecorder is injected, each posted
hypothesis becomes a Hypothesis Log row, each operator action updates it,
each auto-acked feedback records its completed experiment's metrics as a
Backtest Results row, and run completion moves the idea page's Status. All
of it is best-effort — the recorder logs and swallows its own failures.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orchestrator import summary
from orchestrator.config import DEFAULT_MAX_HYPOTHESES
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.promotion import PROMOTABLE_STATUSES, promotion_offer_blocks
from orchestrator.rdagent_client import (
    KIND_BASE_FEATURES,
    KIND_FEEDBACK,
    KIND_HYPOTHESIS,
    KIND_INIT_PARAMS,
    ArtifactNotFoundError,
    PendingInteraction,
    RunArtifacts,
    RunStatus,
    locate_artifacts,
)
from orchestrator.state import PendingInteraction as InteractionRow
from orchestrator.state import Run, StateStore

logger = logging.getLogger(__name__)

# slack_bolt's Say or any equivalent accepting (text=..., thread_ts=...).
SayFn = Callable[..., Any]

DEFAULT_POLL_INTERVAL_SECONDS = 15.0

# Block Kit action ids (app.py registers a Bolt listener per id).
ACTION_APPROVE = "hypo_approve"
ACTION_EDIT = "hypo_edit"
ACTION_REJECT = "hypo_reject"

# Row statuses an operator button/edit-reply may still act on.
ACTIONABLE_STATUSES = frozenset({"pending", "editing"})

# Autonomous loop (US-045): identical-error abort threshold. The hypothesis
# budget default lives in config.py (env-tunable as RDQ_MAX_HYPOTHESES).
DEFAULT_IDENTICAL_ERROR_LIMIT = 3

# Hypothesis row statuses that consumed one iteration of the run (an answer
# was submitted upstream). 'cancelled' rows never reached the run.
SUBMITTED_STATUSES = frozenset({"approved", "edited", "rejected", "auto_approved"})

# Slack section blocks cap text at 3000 chars.
_MAX_SECTION_TEXT = 2900

EDIT_PROMPT = (
    ":pencil2: *Editing.* Reply in this thread with the revised hypothesis"
    " text — your next message here will be submitted as the edit."
)

REJECTION_INSTRUCTION = (
    "The operator REJECTED the proposed hypothesis: {original!r}. Do not"
    " implement it. Treat this iteration as a discard and propose a materially"
    " different hypothesis in the next iteration."
)


class InteractionClient(Protocol):
    """What the poller needs from RdAgentClient (stub-friendly)."""

    def pending(self, trace_id: str) -> list[PendingInteraction]: ...

    def submit(self, trace_id: str, payload: Any) -> None: ...

    def trace_id_of(self, session_path: str) -> str: ...

    def status(self, trace_id: str) -> RunStatus: ...

    def stop(self, trace_id: str) -> None: ...


class SlackPoster(Protocol):
    """The WebClient methods the poller posts and uploads through."""

    def chat_postMessage(self, **kwargs: Any) -> Any: ...  # noqa: N802 - slack_sdk casing

    def files_upload_v2(self, **kwargs: Any) -> Any: ...  # noqa: N802 - slack_sdk casing


def format_hypothesis_text(content: dict[str, Any]) -> str:
    """Slack mrkdwn body for a proposed hypothesis."""
    action = content.get("action")
    header = "*New hypothesis proposed*" + (f" (`{action}`)" if action else "")
    lines = [header, f"*Hypothesis:* {content.get('hypothesis', '')}"]
    reason = content.get("concise_reason") or content.get("reason")
    if reason:
        lines.append(f"*Why:* {reason}")
    text = "\n".join(lines)
    if len(text) > _MAX_SECTION_TEXT:
        text = text[: _MAX_SECTION_TEXT - 1] + "…"
    return text


def hypothesis_blocks(interaction_id: int, content: dict[str, Any]) -> list[dict[str, Any]]:
    """Block Kit blocks: hypothesis summary + Approve/Edit/Reject buttons."""

    def button(label: str, action_id: str, style: str | None = None) -> dict[str, Any]:
        element: dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": label},
            "action_id": action_id,
            "value": str(interaction_id),
        }
        if style is not None:
            element["style"] = style
        return element

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": format_hypothesis_text(content)},
        },
        {
            "type": "actions",
            "block_id": f"hypothesis_{interaction_id}",
            "elements": [
                button("Approve", ACTION_APPROVE, style="primary"),
                button("Edit", ACTION_EDIT),
                button("Reject", ACTION_REJECT, style="danger"),
            ],
        },
    ]


def autonomous_hypothesis_text(
    content: dict[str, Any], iteration: int, maximum: int
) -> str:
    """Narration for an auto-approved hypothesis (no buttons — audit trail only)."""
    action = content.get("action")
    header = (
        f":arrow_forward: *Hypothesis {iteration}/{maximum}*"
        + (f" (`{action}`)" if action else "")
        + " — auto-approved, experiment running."
    )
    lines = [header, f"*Hypothesis:* {content.get('hypothesis', '')}"]
    reason = content.get("concise_reason") or content.get("reason")
    if reason:
        lines.append(f"*Why:* {reason}")
    text = "\n".join(lines)
    if len(text) > _MAX_SECTION_TEXT:
        text = text[: _MAX_SECTION_TEXT - 1] + "…"
    return text


def feedback_outcome_text(content: dict[str, Any]) -> str:
    """One-line experiment verdict for the thread (autonomous runs only)."""
    reason = str(content.get("reason") or "").strip()
    snippet = f"\n_{reason[:280]}…_" if len(reason) > 280 else (f"\n_{reason}_" if reason else "")
    if content.get("decision"):
        return ":white_check_mark: Experiment result: *beat the best so far* — new baseline (SOTA)." + snippet
    return ":heavy_multiplication_x: Experiment result: did not beat the best so far." + snippet


def budget_exhausted_text(maximum: int) -> str:
    return (
        f":checkered_flag: Hypothesis budget reached ({maximum} ideas tried) —"
        " stopping the run. The final summary of the best result found will"
        " post here shortly, with the option to promote it."
    )


def identical_error_text(limit: int, signature: str) -> str:
    sig = signature[:400] + ("…" if len(signature) > 400 else "")
    return (
        f":rotating_light: Stopping the run early: the last {limit} experiments"
        " failed with the *identical* error, which points at an infrastructure"
        " problem rather than the research ideas themselves. An operator should"
        f" take a look; the run can be resumed once it's fixed.\n```{sig}```"
    )


@dataclass(frozen=True)
class LoopState:
    """The autonomous loop's budget/streak view of one thread's history."""

    submitted_hypotheses: int
    error_streak: int
    error_signature: str | None


def failure_signature(content: dict[str, Any]) -> str:
    """Stable identity of a failed feedback, for the identical-error streak.

    Judged-and-rejected ideas carry the summarizer's ``reason`` prose (unique
    every time). Crashed experiments instead arrive with an EMPTY reason and
    the raw failure text in ``observations`` — prefixed by hypothesis-specific
    render context and salted with timestamps/indices. Use the tail (the
    traceback's crash site) with digit runs normalized, so the same crash
    matches across iterations while different crashes don't.
    """
    reason = str(content.get("reason") or "").strip()
    if reason:
        return _normalize_digits(reason)
    observations = str(content.get("observations") or "").strip()
    if not observations:
        return ""
    normalized = _normalize_digits(observations)
    # Upstream truncates observations at arbitrary points, so raw tails
    # misalign between iterations. The innermost traceback frames identify
    # the crash site regardless of where the cut landed.
    frames = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip().startswith('File "')
    ]
    if frames:
        return "\n".join(frames[-3:])
    return normalized[-600:]


def _normalize_digits(text: str) -> str:
    return "".join("#" if ch.isdigit() else ch for ch in text)


def terminal_status(status: RunStatus) -> str:
    """runs.status value for a finished run (upstream END end_code semantics).

    end_code 0 = subprocess completed, -1 = stopped by the operator via
    /control, anything else = the fin_quant subprocess died.
    """
    if status.end_code == 0 or status.end_code is None:
        return "completed"
    if status.end_code == -1:
        return "stopped"
    return "failed"


def _completion_headline(status: RunStatus) -> str:
    kind = terminal_status(status)
    if kind == "completed":
        return ":checkered_flag: *Research run complete.*"
    if kind == "stopped":
        return ":octagonal_sign: *Research run stopped.*"
    detail = f" ({status.error_msg})" if status.error_msg else ""
    return f":x: *Research run failed* (exit code {status.end_code}){detail}."


def edited_payload(content: dict[str, Any], operator_text: str) -> dict[str, Any]:
    """The operator's text merged into the hypothesis dict (Edit action)."""
    return {**content, "hypothesis": operator_text.strip()}


def rejection_payload(content: dict[str, Any]) -> dict[str, Any]:
    """Reject: same constructor keys, hypothesis text replaced by the rejection.

    The upstream queue protocol cannot skip or regenerate a proposal, so the
    rejection instruction rides in the hypothesis text itself.
    """
    original = str(content.get("hypothesis", ""))
    out = dict(content)
    out["hypothesis"] = REJECTION_INSTRUCTION.format(original=original)
    out["reason"] = "Rejected by the operator in Slack; see hypothesis field."
    return out


class HypothesisPoller:
    """Polls active runs for interactions and handles the operator's actions.

    Share one instance per process. ``poll_once`` is what the background
    thread runs; the approve/reject/request_edit/consume_edit_reply methods
    are called from Bolt listeners (button clicks and thread replies). All
    state transitions go through StateStore, so a restart resumes cleanly:
    posted-but-unanswered hypotheses stay pending, dedup lives in the schema.
    """

    def __init__(
        self,
        store: StateStore,
        rdagent: InteractionClient,
        slack: SlackPoster,
        channel_id: str,
        interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        locate: Callable[[str | Path], RunArtifacts] = locate_artifacts,
        recorder: NotionRecorder | None = None,
        max_hypotheses: int = DEFAULT_MAX_HYPOTHESES,
        identical_error_limit: int = DEFAULT_IDENTICAL_ERROR_LIMIT,
    ) -> None:
        self._store = store
        self._rdagent = rdagent
        self._slack = slack
        self._channel_id = channel_id
        self._interval = interval_seconds
        self._locate = locate
        # Optional Notion audit trail (US-027); None disables recording.
        self._recorder = recorder
        # Autonomous loop brakes (US-045); apply to unsupervised runs only.
        self._max_hypotheses = max_hypotheses
        self._identical_error_limit = identical_error_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- background loop ----------------------------------------------------

    def start(self) -> None:
        """Start the daemon polling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hypothesis-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - the poller must survive any poll failure
                logger.exception("hypothesis poll failed")
            self._stop.wait(self._interval)

    # -- polling ------------------------------------------------------------

    def poll_once(self) -> int:
        """Process every active run once; returns how many hypotheses were posted."""
        posted = 0
        for run in self._store.list_runs(status="running"):
            try:
                posted += self._poll_run(run)
            except Exception:  # noqa: BLE001 - one broken run must not starve the others
                logger.exception("polling failed for run %s", run.session_path)
        return posted

    def _poll_run(self, run: Run) -> int:
        trace_id = self._rdagent.trace_id_of(run.session_path)
        status = self._rdagent.status(trace_id)
        if status.finished:
            self._handle_completion(run, status)
            return 0
        posted = 0
        for interaction in self._rdagent.pending(trace_id):
            if interaction.kind in (KIND_INIT_PARAMS, KIND_BASE_FEATURES):
                continue  # pre-answered by start_run
            if interaction.kind == KIND_HYPOTHESIS:
                if run.supervised:
                    row = self._store.get_pending_interaction_by_key(interaction.key)
                    if row is None:
                        posted += self._post_hypothesis(run, trace_id, interaction)
                        break  # the run is blocked on it; nothing later is answerable yet
                    if row.status in ACTIONABLE_STATUSES:
                        break  # still awaiting the operator
                    continue  # resolved — later interactions may now be live
                narrated, proceed = self._advance_autonomous(run, trace_id, interaction)
                posted += narrated
                if not proceed:
                    break
                continue
            if interaction.kind == KIND_FEEDBACK:
                if not self._auto_ack_feedback(run, trace_id, interaction):
                    break  # the run was aborted for repeated identical errors
                continue
            logger.warning(
                "unknown interaction kind %r for run %s (key %s) — skipped",
                interaction.kind,
                run.thread_ts,
                interaction.key,
            )
        return posted

    # -- autonomous loop (US-045) ---------------------------------------------

    def _loop_state(self, thread_ts: str) -> LoopState:
        """Budget/streak state, derived from the thread's interaction history."""
        submitted = 0
        feedbacks: list[dict[str, Any]] = []
        for row in self._store.list_interactions(thread_ts):
            kind = row.payload.get("kind")
            if kind == KIND_HYPOTHESIS and row.status in SUBMITTED_STATUSES:
                submitted += 1
            elif kind == KIND_FEEDBACK and row.status == "auto_approved":
                feedbacks.append(row.payload.get("content") or {})
        streak = 0
        signature: str | None = None
        for content in reversed(feedbacks):
            if content.get("decision"):
                break
            sig = failure_signature(content)
            if not sig or (signature is not None and sig != signature):
                break
            signature = sig
            streak += 1
        return LoopState(
            submitted_hypotheses=submitted,
            error_streak=streak,
            error_signature=signature,
        )

    def _advance_autonomous(
        self, run: Run, trace_id: str, interaction: PendingInteraction
    ) -> tuple[int, bool]:
        """Handle one hypothesis interaction of an unsupervised run.

        Returns ``(narrations_posted, proceed)`` — ``proceed`` is False when
        the run is now blocked (a failed submit will retry next poll) or was
        just stopped (budget spent / error streak).
        """
        row = self._store.get_pending_interaction_by_key(interaction.key)
        if row is not None:
            if row.status == "pending":
                # Posted with buttons before this run went autonomous (mode
                # flip mid-run, e.g. the US-045 deploy): approve it now.
                if not self._submit_quietly(trace_id, row.payload["content"], run):
                    return 0, False
                self._store.resolve_pending_interaction(row.id, "auto_approved")
                if self._recorder is not None:
                    self._recorder.record_hypothesis_action(row.interaction_key, "auto_approved")
                self._narrate(run, autonomous_hypothesis_text(
                    row.payload["content"],
                    self._loop_state(run.thread_ts).submitted_hypotheses,
                    self._max_hypotheses,
                ))
                return 1, True
            if row.status == "editing":
                return 0, False  # the operator is mid-edit; keep the FIFO blocked
            return 0, True  # already resolved — later interactions may be live

        state = self._loop_state(run.thread_ts)
        if state.submitted_hypotheses >= self._max_hypotheses:
            self._halt_run(
                run, trace_id, interaction, budget_exhausted_text(self._max_hypotheses)
            )
            return 0, False
        if state.error_streak >= self._identical_error_limit and state.error_signature:
            # Backstop: the abort at feedback-ack time already tried to stop
            # the run; a proposal arriving anyway means that stop failed.
            self._halt_run(
                run,
                trace_id,
                interaction,
                identical_error_text(self._identical_error_limit, state.error_signature),
            )
            return 0, False

        row = self._store.add_pending_interaction(
            run.thread_ts,
            interaction.key,
            {"trace_id": trace_id, "kind": interaction.kind, "content": interaction.content},
        )
        if row is None:  # lost an insert race with another poller
            return 0, True
        if not self._submit_quietly(trace_id, interaction.content, run):
            self._store.delete_pending_interaction(row.id)
            return 0, False
        self._store.resolve_pending_interaction(row.id, "auto_approved")
        if self._recorder is not None:
            self._recorder.record_hypothesis(run.thread_ts, interaction.key, interaction.content)
            self._recorder.record_hypothesis_action(interaction.key, "auto_approved")
        self._narrate(run, autonomous_hypothesis_text(
            interaction.content, state.submitted_hypotheses + 1, self._max_hypotheses
        ))
        logger.info(
            "auto-approved hypothesis %s (%d/%d) for thread %s",
            interaction.key,
            state.submitted_hypotheses + 1,
            self._max_hypotheses,
            run.thread_ts,
        )
        return 1, True

    def _halt_run(
        self, run: Run, trace_id: str, interaction: PendingInteraction, message: str
    ) -> None:
        """Stop a run instead of answering its newest proposal.

        The proposal's row is recorded as 'cancelled' so it never counts
        against the budget and the halt is not re-attempted every poll. The
        run row deliberately stays 'running': the stop surfaces as an END
        message (end_code -1), and the normal completion path posts the
        summary + Promote offer and flips the status.
        """
        row = self._store.add_pending_interaction(
            run.thread_ts,
            interaction.key,
            {"trace_id": trace_id, "kind": interaction.kind, "content": interaction.content},
        )
        if row is None:
            return  # already handled (e.g. by a concurrent poll)
        try:
            self._rdagent.stop(trace_id)
        except Exception:  # noqa: BLE001 - free the key so the next poll retries the stop
            self._store.delete_pending_interaction(row.id)
            logger.exception("failed to stop run for thread %s", run.thread_ts)
            return
        self._store.resolve_pending_interaction(row.id, "cancelled")
        if self._recorder is not None:
            self._recorder.record_hypothesis(run.thread_ts, interaction.key, interaction.content)
            self._recorder.record_hypothesis_action(interaction.key, "cancelled")
        self._narrate(run, message)
        logger.info("halted run for thread %s: %s", run.thread_ts, message.splitlines()[0])

    def _submit_quietly(self, trace_id: str, payload: dict[str, Any], run: Run) -> bool:
        try:
            self._rdagent.submit(trace_id, payload)
        except Exception:  # noqa: BLE001 - the next poll retries
            logger.exception("autonomous submit failed for thread %s", run.thread_ts)
            return False
        return True

    def _narrate(self, run: Run, text: str) -> None:
        """Best-effort thread narration — never blocks or fails the loop."""
        try:
            self._slack.chat_postMessage(
                channel=self._channel_id, thread_ts=run.thread_ts, text=text
            )
        except Exception:  # noqa: BLE001 - narration is auxiliary to the submit
            logger.exception("failed to narrate to thread %s", run.thread_ts)

    def _post_hypothesis(self, run: Run, trace_id: str, interaction: PendingInteraction) -> int:
        row = self._store.add_pending_interaction(
            run.thread_ts,
            interaction.key,
            {"trace_id": trace_id, "kind": interaction.kind, "content": interaction.content},
        )
        if row is None:  # lost an insert race with another poller — already posted
            return 0
        try:
            self._slack.chat_postMessage(
                channel=self._channel_id,
                thread_ts=run.thread_ts,
                text=f"New hypothesis proposed: {interaction.content.get('hypothesis', '')}",
                blocks=hypothesis_blocks(row.id, interaction.content),
            )
        except Exception:  # noqa: BLE001 - free the key so the next poll retries the post
            self._store.delete_pending_interaction(row.id)
            logger.exception("failed to post hypothesis to thread %s", run.thread_ts)
            return 0
        if self._recorder is not None:
            self._recorder.record_hypothesis(run.thread_ts, interaction.key, interaction.content)
        logger.info("posted hypothesis %s to thread %s", interaction.key, run.thread_ts)
        return 1

    def _auto_ack_feedback(self, run: Run, trace_id: str, interaction: PendingInteraction) -> bool:
        """Ack one feedback; returns False when the run was just aborted."""
        row = self._store.add_pending_interaction(
            run.thread_ts,
            interaction.key,
            {"trace_id": trace_id, "kind": interaction.kind, "content": interaction.content},
        )
        if row is None:
            return True  # already acknowledged
        try:
            self._rdagent.submit(trace_id, interaction.content)
        except Exception:  # noqa: BLE001 - free the key so the next poll retries the ack
            self._store.delete_pending_interaction(row.id)
            logger.exception("failed to auto-ack feedback for thread %s", run.thread_ts)
            return True
        self._store.resolve_pending_interaction(row.id, "auto_approved")
        self._record_experiment(run, interaction)
        logger.info("auto-acknowledged feedback %s for thread %s", interaction.key, run.thread_ts)
        if run.supervised:
            return True
        self._narrate(run, feedback_outcome_text(interaction.content))
        # Identical-error abort (US-045): N consecutive failures with the same
        # reason text is an environment problem repeating, not research.
        state = self._loop_state(run.thread_ts)
        if state.error_streak >= self._identical_error_limit and state.error_signature:
            self._narrate(
                run,
                identical_error_text(self._identical_error_limit, state.error_signature),
            )
            try:
                self._rdagent.stop(trace_id)
            except Exception:  # noqa: BLE001 - the halt check on the next proposal retries
                logger.exception("failed to stop erroring run for thread %s", run.thread_ts)
                return True
            logger.info(
                "aborted run for thread %s after %d identical failures",
                run.thread_ts,
                state.error_streak,
            )
            return False
        return True

    def _record_experiment(self, run: Run, interaction: PendingInteraction) -> None:
        """Record the just-judged experiment as a Backtest Results row (US-027).

        A feedback interaction marks one completed experiment: the runner
        result (and its workspace metrics) already exist, and the feedback's
        ``decision`` is the loop's SOTA verdict on it. Best-effort — missing
        or unparsable artifacts are logged, never raised.
        """
        if self._recorder is None:
            return
        try:
            artifacts = self._locate(run.session_path)
            metrics = summary.load_metrics(artifacts.qlib_res_csv)
        except (ArtifactNotFoundError, OSError, summary.SummaryError) as exc:
            logger.warning(
                "no backtest artifacts to record for run %s: %s", run.session_path, exc
            )
            return
        sharpe = next((metrics[k] for k in summary.SHARPE_CSV_KEYS if k in metrics), None)
        if sharpe is None and artifacts.ret_pkl is not None:
            try:
                sharpe = summary.compute_sharpe(artifacts.ret_pkl)
            except summary.SummaryError:
                sharpe = None
        title = f"Experiment {interaction.timestamp}"
        if run.universe:
            title += f" — {run.universe}"
        self._recorder.record_backtest(
            run.thread_ts,
            title=title,
            metrics=metrics,
            sharpe=sharpe,
            sota=bool(interaction.content.get("decision")),
            workspace_path=str(artifacts.workspace_path),
            universe=run.universe,
        )

    # -- run completion (US-022) ---------------------------------------------

    def _handle_completion(self, run: Run, status: RunStatus) -> None:
        """Post the metrics summary + equity chart, then close out the run row.

        Order matters: the terminal status update comes LAST — it is what
        removes the run from the ``running`` set, so a Slack failure leaves
        the run running and the next poll retries the whole completion (a
        rare transient failure may repost the summary; better than losing
        it). Deterministically-bad artifacts (missing/corrupt) never loop:
        they downgrade to an honest message before anything is posted.
        """
        artifacts: RunArtifacts | None = None
        artifact_problem: str | None = None
        try:
            artifacts = self._locate(run.session_path)
        except (ArtifactNotFoundError, OSError) as exc:
            artifact_problem = str(exc)

        chart_png: bytes | None = None
        text = f"{_completion_headline(status)}\n"
        if artifacts is not None:
            sharpe: float | None = None
            if artifacts.ret_pkl is not None:
                try:
                    sharpe = summary.compute_sharpe(artifacts.ret_pkl)
                except summary.SummaryError:
                    sharpe = None  # Sharpe degrades to n/a; the csv metrics still post
            try:
                metrics = summary.load_metrics(artifacts.qlib_res_csv)
                text += summary.format_summary(
                    metrics, sharpe, workspace_path=artifacts.workspace_path
                )
            except summary.SummaryError as exc:
                text += f"Backtest artifacts could not be parsed: {exc}"
            if artifacts.ret_pkl is not None:
                try:
                    chart_png = summary.render_equity_curve(
                        artifacts.ret_pkl, title=f"Equity curve — {run.universe or 'run'}"
                    )
                except summary.SummaryError as exc:
                    text += f"\n_(equity chart unavailable: {exc})_"
            else:
                text += "\n_(no ret.pkl in the workspace — equity chart unavailable)_"
        else:
            text += (
                "No backtest artifacts were found for this run"
                f" — nothing to summarize. ({artifact_problem})"
            )

        post_kwargs: dict[str, Any] = {
            "channel": self._channel_id,
            "thread_ts": run.thread_ts,
            "text": text,
        }
        # Promotion offer (US-033): completed AND operator-stopped runs with
        # resolvable artifacts are promotable (unbounded orchestrator runs
        # only ever end by an operator stop); failed runs never get the button.
        if terminal_status(status) in PROMOTABLE_STATUSES and artifacts is not None:
            post_kwargs["blocks"] = promotion_offer_blocks(run.thread_ts, text)
        self._slack.chat_postMessage(**post_kwargs)
        if chart_png is not None:
            # The chart is supplementary — never let its upload block
            # finalization. A PERSISTENT failure (e.g. the bot token missing
            # the files:write scope) would otherwise throw before the status
            # flip below and make every 15s poll re-post the whole summary
            # forever. The metrics text (posted above) is the real payload;
            # only its failure should retry.
            try:
                self._slack.files_upload_v2(
                    channel=self._channel_id,
                    thread_ts=run.thread_ts,
                    filename="equity_curve.png",
                    title="Equity curve",
                    file=chart_png,
                )
            except Exception:  # noqa: BLE001 - degrade to text-only, keep finalizing
                logger.exception(
                    "equity chart upload failed for thread %s (summary text still posted)",
                    run.thread_ts,
                )
        self._store.update_run_status(run.thread_ts, terminal_status(status))
        if self._recorder is not None:
            self._recorder.record_idea_status(run.thread_ts, terminal_status(status))
        logger.info(
            "run %s finished (end_code=%s) — summary posted to thread %s",
            run.session_path,
            status.end_code,
            run.thread_ts,
        )

    # -- operator actions (Bolt listeners call these) ------------------------

    def approve(self, interaction_id: int, say: SayFn) -> None:
        """Approve: submit the proposed hypothesis unchanged."""
        row = self._actionable_row(interaction_id, say)
        if row is None:
            return
        if not self._submit_row(row, row.payload["content"], say):
            return
        self._store.resolve_pending_interaction(row.id, "approved")
        if self._recorder is not None:
            self._recorder.record_hypothesis_action(row.interaction_key, "approved")
        say(
            text=":white_check_mark: Hypothesis approved and submitted to the run.",
            thread_ts=row.thread_ts,
        )

    def reject(self, interaction_id: int, say: SayFn) -> None:
        """Reject: tell the loop to discard the idea and propose differently."""
        row = self._actionable_row(interaction_id, say)
        if row is None:
            return
        if not self._submit_row(row, rejection_payload(row.payload["content"]), say):
            return
        self._store.resolve_pending_interaction(row.id, "rejected")
        if self._recorder is not None:
            self._recorder.record_hypothesis_action(row.interaction_key, "rejected")
        say(
            text=(
                ":no_entry: Hypothesis rejected — the run was told to discard it"
                " and propose a different direction."
            ),
            thread_ts=row.thread_ts,
        )

    def request_edit(self, interaction_id: int, say: SayFn) -> None:
        """Edit: start the text round-trip; the next thread reply is the edit."""
        row = self._actionable_row(interaction_id, say)
        if row is None:
            return
        self._store.resolve_pending_interaction(row.id, "editing")
        say(text=EDIT_PROMPT, thread_ts=row.thread_ts)

    def consume_edit_reply(self, thread_ts: str, text: str, say: SayFn) -> bool:
        """If this thread has an interaction in 'editing', submit the reply as the edit.

        Returns True when the message was consumed (the conversational core
        must then NOT see it).
        """
        rows = self._store.list_pending_interactions(thread_ts, status="editing")
        if not rows:
            return False
        row = rows[0]  # oldest first — matches the run's FIFO answer order
        if not self._submit_row(row, edited_payload(row.payload["content"], text), say):
            return True  # consumed (the operator was told the submit failed)
        self._store.resolve_pending_interaction(row.id, "edited")
        if self._recorder is not None:
            self._recorder.record_hypothesis_action(
                row.interaction_key, "edited", operator_input=text.strip()
            )
        say(
            text=":pencil2: Edited hypothesis submitted to the run.",
            thread_ts=row.thread_ts,
        )
        return True

    # -- internals ------------------------------------------------------------

    def _actionable_row(self, interaction_id: int, say: SayFn) -> InteractionRow | None:
        row = self._store.get_pending_interaction(interaction_id)
        if row is None:
            say(text=f"Unknown interaction #{interaction_id} — was the state reset?")
            return None
        if row.status not in ACTIONABLE_STATUSES:
            say(
                text=f"That hypothesis was already handled (status: {row.status}).",
                thread_ts=row.thread_ts,
            )
            return None
        return row

    def _submit_row(self, row: InteractionRow, payload: dict[str, Any], say: SayFn) -> bool:
        """Submit an answer for a row; on failure tell the thread and keep the row live."""
        try:
            self._rdagent.submit(row.payload["trace_id"], payload)
        except Exception as exc:  # noqa: BLE001 - report in-thread, leave the row actionable
            logger.exception("submit failed for interaction %s", row.interaction_key)
            say(
                text=f"Submitting to the research run failed ({exc}). Try again shortly.",
                thread_ts=row.thread_ts,
            )
            return False
        return True

"""US-009: conversational core — idea -> directive -> echo; US-020:
start_research — directive -> run row + duplicate rejection. Mocked Anthropic
(FakeClient from tests/test_llm.py), mocked Slack (a recording say callable),
stubbed GPU backend (StubGpu). No network anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ops.run_lock import RunLock
from orchestrator import prompts
from orchestrator.conversation import (
    DEFAULT_UNIVERSE,
    REFUSAL_REPLY,
    ConversationCore,
    directive_instruction,
    format_directive_summary,
    format_run_started,
)
from orchestrator.llm import ModelRouter
from orchestrator.run_memory import MEMORY_DELIMITER, Digest
from orchestrator.state import Run, StateStore
from tests.test_llm import (
    FakeClient,
    RefusalMessage,
    message,
    text_block,
    tool_use_block,
)
from tests.test_menu import fixture_store as menu_fixture_store

THREAD = "1751900000.000100"


class RecordingSay:
    """Mocked Slack say(): records (text, thread_ts) per call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, text: str, thread_ts: str) -> None:
        self.calls.append({"text": text, "thread_ts": thread_ts})


class StubGpu:
    """Stubbed orchestrator.gpu_backend.GpuBackend: records launches/cancels."""

    def __init__(self) -> None:
        self.status_file = Path("/stub-gpu/pipeline_status.json")
        self.launched: list[dict[str, Any]] = []
        self.stopped_units: list[str] = []
        self.cancels = 0
        self.status: dict[str, Any] | None = None
        self.active = False
        # US-020 global run lock: (active holder, stale lock just broken).
        self.lock: RunLock | None = None
        self.broken_lock: RunLock | None = None

    def launch(
        self,
        thread_ts: str,
        *,
        loop_n: int = 10,
        universe: str | None = None,
        instruction: str | None = None,
    ) -> str:
        self.launched.append(
            {
                "thread_ts": thread_ts,
                "loop_n": loop_n,
                "universe": universe,
                "instruction": instruction,
            }
        )
        return "rdq-gpu-run-" + thread_ts.replace(".", "-")

    def stop_unit(self, unit: str) -> None:
        self.stopped_units.append(unit)

    def unit_active(self, thread_ts: str) -> bool:
        del thread_ts
        return self.active

    def active_run_lock(self) -> tuple[RunLock | None, RunLock | None]:
        broken, self.broken_lock = self.broken_lock, None
        return self.lock, broken

    def cancel(self) -> str:
        self.cancels += 1
        return "cancel signal sent — the pipeline will finalize and tear down"

    def read_status(self) -> dict[str, Any] | None:
        return self.status


def make_core(
    tmp_path: Path,
    client: FakeClient,
    promotions: Any | None = None,
    gpu: StubGpu | None = None,
    digest_builder: Any | None = None,
    menu_builder: Any | None = None,
    field_lister: Any | None = None,
) -> tuple[ConversationCore, StateStore]:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    core = ConversationCore(
        store=store,
        router=ModelRouter(client=client),
        promotions=promotions,
        gpu=gpu if gpu is not None else StubGpu(),
        digest_builder=digest_builder,
        menu_builder=menu_builder,
        field_lister=field_lister,
    )
    return core, store


def save_directive_script(final_reply: str = "Directive saved — ready to research.") -> list[Any]:
    """Model turn 1: call save_directive; turn 2: confirm in text."""
    return [
        message(
            "tool_use",
            [
                tool_use_block(
                    "tu_1",
                    "save_directive",
                    {
                        "objective": "Test whether 12-1 momentum beats SPY",
                        "universe_hint": "US large caps",
                        "constraints": "long-only, monthly rebalance",
                    },
                )
            ],
        ),
        message("end_turn", [text_block(final_reply)]),
    ]


# --- system prompt (acceptance: persona lives in orchestrator/prompts.py) ----


def test_system_prompt_states_persona_and_ground_rules() -> None:
    prompt = prompts.SYSTEM_PROMPT.lower()
    assert "portfolio manager" in prompt and "quant" in prompt
    assert "honest" in prompt  # honest reporting
    assert "never trade" in prompt and "explicit approval" in prompt


# --- idea -> directive -> echo flow ------------------------------------------


def test_idea_to_directive_to_echo_flow(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=save_directive_script())
    core, store = make_core(tmp_path, client)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "momentum on big US names?", say)

    # persisted: {objective, universe_hint, constraints} in the directives table
    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Test whether 12-1 momentum beats SPY"
    assert directive.universe_hint == "US large caps"
    assert directive.constraints == "long-only, monthly rebalance"

    # echoed: formatted summary posted to the thread, then the final reply
    assert [c["thread_ts"] for c in say.calls] == [THREAD, THREAD]
    summary, final = say.calls[0]["text"], say.calls[1]["text"]
    assert summary == format_directive_summary(directive)
    assert "Test whether 12-1 momentum beats SPY" in summary
    assert final == "Directive saved — ready to research." == reply

    # the tool result told the model the save happened
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert f"#{directive.id} saved" in tool_result["content"]


def test_optional_directive_fields_default_to_none(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message(
                "tool_use",
                [tool_use_block("tu_1", "save_directive", {"objective": "Objective only"})],
            ),
            message("end_turn", [text_block("Saved.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    core.handle_message(THREAD, "idea", RecordingSay())
    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Objective only"
    assert directive.universe_hint is None
    assert directive.constraints is None


def test_empty_objective_is_rejected_and_nothing_persisted(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message(
                "tool_use",
                [tool_use_block("tu_1", "save_directive", {"objective": "   "})],
            ),
            message("end_turn", [text_block("That objective was empty.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message(THREAD, "idea", say)

    assert store.get_directive(THREAD) is None
    # the failure went back to the model as an is_error tool_result
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "objective" in tool_result["content"]
    # no summary was posted — only the final reply
    assert [c["text"] for c in say.calls] == ["That objective was empty."]


# --- conversation context ------------------------------------------------------


def test_history_accumulates_across_turns(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message("end_turn", [text_block("What horizon?")]),
            message("end_turn", [text_block("Got it.")]),
        ]
    )
    core, _ = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message(THREAD, "momentum idea", say)
    core.handle_message(THREAD, "12 months", say)

    second_turn = client.stream_calls[1]["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "user"]
    assert second_turn[0]["content"] == "momentum idea"
    assert second_turn[1]["content"] == "What horizon?"
    assert second_turn[2]["content"] == "12 months"


def test_threads_have_independent_histories(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            message("end_turn", [text_block("a")]),
            message("end_turn", [text_block("b")]),
        ]
    )
    core, _ = make_core(tmp_path, client)
    say = RecordingSay()
    core.handle_message("111.000", "first thread", say)
    core.handle_message("222.000", "second thread", say)
    assert client.stream_calls[1]["messages"] == [
        {"role": "user", "content": "second thread"}
    ]


def test_directive_context_reloads_from_sqlite_after_restart(tmp_path: Path) -> None:
    """Acceptance: create directive, recreate the app objects, directive is
    retrievable by thread AND flows back into the model's context."""
    core1, _ = make_core(tmp_path, FakeClient(judgment_messages=save_directive_script()))
    core1.handle_message(THREAD, "momentum on big US names?", RecordingSay())

    # simulated restart: brand-new store + core over the same sqlite file
    client2 = FakeClient(judgment_messages=[message("end_turn", [text_block("Recap...")])])
    core2, store2 = make_core(tmp_path, client2)

    directive = store2.get_directive(THREAD)
    assert directive is not None
    assert directive.objective == "Test whether 12-1 momentum beats SPY"

    core2.handle_message(THREAD, "where were we?", RecordingSay())
    system = client2.stream_calls[0]["system"]
    assert system.startswith(prompts.SYSTEM_PROMPT)
    assert "Test whether 12-1 momentum beats SPY" in system
    assert "US large caps" in system


def test_system_prompt_has_no_directive_context_before_save(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Tell me more.")])])
    core, _ = make_core(tmp_path, client)
    core.handle_message(THREAD, "vague idea", RecordingSay())
    assert client.stream_calls[0]["system"] == prompts.SYSTEM_PROMPT


# --- data menu in the directive-crafting prompt (US-061 / US-002) -----------------


def test_data_menu_context_renders_fixture_store_menu(tmp_path: Path) -> None:
    """Single source of truth: the context is data/menu.py's rendering, so it
    carries the store's field names and PIT notes — no hand-copied list."""
    context = prompts.data_menu_context(menu_fixture_store(tmp_path))
    assert context.startswith(prompts.MENU_HEADER)
    for name in ("$open", "$close", "$volume", "$factor"):
        assert name in context
    assert "tiny_pit" in context  # universes come through too
    assert prompts.MENU_UNAVAILABLE_LINE not in context


def test_data_menu_context_degrades_when_store_unreadable(tmp_path: Path) -> None:
    assert prompts.data_menu_context(tmp_path / "no-store-here") == prompts.MENU_UNAVAILABLE_LINE


def test_crafting_prompt_includes_menu_fields_and_directive(tmp_path: Path) -> None:
    """The system prompt the model crafts directives against carries the menu
    (from a fixture store) alongside persona and saved-directive context."""
    fixture = menu_fixture_store(tmp_path)
    client = FakeClient(
        judgment_messages=save_directive_script()
        + [message("end_turn", [text_block("Recap...")])]
    )
    core, _ = make_core(tmp_path, client, menu_builder=lambda: prompts.data_menu_context(fixture))

    core.handle_message(THREAD, "momentum on big US names?", RecordingSay())
    system = client.stream_calls[0]["system"]
    assert system.startswith(prompts.SYSTEM_PROMPT)
    for name in ("$open", "$close", "$volume", "$factor"):
        assert name in system

    # After the save, menu and directive context coexist.
    core.handle_message(THREAD, "where were we?", RecordingSay())
    system = client.stream_calls[-1]["system"]
    assert prompts.MENU_HEADER in system
    assert "Test whether 12-1 momentum beats SPY" in system


def test_unreadable_store_degrades_to_fallback_line_in_prompt(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Noted.")])])
    core, _ = make_core(
        tmp_path,
        client,
        menu_builder=lambda: prompts.data_menu_context(tmp_path / "no-store-here"),
    )
    reply = core.handle_message(THREAD, "vague idea", RecordingSay())
    assert reply == "Noted."  # degraded, never raised into the handler
    assert prompts.MENU_UNAVAILABLE_LINE in client.stream_calls[0]["system"]


def test_menu_builder_exception_never_reaches_the_handler(tmp_path: Path) -> None:
    def exploding_builder() -> str:
        raise RuntimeError("store swap mid-read")

    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Noted.")])])
    core, _ = make_core(tmp_path, client, menu_builder=exploding_builder)
    reply = core.handle_message(THREAD, "vague idea", RecordingSay())
    assert reply == "Noted."
    assert prompts.MENU_UNAVAILABLE_LINE in client.stream_calls[0]["system"]


# --- failure handling ------------------------------------------------------------


def test_refusal_posts_notice_and_keeps_history_clean(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=[
            RefusalMessage(),
            message("end_turn", [text_block("Happy to help with research.")]),
        ]
    )
    core, store = make_core(tmp_path, client)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "do something sketchy", say)
    assert reply == REFUSAL_REPLY
    assert say.calls == [{"text": REFUSAL_REPLY, "thread_ts": THREAD}]
    assert store.get_directive(THREAD) is None

    # the refused turn was rolled back — next turn starts a clean transcript
    core.handle_message(THREAD, "ok, a real idea", say)
    assert client.stream_calls[1]["messages"] == [
        {"role": "user", "content": "ok, a real idea"}
    ]


# --- start_research (US-020) --------------------------------------------------


def start_research_script(
    final_reply: str = "Run started — watch this thread.",
    tool_input: dict[str, Any] | None = None,
) -> list[Any]:
    """Model turn 1: call start_research; turn 2: confirm in text."""
    return [
        message("tool_use", [tool_use_block("tu_sr", "start_research", tool_input or {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def test_start_research_launches_gpu_pipeline_and_writes_row(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=start_research_script())
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(
        THREAD,
        objective="Test whether 12-1 momentum beats SPY",
        universe_hint="US large caps",
        constraints="long-only, monthly rebalance",
    )
    say = RecordingSay()

    reply = core.handle_message(THREAD, "research it", say)

    # the GPU pipeline was launched with the thread's directive
    assert gpu.launched == [
        {
            "thread_ts": THREAD,
            "loop_n": 10,
            "universe": DEFAULT_UNIVERSE,
            "instruction": (
                "Test whether 12-1 momentum beats SPY\nConstraints: long-only, monthly rebalance"
            ),
        }
    ]

    # the run row points at the pipeline status file until fetch rewrites it
    run = store.get_run(THREAD)
    assert run is not None
    assert run.session_path == str(gpu.status_file)
    assert run.status == "running"
    assert run.universe == DEFAULT_UNIVERSE
    assert run.backend == "gpu"

    # start notice posted in-thread, then the model's final reply
    assert [c["thread_ts"] for c in say.calls] == [THREAD, THREAD]
    assert say.calls[0]["text"] == format_run_started(run)
    assert reply == "Run started — watch this thread."
    assert gpu.stopped_units == []

    # GPU runs are always autonomous.
    assert run.supervised is False
    assert "no approvals needed" in say.calls[0]["text"]


def test_start_research_injects_run_memory_digest(tmp_path: Path) -> None:
    """US-015: directive + MEMORY_DELIMITER + digest reaches launch unchanged
    and the start notice states how many prior runs rode along."""
    digest = Digest(
        text=(
            "Run-history digest (prior research runs, newest first):\n\n"
            "[2026-08-14 | completed] directive: try downside-share factors"
        ),
        runs=3,
    )
    client = FakeClient(judgment_messages=start_research_script())
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu, digest_builder=lambda: digest)
    store.create_directive(
        THREAD,
        objective="Test whether 12-1 momentum beats SPY",
        constraints="long-only, monthly rebalance",
    )
    say = RecordingSay()

    core.handle_message(THREAD, "research it", say)

    directive_text = (
        "Test whether 12-1 momentum beats SPY\nConstraints: long-only, monthly rebalance"
    )
    assert gpu.launched[0]["instruction"] == directive_text + MEMORY_DELIMITER + digest.text
    run = store.get_run(THREAD)
    assert run is not None
    assert say.calls[0]["text"] == format_run_started(run, memory_runs=3)
    assert "3 prior run(s) included as context" in say.calls[0]["text"]


def test_start_research_include_memory_false_skips_digest(tmp_path: Path) -> None:
    builder_calls: list[int] = []

    def builder() -> Digest:
        builder_calls.append(1)
        return Digest("should never appear", 5)

    client = FakeClient(
        judgment_messages=start_research_script(tool_input={"include_memory": False})
    )
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu, digest_builder=builder)
    store.create_directive(THREAD, objective="Clean-slate objective")
    say = RecordingSay()

    core.handle_message(THREAD, "research it fresh, no memory", say)

    assert builder_calls == []  # clean slate: the digest is never even built
    assert gpu.launched[0]["instruction"] == "Clean-slate objective"
    assert "not included (clean-slate run)" in say.calls[0]["text"]


def test_start_research_without_digest_builder_launches_bare_directive(tmp_path: Path) -> None:
    """Unwired digest builder (tests, partial deployments) must not block a
    launch — the instruction is the bare directive."""
    client = FakeClient(judgment_messages=start_research_script())
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Objective only")
    say = RecordingSay()

    core.handle_message(THREAD, "research it", say)

    assert gpu.launched[0]["instruction"] == "Objective only"
    assert "not included (clean-slate run)" in say.calls[0]["text"]


def test_start_research_refuses_while_another_thread_holds_the_lock(tmp_path: Path) -> None:
    """US-020: the global lock, not just this thread's run row, gates launches."""
    client = FakeClient(judgment_messages=start_research_script("Can't start yet."))
    gpu = StubGpu()
    gpu.lock = RunLock(unit="rdq-gpu-run-9999-0001", thread_ts="9999.0001")
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Test something")

    core.handle_message(THREAD, "research it", RecordingSay())

    assert gpu.launched == []
    assert store.get_run(THREAD) is None
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "9999.0001" in tool_result["content"]  # names the active run's thread
    assert "one GPU worker" in tool_result["content"]


def test_start_research_breaks_stale_lock_with_note_then_launches(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=start_research_script())
    gpu = StubGpu()
    gpu.broken_lock = RunLock(unit="rdq-gpu-run-9999-0001", thread_ts="9999.0001")
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Test something")
    say = RecordingSay()

    core.handle_message(THREAD, "research it", say)

    assert len(gpu.launched) == 1
    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"
    assert ":broom:" in say.calls[0]["text"]
    assert "rdq-gpu-run-9999-0001" in say.calls[0]["text"]


def test_start_research_supervised_is_refused_on_gpu(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=start_research_script(
            "Supervised runs aren't available.", tool_input={"supervised": True}
        )
    )
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Test something")

    core.handle_message(THREAD, "research it, I want to approve each hypothesis", RecordingSay())

    assert gpu.launched == []
    assert store.get_run(THREAD) is None
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "autonomous" in tool_result["content"]


def test_directive_instruction_omits_missing_constraints(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    bare = store.create_directive(THREAD, objective="Objective only")
    assert directive_instruction(bare) == "Objective only"


def test_start_research_without_directive_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=start_research_script("Save a directive first.")
    )
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)

    core.handle_message(THREAD, "research it", RecordingSay())

    assert gpu.launched == []
    assert store.get_run(THREAD) is None
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "save_directive" in tool_result["content"]


def test_duplicate_start_rejected_pointing_at_active_run(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=start_research_script("A run is already going here.")
    )
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Momentum on US large caps")
    existing = store.create_run(THREAD, "/stub-traces/existing/run", universe="us_liquid")

    core.handle_message(THREAD, "research it again", RecordingSay())

    # nothing new was launched; the existing row is untouched
    assert gpu.launched == []
    assert store.get_run(THREAD) == existing

    # the rejection points the model at the active run
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert existing.session_path in tool_result["content"]
    assert existing.status in tool_result["content"]


def test_start_research_works_again_after_a_run_is_reaped(tmp_path: Path) -> None:
    """US-021: a reaped (failed) GPU run row no longer bricks its thread —
    the full loop: stranded running row -> reaper marks it failed ->
    start_research launches a fresh run in the same thread."""
    from orchestrator.run_reaper import GpuRunReaper

    client = FakeClient(judgment_messages=start_research_script())
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_directive(THREAD, objective="Momentum on US large caps")
    store.create_run(THREAD, "/stub-gpu/pipeline_status.json", backend="gpu")

    class ReapSlack:
        def chat_postMessage(self, **kwargs: Any) -> None:  # noqa: N802
            pass

    clock_now = [0.0]
    reaper = GpuRunReaper(
        store,
        ReapSlack(),
        "C0TEST",
        unit_active=lambda thread_ts: False,  # the pipeline unit died
        grace_seconds=60.0,
        clock=lambda: clock_now[0],
    )
    reaper.tick()  # grace starts
    clock_now[0] = 61.0
    assert reaper.tick() == [THREAD]
    reaped = store.get_run(THREAD)
    assert reaped is not None and reaped.status == "failed"

    reply = core.handle_message(THREAD, "research it again", RecordingSay())

    assert len(gpu.launched) == 1
    run = store.get_run(THREAD)
    assert run is not None
    assert run.status == "running"
    assert run.backend == "gpu"
    assert reply == "Run started — watch this thread."


def test_start_research_still_blocked_by_completed_and_stopped_runs(tmp_path: Path) -> None:
    """Only a failed run frees the thread — terminal-but-promotable runs keep
    the one-run-per-thread rule (promotion reads the row's session_path)."""
    for status in ("completed", "stopped"):
        client = FakeClient(judgment_messages=start_research_script("Already ran here."))
        gpu = StubGpu()
        (tmp_path / status).mkdir()
        core, store = make_core(tmp_path / status, client, gpu=gpu)
        store.create_directive(THREAD, objective="Momentum on US large caps")
        store.create_run(THREAD, "/stub-gpu/trace", backend="gpu")
        store.update_run_status(THREAD, status)

        core.handle_message(THREAD, "research it again", RecordingSay())

        assert gpu.launched == []
        existing = store.get_run(THREAD)
        assert existing is not None and existing.status == status
        tool_result = client.stream_calls[1]["messages"][2]["content"][0]
        assert tool_result["is_error"] is True


class RaceyStore(StateStore):
    """Simulates a concurrent start: the duplicate pre-check misses the other
    run (first get_run returns None), then create_run hits the PK conflict."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path=db_path)
        self._get_run_calls = 0

    def get_run(self, thread_ts: str) -> Run | None:
        self._get_run_calls += 1
        if self._get_run_calls == 1:
            return None
        return super().get_run(thread_ts)


def test_lost_start_race_stops_the_orphan_pipeline(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=start_research_script("Already running."))
    gpu = StubGpu()
    store = RaceyStore(db_path=tmp_path / "state.sqlite")
    core = ConversationCore(store=store, router=ModelRouter(client=client), gpu=gpu)
    store.create_directive(THREAD, objective="Momentum on US large caps")
    existing = store.create_run(THREAD, "/stub-traces/winner/run", universe="us_liquid")

    core.handle_message(THREAD, "research it", RecordingSay())

    # the racing pipeline WAS launched, then its unit stopped on the conflict
    assert len(gpu.launched) == 1
    assert gpu.stopped_units == ["rdq-gpu-run-" + THREAD.replace(".", "-")]
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert existing.session_path in tool_result["content"]


# --- stop_run (US-024, GPU-only since US-028) ------------------------------------


# A legacy pre-GPU row's stored session path (backend 'server_ui'): the
# control plane is gone, but the row must stay readable and tolerated.
SESSION_PATH = "/stub-traces/Finance Whole Pipeline/trace_9"


def lifecycle_script(tool: str, final_reply: str) -> list[Any]:
    """Model turn 1: call *tool* (e.g. stop_run); turn 2: confirm in text."""
    return [
        message("tool_use", [tool_use_block("tu_lc", tool, {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def test_stop_run_on_gpu_backend_sends_cancel_and_keeps_row_running(tmp_path: Path) -> None:
    """The pipeline (not the tool) flips the row when it finalizes."""
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Cancelling."))
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, str(gpu.status_file), universe="us_liquid", backend="gpu")
    say = RecordingSay()

    core.handle_message(THREAD, "cancel the run", say)

    assert gpu.cancels == 1
    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"
    assert "cancel signal sent" in say.calls[0]["text"]


def test_stop_run_refuses_when_lock_owned_by_another_thread(tmp_path: Path) -> None:
    """US-020: cancel kills THE shared worker — only the owning thread may."""
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Can't stop."))
    gpu = StubGpu()
    gpu.lock = RunLock(unit="rdq-gpu-run-9999-0001", thread_ts="9999.0001")
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, str(gpu.status_file), universe="us_liquid", backend="gpu")

    core.handle_message(THREAD, "cancel the run", RecordingSay())

    assert gpu.cancels == 0
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "9999.0001" in tool_result["content"]  # names the owning thread
    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"


def test_stop_run_acts_when_this_thread_owns_the_lock(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Cancelling."))
    gpu = StubGpu()
    gpu.lock = RunLock(unit="rdq-gpu-run-" + THREAD.replace(".", "-"), thread_ts=THREAD)
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, str(gpu.status_file), universe="us_liquid", backend="gpu")

    core.handle_message(THREAD, "cancel the run", RecordingSay())

    assert gpu.cancels == 1


def check_status_script(final_reply: str = "Here's the status.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_st", "check_research_status", {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def test_check_research_status_reports_gpu_progress(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=check_status_script())
    gpu = StubGpu()
    gpu.status = {
        "thread_ts": THREAD,
        "stage": "running",
        "loops": [
            {"loop": 0, "decision": True, "metrics": {"IC": 0.02, "ARR": 0.5, "MDD": -0.1}},
            {"loop": 1, "decision": None},
        ],
        "exit": None,
    }
    gpu.active = True
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, str(gpu.status_file), universe="us_liquid", backend="gpu")

    core.handle_message(THREAD, "how's the run going?", RecordingSay())

    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result.get("is_error") is not True
    assert "stage: running" in tool_result["content"]
    assert "loops finished: 1 (1 SOTA)" in tool_result["content"]


def test_check_research_status_flags_foreign_pipeline(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=check_status_script())
    gpu = StubGpu()
    gpu.status = {"thread_ts": "9999.0001", "stage": "running"}
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, str(gpu.status_file), universe="us_liquid", backend="gpu")

    core.handle_message(THREAD, "status?", RecordingSay())

    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert "another thread" in tool_result["content"]
    # US-020: runs are never queued — the old text lied about that.
    assert "queued" not in tool_result["content"] or "never queued" in tool_result["content"]


def test_stop_run_without_run_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(
        judgment_messages=lifecycle_script("stop_run", "There is nothing to stop.")
    )
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)

    core.handle_message(THREAD, "stop it", RecordingSay())

    assert gpu.cancels == 0
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "nothing to stop" in tool_result["content"]


def test_stop_run_on_non_running_run_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Not running."))
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, SESSION_PATH, universe="us_liquid", status="completed")

    core.handle_message(THREAD, "stop it", RecordingSay())

    assert gpu.cancels == 0
    run = store.get_run(THREAD)
    assert run is not None and run.status == "completed"  # status untouched
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "completed" in tool_result["content"]


def test_stop_run_refuses_legacy_backend_run(tmp_path: Path) -> None:
    """US-028: a legacy 'server_ui' row stays readable, but its control plane
    is gone — stop_run explains that instead of crashing or touching the row."""
    client = FakeClient(judgment_messages=lifecycle_script("stop_run", "Can't stop that."))
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(THREAD, SESSION_PATH, universe="us_liquid", backend="server_ui")

    core.handle_message(THREAD, "stop it", RecordingSay())

    assert gpu.cancels == 0
    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"  # row untouched
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "decommissioned" in tool_result["content"]
    assert "server_ui" in tool_result["content"]


def test_check_research_status_tolerates_legacy_backend_row(tmp_path: Path) -> None:
    """US-028: iterating/reporting code paths tolerate backend='server_ui'."""
    client = FakeClient(judgment_messages=check_status_script())
    gpu = StubGpu()
    core, store = make_core(tmp_path, client, gpu=gpu)
    store.create_run(
        THREAD, SESSION_PATH, universe="us_liquid", backend="server_ui", status="stopped"
    )

    core.handle_message(THREAD, "status?", RecordingSay())

    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert tool_result.get("is_error") is not True
    assert "stopped" in tool_result["content"]
    assert "server_ui" in tool_result["content"]


# --- US-044: conversational promotion ----------------------------------------


class StubPromotions:
    """Stubbed PromotionFlow: records calls, posts a canned outcome in-thread."""

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.requested: list[str] = []
        self.confirmed: list[str] = []

    def request_promotion(self, thread_ts: str, say: Any) -> None:
        self.requested.append(thread_ts)
        if self.refuse:
            say(text=":no_entry: Cannot promote: the run is 'running'",
                thread_ts=thread_ts)
        else:
            say(text=":rocket: *Confirm promotion to paper trading*",
                thread_ts=thread_ts)

    def confirm_promotion(self, thread_ts: str, say: Any) -> None:
        self.confirmed.append(thread_ts)
        say(text=":rocket: *Strategy promoted to paper trading.*",
            thread_ts=thread_ts)


def test_promote_run_relays_the_posted_confirmation(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("promote_run", "Please confirm."))
    promotions = StubPromotions()
    core, store = make_core(tmp_path, client, promotions=promotions)
    say = RecordingSay()

    reply = core.handle_message(THREAD, "promote this run", say)

    assert promotions.requested == [THREAD]
    assert promotions.confirmed == []  # request never promotes by itself
    assert ":rocket:" in say.calls[0]["text"]
    # the tool result carries what was posted so the model can relay it
    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert "Confirm promotion" in tool_result["content"]
    assert reply == "Please confirm."


def test_promote_run_relays_a_refusal(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("promote_run", "Cannot promote."))
    promotions = StubPromotions(refuse=True)
    core, store = make_core(tmp_path, client, promotions=promotions)

    core.handle_message(THREAD, "promote it", RecordingSay())

    tool_result = client.stream_calls[1]["messages"][2]["content"][0]
    assert "Cannot promote" in tool_result["content"]
    assert promotions.confirmed == []


def test_confirm_promotion_pins_via_the_flow(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=lifecycle_script("confirm_promotion", "Promoted."))
    promotions = StubPromotions()
    core, store = make_core(tmp_path, client, promotions=promotions)
    say = RecordingSay()

    core.handle_message(THREAD, "yes, confirm the promotion", say)

    assert promotions.confirmed == [THREAD]
    assert ":rocket: *Strategy promoted" in say.calls[0]["text"]


def test_promotion_tools_absent_when_not_wired(tmp_path: Path) -> None:
    """A core without promotion wiring never offers the tools."""
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Hi.")])])
    core, store = make_core(tmp_path, client)

    core.handle_message(THREAD, "hello", RecordingSay())

    offered = {tool["name"] for tool in client.stream_calls[0]["tools"]}
    assert offered.isdisjoint({"promote_run", "confirm_promotion"})


def test_promotion_tools_offered_when_wired(tmp_path: Path) -> None:
    client = FakeClient(judgment_messages=[message("end_turn", [text_block("Hi.")])])
    core, store = make_core(tmp_path, client, promotions=StubPromotions())

    core.handle_message(THREAD, "hello", RecordingSay())

    offered = {tool["name"] for tool in client.stream_calls[0]["tools"]}
    assert {"promote_run", "confirm_promotion"} <= offered


# --- directive data pre-flight (US-062) ---------------------------------------


def save_with_data_script(
    data_required: list[str], final_reply: str = "Saved."
) -> list[Any]:
    """Model turn 1: save_directive declaring data_required; turn 2: text."""
    return [
        message(
            "tool_use",
            [
                tool_use_block(
                    "tu_dr",
                    "save_directive",
                    {
                        "objective": "Condition 12-1 momentum on crude strength",
                        "data_required": data_required,
                    },
                )
            ],
        ),
        message("end_turn", [text_block(final_reply)]),
    ]


def fixture_field_lister(tmp_path: Path) -> Any:
    """Field names from a real fixture store, through data/menu.py itself."""
    from data.menu import build_menu

    store = menu_fixture_store(tmp_path)
    return lambda: build_menu(store).field_names()


def last_tool_result(client: FakeClient) -> dict[str, Any]:
    return client.stream_calls[-1]["messages"][-1]["content"][0]


def test_data_required_all_present_renders_line_and_stays_startable(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        judgment_messages=save_with_data_script(["$close", "$volume"])
        + start_research_script()
    )
    gpu = StubGpu()
    core, store = make_core(
        tmp_path, client, gpu=gpu, field_lister=fixture_field_lister(tmp_path)
    )
    say = RecordingSay()

    core.handle_message(THREAD, "momentum conditioned on volume?", say)

    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.data_required == ("$close", "$volume")
    assert directive.missing_data == ()
    assert directive.parked is False
    summary = say.calls[0]["text"]
    line = next(
        ln for ln in summary.splitlines() if ln.startswith("*Data required:*")
    )
    assert "$close, $volume" in line
    assert line.endswith("— all present in store")
    assert "parked" not in summary

    # startable: the pre-flight passed, so start_research launches normally
    core.handle_message(THREAD, "research it", say)
    assert len(gpu.launched) == 1
    run = store.get_run(THREAD)
    assert run is not None
    assert run.status == "running"


def test_missing_data_parks_directive_and_blocks_start(tmp_path: Path) -> None:
    # $news_ct_1d is absent from the fixture store until US-014 lands.
    client = FakeClient(
        judgment_messages=save_with_data_script(["$close", "$news_ct_1d"], "Parked.")
        + start_research_script("Can't start — parked.")
    )
    gpu = StubGpu()
    core, store = make_core(
        tmp_path, client, gpu=gpu, field_lister=fixture_field_lister(tmp_path)
    )
    say = RecordingSay()

    core.handle_message(THREAD, "momentum conditioned on news attention?", say)

    # saved (objective persisted) ...
    directive = store.get_directive(THREAD)
    assert directive is not None
    assert directive.data_required == ("$close", "$news_ct_1d")
    # ... but parked, and the thread reply names each missing series
    assert directive.missing_data == ("$news_ct_1d",)
    assert directive.parked is True
    assert "parked — needs ingestion: $news_ct_1d" in say.calls[0]["text"]
    # the model's tool result states the parked outcome too
    assert "PARKED" in last_tool_result(client)["content"]

    # NOT startable: start_research refuses with the same parked message
    core.handle_message(THREAD, "research it anyway", say)
    tool_result = last_tool_result(client)
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("is_error") is True
    assert "parked — needs ingestion: $news_ct_1d" in tool_result["content"]
    assert gpu.launched == []
    assert store.get_run(THREAD) is None


def test_parked_start_refusal_is_state_enforced_across_restart(
    tmp_path: Path,
) -> None:
    """Parking lives in SQLite: a fresh core (new process) still refuses."""
    store = StateStore(db_path=tmp_path / "state.sqlite")
    store.create_directive(
        THREAD,
        objective="Trade the crack spread",
        data_required=["$close", "$mkt_wti"],
        missing_data=["$mkt_wti"],
    )
    client = FakeClient(judgment_messages=start_research_script("Refused."))
    gpu = StubGpu()
    core = ConversationCore(
        store=store, router=ModelRouter(client=client), gpu=gpu
    )

    core.handle_message(THREAD, "start the run", RecordingSay())

    tool_result = last_tool_result(client)
    assert tool_result.get("is_error") is True
    assert "parked — needs ingestion: $mkt_wti" in tool_result["content"]
    assert gpu.launched == []
    assert store.get_run(THREAD) is None
    # the reloaded system prompt carries the parked state for the model
    assert "PARKED — needs ingestion: $mkt_wti" in client.stream_calls[0]["system"]


def test_unverifiable_data_required_fails_the_save_loud(tmp_path: Path) -> None:
    """Store unreadable at save time: no silent 'all present' — the tool errors
    and nothing is persisted."""

    def boom() -> list[str]:
        raise RuntimeError("store offline")

    client = FakeClient(
        judgment_messages=save_with_data_script(["$close"], "Try again later.")
    )
    core, store = make_core(tmp_path, client, field_lister=boom)

    core.handle_message(THREAD, "an idea", RecordingSay())

    assert store.get_directive(THREAD) is None
    tool_result = last_tool_result(client)
    assert tool_result.get("is_error") is True
    assert "could not verify data_required" in tool_result["content"]

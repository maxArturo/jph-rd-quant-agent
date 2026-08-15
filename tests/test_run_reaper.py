"""US-021: stale GPU run-row reaper — a dead pipeline unit's 'running' row is
finalized as 'failed' after one grace period, with a note in the run's thread.
No network, no systemd: unit_active and the clock are injected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ops.run_lock import unit_name
from orchestrator.run_reaper import GpuRunReaper, reap_note
from orchestrator.state import StateStore

THREAD = "1751900000.000100"
OTHER_THREAD = "1751900000.000200"
CHANNEL = "C0TEST"
GRACE = 900.0


class StubSlack:
    """Records chat_postMessage calls; optionally fails."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.fail = False

    def chat_postMessage(self, **kwargs: Any) -> Any:  # noqa: N802 - slack_sdk casing
        if self.fail:
            raise RuntimeError("slack down")
        self.posts.append(kwargs)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_reaper(
    tmp_path: Path,
    *,
    dead_units: set[str] | None = None,
) -> tuple[GpuRunReaper, StateStore, StubSlack, Clock, set[str]]:
    """Reaper over a temp DB; a thread_ts in ``dead`` reads as unit-inactive."""
    store = StateStore(db_path=tmp_path / "state.sqlite")
    slack = StubSlack()
    clock = Clock()
    dead = dead_units if dead_units is not None else set()
    reaper = GpuRunReaper(
        store,
        slack,
        CHANNEL,
        unit_active=lambda thread_ts: thread_ts not in dead,
        grace_seconds=GRACE,
        clock=clock,
    )
    return reaper, store, slack, clock, dead


def seed_gpu_run(store: StateStore, thread_ts: str = THREAD) -> None:
    store.create_run(thread_ts, "/stub/pipeline_status.json", backend="gpu")


# --- active unit untouched ----------------------------------------------------


def test_active_unit_is_never_touched(tmp_path: Path) -> None:
    reaper, store, slack, clock, _ = make_reaper(tmp_path)
    seed_gpu_run(store)

    clock.now += 10 * GRACE
    assert reaper.tick() == []

    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"
    assert slack.posts == []


def test_server_ui_and_terminal_rows_are_ignored(tmp_path: Path) -> None:
    reaper, store, slack, clock, dead = make_reaper(tmp_path)
    store.create_run(THREAD, "/stub/legacy-trace", backend="server_ui")
    store.create_run(OTHER_THREAD, "/stub/pipeline_status.json", backend="gpu")
    store.update_run_status(OTHER_THREAD, "completed")
    dead.update({THREAD, OTHER_THREAD})

    assert reaper.tick() == []
    clock.now += GRACE + 1
    assert reaper.tick() == []

    legacy = store.get_run(THREAD)
    assert legacy is not None and legacy.status == "running"
    done = store.get_run(OTHER_THREAD)
    assert done is not None and done.status == "completed"
    assert slack.posts == []


# --- grace period ---------------------------------------------------------------


def test_dead_unit_within_grace_is_not_reaped(tmp_path: Path) -> None:
    reaper, store, slack, clock, _ = make_reaper(tmp_path, dead_units={THREAD})
    seed_gpu_run(store)

    assert reaper.tick() == []  # first sighting starts the grace clock
    clock.now += GRACE - 1
    assert reaper.tick() == []

    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"
    assert slack.posts == []


def test_dead_unit_reaped_after_grace_with_thread_note(tmp_path: Path) -> None:
    reaper, store, slack, clock, _ = make_reaper(tmp_path, dead_units={THREAD})
    seed_gpu_run(store)

    assert reaper.tick() == []
    clock.now += GRACE
    assert reaper.tick() == [THREAD]

    run = store.get_run(THREAD)
    assert run is not None and run.status == "failed"
    assert len(slack.posts) == 1
    post = slack.posts[0]
    assert post["channel"] == CHANNEL
    assert post["thread_ts"] == THREAD
    assert post["text"] == reap_note(run)
    assert unit_name(THREAD) in post["text"]
    assert "start_research" in post["text"]

    # a reaped run is out of the candidate set — nothing further happens
    clock.now += GRACE
    assert reaper.tick() == []
    assert len(slack.posts) == 1


def test_unit_recovery_resets_the_grace_clock(tmp_path: Path) -> None:
    reaper, store, slack, clock, dead = make_reaper(tmp_path, dead_units={THREAD})
    seed_gpu_run(store)

    assert reaper.tick() == []  # dead: grace starts
    dead.discard(THREAD)
    clock.now += GRACE + 1
    assert reaper.tick() == []  # recovered: grace forgotten
    dead.add(THREAD)
    assert reaper.tick() == []  # dead again: grace restarts from here
    clock.now += GRACE - 1
    assert reaper.tick() == []

    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"
    assert slack.posts == []


# --- failure behavior -----------------------------------------------------------


def test_slack_failure_leaves_row_running_and_retries_next_tick(tmp_path: Path) -> None:
    reaper, store, slack, clock, _ = make_reaper(tmp_path, dead_units={THREAD})
    seed_gpu_run(store)

    reaper.tick()
    clock.now += GRACE
    slack.fail = True
    assert reaper.tick() == []  # note couldn't post -> no status flip

    run = store.get_run(THREAD)
    assert run is not None and run.status == "running"

    slack.fail = False
    assert reaper.tick() == [THREAD]
    run = store.get_run(THREAD)
    assert run is not None and run.status == "failed"


def test_one_bad_run_does_not_block_the_rest(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    seed_gpu_run(store, THREAD)
    seed_gpu_run(store, OTHER_THREAD)
    slack = StubSlack()
    clock = Clock()
    probes: list[str] = []

    def unit_active(thread_ts: str) -> bool:
        probes.append(thread_ts)
        if thread_ts == THREAD:
            raise RuntimeError("systemctl exploded")
        return False

    reaper = GpuRunReaper(
        store, slack, CHANNEL, unit_active=unit_active, grace_seconds=GRACE, clock=clock
    )

    reaper.tick()
    clock.now += GRACE
    reaped = reaper.tick()

    assert reaped == [OTHER_THREAD]
    assert THREAD in probes  # the bad run was attempted, not skipped
    survivor = store.get_run(OTHER_THREAD)
    assert survivor is not None and survivor.status == "failed"
    stuck = store.get_run(THREAD)
    assert stuck is not None and stuck.status == "running"

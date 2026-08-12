"""US-009: live promotion backend (LivePromotion.promote / .demote).

Covers candidate resolution (explicit reference / current thread / paper-slot
copy — never guessing), the PROMOTABLE_STATUSES refusal, provenance reuse
(conf-derived market + instruments-file tickers via PromotionFlow), the
pred-refresh snapshot rules (create when missing, never touch a complete set,
warn on regenerating a partial one, refuse on failure), the live-slot write
with allocation pct, and the Decision Log rows — all against fake
state/store/Notion recorders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from execution.allocation import AllocationConfigError, LiveAllocation
from execution.pred_refresh import (
    SNAPSHOT_CONF_NAME,
    SNAPSHOT_ENV_NAME,
    SNAPSHOT_PARAMS_NAME,
)
from orchestrator.notion_client import NotionClient
from orchestrator.notion_recorder import NotionRecorder
from orchestrator.promotion import (
    LivePromotion,
    LivePromotionError,
    PromotionFlow,
)
from orchestrator.rdagent_client import RunArtifacts
from orchestrator.state import StateStore
from tests.test_notion_client import FakeSession
from tests.test_notion_recorder import DBS, page_response
from tests.test_poller import SESSION, THREAD
from tests.test_promotion import promotable_artifacts

ALLOCATION = LiveAllocation(live_equity_allocation_pct=10.0)
PERMALINK = "https://example.slack.com/archives/C0LIVE/p1751900000000100"


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    s = StateStore(tmp_path / "state.sqlite")
    s.create_run(THREAD, SESSION, universe="us_liquid", universe_tickers=("AAPL", "MSFT"))
    s.update_run_status(THREAD, "completed")
    return s


def make_live(
    store: StateStore,
    artifacts: RunArtifacts,
    recorder: NotionRecorder | None = None,
    snapshot: Any = None,
    load_allocation: Any = None,
) -> LivePromotion:
    """Backend over stub artifacts; snapshot defaults to a no-op (fixture
    workspaces have no docker logs for the real US-048 snapshot)."""
    flow = PromotionFlow(
        store,
        locate=lambda _session: artifacts,
        snapshot=lambda _workspace: None,
        instruments_dir=Path(artifacts.workspace_path).parent / "instruments",
    )
    return LivePromotion(
        store,
        flow,
        recorder=recorder,
        snapshot=snapshot if snapshot is not None else lambda _workspace: None,
        load_allocation=load_allocation if load_allocation is not None else lambda: ALLOCATION,
    )


# --- candidate resolution ------------------------------------------------------


def test_promotes_the_current_threads_run(store: StateStore, tmp_path: Path) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    result = live.promote(thread_ts=THREAD)

    assert result.source == "run"
    assert result.source_thread_ts == THREAD
    pinned = store.get_promoted_strategy_live()
    assert pinned is not None
    assert pinned.workspace_path == str(tmp_path / "workspace")
    assert pinned.config["universe"] == "us_liquid"
    assert pinned.config["universe_tickers"] == ["AAPL", "MSFT"]
    assert pinned.config["topk"] == 50
    assert pinned.config["n_drop"] == 5
    assert pinned.config["live_equity_allocation_pct"] == 10.0
    # Direct-run promotion touches ONLY the live slot.
    assert store.get_promoted_strategy() is None
    # Headline metrics come from the run artifacts, like paper promotion.
    assert result.metrics.get("IC") == pytest.approx(0.0432)


def test_promotes_an_explicitly_named_run_by_thread_ts(
    store: StateStore, tmp_path: Path
) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    result = live.promote(reference=THREAD, thread_ts="9999999999.000001")
    assert result.source == "run"
    assert result.source_thread_ts == THREAD


def test_promotes_an_explicitly_named_run_by_session_fragment(
    store: StateStore, tmp_path: Path
) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    result = live.promote(reference="2026-07-08_10-00-00")
    assert result.source_thread_ts == THREAD


def test_ambiguous_reference_lists_the_matches(store: StateStore, tmp_path: Path) -> None:
    other = "1751900001.000200"
    store.create_run(other, SESSION + "-b", universe="us_liquid")
    store.update_run_status(other, "completed")
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError) as exc:
        live.promote(reference="2026-07-08_10-00-00")
    assert "ambiguous" in str(exc.value)
    assert THREAD in str(exc.value) and other in str(exc.value)
    assert store.get_promoted_strategy_live() is None


def test_unknown_reference_lists_promotable_runs(store: StateStore, tmp_path: Path) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError) as exc:
        live.promote(reference="no-such-run-anywhere")
    assert "no run matches" in str(exc.value)
    assert THREAD in str(exc.value)  # the promotable run is listed
    assert store.get_promoted_strategy_live() is None


def test_short_fragment_never_fuzzy_matches(store: StateStore, tmp_path: Path) -> None:
    """A tiny fragment ('26') would substring-match half the store — refused."""
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError, match="no run matches"):
        live.promote(reference="26")


def test_non_promotable_status_is_named(store: StateStore, tmp_path: Path) -> None:
    store.update_run_status(THREAD, "running")
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError, match="'running'"):
        live.promote(thread_ts=THREAD)
    with pytest.raises(LivePromotionError, match="'running'"):
        live.promote(reference=THREAD)
    assert store.get_promoted_strategy_live() is None


def test_conf_market_wins_over_run_label(store: StateStore, tmp_path: Path) -> None:
    """US-023 rule carries over: the pinned universe is the conf's market line."""
    mislabeled = "1751900002.000300"
    store.create_run(mislabeled, SESSION + "-c", universe="ai_deployers")
    store.update_run_status(mislabeled, "completed")
    live = make_live(store, promotable_artifacts(tmp_path))  # conf says us_liquid
    result = live.promote(thread_ts=mislabeled)
    assert result.universe == "us_liquid"
    assert result.universe_label == "ai_deployers"
    assert result.label_mismatch


# --- paper-slot copy -------------------------------------------------------------


def paper_config(workspace: Path) -> dict[str, Any]:
    return {
        "universe": "us_liquid_promoted_30",
        "universe_tickers": ["AAPL", "MSFT", "NVDA"],
        "topk": 30,
        "n_drop": 3,
        "thread_ts": THREAD,
        "session_path": str(workspace.parent),
    }


def test_copies_the_paper_slot_when_nothing_else_given(
    store: StateStore, tmp_path: Path
) -> None:
    artifacts = promotable_artifacts(tmp_path)
    workspace = Path(artifacts.workspace_path)
    store.set_promoted_strategy(str(workspace), paper_config(workspace))
    live = make_live(store, artifacts)

    result = live.promote(thread_ts="9999999999.000001")  # thread with no run

    assert result.source == "paper"
    pinned = store.get_promoted_strategy_live()
    assert pinned is not None
    # Copied IN FULL, provenance included, plus the allocation pct.
    assert pinned.config == {**paper_config(workspace), "live_equity_allocation_pct": 10.0}
    # The paper slot itself is untouched.
    paper = store.get_promoted_strategy()
    assert paper is not None
    assert "live_equity_allocation_pct" not in paper.config
    # Metrics degrade-or-load from the workspace's own qlib_res.csv.
    assert result.metrics.get("IC") == pytest.approx(0.0432)


def test_no_candidate_anywhere_is_a_refusal(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite")
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError) as exc:
        live.promote(thread_ts="9999999999.000001")
    assert "no paper-promoted strategy" in str(exc.value)
    assert store.get_promoted_strategy_live() is None


def test_paper_copy_with_missing_workspace_is_refused(
    store: StateStore, tmp_path: Path
) -> None:
    gone = tmp_path / "deleted-workspace"
    store.set_promoted_strategy(str(gone), paper_config(gone))
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError, match="missing on disk"):
        live.promote()
    assert store.get_promoted_strategy_live() is None


# --- pred-refresh snapshot rules ---------------------------------------------------


def snapshot_files(workspace: Path) -> tuple[Path, Path, Path]:
    return (
        workspace / SNAPSHOT_CONF_NAME,
        workspace / SNAPSHOT_ENV_NAME,
        workspace / SNAPSHOT_PARAMS_NAME,
    )


def test_snapshot_created_when_artifacts_missing(store: StateStore, tmp_path: Path) -> None:
    snapshotted: list[Path] = []
    live = make_live(store, promotable_artifacts(tmp_path), snapshot=snapshotted.append)
    result = live.promote(thread_ts=THREAD)
    assert snapshotted == [tmp_path / "workspace"]
    assert result.warnings == ()


def test_complete_snapshot_is_never_touched(store: StateStore, tmp_path: Path) -> None:
    """A full snapshot may carry an operator-pinned market — never regenerate it."""
    artifacts = promotable_artifacts(tmp_path)
    for path in snapshot_files(Path(artifacts.workspace_path)):
        path.write_text("pinned")
    snapshotted: list[Path] = []
    live = make_live(store, artifacts, snapshot=snapshotted.append)
    result = live.promote(thread_ts=THREAD)
    assert snapshotted == []
    assert result.warnings == ()
    conf, _env, _params = snapshot_files(Path(artifacts.workspace_path))
    assert conf.read_text() == "pinned"


def test_partial_snapshot_regenerates_with_a_warning(
    store: StateStore, tmp_path: Path
) -> None:
    """conf present but env/params missing: the set must be completed, which
    overwrites the conf — warned about, never silent (promote_fetched rule)."""
    artifacts = promotable_artifacts(tmp_path)
    conf, _env, _params = snapshot_files(Path(artifacts.workspace_path))
    conf.write_text("market: us_liquid_promoted_30\n")
    snapshotted: list[Path] = []
    live = make_live(store, artifacts, snapshot=snapshotted.append)
    result = live.promote(thread_ts=THREAD)
    assert snapshotted == [Path(artifacts.workspace_path)]
    (warning,) = result.warnings
    assert SNAPSHOT_CONF_NAME in warning
    assert "operator-pinned" in warning


def test_snapshot_failure_refuses_and_writes_nothing(
    store: StateStore, tmp_path: Path
) -> None:
    def broken_snapshot(_workspace: Path) -> None:
        raise RuntimeError("no docker logs")

    live = make_live(store, promotable_artifacts(tmp_path), snapshot=broken_snapshot)
    with pytest.raises(LivePromotionError, match="no docker logs"):
        live.promote(thread_ts=THREAD)
    assert store.get_promoted_strategy_live() is None


# --- allocation + replacement ------------------------------------------------------


def test_unusable_allocation_config_refuses(store: StateStore, tmp_path: Path) -> None:
    def broken_allocation() -> LiveAllocation:
        raise AllocationConfigError("missing keys: live_equity_allocation_pct")

    live = make_live(
        store, promotable_artifacts(tmp_path), load_allocation=broken_allocation
    )
    with pytest.raises(LivePromotionError, match="allocation config"):
        live.promote(thread_ts=THREAD)
    assert store.get_promoted_strategy_live() is None


def test_re_promoting_reports_what_was_replaced(store: StateStore, tmp_path: Path) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    first = live.promote(thread_ts=THREAD)
    assert first.replaced is None
    second = live.promote(thread_ts=THREAD)
    assert second.replaced is not None
    assert second.replaced.workspace_path == first.promoted.workspace_path


# --- Decision Log (mocked HTTP) -----------------------------------------------------


def make_recorder(store: StateStore, session: FakeSession) -> NotionRecorder:
    return NotionRecorder(
        NotionClient(session=session, sleep=lambda _s: None, max_retries=0), DBS, store
    )


def test_promote_writes_a_decision_log_row(store: StateStore, tmp_path: Path) -> None:
    store.create_directive(THREAD, "12-1 momentum in liquid US names")
    store.set_notion_page("idea", THREAD, "page-idea")
    session = FakeSession([page_response("page-dec")])
    live = make_live(
        store, promotable_artifacts(tmp_path), recorder=make_recorder(store, session)
    )
    live.promote(thread_ts=THREAD, trigger_permalink=PERMALINK)

    (create,) = session.calls
    body = create["json"]
    assert body["parent"] == {"type": "database_id", "database_id": DBS.decision_log}
    props = body["properties"]
    title = props["Decision"]["title"][0]["text"]["content"]
    assert title == "Promote '12-1 momentum in liquid US names' to LIVE trading"
    assert props["Type"] == {"select": {"name": "promote_live"}}
    assert props["Idea"] == {"relation": [{"id": "page-idea"}]}
    details = props["Details"]["rich_text"][0]["text"]["content"]
    assert str(tmp_path / "workspace") in details
    assert "us_liquid (2 tickers pinned)" in details
    assert "Allocation: 10.0% of live equity" in details
    assert f"Triggered by: {PERMALINK}" in details


def test_paper_copy_decision_names_the_source(store: StateStore, tmp_path: Path) -> None:
    artifacts = promotable_artifacts(tmp_path)
    workspace = Path(artifacts.workspace_path)
    store.set_promoted_strategy(str(workspace), paper_config(workspace))
    session = FakeSession([page_response("page-dec")])
    live = make_live(store, artifacts, recorder=make_recorder(store, session))
    live.promote()

    (create,) = session.calls
    details = create["json"]["properties"]["Details"]["rich_text"][0]["text"]["content"]
    assert "paper-promoted strategy (copied in full)" in details
    assert "us_liquid_promoted_30 (3 tickers pinned)" in details


def test_demote_clears_only_the_live_slot_and_records(
    store: StateStore, tmp_path: Path
) -> None:
    artifacts = promotable_artifacts(tmp_path)
    workspace = Path(artifacts.workspace_path)
    store.set_promoted_strategy(str(workspace), paper_config(workspace))
    session = FakeSession([page_response("page-p"), page_response("page-d")])
    live = make_live(store, artifacts, recorder=make_recorder(store, session))
    live.promote(thread_ts=THREAD)

    demoted = live.demote(trigger_permalink=PERMALINK)

    assert demoted.workspace_path == str(workspace)
    assert store.get_promoted_strategy_live() is None
    assert store.get_promoted_strategy() is not None  # paper slot untouched
    demote_call = session.calls[-1]
    props = demote_call["json"]["properties"]
    assert props["Type"] == {"select": {"name": "demote_live"}}
    details = props["Details"]["rich_text"][0]["text"]["content"]
    assert "next live rebalance will abort" in details
    assert f"Triggered by: {PERMALINK}" in details


def test_demote_with_empty_slot_refuses(store: StateStore, tmp_path: Path) -> None:
    live = make_live(store, promotable_artifacts(tmp_path))
    with pytest.raises(LivePromotionError, match="already empty"):
        live.demote()

"""Tests for ops/sweep.py — workspace retention sweep (US-041).

Fixture directory trees mirror the real layout: workspace dirs under
``<run_root>/*/workspace/``, trace logs under
``<run_root>/server_ui/traces/<trace>/Loop_<n>/<step>/<tag>/<pid>/<ts>.pkl``
(the FileStorage layout locate_artifacts reads). SOTA pkls use
SimpleNamespace stand-ins, same as tests/test_gpu_backend.py.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.sweep import (
    DEFAULT_MAX_AGE_DAYS,
    REASON_PROMOTED,
    REASON_RECENT_PROMOTION,
    REASON_SOTA,
    SweepError,
    SweepPlan,
    build_plan,
    execute,
    main,
    newest_mtime,
    promoted_workspace,
    recent_promotion_workspaces,
    repo_prune_actions,
    sota_workspaces,
)
from orchestrator.state import StateStore

NOW = time.time()
OLD = NOW - 30 * 86400.0  # well past the default 14-day cutoff
FRESH = NOW - 1 * 86400.0


def write_pkl(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


def experiment(workspace: Path, subs: list[Path] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_workspace=SimpleNamespace(workspace_path=str(workspace)),
        sub_workspace_list=[
            SimpleNamespace(workspace_path=str(sub)) for sub in (subs or [])
        ],
    )


def age_tree(path: Path, mtime: float) -> None:
    """Force every file AND dir in the tree to one mtime (dirs last —
    creating children bumps parent dir mtimes)."""
    entries = [path, *path.rglob("*")]
    for entry in sorted(entries, key=lambda p: len(p.parts), reverse=True):
        os.utime(entry, (mtime, mtime))
    os.utime(path, (mtime, mtime))


def make_workspace(root: Path, name: str, mtime: float) -> Path:
    workspace = root / "server_ui" / "workspace" / name
    workspace.mkdir(parents=True)
    (workspace / "qlib_res.csv").write_text("metric,0.1\n")
    age_tree(workspace, mtime)
    return workspace


def log_loop(
    trace: Path, loop: int, exp: SimpleNamespace | None, decision: bool | None
) -> None:
    """One loop's runner-result + feedback pkls in the FileStorage layout."""
    loop_dir = trace / f"Loop_{loop}"
    if exp is not None:
        write_pkl(loop_dir / "running" / "runner result" / "123" / "0001.pkl", exp)
    if decision is not None:
        write_pkl(
            loop_dir / "feedback" / "feedback" / "123" / "0001.pkl",
            SimpleNamespace(decision=decision),
        )


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "rdq-runs"
    (root / "server_ui" / "traces").mkdir(parents=True)
    (root / "server_ui" / "workspace").mkdir(parents=True)
    return root


def trace_dir(run_root: Path, name: str = "Finance Whole Pipeline/t1") -> Path:
    return run_root / "server_ui" / "traces" / name


class TestSotaWorkspaces:
    def test_decision_true_protects_experiment_and_sub_workspaces(
        self, run_root: Path
    ) -> None:
        sota = make_workspace(run_root, "ws_sota", OLD)
        sub = make_workspace(run_root, "ws_sub", OLD)
        rejected = make_workspace(run_root, "ws_rejected", OLD)
        trace = trace_dir(run_root)
        log_loop(trace, 0, experiment(sota, subs=[sub]), decision=True)
        log_loop(trace, 1, experiment(rejected), decision=False)

        protected = sota_workspaces([run_root / "server_ui" / "traces"])
        assert sota.resolve() in protected
        assert sub.resolve() in protected
        assert rejected.resolve() not in protected

    def test_unreadable_feedback_is_treated_as_sota(self, run_root: Path) -> None:
        workspace = make_workspace(run_root, "ws_unknown", OLD)
        trace = trace_dir(run_root)
        log_loop(trace, 0, experiment(workspace), decision=None)
        corrupt = trace / "Loop_0" / "feedback" / "feedback" / "123" / "0001.pkl"
        corrupt.parent.mkdir(parents=True)
        corrupt.write_bytes(b"not a pickle")

        assert workspace.resolve() in sota_workspaces([run_root / "server_ui" / "traces"])

    def test_loopless_runner_result_is_protected(self, run_root: Path) -> None:
        """A runner result that can't be tied to a loop keeps its workspace."""
        workspace = make_workspace(run_root, "ws_loopless", OLD)
        trace = trace_dir(run_root)
        write_pkl(trace / "runner result" / "123" / "0001.pkl", experiment(workspace))

        assert workspace.resolve() in sota_workspaces([run_root / "server_ui" / "traces"])

    def test_no_feedback_loop_is_not_sota(self, run_root: Path) -> None:
        workspace = make_workspace(run_root, "ws_nofb", OLD)
        log_loop(trace_dir(run_root), 0, experiment(workspace), decision=None)

        assert workspace.resolve() not in sota_workspaces(
            [run_root / "server_ui" / "traces"]
        )


class TestPromotedWorkspace:
    def test_missing_db_means_nothing_promoted(self, tmp_path: Path) -> None:
        assert promoted_workspace(tmp_path / "absent.sqlite") is None
        # reading must never create the orchestrator's database
        assert not (tmp_path / "absent.sqlite").exists()

    def test_reads_the_promoted_row(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        workspace = tmp_path / "ws_promoted"
        workspace.mkdir()
        StateStore(db).set_promoted_strategy(str(workspace), {"topk": 5})

        assert promoted_workspace(db) == workspace.resolve()

    def test_unreadable_db_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        db.write_bytes(b"garbage, not sqlite")
        with pytest.raises(SweepError, match="refusing to sweep"):
            promoted_workspace(db)


class TestRecentPromotionWorkspaces:
    def test_missing_db_means_no_history(self, tmp_path: Path) -> None:
        assert recent_promotion_workspaces(tmp_path / "absent.sqlite") == set()
        assert not (tmp_path / "absent.sqlite").exists()

    def test_reads_the_newest_limit_entries(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        store = StateStore(db)
        workspaces = [tmp_path / f"ws_{i}" for i in range(4)]
        for workspace in workspaces:
            workspace.mkdir()
            store.set_promoted_strategy(str(workspace), {"topk": 5})

        recent = recent_promotion_workspaces(db)
        assert recent == {w.resolve() for w in workspaces[1:]}  # last 3, oldest dropped

    def test_unreadable_db_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "state.sqlite"
        db.write_bytes(b"garbage, not sqlite")
        with pytest.raises(SweepError, match="refusing to sweep"):
            recent_promotion_workspaces(db)


class TestBuildPlan:
    def test_last_three_promotion_history_workspaces_protected(
        self, run_root: Path, tmp_path: Path
    ) -> None:
        """AC US-006: rollback targets (last 3 history entries) survive the
        sweep however old; older history entries do not."""
        ancient, previous2, previous1, current = (
            make_workspace(run_root, name, OLD)
            for name in ("ws_ancient", "ws_prev2", "ws_prev1", "ws_current")
        )
        db = tmp_path / "state.sqlite"
        store = StateStore(db)
        for workspace in (ancient, previous2, previous1, current):
            store.set_promoted_strategy(str(workspace), {"topk": 5})

        plan = build_plan(run_root, state_db=db, now=NOW)

        assert plan.protected == {
            current: REASON_PROMOTED,
            previous1: REASON_RECENT_PROMOTION,
            previous2: REASON_RECENT_PROMOTION,
        }
        assert [a.path for a in plan.delete] == [ancient]


    def test_old_unprotected_deleted_sota_and_promoted_never(
        self, run_root: Path, tmp_path: Path
    ) -> None:
        """AC: promoted + SOTA workspaces are never deleted, however old."""
        sota = make_workspace(run_root, "ws_sota", OLD)
        promoted = make_workspace(run_root, "ws_promoted", OLD)
        rejected = make_workspace(run_root, "ws_rejected", OLD)
        orphan = make_workspace(run_root, "ws_orphan", OLD)  # no trace reference
        young = make_workspace(run_root, "ws_young", FRESH)
        trace = trace_dir(run_root)
        log_loop(trace, 0, experiment(sota), decision=True)
        log_loop(trace, 1, experiment(rejected), decision=False)
        db = tmp_path / "state.sqlite"
        StateStore(db).set_promoted_strategy(str(promoted), {"topk": 5})

        plan = build_plan(run_root, state_db=db, now=NOW)

        assert plan.protected == {sota: REASON_SOTA, promoted: REASON_PROMOTED}
        assert {a.path for a in plan.delete} == {rejected, orphan}
        assert all(a.kind == "workspace" for a in plan.delete)
        assert plan.kept_recent == [young]

    def test_age_cutoff_is_configurable(self, run_root: Path, tmp_path: Path) -> None:
        workspace = make_workspace(run_root, "ws", NOW - 3 * 86400.0)
        db = tmp_path / "state.sqlite"

        keep = build_plan(run_root, max_age_days=7, state_db=db, now=NOW)
        assert keep.delete == [] and keep.kept_recent == [workspace]

        drop = build_plan(run_root, max_age_days=2, state_db=db, now=NOW)
        assert [a.path for a in drop.delete] == [workspace]

    def test_recent_write_anywhere_in_tree_keeps_the_workspace(
        self, run_root: Path, tmp_path: Path
    ) -> None:
        """Age = newest mtime in the tree: an active run's workspace with an
        old top dir but a fresh deep file must survive."""
        workspace = make_workspace(run_root, "ws_active", OLD)
        deep = workspace / "mlruns" / "1" / "r1" / "artifacts"
        deep.mkdir(parents=True)
        (deep / "pred.pkl").write_bytes(b"x")
        age_tree(workspace, OLD)
        os.utime(deep / "pred.pkl", (FRESH, FRESH))

        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert plan.delete == []
        assert plan.kept_recent == [workspace]

    def test_stale_mlruns_swept_inside_surviving_unprotected_workspace(
        self, run_root: Path, tmp_path: Path
    ) -> None:
        workspace = make_workspace(run_root, "ws", FRESH)
        old_run = workspace / "mlruns" / "1" / "old_run"
        new_run = workspace / "mlruns" / "1" / "new_run"
        old_run.mkdir(parents=True)
        new_run.mkdir(parents=True)
        age_tree(old_run, OLD)
        os.utime(new_run, (FRESH, FRESH))

        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert [(a.path, a.kind) for a in plan.delete] == [(old_run, "mlrun")]

    def test_protected_workspace_mlruns_untouched(
        self, run_root: Path, tmp_path: Path
    ) -> None:
        """Never reach inside a SOTA/promoted workspace — its mlruns hold the
        pred.pkl a promotion needs."""
        workspace = make_workspace(run_root, "ws_sota", FRESH)
        old_run = workspace / "mlruns" / "1" / "old_run"
        old_run.mkdir(parents=True)
        age_tree(old_run, OLD)
        log_loop(trace_dir(run_root), 0, experiment(workspace), decision=True)

        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert plan.delete == []
        assert plan.protected == {workspace: REASON_SOTA}

    def test_symlinked_candidate_is_skipped(self, run_root: Path, tmp_path: Path) -> None:
        target = tmp_path / "outside"
        target.mkdir()
        link = run_root / "server_ui" / "workspace" / "ws_link"
        link.symlink_to(target)
        os.utime(link, (OLD, OLD), follow_symlinks=False)

        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert plan.delete == []
        assert target.exists()

    def test_refuses_unsafe_run_root(self, tmp_path: Path) -> None:
        with pytest.raises(SweepError, match="unsafe run root"):
            build_plan(Path("/"), state_db=tmp_path / "db.sqlite", now=NOW)

    def test_cli_wrapper_log_traces_also_count(self, run_root: Path, tmp_path: Path) -> None:
        """SOTA evidence in a run_us_quant.sh LOG_TRACE_PATH (~/rdq-runs/
        us_quant/log/<ts>/) protects workspaces too."""
        workspace_root = run_root / "us_quant" / "workspace"
        workspace_root.mkdir(parents=True)
        workspace = workspace_root / "ws_cli_sota"
        workspace.mkdir()
        age_tree(workspace, OLD)
        log_loop(run_root / "us_quant" / "log" / "2026-07-01", 0, experiment(workspace), True)

        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert plan.protected == {workspace: REASON_SOTA}
        assert plan.delete == []


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """A stand-in checkout with rdagent's CWD-relative droppings (US-030)."""
    root = tmp_path / "repo"
    (root / "log").mkdir(parents=True)
    (root / "pickle_cache" / "utils.env.run").mkdir(parents=True)
    return root


def make_log_trace(repo_root: Path, name: str, mtime: float) -> Path:
    trace = repo_root / "log" / name
    (trace / "sub").mkdir(parents=True)
    (trace / "sub" / "0001.pkl").write_bytes(b"x")
    age_tree(trace, mtime)
    return trace


def make_cache_file(repo_root: Path, name: str, mtime: float) -> Path:
    cached = repo_root / "pickle_cache" / "utils.env.run" / name
    cached.write_bytes(b"x")
    os.utime(cached, (mtime, mtime))
    return cached


class TestRepoPruneActions:
    def test_stale_log_traces_and_cache_files_pruned(self, repo_root: Path) -> None:
        old_trace = make_log_trace(repo_root, "2026-07-01_00-00-00-000000", OLD)
        fresh_trace = make_log_trace(repo_root, "2026-08-14_00-00-00-000000", FRESH)
        old_cache = make_cache_file(repo_root, "aa.pkl", OLD)
        old_lock = make_cache_file(repo_root, "aa.lock", OLD)
        fresh_cache = make_cache_file(repo_root, "bb.pkl", FRESH)

        actions = repo_prune_actions(repo_root, now=NOW)
        assert {a.path for a in actions} == {old_trace, old_cache, old_lock}
        kinds = {a.path: a.kind for a in actions}
        assert kinds[old_trace] == "log trace"
        assert kinds[old_cache] == "cache file"
        assert fresh_trace.is_dir() and fresh_cache.is_file()

    def test_retention_boundary_at_cutoff_is_kept(self, repo_root: Path) -> None:
        cutoff = NOW - DEFAULT_MAX_AGE_DAYS * 86400.0
        make_log_trace(repo_root, "at_cutoff", cutoff)
        make_cache_file(repo_root, "at.pkl", cutoff)
        just_over_trace = make_log_trace(repo_root, "just_over", cutoff - 1)
        just_over_cache = make_cache_file(repo_root, "over.pkl", cutoff - 1)

        actions = repo_prune_actions(repo_root, now=NOW)
        assert {a.path for a in actions} == {just_over_trace, just_over_cache}

    def test_fresh_write_inside_old_trace_keeps_it(self, repo_root: Path) -> None:
        trace = make_log_trace(repo_root, "active_run", OLD)
        os.utime(trace / "sub" / "0001.pkl", (FRESH, FRESH))
        assert repo_prune_actions(repo_root, now=NOW) == []

    def test_cache_function_dirs_and_other_paths_untouched(self, repo_root: Path) -> None:
        """Only entries INSIDE log/ and pickle_cache/<fn>/ are planned — the
        function dirs themselves and everything else in the checkout stay."""
        function_dir = repo_root / "pickle_cache" / "utils.env.run"
        old_cache = make_cache_file(repo_root, "aa.pkl", OLD)
        other = repo_root / "orchestrator"
        other.mkdir()
        (other / "state.sqlite").write_bytes(b"x")
        stray = repo_root / "selector.log"
        stray.write_text("x")
        age_tree(repo_root, OLD)

        actions = repo_prune_actions(repo_root, now=NOW)
        assert [a.path for a in actions] == [old_cache]
        assert function_dir.is_dir() and other.is_dir() and stray.is_file()

    def test_missing_dirs_are_a_noop(self, tmp_path: Path) -> None:
        assert repo_prune_actions(tmp_path / "bare_repo", now=NOW) == []

    def test_symlinks_never_followed_or_planned(self, repo_root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "f.pkl").write_bytes(b"x")
        age_tree(outside, OLD)
        (repo_root / "log" / "linked").symlink_to(outside)
        (repo_root / "pickle_cache" / "utils.env.run" / "l.pkl").symlink_to(outside / "f.pkl")

        assert repo_prune_actions(repo_root, now=NOW) == []
        assert (outside / "f.pkl").is_file()

    def test_max_age_days_is_configurable(self, repo_root: Path) -> None:
        cached = make_cache_file(repo_root, "aa.pkl", NOW - 5 * 86400.0)
        assert repo_prune_actions(repo_root, max_age_days=7, now=NOW) == []
        actions = repo_prune_actions(repo_root, max_age_days=2, now=NOW)
        assert [a.path for a in actions] == [cached]

    def test_refuses_unsafe_repo_root(self) -> None:
        with pytest.raises(SweepError, match="unsafe repo root"):
            repo_prune_actions(Path("/"), now=NOW)

    def test_execute_unlinks_planned_files(self, repo_root: Path) -> None:
        old_trace = make_log_trace(repo_root, "old", OLD)
        old_cache = make_cache_file(repo_root, "aa.pkl", OLD)
        actions = repo_prune_actions(repo_root, now=NOW)

        plan = SweepPlan(delete=actions, protected={}, kept_recent=[])
        assert execute(plan, dry_run=True) == []
        assert old_trace.is_dir() and old_cache.is_file()
        assert execute(plan, dry_run=False) == []
        assert not old_trace.exists() and not old_cache.exists()
        assert (repo_root / "pickle_cache" / "utils.env.run").is_dir()


class TestExecuteAndCli:
    def test_dry_run_deletes_nothing(self, run_root: Path, tmp_path: Path) -> None:
        workspace = make_workspace(run_root, "ws_old", OLD)
        plan = build_plan(run_root, state_db=tmp_path / "db.sqlite", now=NOW)
        assert [a.path for a in plan.delete] == [workspace]

        assert execute(plan, dry_run=True) == []
        assert workspace.is_dir()

        assert execute(plan, dry_run=False) == []
        assert not workspace.exists()

    def test_main_sweeps_and_reports(
        self, run_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sota = make_workspace(run_root, "ws_sota", OLD)
        old = make_workspace(run_root, "ws_old", OLD)
        log_loop(trace_dir(run_root), 0, experiment(sota), decision=True)

        code = main(
            [
                "--run-root",
                str(run_root),
                "--state-db",
                str(tmp_path / "db.sqlite"),
                "--repo-root",
                str(tmp_path / "repo"),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert not old.exists()
        assert sota.is_dir()
        assert f"protected ({REASON_SOTA}): {sota}" in out
        assert "deleting workspace" in out

    def test_main_dry_run_flag(
        self, run_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = make_workspace(run_root, "ws_old", OLD)
        code = main(
            [
                "--run-root",
                str(run_root),
                "--state-db",
                str(tmp_path / "db.sqlite"),
                "--repo-root",
                str(tmp_path / "repo"),
                "--dry-run",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert old.is_dir()
        assert "would delete workspace" in out
        assert "nothing deleted" in out

    def test_main_prunes_repo_droppings(
        self, run_root: Path, repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old_trace = make_log_trace(repo_root, "2026-07-01_00-00-00-000000", OLD)
        fresh_trace = make_log_trace(repo_root, "2026-08-14_00-00-00-000000", FRESH)
        old_cache = make_cache_file(repo_root, "aa.pkl", OLD)

        code = main(
            [
                "--run-root",
                str(run_root),
                "--state-db",
                str(tmp_path / "db.sqlite"),
                "--repo-root",
                str(repo_root),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert not old_trace.exists() and not old_cache.exists()
        assert fresh_trace.is_dir()
        assert "deleting log trace" in out
        assert "deleting cache file" in out

    def test_main_missing_run_root_still_prunes_repo(
        self, repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old_cache = make_cache_file(repo_root, "aa.pkl", OLD)
        code = main(
            [
                "--run-root",
                str(tmp_path / "absent"),
                "--state-db",
                str(tmp_path / "db.sqlite"),
                "--repo-root",
                str(repo_root),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "nothing to sweep" in out
        assert not old_cache.exists()

    def test_main_missing_run_root_is_a_noop(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["--run-root", str(tmp_path / "absent"), "--repo-root", str(tmp_path / "repo")]
        )
        assert code == 0
        assert "nothing to sweep" in capsys.readouterr().out

    def test_main_rejects_nonpositive_days(self, run_root: Path) -> None:
        assert main(["--run-root", str(run_root), "--days", "0"]) == 1


class TestNewestMtime:
    def test_walks_the_whole_tree(self, tmp_path: Path) -> None:
        root = tmp_path / "tree"
        deep = root / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "f.txt").write_text("x")
        age_tree(root, OLD)
        assert newest_mtime(root) == pytest.approx(OLD)
        os.utime(deep / "f.txt", (FRESH, FRESH))
        assert newest_mtime(root) == pytest.approx(FRESH)

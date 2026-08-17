"""Workspace retention sweep (US-041): reclaim disk, never touch tradable state.

RD-Agent runs leave one workspace directory per experiment under
``~/rdq-runs/*/workspace/`` (plus mlflow run dirs inside long-lived
workspaces). This sweep deletes the ones nothing can ever need again:

- workspace dirs older than ``--days`` that are neither the PROMOTED
  strategy's workspace (the ``promoted_strategy`` row in the orchestrator's
  SQLite — what the nightly rebalancer trades) nor one of the last
  ``RECENT_PROMOTIONS_KEPT`` promotion_history workspaces (US-006 rollback
  targets) nor a SOTA workspace,
- mlflow run dirs (``<workspace>/mlruns/<exp>/<run>``) older than ``--days``
  inside surviving unprotected workspaces,
- repo-root rdagent droppings (US-030): ``log/<run-ts>/`` trace trees and
  ``pickle_cache/<function>/<hash>.{pkl,lock,zip}`` cache files under this
  checkout older than ``--days``. Only those two directories are ever
  touched; the pickle_cache function dirs themselves are kept (rdagent
  re-creates them anyway, and keeping them makes the prune idempotent).

SOTA is derived from the on-disk trace logs, the same FileStorage layout
``locate_artifacts`` reads: each loop logs its finished experiment under
``Loop_<n>/**/runner result/**/*.pkl`` (object carries
``experiment_workspace.workspace_path`` + ``sub_workspace_list``) and its
verdict under ``Loop_<n>/**/feedback/**/*.pkl`` (object carries
``decision``). A loop whose feedback decision is truthy protects every
workspace its experiment references. Unreadable feedback pkls and
uncorrelatable runner results are treated as SOTA — when in doubt, keep.

Age = the NEWEST lstat mtime anywhere in the tree, so anything an active run
is still writing into can never look old, whatever ``--days`` is.

Usage (also ops/rdq-sweep.timer, weekly):
    .venv/bin/python -m ops.sweep [--days N] [--dry-run] [--repo-root DIR]

Exit codes: 0 = swept (or nothing to do), 1 = operational failure.
"""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.state import DEFAULT_DB_PATH, StateStore

DEFAULT_RUN_ROOT = Path("~/rdq-runs")
DEFAULT_MAX_AGE_DAYS = 14.0

# The checkout this module lives in — rdagent drops CWD-relative log/ traces
# and pickle_cache/ entries here on every invocation (both gitignored).
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

_SECONDS_PER_DAY = 86400.0
_LOOP_DIR_RE = re.compile(r"^Loop_\d+$")

# Protection reasons (report labels; tests assert on them).
REASON_PROMOTED = "promoted"
REASON_RECENT_PROMOTION = "recent promotion"
REASON_SOTA = "SOTA"

# How many promotion_history entries keep their workspaces sweep-proof
# (US-006: ops.rollback_promotion can only restore a workspace that still
# exists on disk).
RECENT_PROMOTIONS_KEPT = 3


class SweepError(RuntimeError):
    """The sweep could not determine what is safe to delete."""


@dataclass(frozen=True)
class SweepAction:
    """One directory tree the plan wants gone."""

    path: Path
    kind: str  # "workspace" | "mlrun" | "log trace" | "cache file"
    age_days: float


@dataclass(frozen=True)
class SweepPlan:
    delete: list[SweepAction]
    protected: dict[Path, str]  # workspace dir -> protection reason
    kept_recent: list[Path]  # unprotected but younger than the cutoff


def newest_mtime(path: Path) -> float:
    """Newest lstat mtime in the tree (symlinks not followed)."""
    newest = path.lstat().st_mtime
    if not path.is_dir() or path.is_symlink():
        return newest
    stack = [path]
    while stack:
        current = stack.pop()
        for entry in current.iterdir():
            newest = max(newest, entry.lstat().st_mtime)
            if entry.is_dir() and not entry.is_symlink():
                stack.append(entry)
    return newest


def promoted_workspace(db_path: Path = DEFAULT_DB_PATH) -> Path | None:
    """The promoted strategy's workspace path, or None when nothing is promoted.

    Never creates the orchestrator's database (StateStore migrates on init,
    so only open a file that already exists). A present-but-unreadable DB is
    a SweepError: without it we cannot prove the promoted workspace is safe.
    """
    db_path = Path(db_path).expanduser()
    if not db_path.is_file():
        return None
    try:
        promoted = StateStore(db_path).get_promoted_strategy()
    except Exception as exc:
        raise SweepError(
            f"cannot read the promoted strategy from {db_path} ({exc}); "
            "refusing to sweep without it"
        ) from exc
    if promoted is None:
        return None
    return Path(promoted.workspace_path).expanduser().resolve()


def recent_promotion_workspaces(
    db_path: Path = DEFAULT_DB_PATH, limit: int = RECENT_PROMOTIONS_KEPT
) -> set[Path]:
    """Workspaces of the most recent ``limit`` promotion_history entries.

    These stay sweep-proof so ops.rollback_promotion always has somewhere to
    roll back to. Same guard rails as promoted_workspace: never create the
    DB, and a present-but-unreadable DB aborts the sweep.
    """
    db_path = Path(db_path).expanduser()
    if not db_path.is_file():
        return set()
    try:
        history = StateStore(db_path).list_promotion_history(limit=limit)
    except Exception as exc:
        raise SweepError(
            f"cannot read promotion history from {db_path} ({exc}); "
            "refusing to sweep without it"
        ) from exc
    return {Path(entry.workspace_path).expanduser().resolve() for entry in history}


def _loop_dir_of(pkl_file: Path) -> Path | None:
    """The pkl's ``Loop_<n>`` ancestor dir — the key runner results and
    feedbacks of the same loop share."""
    for ancestor in pkl_file.parents:
        if _LOOP_DIR_RE.match(ancestor.name):
            return ancestor
    return None


def _load_pkl(pkl_file: Path) -> object | None:
    try:
        with pkl_file.open("rb") as handle:
            return pickle.load(handle)
    except Exception:  # noqa: BLE001 - any unpickle failure means "unknown"
        return None


def _experiment_workspaces(exp: object) -> set[Path]:
    """Every workspace dir a logged experiment object references."""
    found: set[Path] = set()
    workspace = getattr(getattr(exp, "experiment_workspace", None), "workspace_path", None)
    if workspace is not None:
        found.add(Path(workspace).expanduser().resolve())
    for sub in getattr(exp, "sub_workspace_list", None) or []:
        sub_path = getattr(sub, "workspace_path", None)
        if sub_path is not None:
            found.add(Path(sub_path).expanduser().resolve())
    return found


def sota_workspaces(trace_roots: list[Path]) -> set[Path]:
    """Workspace dirs referenced by loops whose feedback decision was truthy.

    Conservative on every unknown: an unreadable feedback pkl counts as
    SOTA, and a runner result that cannot be tied to a loop protects its
    workspaces unconditionally.
    """
    protected: set[Path] = set()
    for root in trace_roots:
        if not root.is_dir():
            continue
        decisions: dict[Path, bool] = {}
        for pkl_file in root.glob("**/feedback/**/*.pkl"):
            loop_dir = _loop_dir_of(pkl_file)
            if loop_dir is None:
                continue
            feedback = _load_pkl(pkl_file)
            decision = True if feedback is None else bool(getattr(feedback, "decision", False))
            decisions[loop_dir] = decisions.get(loop_dir, False) or decision
        for pkl_file in root.glob("**/runner result/**/*.pkl"):
            exp = _load_pkl(pkl_file)
            if exp is None:
                continue
            loop_dir = _loop_dir_of(pkl_file)
            if loop_dir is None or decisions.get(loop_dir, False):
                protected |= _experiment_workspaces(exp)
    return protected


def discover_workspace_roots(run_root: Path) -> list[Path]:
    return sorted(p for p in run_root.glob("*/workspace") if p.is_dir())


def discover_trace_roots(run_root: Path) -> list[Path]:
    """server_ui trace folder + the CLI wrappers' LOG_TRACE_PATH parents."""
    roots = [run_root / "server_ui" / "traces"]
    roots.extend(sorted(run_root.glob("*/log")))
    return [p for p in roots if p.is_dir()]


def _is_protected(candidate: Path, protected: set[Path]) -> bool:
    for path in protected:
        if path == candidate or path.is_relative_to(candidate):
            return True
    return False


def _stale_mlruns(workspace: Path, cutoff: float, now: float) -> list[SweepAction]:
    """mlflow run dirs (mlruns/<exp>/<run>) whose whole tree predates cutoff."""
    stale: list[SweepAction] = []
    mlruns = workspace / "mlruns"
    if not mlruns.is_dir() or mlruns.is_symlink():
        return stale
    for exp_dir in sorted(mlruns.iterdir()):
        if not exp_dir.is_dir() or exp_dir.is_symlink():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            mtime = newest_mtime(run_dir)
            if mtime < cutoff:
                age = (now - mtime) / _SECONDS_PER_DAY
                stale.append(SweepAction(path=run_dir, kind="mlrun", age_days=age))
    return stale


def repo_prune_actions(
    repo_root: Path = DEFAULT_REPO_ROOT,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    now: float | None = None,
) -> list[SweepAction]:
    """Stale rdagent droppings under the repo checkout (US-030).

    Two directories, nothing else: ``log/`` (one trace tree per invocation —
    pruned per entry, aged by the NEWEST mtime so an active run's trace never
    looks old) and ``pickle_cache/`` (function-result cache — pruned per FILE
    inside the per-function dirs, which themselves are kept). Absent dirs are
    a no-op; symlinks are never followed or deleted.
    """
    repo_root = repo_root.expanduser().resolve()
    if repo_root in (Path("/"), Path.home().resolve()):
        raise SweepError(f"refusing to prune {repo_root} (unsafe repo root)")
    now = time.time() if now is None else now
    cutoff = now - max_age_days * _SECONDS_PER_DAY

    actions: list[SweepAction] = []

    def consider(path: Path, kind: str, mtime: float) -> None:
        if mtime < cutoff:
            age = (now - mtime) / _SECONDS_PER_DAY
            actions.append(SweepAction(path=path, kind=kind, age_days=age))

    log_root = repo_root / "log"
    if log_root.is_dir() and not log_root.is_symlink():
        for entry in sorted(log_root.iterdir()):
            if entry.is_symlink():
                continue
            consider(entry, "log trace", newest_mtime(entry))

    cache_root = repo_root / "pickle_cache"
    if cache_root.is_dir() and not cache_root.is_symlink():
        for entry in sorted(cache_root.iterdir()):
            if entry.is_symlink():
                continue
            if not entry.is_dir():
                consider(entry, "cache file", entry.lstat().st_mtime)
                continue
            for cached in sorted(entry.iterdir()):
                if cached.is_symlink() or cached.is_dir():
                    continue
                consider(cached, "cache file", cached.lstat().st_mtime)
    return actions


def build_plan(
    run_root: Path,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    state_db: Path = DEFAULT_DB_PATH,
    now: float | None = None,
) -> SweepPlan:
    run_root = run_root.expanduser().resolve()
    if run_root in (Path("/"), Path.home().resolve()):
        raise SweepError(f"refusing to sweep {run_root} (unsafe run root)")
    now = time.time() if now is None else now
    cutoff = now - max_age_days * _SECONDS_PER_DAY

    protected_paths: set[Path] = set()
    pinned = promoted_workspace(state_db)
    if pinned is not None:
        protected_paths.add(pinned)
    recent_promotions = recent_promotion_workspaces(state_db)
    protected_paths |= recent_promotions
    protected_paths |= sota_workspaces(discover_trace_roots(run_root))

    delete: list[SweepAction] = []
    protected: dict[Path, str] = {}
    kept_recent: list[Path] = []
    for root in discover_workspace_roots(run_root):
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if _is_protected(resolved, protected_paths):
                if pinned is not None and (
                    pinned == resolved or pinned.is_relative_to(resolved)
                ):
                    protected[candidate] = REASON_PROMOTED
                elif _is_protected(resolved, recent_promotions):
                    protected[candidate] = REASON_RECENT_PROMOTION
                else:
                    protected[candidate] = REASON_SOTA
                continue
            mtime = newest_mtime(candidate)
            if mtime < cutoff:
                age = (now - mtime) / _SECONDS_PER_DAY
                delete.append(SweepAction(path=candidate, kind="workspace", age_days=age))
            else:
                kept_recent.append(candidate)
                delete.extend(_stale_mlruns(candidate, cutoff, now))
    return SweepPlan(delete=delete, protected=protected, kept_recent=kept_recent)


def execute(plan: SweepPlan, *, dry_run: bool) -> list[str]:
    """Delete the plan's trees; returns failure messages (empty = clean)."""
    failures: list[str] = []
    for action in plan.delete:
        if dry_run:
            continue
        try:
            if action.path.is_dir() and not action.path.is_symlink():
                shutil.rmtree(action.path)
            else:
                action.path.unlink()
        except OSError as exc:
            failures.append(f"failed to delete {action.path}: {exc}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ops.sweep",
        description="Workspace retention sweep: reclaim disk, never touch tradable state.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"delete unprotected trees older than this (default {DEFAULT_MAX_AGE_DAYS:g})",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="root holding */workspace and trace dirs (default ~/rdq-runs)",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="orchestrator SQLite holding the promoted strategy",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="checkout whose log/ and pickle_cache/ droppings get pruned (default: this repo)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without deleting anything"
    )
    args = parser.parse_args(argv)
    if args.days <= 0:
        print("--days must be positive", file=sys.stderr)
        return 1

    run_root = args.run_root.expanduser()
    try:
        if run_root.is_dir():
            plan = build_plan(run_root, max_age_days=args.days, state_db=args.state_db)
        else:
            print(f"nothing to sweep: {run_root} does not exist")
            plan = SweepPlan(delete=[], protected={}, kept_recent=[])
        repo_actions = repo_prune_actions(args.repo_root, max_age_days=args.days)
    except SweepError as exc:
        print(f"sweep aborted: {exc}", file=sys.stderr)
        return 1
    plan = SweepPlan(
        delete=[*plan.delete, *repo_actions],
        protected=plan.protected,
        kept_recent=plan.kept_recent,
    )

    for path, reason in sorted(plan.protected.items()):
        print(f"protected ({reason}): {path}")
    for path in plan.kept_recent:
        print(f"kept (recent): {path}")
    verb = "would delete" if args.dry_run else "deleting"
    for action in plan.delete:
        print(f"{verb} {action.kind} (age {action.age_days:.1f}d): {action.path}")

    failures = execute(plan, dry_run=args.dry_run)
    for failure in failures:
        print(failure, file=sys.stderr)
    summary = (
        f"{len(plan.delete)} deletable, {len(plan.protected)} protected, "
        f"{len(plan.kept_recent)} kept recent"
    )
    print(f"dry run: {summary}; nothing deleted" if args.dry_run else f"swept: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Automated daily prediction refresh for the promoted strategy (US-048).

Two halves, both promoted-workspace-local:

* **Snapshot (promote time).** ``snapshot_pred_refresh(workspace)`` writes the
  two files a refresh needs so there is no log archaeology later:
  ``conf_pred_refresh.yaml`` — the conf the SOTA run actually used (the sota
  variant when its combined-factors parquet still exists, else the baseline)
  with ``record:`` reduced to SignalRecord ONLY — and ``pred_refresh.env`` —
  the rendered jinja context recovered from the workspace's
  docker_execution logs (plus ``num_features`` from the training log, and the
  container env QTDockerEnv always sets). Called by the promotion flow
  (orchestrator/promotion.py); a failure there warns the operator instead of
  blocking the promotion.

* **Refresh (every trading morning).** ``run_pred_refresh()`` re-runs the
  snapshot conf in the local_qlib docker image with ``test_end=<today NY>``
  (same mounts as QTDockerEnv), self-checks that the newest pred.pkl now
  passes the rebalancer's freshness gate, prunes old refresh runs, and posts
  the outcome to Slack. Exit codes: 0 = refreshed / already fresh / nothing
  promoted (clean skip); 1 = failed (the rebalancer's stale-pred abort is the
  backstop). Wired to rdq-pred-refresh.timer between the 06:30 data refresh
  and the 08:00 rebalance.

Deliberate semantics (task doc "design decisions"):

* The refresh RE-FITS the model (~13 min GRU on CPU) — the traded model is a
  fresh stochastic re-fit, not the exact promoted weights. Exact-weights
  re-predict from params.pkl is an explicit non-goal/follow-up.
* SignalRecord-only sidesteps qlib's end-of-calendar IndexError (the backtest
  indexes calendar_index + 1 past test_end), so ``test_end=today`` is safe
  even though the store ends yesterday pre-open.
* ``chmod -R 777 mlruns`` runs INSIDE the container (QTDockerEnv parity) —
  the new files are root-owned, so a host-side chmod would be too late.
* qrun output streams to ``logs/pred_refresh_<date>.log`` (a real file, never
  a pipe: an attached container whose stdout pipe loses its reader freezes
  qlib training mid-epoch — see docs 2026-07-15 incident).
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from execution import signal
from execution.promoted import NoPromotedStrategyError, load_promoted_strategy
from execution.rebalance import DEFAULT_STORE_PATH, MARKET_TZ, Notify, _safe_notify
from orchestrator.state import DEFAULT_DB_PATH, PromotedStrategy

# Single source of truth lives in signal.py: load_market must skip this conf
# (its market may be operator-pinned to freeze the traded universe).
SNAPSHOT_CONF_NAME = signal.PRED_REFRESH_CONF_NAME
SNAPSHOT_ENV_NAME = "pred_refresh.env"

DEFAULT_IMAGE = "local_qlib:latest"
DEFAULT_QLIB_DIR = Path("~/.qlib")
# 06:45 start + 50 min still ends comfortably before the 08:00 rebalance.
DEFAULT_TIMEOUT_MINUTES = 50.0
DEFAULT_KEEP_REFRESH_RUNS = 5

# qrun logs the jinja context it rendered with; this is the recovery anchor.
_CONTEXT_MARKER = "Render the template with the context: "

# QTDockerEnv always sets these for the workspace container; the snapshot
# persists them so the refresh needs nothing beyond the two files.
_CONTAINER_ENV = {
    "PYTHONPATH": "/workspace/qlib_workspace",
    "MLFLOW_ALLOW_FILE_STORE": "true",
}

# A refresh (SignalRecord-only) mlflow run logs exactly these artifacts; any
# run with anything else (portfolio_analysis, sig_analysis, ...) is a real
# research run and must never be pruned.
_REFRESH_ONLY_ARTIFACTS = frozenset({"pred.pkl", "label.pkl", "params.pkl"})

# Runner signature: (docker command, qrun log file, timeout seconds) -> rc.
Runner = Callable[[Sequence[str], Path, float], int]


class PredRefreshError(RuntimeError):
    """Any condition that must fail the refresh (the rebalance gate backstops)."""


# -- snapshot (promote time) --------------------------------------------------


def _static_loader_deps(conf_text: str) -> list[str]:
    """Workspace-relative files a conf's StaticDataLoader blocks reference."""
    return re.findall(r'config:\s*"([^"]+\.parquet)"', conf_text)


# Candidate source confs, most-specific first, per workspace family: model
# experiments write conf_{sota,baseline}_factors_model.yaml; factor
# experiments write conf_combined_factors{_sota_model,}.yaml + conf_baseline.yaml.
_SOURCE_CONF_CANDIDATES = (
    "conf_sota_factors_model.yaml",
    "conf_baseline_factors_model.yaml",
    "conf_combined_factors_sota_model.yaml",
    "conf_combined_factors.yaml",
    "conf_baseline.yaml",
)


def choose_source_conf(workspace: Path) -> Path:
    """The conf the promoted run actually executed.

    The combined/sota variants need combined_factors_df.parquet, which
    RD-Agent cleans up after runs without SOTA factor experiments — a conf is
    only the right source when its data dependencies still exist; otherwise
    the plain baseline conf is what produced the promoted artifacts.
    """
    for name in _SOURCE_CONF_CANDIDATES:
        conf = workspace / name
        if not conf.is_file():
            continue
        deps = _static_loader_deps(conf.read_text())
        if all((workspace / dep).is_file() for dep in deps):
            return conf
    raise PredRefreshError(
        f"no usable source conf in {workspace}: need one of "
        f"{', '.join(_SOURCE_CONF_CANDIDATES)} with its parquet dependencies "
        "still on disk"
    )


def reduce_records(conf_text: str) -> str:
    """Drop every ``record:`` entry except SignalRecord, preserving the text.

    Text-level surgery on purpose: the conf keeps its jinja placeholders
    (qrun renders them at run time), so a yaml round-trip would destroy it.
    Dropping SigAnaRecord/PortAnaRecord is what makes ``test_end=today`` safe
    when the store's calendar ends yesterday (no backtest, no end-of-calendar
    IndexError).
    """
    lines = conf_text.splitlines()
    record_idx: int | None = None
    record_indent = 0
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)record:\s*$", line)
        if match:
            record_idx = i
            record_indent = len(match.group(1))
    if record_idx is None:
        raise PredRefreshError("conf has no record: block to reduce")

    end = len(lines)
    for j in range(record_idx + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped and len(lines[j]) - len(lines[j].lstrip()) <= record_indent:
            end = j
            break

    items: list[list[str]] = []
    for line in lines[record_idx + 1 : end]:
        if re.match(r"^\s*-\s+class:", line):
            items.append([line])
        elif items:
            items[-1].append(line)
        elif line.strip():
            raise PredRefreshError(f"unexpected line inside record block: {line!r}")
    kept = [item for item in items if re.search(r"class:\s*SignalRecord\s*$", item[0])]
    if not kept:
        raise PredRefreshError("conf's record block has no SignalRecord entry")

    reduced = lines[: record_idx + 1] + [line for item in kept for line in item] + lines[end:]
    return "\n".join(reduced).rstrip("\n") + "\n"


def recover_context(workspace: Path) -> dict[str, str]:
    """The jinja context the promoted run rendered with, from its docker logs.

    qrun logs the full context dict; the newest log wins and the last
    occurrence within it is the run that produced the artifacts.
    ``num_features`` is not always a context var (the baseline conf hardcodes
    it) — when absent, the training log's ``'num_features': N`` is used.
    """
    logs = sorted(
        (workspace / "logs").glob("docker_execution_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    context: dict[str, str] | None = None
    for log in logs:
        text = log.read_text(errors="replace")
        pos = text.rfind(_CONTEXT_MARKER)
        if pos < 0:
            continue
        line = text[pos + len(_CONTEXT_MARKER) :].splitlines()[0].strip()
        try:
            parsed = ast.literal_eval(line)
        except (ValueError, SyntaxError) as exc:
            raise PredRefreshError(f"cannot parse the render context in {log}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise PredRefreshError(f"render context in {log} is not a dict: {line[:120]}")
        context = {str(key): str(value) for key, value in parsed.items()}
        break
    if context is None:
        raise PredRefreshError(
            f"no '{_CONTEXT_MARKER.rstrip(': ')}' line under {workspace / 'logs'} — "
            "cannot recover the jinja context the promoted run used"
        )
    if "num_features" not in context:
        for log in logs:
            match = re.search(r"'num_features':\s*(\d+)", log.read_text(errors="replace"))
            if match:
                context["num_features"] = match.group(1)
                break
    return context


def snapshot_pred_refresh(workspace: Path) -> tuple[Path, Path]:
    """Write conf_pred_refresh.yaml + pred_refresh.env into the workspace.

    Everything a later refresh needs, captured while the run's logs still
    exist. Overwrites any previous snapshot (promotion is the source of
    truth). Returns (conf_path, env_path).
    """
    workspace = workspace.expanduser()
    source = choose_source_conf(workspace)
    conf_text = reduce_records(source.read_text())
    env = recover_context(workspace)
    env.update(_CONTAINER_ENV)
    for key, value in env.items():
        if "=" in key or any("\n" in part for part in (key, value)):
            raise PredRefreshError(f"context entry {key!r} cannot be stored as KEY=VALUE")
    conf_path = workspace / SNAPSHOT_CONF_NAME
    env_path = workspace / SNAPSHOT_ENV_NAME
    conf_path.write_text(conf_text)
    env_path.write_text("".join(f"{key}={value}\n" for key, value in env.items()))
    return conf_path, env_path


# -- refresh (every trading morning) ------------------------------------------


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a pred_refresh.env snapshot (KEY=VALUE lines, # comments)."""
    entries: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PredRefreshError(f"bad line in {path}: {raw!r}")
        key, value = line.split("=", 1)
        entries[key] = value
    if not entries:
        raise PredRefreshError(f"snapshot env file {path} is empty")
    return entries


def docker_command(
    workspace: Path,
    env: dict[str, str],
    image: str = DEFAULT_IMAGE,
    qlib_dir: Path = DEFAULT_QLIB_DIR,
    name: str | None = None,
) -> list[str]:
    """The docker invocation (QTDockerEnv mount parity, chmod inside)."""
    command = ["docker", "run", "--rm"]
    if name is not None:
        command += ["--name", name]
    command += [
        "-v", f"{workspace}:/workspace/qlib_workspace",
        "-v", f"{qlib_dir.expanduser()}:/root/.qlib",
        "-w", "/workspace/qlib_workspace",
    ]
    for key in sorted(env):
        command += ["-e", f"{key}={env[key]}"]
    command += [
        image,
        "sh",
        "-c",
        f"qrun {SNAPSHOT_CONF_NAME}; rc=$?;"
        " chmod -R 777 /workspace/qlib_workspace/mlruns; exit $rc",
    ]
    return command


def _docker_runner(command: Sequence[str], log_path: Path, timeout_seconds: float) -> int:
    """Run docker with qrun output appended to a real file (never a pipe)."""
    with log_path.open("ab") as log:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PredRefreshError(
                f"docker qrun exceeded {timeout_seconds / 60:.0f} min — the container may "
                "still be running; check `docker ps` and kill any stuck rdq-pred-refresh "
                "container before rerunning"
            ) from exc
    return proc.returncode


def _fresh_cross_section(
    workspace: Path, as_of: dt.date, calendar_path: Path
) -> tuple[dt.date, object] | None:
    """(pred_date, scores) if the rebalancer's freshness gate passes right now."""
    try:
        pred_date, scores = signal.load_latest_cross_section(signal.locate_pred(workspace))
        signal.assert_fresh(pred_date, as_of, calendar_path)
    except signal.SignalError:
        return None
    return pred_date, scores


def _model_class(conf_path: Path) -> str | None:
    """The model class the snapshot conf trains (e.g. GeneralPTNN).

    Matches only a bare ``model:`` block header — the SignalRecord kwarg
    ``model: <MODEL>`` has text after the colon and never matches.
    """
    if not conf_path.is_file():
        return None
    match = re.search(r"^\s*model:\s*\n\s*class:\s*(\S+)", conf_path.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def describe_promoted(promoted: PromotedStrategy) -> str:
    """One Slack line of recall: WHICH strategy these predictions belong to.

    Every field is optional — the line degrades gracefully rather than ever
    failing a refresh over a sparse promoted config.
    """
    workspace = Path(promoted.workspace_path).expanduser()
    config = promoted.config or {}
    bits: list[str] = []
    model = _model_class(workspace / SNAPSHOT_CONF_NAME)
    universe = config.get("universe")
    tickers = config.get("universe_tickers") or []
    if universe:
        scope = f"{model} on {universe}" if model else str(universe)
        if tickers:
            scope += f" ({len(tickers)} tickers)"
        bits.append(scope)
    elif model:
        bits.append(model)
    topk, n_drop = config.get("topk"), config.get("n_drop")
    if topk is not None and n_drop is not None:
        bits.append(f"top-{topk}/drop-{n_drop} selection")
    bits.append(f"workspace {workspace.name[:8]}")
    if promoted.promoted_at:
        bits.append(f"promoted {promoted.promoted_at[:10]}")
    return "strategy: " + ", ".join(bits)


def describe_cross_section(pred_date: dt.date, scores: object, top_n: int = 5) -> str:
    """One Slack line on the scores the rebalancer will trade from."""
    ordered = scores.sort_values(ascending=False)  # type: ignore[attr-defined]
    shown = ordered.head(top_n)
    top = ", ".join(f"{sym} {val:+.3f}" for sym, val in shown.items())
    count = len(ordered)
    return (
        f"latest cross-section {pred_date}: {count} name{'s' if count != 1 else ''} scored; "
        f"top {len(shown)}: {top}"
    )


def prune_refresh_runs(workspace: Path, keep: int = DEFAULT_KEEP_REFRESH_RUNS) -> int:
    """Bound disk growth: keep the newest ``keep`` refresh runs, delete the rest.

    Only runs whose artifacts are exactly the SignalRecord set are candidates
    — anything with extra artifacts is a real research run (the promoted
    backtest itself) and is never touched. Returns the number removed.
    """
    if keep < 1:
        raise PredRefreshError(f"keep must be >= 1, got {keep}")
    candidates: list[tuple[float, Path]] = []
    for pred in workspace.glob("mlruns/*/*/artifacts/pred.pkl"):
        if ".trash" in pred.parts:
            continue
        artifacts = pred.parent
        children = list(artifacts.iterdir())
        if any(child.is_dir() or child.name not in _REFRESH_ONLY_ARTIFACTS for child in children):
            continue
        candidates.append((pred.stat().st_mtime, artifacts.parent))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    removed = 0
    for _, run_dir in candidates[keep:]:
        shutil.rmtree(run_dir)
        removed += 1
    return removed


def run_pred_refresh(
    notify: Notify,
    as_of: dt.date | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    store_path: Path = DEFAULT_STORE_PATH,
    qlib_dir: Path = DEFAULT_QLIB_DIR,
    image: str = DEFAULT_IMAGE,
    timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES,
    keep_runs: int = DEFAULT_KEEP_REFRESH_RUNS,
    runner: Runner = _docker_runner,
) -> int:
    """Refresh the promoted workspace's predictions for as_of; exit code.

    0 = refreshed (self-check passed), already fresh, or nothing promoted
    (clean skip); 1 = failed — posted to Slack, and the rebalancer's own
    stale-pred abort remains the backstop.
    """
    if as_of is None:
        as_of = dt.datetime.now(MARKET_TZ).date()
    calendar_path = store_path.expanduser() / "calendars" / "day.txt"
    try:
        try:
            promoted = load_promoted_strategy(db_path)
        except NoPromotedStrategyError as exc:
            message = f"pred refresh skipped ({as_of}): {exc}"
            _safe_notify(notify, message)
            print(message)
            return 0
        workspace = Path(promoted.workspace_path).expanduser()
        conf_path = workspace / SNAPSHOT_CONF_NAME
        env_path = workspace / SNAPSHOT_ENV_NAME
        if not conf_path.is_file() or not env_path.is_file():
            raise PredRefreshError(
                f"pred-refresh snapshot missing in {workspace}: need {SNAPSHOT_CONF_NAME} "
                f"and {SNAPSHOT_ENV_NAME} — re-promote the strategy (promotion snapshots "
                "them) or run execution.pred_refresh.snapshot_pred_refresh by hand"
            )
        fresh = _fresh_cross_section(workspace, as_of, calendar_path)
        if fresh is not None:
            pred_date, scores = fresh
            message = (
                f"pred refresh ({as_of}): predictions already fresh — nothing to do"
                f"\n• {describe_promoted(promoted)}"
                f"\n• {describe_cross_section(pred_date, scores)}"
            )
            _safe_notify(notify, message)
            print(message)
            return 0

        env = load_env_file(env_path)
        env.update(_CONTAINER_ENV)
        env["test_end"] = as_of.isoformat()
        log_path = workspace / "logs" / f"pred_refresh_{as_of:%Y%m%d}.log"
        log_path.parent.mkdir(exist_ok=True)
        command = docker_command(
            workspace, env, image=image, qlib_dir=qlib_dir, name=f"rdq-pred-refresh-{as_of}"
        )
        started = time.monotonic()
        returncode = runner(command, log_path, timeout_minutes * 60)
        if returncode != 0:
            raise PredRefreshError(f"docker qrun exited {returncode} — see {log_path}")

        # Self-check: the exact gate the rebalancer will apply.
        pred_path = signal.locate_pred(workspace)
        pred_date, scores = signal.load_latest_cross_section(pred_path)
        signal.assert_fresh(pred_date, as_of, calendar_path)
        pruned = prune_refresh_runs(workspace, keep=keep_runs)

        minutes = (time.monotonic() - started) / 60
        message = (
            f"pred refresh ({as_of}): regenerated promoted-strategy predictions "
            f"for the 08:00 ET paper rebalance — {minutes:.0f} min"
            f"\n• {describe_promoted(promoted)}"
            f"\n• {describe_cross_section(pred_date, scores)}"
        )
        if pruned:
            message += f"\n• pruned {pruned} old refresh run(s)"
        _safe_notify(notify, message)
        print(message)
        return 0
    except (PredRefreshError, signal.SignalError) as exc:
        message = f"pred refresh FAILED ({as_of}): {exc}"
        _safe_notify(notify, message)
        print(message, file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected bug: tell the operator, then crash loudly
        _safe_notify(notify, f"pred refresh CRASHED ({as_of}): {exc!r}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the promoted strategy's predictions for today (US-048)"
    )
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        help="YYYY-MM-DD (default: today in America/New_York)",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    parser.add_argument("--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout-minutes", type=float, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--keep-runs", type=int, default=DEFAULT_KEEP_REFRESH_RUNS)
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="print notices to stderr instead of Slack (supervised local runs)",
    )
    args = parser.parse_args(argv)

    from execution.rebalance import slack_notifier, stderr_notifier
    from orchestrator.config import ConfigError

    if args.no_slack:
        notify = stderr_notifier()
    else:
        try:
            notify = slack_notifier()
        except ConfigError as exc:
            print(
                f"ERROR: {exc}\nRefusing to run unattended without a Slack channel for "
                "failure notices; pass --no-slack for a supervised local run.",
                file=sys.stderr,
            )
            return 1

    return run_pred_refresh(
        notify,
        as_of=args.as_of,
        db_path=args.db_path,
        store_path=args.store,
        qlib_dir=args.qlib_dir,
        image=args.image,
        timeout_minutes=args.timeout_minutes,
        keep_runs=args.keep_runs,
    )


if __name__ == "__main__":
    sys.exit(main())

"""Refresh-parity harness for new store fields (US-069).

Proof that a factor built on a given store field survives the exact-weights
re-predict path — the c9587797 failure class (2026-08-15: a refresh re-predict
whose scores no longer reproduced the backtested strategy), closed by
measurement instead of hope. One run per field:

1. **Train + backtest.** A throwaway workspace under ``~/rdq-runs/parity_check/
   <field>/`` gets a fully specified LGBM conf (``conf_baseline.yaml`` — one of
   the names ``choose_source_conf`` recognizes) whose features are minimal
   expressions REFERENCING the field (market-level fields enter as excess
   momentum / rolling correlation against ``$close`` so the factor still has a
   cross-section; per-ticker fields enter directly). ``qrun`` runs it in the
   same local_qlib image the research/refresh stack uses; qrun's own "Render
   the template with the context" log line lands in ``logs/`` where
   ``recover_context`` expects it.
2. **Snapshot.** ``execution.pred_refresh.snapshot_pred_refresh`` — the real
   promotion-time path, including the task-signature conf match that the
   c9587797 fix added.
3. **Exact-weights re-predict.** ``ops.confirm_window._run_repredict`` with
   ``test_end`` pushed to the store calendar end — byte-for-byte the daily
   refresh / confirmation machinery.
4. **Compare.** Mean per-day Spearman between the re-predicted pred and the
   original backtested pred over every overlapping cross-section; the run
   passes at >= ``--min-spearman`` (default 0.98, the confirmation gate's
   ``min_reproduction``). Failure output names the field and the measured
   value.

CPU-only and control-box-local. Entry points::

    python -m ops.parity_check --field '$mkt_brent'
    python -m ops.parity_check --all-mkt          # sweep every $mkt_* field
    python -m ops.parity_check --field '$close'   # OHLCV-only regression guard

The dates default to a 50/25/25 train/valid/test split of the store trading
days from ``--start`` (default MARKET_SERIES_START, 2025-01-02 — the market
series carry no data before it), with the last ``TEST_END_LAG_DAYS`` trading
days held out so the training run's PortAnaRecord backtest never indexes past
the calendar end. Workspaces are recreated from scratch on every run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import shutil
import subprocess
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.build_store import MARKET_FIELDS, MARKET_SERIES_START
from execution import pred_refresh
from execution.pred_refresh import (
    PredRefreshError,
    Runner,
    _docker_runner,
    snapshot_pred_refresh,
)
from execution.rebalance import DEFAULT_STORE_PATH
from ops.confirm_window import (
    ConfirmWindowError,
    _calendar,
    _cross_section,
    _load_pred_series,
    _pred_paths,
    _run_repredict,
)

# Must be a name choose_source_conf recognizes (a test binds them).
CONF_NAME = "conf_baseline.yaml"
# Named docker_execution_* so recover_context's glob finds qrun's context line.
TRAIN_LOG_NAME = "docker_execution_train.log"

DEFAULT_ROOT = Path("~/rdq-runs/parity_check")
DEFAULT_MARKET = "us_liquid"
DEFAULT_MIN_SPEARMAN = 0.98
DEFAULT_TIMEOUT_MINUTES = 40.0
DEFAULT_TOPK = 50
DEFAULT_N_DROP = 5
# Trading days held out past test_end so PortAnaRecord (which indexes one
# calendar entry past its end_time) never falls off the store calendar, and so
# the re-predict demonstrably EXTENDS the pred beyond the backtested window.
TEST_END_LAG_DAYS = 5
_MIN_COMMON_NAMES = 3


class ParityError(RuntimeError):
    """Any condition that must fail the parity check loudly (never silent)."""


@dataclass(frozen=True)
class ParityDates:
    """Train/valid/test segments, all inclusive ISO-friendly dates."""

    train_start: dt.date
    train_end: dt.date
    valid_start: dt.date
    valid_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def context(self) -> dict[str, str]:
        """The jinja context qrun renders the conf with (env-var transport)."""
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "valid_start": self.valid_start.isoformat(),
            "valid_end": self.valid_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass(frozen=True)
class ParityResult:
    field: str
    spearman: float
    days: int  # overlap days the spearman was averaged over
    passed: bool


def derive_dates(
    calendar: Sequence[dt.date],
    start: dt.date,
    lag: int = TEST_END_LAG_DAYS,
) -> ParityDates:
    """50/25/25 train/valid/test split of the trading days from ``start``,
    holding the last ``lag`` days out past test_end."""
    days = [day for day in calendar if day >= start]
    usable = days[: len(days) - lag] if lag else days
    n = len(usable)
    train_n = n // 2
    valid_n = n // 4
    test_n = n - train_n - valid_n
    if train_n < 2 or valid_n < 1 or test_n < 2:
        raise ParityError(
            f"only {len(days)} trading day(s) on/after {start} (lag {lag}) — "
            "not enough for a train/valid/test split"
        )
    train = usable[:train_n]
    valid = usable[train_n : train_n + valid_n]
    test = usable[train_n + valid_n :]
    return ParityDates(train[0], train[-1], valid[0], valid[-1], test[0], test[-1])


def feature_spec(field: str, market_level: bool) -> tuple[list[str], list[str]]:
    """Minimal qlib feature expressions referencing ``field`` (with ``$``).

    Market-level series are identical across instruments, so raw use would be
    cross-sectionally constant (degenerate scores, unverifiable spearman) —
    they enter relative to each ticker's own $close instead. Per-ticker fields
    enter directly, denominator-guarded so zero-heavy series (news counts)
    stay finite.
    """
    if market_level:
        return (
            [
                f"$close/Ref($close, 5) - {field}/Ref({field}, 5)",
                f"$close/Ref($close, 20) - {field}/Ref({field}, 20)",
                f"Corr($close/Ref($close, 1), {field}/Ref({field}, 1), 20)",
            ],
            ["PARITY_EXCESS5", "PARITY_EXCESS20", "PARITY_CORR20"],
        )
    return (
        [
            f"{field}/(Mean({field}, 20)+1e-12)",
            f"{field}/(Ref({field}, 5)+1e-12)",
            f"Std({field}, 10)/(Mean({field}, 20)+1e-12)",
        ],
        ["PARITY_REL20", "PARITY_MOM5", "PARITY_VOL10"],
    )


# Jinja placeholders (rendered by qrun from the container env, exactly like the
# research templates) — that is what makes the snapshotted conf re-renderable
# with test_end overridden at re-predict time. Model/handler blocks mirror
# research/us_templates/factor_template/conf_baseline.yaml except num_threads
# (the control box has 4 cores).
_CONF_TEMPLATE = """\
qlib_init:
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us

market: &market {market}
benchmark: &benchmark SPY

data_handler_config: &data_handler_config
    start_time: {{{{ train_start | default("2025-01-02", true) }}}}
    end_time: {{{{ test_end | default("null", true) }}}}
    instruments: *market
    data_loader:
        class: NestedDataLoader
        kwargs:
            dataloader_l:
                - class: qlib.contrib.data.loader.Alpha158DL
                  kwargs:
                    config:
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                            - ["LABEL0"]
                        feature:
                            - [{expressions}]
                            - [{names}]
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
              fit_start_time: {{{{ train_start | default("2025-01-02", true) }}}}
              fit_end_time: {{{{ train_end | default("2025-12-31", true) }}}}
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: {topk}
            n_drop: {n_drop}
    backtest:
        start_time: {{{{ test_start | default("2026-04-01", true) }}}}
        end_time: {{{{ test_end | default("null", true) }}}}
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0005
            min_cost: 0

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 4
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train:
                    - {{{{ train_start | default("2025-01-02", true) }}}}
                    - {{{{ train_end | default("2025-12-31", true) }}}}
                valid:
                    - {{{{ valid_start | default("2026-01-02", true) }}}}
                    - {{{{ valid_end | default("2026-03-31", true) }}}}
                test:
                    - {{{{ test_start | default("2026-04-01", true) }}}}
                    - {{{{ test_end | default("null", true) }}}}
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""


def render_conf(
    field: str,
    market_level: bool,
    market: str = DEFAULT_MARKET,
    topk: int = DEFAULT_TOPK,
    n_drop: int = DEFAULT_N_DROP,
) -> str:
    expressions, names = feature_spec(field, market_level)
    return _CONF_TEMPLATE.format(
        market=market,
        expressions=", ".join(f'"{expr}"' for expr in expressions),
        names=", ".join(f'"{name}"' for name in names),
        topk=topk,
        n_drop=n_drop,
    )


def resolve_field(store: Path, field: str) -> tuple[str, bool]:
    """(canonical $-name, is-market-level) of a field, from the data menu."""
    from data import menu as data_menu

    name = field if field.startswith("$") else f"${field}"
    try:
        menu = data_menu.build_menu(store)
    except data_menu.MenuError as exc:
        raise ParityError(f"cannot read the store menu at {store}: {exc}") from exc
    for entry in menu.fields:
        if entry.name == name:
            return name, entry.kind == data_menu.KIND_MARKET
    raise ParityError(
        f"field {name} is not in the store at {store} — store fields: "
        f"{', '.join(menu.field_names())}"
    )


def train_command(
    workspace: Path,
    env: dict[str, str],
    image: str = pred_refresh.DEFAULT_IMAGE,
    qlib_dir: Path = pred_refresh.DEFAULT_QLIB_DIR,
) -> list[str]:
    """The qrun docker invocation (pred_refresh.docker_command mount parity)."""
    command = [
        "docker", "run", "--rm",
        "--name", f"rdq-parity-{workspace.name[:24]}",
        "--shm-size", "2g",
        "-v", f"{workspace}:/workspace/qlib_workspace",
        "-v", f"{qlib_dir.expanduser()}:/root/.qlib",
        "-w", "/workspace/qlib_workspace",
    ]  # fmt: skip
    for key in sorted(env):
        command += ["-e", f"{key}={env[key]}"]
    command += [
        image,
        "sh",
        "-c",
        # chmod the WHOLE workspace, not just mlruns — qrun's backtest also
        # writes a root-owned nested workspace/qlib_workspace/mlruns/filelock
        # that would otherwise block the next run's rmtree.
        f"qrun {CONF_NAME}; rc=$?;"
        " chmod -R 777 /workspace/qlib_workspace; exit $rc",
    ]
    return command


def _force_writable(workspace: Path, image: str) -> None:
    """chmod a stale workspace from a throwaway container.

    An interrupted run's container files are root-owned (docker runs as root)
    and block host-side rmtree — reclaim them the same way the training
    command does on a clean exit.
    """
    command = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/workspace/qlib_workspace",
        image, "chmod", "-R", "777", "/workspace/qlib_workspace",
    ]  # fmt: skip
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ParityError(
            f"cannot reclaim root-owned files under {workspace}: docker chmod "
            f"exited {completed.returncode}: {completed.stderr.strip()[:200]}"
        )


def _remove_workspace(workspace: Path, image: str) -> None:
    """Delete a prior run's workspace, reclaiming root-owned docker leftovers."""
    try:
        shutil.rmtree(workspace)
    except PermissionError:
        _force_writable(workspace, image)
        shutil.rmtree(workspace)


def measured_reproduction(workspace: Path) -> tuple[float, int]:
    """(mean spearman, overlap days) of the newest pred vs the original.

    Same comparison as confirm_window's reproduction check but over EVERY
    overlapping cross-section (the harness wants the measured value, not just
    a pass/fail at a sample).
    """
    paths = _pred_paths(workspace)
    if len(paths) < 2:
        raise ParityError(
            f"need the backtested pred AND a re-predicted pred under {workspace}/mlruns "
            f"— found {len(paths)} pred.pkl file(s)"
        )
    original = _load_pred_series(paths[0])
    latest = _load_pred_series(paths[-1])
    if original is None or latest is None:
        raise ParityError(f"unreadable pred.pkl under {workspace}/mlruns")
    overlap = sorted(original.dates & latest.dates)
    if not overlap:
        raise ParityError(
            "re-predicted pred shares no prediction days with the backtested pred"
        )
    correlations: list[float] = []
    for day in overlap:
        original_cross = _cross_section(original.series, original.level, day)
        latest_cross = _cross_section(latest.series, latest.level, day)
        if original_cross is None or latest_cross is None:
            continue
        common = original_cross.index.intersection(latest_cross.index)
        if len(common) < _MIN_COMMON_NAMES:
            continue
        with warnings.catch_warnings():
            # A constant side makes scipy warn and return NaN — dropped below,
            # and an all-NaN comparison stays a loud error.
            warnings.simplefilter("ignore")
            corr = float(
                original_cross.loc[common].corr(latest_cross.loc[common], method="spearman")
            )
        if not math.isnan(corr):
            correlations.append(corr)
    if not correlations:
        raise ParityError(
            "no comparable cross-sections between the re-predicted and backtested "
            "preds (constant scores or disjoint instruments)"
        )
    return sum(correlations) / len(correlations), len(correlations)


def run_field(
    field: str,
    *,
    store: Path = DEFAULT_STORE_PATH,
    root: Path = DEFAULT_ROOT,
    market: str = DEFAULT_MARKET,
    start: dt.date = MARKET_SERIES_START,
    image: str = pred_refresh.DEFAULT_IMAGE,
    qlib_dir: Path = pred_refresh.DEFAULT_QLIB_DIR,
    timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES,
    min_spearman: float = DEFAULT_MIN_SPEARMAN,
    dates: ParityDates | None = None,
    runner: Runner = _docker_runner,
) -> ParityResult:
    """Train → snapshot → exact-weights re-predict → spearman, for one field.

    The workspace ``root/<field>`` is deleted and recreated. Raises
    ``ParityError`` on any gap; a completed comparison returns a
    ``ParityResult`` (pass or fail) instead of raising.
    """
    store = Path(store).expanduser()
    name, market_level = resolve_field(store, field)
    try:
        calendar = _calendar(store)
    except ConfirmWindowError as exc:
        raise ParityError(str(exc)) from exc
    if dates is None:
        dates = derive_dates(calendar, start)

    workspace = root.expanduser() / name.lstrip("$")
    if workspace.exists():
        _remove_workspace(workspace, image)
    (workspace / "logs").mkdir(parents=True)
    (workspace / CONF_NAME).write_text(render_conf(name, market_level, market=market))

    env = dates.context()
    env.update(pred_refresh._CONTAINER_ENV)
    log_path = workspace / "logs" / TRAIN_LOG_NAME
    command = train_command(workspace, env, image, qlib_dir)
    try:
        returncode = runner(command, log_path, timeout_minutes * 60)
    except PredRefreshError as exc:
        raise ParityError(f"{name}: training qrun failed: {exc}") from exc
    if returncode != 0:
        raise ParityError(f"{name}: training qrun exited {returncode} — see {log_path}")
    if not _pred_paths(workspace):
        raise ParityError(f"{name}: training produced no pred.pkl under {workspace}/mlruns")

    try:
        snapshot_pred_refresh(workspace)
    except PredRefreshError as exc:
        raise ParityError(f"{name}: pred-refresh snapshot failed: {exc}") from exc

    try:
        _run_repredict(
            workspace,
            calendar[-1],
            qlib_dir=qlib_dir,
            image=image,
            timeout_minutes=timeout_minutes,
            runner=runner,
        )
    except ConfirmWindowError as exc:
        raise ParityError(f"{name}: exact-weights re-predict failed: {exc}") from exc

    try:
        spearman, days = measured_reproduction(workspace)
    except ParityError as exc:
        raise ParityError(f"{name}: {exc}") from exc
    return ParityResult(name, spearman, days, spearman >= min_spearman)


def result_line(result: ParityResult, min_spearman: float) -> str:
    """One operator-facing line naming the field and the measured spearman."""
    if result.passed:
        return (
            f"PARITY PASS {result.field}: spearman {result.spearman:.4f} over "
            f"{result.days} overlap day(s) (>= {min_spearman:g})"
        )
    return (
        f"PARITY FAIL {result.field}: spearman {result.spearman:.4f} over "
        f"{result.days} overlap day(s) — need >= {min_spearman:g}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh-parity harness: train a minimal factor on a store field, "
            "re-predict from the exact weights, and require the scores to reproduce"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--field", help="store field to check, e.g. '$mkt_brent' or '$close'")
    group.add_argument(
        "--all-mkt", action="store_true", help="sweep every $mkt_* market broadcast field"
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE_PATH)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=MARKET_SERIES_START,
        help=f"first trading day of the split (default {MARKET_SERIES_START})",
    )
    parser.add_argument("--image", default=pred_refresh.DEFAULT_IMAGE)
    parser.add_argument("--qlib-dir", type=Path, default=pred_refresh.DEFAULT_QLIB_DIR)
    parser.add_argument("--timeout-minutes", type=float, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--min-spearman", type=float, default=DEFAULT_MIN_SPEARMAN)
    args = parser.parse_args(argv)

    fields = [f"${name}" for name in MARKET_FIELDS] if args.all_mkt else [args.field]
    failures: list[str] = []
    for field in fields:
        try:
            result = run_field(
                field,
                store=args.store,
                root=args.root,
                market=args.market,
                start=args.start,
                image=args.image,
                qlib_dir=args.qlib_dir,
                timeout_minutes=args.timeout_minutes,
                min_spearman=args.min_spearman,
            )
        except ParityError as exc:
            print(f"PARITY ERROR {field}: {exc}", file=sys.stderr)
            failures.append(field)
            continue
        line = result_line(result, args.min_spearman)
        if result.passed:
            print(line)
        else:
            print(line, file=sys.stderr)
            failures.append(field)
    print(f"parity: {len(fields) - len(failures)}/{len(fields)} field(s) passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

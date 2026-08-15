"""Codified candidate-vs-incumbent promotion gate (US-007).

"Beats the incumbent" is defined here, by code — not by judgment in a Slack
thread. ``evaluate_gate`` is PURE (no IO, no state): callers load one
``MetricBundle`` per strategy (``load_metric_bundle`` does the artifact IO)
and get a ``GateVerdict`` back, which serializes to JSON (the
promotion_history.gate_verdict payload) and renders to a Slack text block.

The decision has two layers:

1. Parity — the two backtests must be comparable at all: same test window,
   market, instrument-list hash, topk/n_drop, and cost params. Any mismatch,
   or either side MISSING one of those inputs, fails parity — and a parity
   failure fails the gate no matter how good the metrics look (the
   2026-08-12 no-promote was a window mismatch nothing surfaced).
2. Criteria — thresholds read from the ``promotion_gate:`` section of
   orchestrator/config.yaml (``load_gate_config``):
   - candidate IR strictly > incumbent IR × ir_margin (default 1.05)
   - candidate |MDD| ≤ incumbent |MDD| × mdd_tolerance (default 1.25)
   - candidate IC strictly > min_ic (default 0)
   - candidate confirmation-window IR strictly > incumbent's ×
     confirm_ir_margin (default 1.0 — no margin on the out-of-search-sample
     leg, US-010). Evidence comes from ``load_confirmation_evidence`` (the
     US-009 re-predict helper on both workspaces); a technical failure —
     re-predict error, degenerate returns, or NO evidence supplied — fails
     the criterion with reason ``confirmation_unavailable``, never a silent
     skip.
   With no incumbent on record nothing passes unless ``allow_first`` is set;
   allow_first waives the comparisons (confirmation included), not quality —
   IC is still required.

Boundary semantics follow the repo convention (order gate / breaker): exactly
AT the MDD tolerance passes; the IR margin and min_ic are strict ``>`` per
the PRD. Metric extraction reuses orchestrator/summary.py (METRIC_SPECS
labelling via ops.gpu_trace.workspace_metrics, ret.pkl Sharpe via
summary.compute_sharpe); topk/n_drop/market/costs come from
execution.signal's conf loaders.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "orchestrator" / "config.yaml"

PASS_MARK = ":white_check_mark:"
FAIL_MARK = ":x:"


class GateConfigError(RuntimeError):
    """The promotion_gate: section of config.yaml is malformed."""


@dataclass(frozen=True)
class GateConfig:
    """Pass criteria — defaults mirror the promotion_gate: section shipped
    in orchestrator/config.yaml (PRD US-002)."""

    ir_margin: float = 1.05
    mdd_tolerance: float = 1.25
    min_ic: float = 0.0
    allow_first: bool = False
    confirm_ir_margin: float = 1.0
    # Kill-switch (US-011): False = report-only, the pipeline posts the
    # verdict but never promotes. Does not affect the verdict itself.
    auto_promote: bool = True


def load_gate_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GateConfig:
    """Read the promotion_gate: section; a missing file/section means defaults."""
    import yaml

    loaded: Any = None
    if config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text())
        except Exception as exc:  # noqa: BLE001 — one actionable error type for callers
            raise GateConfigError(f"cannot parse {config_path}: {exc}") from exc
    section = loaded.get("promotion_gate") if isinstance(loaded, dict) else None
    if section is None:
        return GateConfig()
    if not isinstance(section, dict):
        raise GateConfigError(f"promotion_gate section in {config_path} must be a mapping")
    return GateConfig(
        ir_margin=_config_float(section, "ir_margin", GateConfig.ir_margin),
        mdd_tolerance=_config_float(section, "mdd_tolerance", GateConfig.mdd_tolerance),
        min_ic=_config_float(section, "min_ic", GateConfig.min_ic),
        allow_first=_config_bool(section, "allow_first", GateConfig.allow_first),
        confirm_ir_margin=_config_float(
            section, "confirm_ir_margin", GateConfig.confirm_ir_margin
        ),
        auto_promote=_config_bool(section, "auto_promote", GateConfig.auto_promote),
    )


def _config_float(section: Mapping[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise GateConfigError(f"promotion_gate.{key} must be a number, got {raw!r}")
    return float(raw)


def _config_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = section.get(key, default)
    if not isinstance(raw, bool):
        raise GateConfigError(f"promotion_gate.{key} must be a boolean, got {raw!r}")
    return raw


def hash_instruments(symbols: Iterable[str]) -> str:
    """Canonical instrument-list hash: the gate's universe-parity input.

    Sorted + deduped + newline-joined, sha256, first 16 hex chars. US-008
    records this at launch (pipeline_status.json); anything comparing two
    universes must hash through here so digests stay comparable.
    """
    names = sorted({str(symbol).strip() for symbol in symbols} - {""})
    if not names:
        raise ValueError("cannot hash an empty instrument list")
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MetricBundle:
    """One strategy's comparison inputs.

    ``metrics`` uses the operator-facing labels from summary.METRIC_SPECS
    (IC/ICIR/Rank IC/ARR/IR/MDD) plus ``Sharpe``. Parity fields are None when
    an artifact could not provide them — which FAILS parity against an
    incumbent (the gate never guesses). ``daily_returns`` is the net-of-cost
    ret.pkl series; unused by today's criteria, it is the confirmation-window
    input US-009/US-010 evaluate on.
    """

    workspace: str
    metrics: Mapping[str, float]
    window: tuple[str, str] | None = None  # first/last backtest day, ISO dates
    market: str | None = None
    instrument_hash: str | None = None
    topk: int | None = None
    n_drop: int | None = None
    cost_params: Mapping[str, float] | None = None
    daily_returns: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ConfirmationSide:
    """One strategy's confirmation-window evaluation (US-010).

    ``ir`` is ops.confirm_window.annualized_ir over the window's net daily
    returns — None when the window is degenerate (<2 days, zero variance),
    which the gate treats as confirmation_unavailable. ``reproduction`` is
    the re-predict fidelity score (spearman vs the original backtested pred;
    None when the used pred IS the original)."""

    workspace: str
    ir: float | None
    window: tuple[str, str]  # trading days actually evaluated, ISO, inclusive
    days: int
    repredicted: bool
    reproduction: float | None = None


@dataclass(frozen=True)
class ConfirmationEvidence:
    """Both strategies evaluated on the reserved confirmation window.

    ``error`` set means the evaluation itself failed (re-predict error,
    missing snapshot, non-reproducing pred, …) — the gate maps that to a
    failing ``confirmation_unavailable`` criterion, never a silent skip."""

    window: tuple[str, str]  # the requested confirmation window, ISO
    candidate: ConfirmationSide | None = None
    incumbent: ConfirmationSide | None = None
    error: str | None = None


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's outcome; ``reason`` always carries the numbers."""

    name: str
    passed: bool
    reason: str
    candidate: float | None = None
    incumbent: float | None = None


@dataclass(frozen=True)
class GateVerdict:
    parity_ok: bool
    passed: bool
    parity_mismatches: tuple[str, ...]
    criteria: tuple[CriterionResult, ...]
    candidate_workspace: str
    incumbent_workspace: str | None
    config: GateConfig
    confirmation: ConfirmationEvidence | None = None
    window: tuple[str, str] | None = None  # the (parity-checked) test window

    def to_dict(self) -> dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "pass": self.passed,
            "parity_mismatches": list(self.parity_mismatches),
            "criteria": [asdict(criterion) for criterion in self.criteria],
            "candidate_workspace": self.candidate_workspace,
            "incumbent_workspace": self.incumbent_workspace,
            "config": asdict(self.config),
            "confirmation": asdict(self.confirmation) if self.confirmation else None,
            "window": list(self.window) if self.window else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def slack_text(self) -> str:
        """Slack mrkdwn block: overall verdict, parity, one line per criterion."""
        candidate = _workspace_tag(self.candidate_workspace)
        incumbent = (
            _workspace_tag(self.incumbent_workspace) if self.incumbent_workspace else "none"
        )
        head = f"{PASS_MARK} PASS" if self.passed else f"{FAIL_MARK} FAIL"
        lines = [f"*Promotion gate:* {head} — candidate `{candidate}` vs incumbent `{incumbent}`"]
        if self.parity_mismatches:
            lines.extend(f"• {FAIL_MARK} parity: {text}" for text in self.parity_mismatches)
        elif self.incumbent_workspace is None:
            lines.append("• parity: no incumbent to compare against")
        else:
            parity = "window/market/instruments/topk-n_drop/costs all match"
            if self.window is not None:
                parity += f" (test window {self.window[0]} → {self.window[1]})"
            lines.append(f"• {PASS_MARK} parity: {parity}")
        for criterion in self.criteria:
            mark = PASS_MARK if criterion.passed else FAIL_MARK
            lines.append(f"• {mark} {criterion.name}: {criterion.reason}")
        if self.confirmation is not None and self.confirmation.error is None:
            window = self.confirmation.window
            sides = ", ".join(
                _confirmation_side_text(role, side)
                for role, side in (
                    ("candidate", self.confirmation.candidate),
                    ("incumbent", self.confirmation.incumbent),
                )
                if side is not None
            )
            if sides:
                lines.append(f"• confirmation window {window[0]} → {window[1]}: {sides}")
        return "\n".join(lines)


def _workspace_tag(workspace: str) -> str:
    return Path(workspace).name[:8] or "unknown"


def _confirmation_side_text(role: str, side: ConfirmationSide) -> str:
    ir = f"{side.ir:.4f}" if side.ir is not None else "n/a"
    source = "re-predicted" if side.repredicted else "cached pred"
    return f"{role} `{_workspace_tag(side.workspace)}` IR {ir} ({side.days}d, {source})"


# (bundle attribute, human label) — the inputs that must match for the two
# backtests to be comparable at all.
_PARITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("window", "test window"),
    ("market", "market"),
    ("instrument_hash", "instrument list"),
    ("topk", "topk"),
    ("n_drop", "n_drop"),
    ("cost_params", "cost params"),
)


def _format_parity(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return " → ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


def _parity_value(value: Any) -> Any:
    """Normalize for equality: sequences compare as tuples, mappings as dicts."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _check_parity(candidate: MetricBundle, incumbent: MetricBundle) -> list[str]:
    mismatches: list[str] = []
    for attribute, label in _PARITY_FIELDS:
        cand = getattr(candidate, attribute)
        inc = getattr(incumbent, attribute)
        if cand is None or inc is None:
            missing = (
                "both sides"
                if cand is None and inc is None
                else ("candidate" if cand is None else "incumbent")
            )
            mismatches.append(f"{label} unavailable on {missing} — parity cannot be verified")
        elif _parity_value(cand) != _parity_value(inc):
            mismatches.append(
                f"{label} mismatch (candidate {_format_parity(cand)}, "
                f"incumbent {_format_parity(inc)})"
            )
    return mismatches


def _ic_criterion(candidate: MetricBundle, config: GateConfig) -> CriterionResult:
    ic = candidate.metrics.get("IC")
    if ic is None:
        return CriterionResult("IC", False, "candidate IC unavailable")
    passed = ic > config.min_ic
    return CriterionResult(
        "IC",
        passed,
        f"candidate {ic:.4f} (need > {config.min_ic:.4f})",
        candidate=ic,
    )


def _ir_criterion(
    candidate: MetricBundle, incumbent: MetricBundle, config: GateConfig
) -> CriterionResult:
    cand = candidate.metrics.get("IR")
    inc = incumbent.metrics.get("IR")
    if cand is None or inc is None:
        side = "candidate" if cand is None else "incumbent"
        return CriterionResult("IR", False, f"{side} IR unavailable", cand, inc)
    threshold = inc * config.ir_margin
    return CriterionResult(
        "IR",
        cand > threshold,
        f"candidate {cand:.4f} vs incumbent {inc:.4f} × {config.ir_margin:g} "
        f"= {threshold:.4f} (need >)",
        candidate=cand,
        incumbent=inc,
    )


def _mdd_criterion(
    candidate: MetricBundle, incumbent: MetricBundle, config: GateConfig
) -> CriterionResult:
    cand = candidate.metrics.get("MDD")
    inc = incumbent.metrics.get("MDD")
    if cand is None or inc is None:
        side = "candidate" if cand is None else "incumbent"
        return CriterionResult("MDD", False, f"{side} MDD unavailable", cand, inc)
    # qlib reports MDD as a negative return; compare magnitudes so the sign
    # convention can never flip the verdict. At the tolerance passes.
    limit = abs(inc) * config.mdd_tolerance
    return CriterionResult(
        "MDD",
        abs(cand) <= limit,
        f"candidate {cand:+.2%} vs limit {-limit:.2%} "
        f"(incumbent {inc:+.2%} × {config.mdd_tolerance:g})",
        candidate=cand,
        incumbent=inc,
    )


def _confirmation_criterion(
    evidence: ConfirmationEvidence | None, config: GateConfig
) -> CriterionResult:
    """The out-of-search-sample leg (US-010): candidate must beat the
    incumbent on the reserved confirmation window. Anything short of two
    computed IRs is ``confirmation_unavailable`` — a re-predict failure can
    never be mistaken for a pass."""
    name = "confirmation"
    if evidence is None:
        return CriterionResult(
            name, False, "confirmation_unavailable — confirmation window was not evaluated"
        )
    if evidence.error is not None:
        return CriterionResult(name, False, f"confirmation_unavailable — {evidence.error}")
    irs: dict[str, float] = {}
    for role, side in (("candidate", evidence.candidate), ("incumbent", evidence.incumbent)):
        if side is None or side.ir is None:
            detail = (
                "not evaluated"
                if side is None
                else f"IR degenerate over {side.days} trading day(s)"
            )
            return CriterionResult(
                name, False, f"confirmation_unavailable — {role} window returns {detail}"
            )
        irs[role] = side.ir
    cand, inc = irs["candidate"], irs["incumbent"]
    threshold = inc * config.confirm_ir_margin
    return CriterionResult(
        name,
        cand > threshold,
        f"window {evidence.window[0]} → {evidence.window[1]}: candidate IR {cand:.4f} "
        f"vs incumbent {inc:.4f} × {config.confirm_ir_margin:g} = {threshold:.4f} (need >)",
        candidate=cand,
        incumbent=inc,
    )


def evaluate_gate(
    candidate: MetricBundle,
    incumbent: MetricBundle | None,
    config: GateConfig | None = None,
    confirmation: ConfirmationEvidence | None = None,
) -> GateVerdict:
    """PURE comparison: does the candidate beat the incumbent on this config?

    ``incumbent=None`` means nothing is promoted: the comparison legs are
    waived only under ``allow_first`` (IC still applies), otherwise the gate
    fails on the missing incumbent alone. With an incumbent, ``confirmation``
    evidence is REQUIRED — omitting it fails the confirmation criterion as
    confirmation_unavailable (the PRD forbids a silent skip).
    """
    config = config if config is not None else GateConfig()
    criteria: list[CriterionResult] = []
    if incumbent is None:
        mismatches: list[str] = []
        criteria.append(
            CriterionResult(
                "incumbent",
                config.allow_first,
                "no incumbent on record — comparison waived (allow_first)"
                if config.allow_first
                else "no incumbent on record and allow_first is off",
            )
        )
    else:
        mismatches = _check_parity(candidate, incumbent)
        criteria.append(_ir_criterion(candidate, incumbent, config))
        criteria.append(_mdd_criterion(candidate, incumbent, config))
    criteria.append(_ic_criterion(candidate, config))
    if incumbent is not None:
        criteria.append(_confirmation_criterion(confirmation, config))
    parity_ok = not mismatches
    return GateVerdict(
        parity_ok=parity_ok,
        passed=parity_ok and all(criterion.passed for criterion in criteria),
        parity_mismatches=tuple(mismatches),
        criteria=tuple(criteria),
        candidate_workspace=candidate.workspace,
        incumbent_workspace=incumbent.workspace if incumbent is not None else None,
        config=config,
        confirmation=confirmation,
        # tuple() re-wrap: workspace_window hands loaders a JSON list.
        window=(candidate.window[0], candidate.window[1])
        if candidate.window is not None
        else None,
    )


# -- bundle loading (the IO half) ----------------------------------------------


def load_metric_bundle(
    workspace: str | Path,
    *,
    instrument_hash: str | None = None,
    config_name: str | None = None,
) -> MetricBundle:
    """Best-effort bundle from a workspace's artifacts.

    Every field degrades independently (metrics to ``{}``, the rest to None)
    when its artifact is missing or unreadable — the pure gate then reports
    the gap honestly: a missing parity input fails parity, a missing metric
    fails its criterion. The instrument hash cannot be derived from the
    workspace alone (the conf names the market, not the resolved list) —
    pass the launch-recorded hash (US-008 pipeline_status.json).
    """
    from ops.gpu_trace import workspace_metrics, workspace_window

    ws = Path(workspace).expanduser()
    metrics = dict(workspace_metrics(str(ws)) or {})
    if "Sharpe" not in metrics:
        sharpe = _derived_sharpe(ws)
        if sharpe is not None:
            metrics["Sharpe"] = sharpe
    window_list = workspace_window(str(ws))
    window = (window_list[0], window_list[1]) if window_list else None

    from execution.signal import SignalError, load_cost_params, load_market, load_strategy_params

    market: str | None = None
    topk: int | None = None
    n_drop: int | None = None
    cost_params: dict[str, float] | None = None
    try:
        market = load_market(ws, config_name)
    except SignalError:
        pass
    try:
        params = load_strategy_params(ws, config_name)
        topk, n_drop = params.topk, params.n_drop
    except SignalError:
        pass
    try:
        cost_params = load_cost_params(ws, config_name)
    except SignalError:
        pass
    return MetricBundle(
        workspace=str(ws),
        metrics=metrics,
        window=window,
        market=market,
        instrument_hash=instrument_hash,
        topk=topk,
        n_drop=n_drop,
        cost_params=cost_params,
        daily_returns=_net_daily_returns(ws / "ret.pkl"),
    )


def load_confirmation_evidence(
    candidate_workspace: str | Path,
    incumbent_workspace: str | Path,
    window_start: dt.date,
    window_end: dt.date,
    **confirm_kwargs: Any,
) -> ConfirmationEvidence:
    """Evaluate both strategies on the confirmation window (US-009 helper).

    Never raises for evaluation problems: any ``ConfirmWindowError`` (missing
    snapshot, re-predict failure, non-reproducing pred, window outside the
    store) lands in ``ConfirmationEvidence.error`` naming the failing side,
    which the gate renders as a failing confirmation_unavailable criterion.
    The incumbent runs first — its docker re-predict is usually skipped
    (daily refresh covers), so its failures are cheap to hit before the
    candidate's ~minutes-long re-predict. ``confirm_kwargs`` pass through to
    ``ops.confirm_window.confirmation_returns`` (store_path, runner, …).
    """
    from ops.confirm_window import ConfirmWindowError, annualized_ir, confirmation_returns

    window = (window_start.isoformat(), window_end.isoformat())
    sides: dict[str, ConfirmationSide] = {}
    for role, workspace in (
        ("incumbent", incumbent_workspace),
        ("candidate", candidate_workspace),
    ):
        ws = Path(workspace).expanduser()
        try:
            returns = confirmation_returns(ws, window_start, window_end, **confirm_kwargs)
        except ConfirmWindowError as exc:
            return ConfirmationEvidence(
                window=window,
                candidate=sides.get("candidate"),
                incumbent=sides.get("incumbent"),
                error=f"{role} `{_workspace_tag(str(ws))}`: {exc}",
            )
        sides[role] = ConfirmationSide(
            workspace=str(ws),
            ir=annualized_ir(returns.daily_returns),
            window=returns.window,
            days=len(returns.daily_returns),
            repredicted=returns.repredicted,
            reproduction=returns.reproduction,
        )
    return ConfirmationEvidence(
        window=window, candidate=sides["candidate"], incumbent=sides["incumbent"]
    )


def _derived_sharpe(workspace: Path) -> float | None:
    """Sharpe via summary's existing derivation: csv key if logged, else ret.pkl."""
    from orchestrator.summary import SHARPE_CSV_KEYS, SummaryError, compute_sharpe, load_metrics

    csv = workspace / "qlib_res.csv"
    if csv.is_file():
        try:
            raw = load_metrics(csv)
            for key in SHARPE_CSV_KEYS:
                if key in raw:
                    return raw[key]
        except SummaryError:
            pass
    ret = workspace / "ret.pkl"
    if not ret.is_file():
        return None
    try:
        return compute_sharpe(ret)
    except SummaryError:
        return None


def _net_daily_returns(ret_pkl: Path) -> tuple[float, ...] | None:
    """ret.pkl -> net-of-cost daily return tuple; None on any unusable artifact."""
    if not ret_pkl.is_file():
        return None
    import pandas as pd  # lazy: keeps offline imports fast, like gpu_trace

    try:
        frame = pd.read_pickle(ret_pkl)
    except Exception:  # noqa: BLE001 — degrade, the gate reports the gap
        return None
    if not isinstance(frame, pd.DataFrame) or "return" not in frame.columns:
        return None
    net = frame["return"].astype(float)
    if "cost" in frame.columns:
        net = net - frame["cost"].astype(float)
    return tuple(float(value) for value in net.dropna())

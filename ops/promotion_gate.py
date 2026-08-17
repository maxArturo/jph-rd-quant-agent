"""Codified candidate-vs-incumbent promotion gate (US-007).

"Beats the incumbent" is defined here, by code — not by judgment in a Slack
thread. ``evaluate_gate`` is PURE (no IO, no state): callers load one
``MetricBundle`` per strategy (``load_metric_bundle`` does the artifact IO)
and get a ``GateVerdict`` back, which serializes to JSON (the
promotion_history.gate_verdict payload) and renders to a Slack text block.

The decision has two layers:

1. Parity — checked on the inputs that must never drift silently: market,
   topk/n_drop, and cost params. Any mismatch, or either side MISSING one of
   those inputs, fails parity — and a parity failure fails the gate no
   matter how good the metrics look. Test window and instrument-list hash
   are NOT parity fields (US-031): both drift by design (the window rolls
   with the store, the PIT universe evolves), so requiring recorded equality
   made auto-promotion structurally unreachable after the first promotion.
   Instead comparability is CONSTRUCTED: the incumbent-relative legs are
   computed from both strategies' daily returns over their shared trading
   days (``align_overlap``), and window/universe drift is surfaced as
   verdict drift notes.
2. Criteria — thresholds read from the ``promotion_gate:`` section of
   orchestrator/config.yaml (``load_gate_config``):
   - candidate IR strictly > incumbent IR × ir_margin (default 1.05), both
     computed over the shared overlap window (≥ min_overlap_days, default
     126, else the leg fails as ``overlap_unavailable``)
   - candidate |MDD| ≤ incumbent |MDD| × mdd_tolerance (default 1.25), same
     overlap window
   - candidate IC strictly > min_ic (default 0) — recorded value, candidate
     only
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

# Standing survivorship caveat (US-025): every gate verdict and Notion
# write-up carries this line until the delisted-names backfill lands.
SURVIVORSHIP_CAVEAT = (
    "Caveat: the universe holds only never-delisted names, so backtest ARR is "
    "flattered (survivorship — docs/decisions.md 2026-08-15 US-025)."
)


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
    # US-031: the IR/MDD legs compare over the two strategies' shared trading
    # days; fewer shared days than this fails those legs as
    # overlap_unavailable (about half a trading year by default).
    min_overlap_days: int = 126


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
        min_overlap_days=_config_int(section, "min_overlap_days", GateConfig.min_overlap_days),
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


def _config_int(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise GateConfigError(f"promotion_gate.{key} must be a positive integer, got {raw!r}")
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
    (IC/ICIR/Rank IC/ARR/IR/MDD) plus ``Sharpe``. Hard-parity fields (market,
    topk, n_drop, cost_params) are None when an artifact could not provide
    them — which FAILS parity against an incumbent (the gate never guesses).
    ``dated_returns`` is the net-of-cost ret.pkl series as (ISO date, return)
    pairs — the ``align_overlap`` input the IR/MDD legs are computed from
    (US-031); missing means those legs fail as overlap_unavailable. ``window``
    and ``instrument_hash`` are informational since US-031 (drift notes).
    """

    workspace: str
    metrics: Mapping[str, float]
    window: tuple[str, str] | None = None  # first/last backtest day, ISO dates
    market: str | None = None
    instrument_hash: str | None = None
    topk: int | None = None
    n_drop: int | None = None
    cost_params: Mapping[str, float] | None = None
    dated_returns: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class OverlapComparison:
    """Both strategies' IR/MDD computed over their shared trading days (US-031).

    Built by ``align_overlap``. ``error`` set means the two return series
    could not be aligned (missing dated returns, or fewer shared days than
    ``min_overlap_days``) — the gate maps that to failing IR and MDD criteria
    with reason ``overlap_unavailable``, never a silent skip.
    """

    window: tuple[str, str] | None = None  # first/last shared day, ISO
    days: int = 0
    candidate_ir: float | None = None
    incumbent_ir: float | None = None
    candidate_mdd: float | None = None
    incumbent_mdd: float | None = None
    error: str | None = None


def max_drawdown(returns: Iterable[float]) -> float:
    """Worst peak-to-trough drawdown of a compounded daily-return series.

    Returns 0.0 for a series that never draws down; always ≤ 0 (qlib's MDD
    sign convention, so overlap MDDs read like recorded ones)."""
    equity = peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def align_overlap(
    candidate: MetricBundle,
    incumbent: MetricBundle,
    min_days: int = GateConfig.min_overlap_days,
) -> OverlapComparison:
    """PURE alignment: IR/MDD for both strategies on their shared trading days.

    This is US-031's comparability-by-construction: recorded windows may
    legitimately differ (the test window rolls with the store), so the
    incumbent-relative legs compare returns on the DATES BOTH STRATEGIES WERE
    MEASURED ON instead of requiring recorded windows to match. IR uses the
    confirmation leg's convention (ops.confirm_window.annualized_ir) so every
    IR in a verdict is computed identically.
    """
    from ops.confirm_window import annualized_ir

    for role, bundle in (("candidate", candidate), ("incumbent", incumbent)):
        if not bundle.dated_returns:
            return OverlapComparison(error=f"dated daily returns unavailable on {role}")
    candidate_map = {day: value for day, value in candidate.dated_returns or ()}
    incumbent_map = {day: value for day, value in incumbent.dated_returns or ()}
    shared = sorted(set(candidate_map) & set(incumbent_map))
    if len(shared) < min_days:
        return OverlapComparison(
            error=(
                f"insufficient overlap — {len(shared)} shared trading day(s), "
                f"need >= {min_days}"
            )
        )
    candidate_returns = [candidate_map[day] for day in shared]
    incumbent_returns = [incumbent_map[day] for day in shared]
    return OverlapComparison(
        window=(shared[0], shared[-1]),
        days=len(shared),
        candidate_ir=annualized_ir(candidate_returns),
        incumbent_ir=annualized_ir(incumbent_returns),
        candidate_mdd=max_drawdown(candidate_returns),
        incumbent_mdd=max_drawdown(incumbent_returns),
    )


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
    window: tuple[str, str] | None = None  # the candidate's test window
    overlap: OverlapComparison | None = None  # the shared-window comparison (US-031)
    drift_notes: tuple[str, ...] = ()  # window/universe drift, informational

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
            "overlap": asdict(self.overlap) if self.overlap else None,
            "drift_notes": list(self.drift_notes),
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
            parity = "market/topk-n_drop/costs match"
            if self.overlap is not None and self.overlap.window is not None:
                parity += (
                    f" — compared on the shared window {self.overlap.window[0]} → "
                    f"{self.overlap.window[1]} ({self.overlap.days} trading days)"
                )
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
        lines.extend(f"• :information_source: {note}" for note in self.drift_notes)
        lines.append(f"_{SURVIVORSHIP_CAVEAT}_")
        return "\n".join(lines)


def _workspace_tag(workspace: str) -> str:
    return Path(workspace).name[:8] or "unknown"


# The conversational promotion path (orchestrator/promotion.py) never runs the
# gate — its Decision Log rows carry this line so every source states its gate
# standing explicitly (US-012).
GATE_NOT_EVALUATED = "Gate: not evaluated (conversational promotion — operator judgment)"


def gate_summary_line(
    verdict: Mapping[str, Any] | None, *, forced: bool = False, error: str | None = None
) -> str:
    """One-line gate standing for audit records (Notion Decision Log, US-012).

    ``verdict`` is a GateVerdict.to_dict() payload (also what
    promotion_history.gate_verdict stores); ``error`` covers the
    could-not-evaluate case; ``forced`` marks an operator override that
    promoted despite the verdict.
    """
    if verdict is None:
        base = f"Gate: ERROR — {error}" if error else "Gate: not evaluated"
    elif verdict.get("pass"):
        base = "Gate: PASS — parity ok, all criteria met"
    else:
        failing = [
            str(criterion.get("name"))
            for criterion in verdict.get("criteria") or []
            if not criterion.get("passed")
        ]
        if not verdict.get("parity_ok", True):
            failing.insert(0, "parity")
        base = f"Gate: FAIL — {', '.join(failing) or 'unspecified'}"
    return f"{base} — promoted anyway (operator --force)" if forced else base


def _confirmation_side_text(role: str, side: ConfirmationSide) -> str:
    ir = f"{side.ir:.4f}" if side.ir is not None else "n/a"
    source = "re-predicted" if side.repredicted else "cached pred"
    return f"{role} `{_workspace_tag(side.workspace)}` IR {ir} ({side.days}d, {source})"


# (bundle attribute, human label) — the configuration inputs that must never
# drift silently. Test window and instrument hash are NOT here (US-031): both
# drift by design, so they surface as drift notes and the incumbent-relative
# legs compare on the constructed overlap window instead.
_PARITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("market", "market"),
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


def _drift_notes(candidate: MetricBundle, incumbent: MetricBundle) -> list[str]:
    """US-031: window/universe differences are informational, not vetoes —
    the IR/MDD legs compare on the constructed overlap and the confirmation
    leg re-predicts each strategy on its own deployed conf. Still worth the
    operator's eyes, so every difference (or unknowable) is stated."""
    notes: list[str] = []
    cand_window, inc_window = candidate.window, incumbent.window
    if cand_window is not None and inc_window is not None:
        if _parity_value(cand_window) != _parity_value(inc_window):
            notes.append(
                f"window drift: candidate {_format_parity(cand_window)}, incumbent "
                f"{_format_parity(inc_window)} — IR/MDD compared on the shared window only"
            )
    else:
        side = "both sides" if cand_window is None and inc_window is None else (
            "candidate" if cand_window is None else "incumbent"
        )
        notes.append(f"window drift unknown — test window unrecorded on {side}")
    cand_hash, inc_hash = candidate.instrument_hash, incumbent.instrument_hash
    if cand_hash is not None and inc_hash is not None:
        if cand_hash != inc_hash:
            notes.append(
                f"universe drift: instrument hash candidate {cand_hash}, incumbent "
                f"{inc_hash} — recorded returns come from each strategy's own universe; "
                "the confirmation leg re-predicts each on its deployed conf"
            )
    else:
        side = "both sides" if cand_hash is None and inc_hash is None else (
            "candidate" if cand_hash is None else "incumbent"
        )
        notes.append(f"universe drift unknown — instrument hash unavailable on {side}")
    return notes


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


def _overlap_prefix(overlap: OverlapComparison) -> str:
    window = overlap.window or ("?", "?")
    return f"shared window {window[0]} → {window[1]} ({overlap.days}d)"


def _ir_criterion(overlap: OverlapComparison | None, config: GateConfig) -> CriterionResult:
    """US-031: IRs computed over the shared trading days, never recorded
    scalars from different windows. Missing/failed alignment fails the leg
    as overlap_unavailable — the exact mirror of confirmation_unavailable."""
    name = "IR"
    if overlap is None:
        return CriterionResult(name, False, "overlap_unavailable — overlap was not evaluated")
    if overlap.error is not None:
        return CriterionResult(name, False, f"overlap_unavailable — {overlap.error}")
    cand, inc = overlap.candidate_ir, overlap.incumbent_ir
    if cand is None or inc is None:
        side = "candidate" if cand is None else "incumbent"
        return CriterionResult(
            name, False, f"overlap_unavailable — {side} IR degenerate on the shared window"
        )
    threshold = inc * config.ir_margin
    return CriterionResult(
        name,
        cand > threshold,
        f"{_overlap_prefix(overlap)}: candidate {cand:.4f} vs incumbent {inc:.4f} × "
        f"{config.ir_margin:g} = {threshold:.4f} (need >)",
        candidate=cand,
        incumbent=inc,
    )


def _mdd_criterion(overlap: OverlapComparison | None, config: GateConfig) -> CriterionResult:
    name = "MDD"
    if overlap is None:
        return CriterionResult(name, False, "overlap_unavailable — overlap was not evaluated")
    if overlap.error is not None:
        return CriterionResult(name, False, f"overlap_unavailable — {overlap.error}")
    cand, inc = overlap.candidate_mdd, overlap.incumbent_mdd
    if cand is None or inc is None:
        side = "candidate" if cand is None else "incumbent"
        return CriterionResult(name, False, f"overlap_unavailable — {side} MDD unavailable")
    # MDD is negative by convention; compare magnitudes so the sign
    # convention can never flip the verdict. At the tolerance passes.
    limit = abs(inc) * config.mdd_tolerance
    return CriterionResult(
        name,
        abs(cand) <= limit,
        f"{_overlap_prefix(overlap)}: candidate {cand:+.2%} vs limit {-limit:.2%} "
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
    overlap: OverlapComparison | None = None,
) -> GateVerdict:
    """PURE comparison: does the candidate beat the incumbent on this config?

    ``incumbent=None`` means nothing is promoted: the comparison legs are
    waived only under ``allow_first`` (IC still applies), otherwise the gate
    fails on the missing incumbent alone. With an incumbent, ``confirmation``
    evidence AND ``overlap`` (US-031, from ``align_overlap``) are REQUIRED —
    omitting either fails its criteria as
    confirmation_unavailable/overlap_unavailable (the PRD forbids a silent
    skip).
    """
    config = config if config is not None else GateConfig()
    criteria: list[CriterionResult] = []
    drift: list[str] = []
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
        drift = _drift_notes(candidate, incumbent)
        criteria.append(_ir_criterion(overlap, config))
        criteria.append(_mdd_criterion(overlap, config))
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
        overlap=overlap if incumbent is not None else None,
        drift_notes=tuple(drift),
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
        dated_returns=_net_dated_returns(ws / "ret.pkl"),
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


def _net_dated_returns(ret_pkl: Path) -> tuple[tuple[str, float], ...] | None:
    """ret.pkl -> net-of-cost (ISO date, return) pairs; None on any unusable
    artifact. Dates ride along so align_overlap (US-031) can intersect two
    strategies' series on their shared trading days."""
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
    net = net.dropna()
    try:
        return tuple(
            (pd.Timestamp(day).date().isoformat(), float(value))  # pyright: ignore[reportArgumentType]
            for day, value in net.items()
        )
    except (TypeError, ValueError):  # non-date index — unusable for alignment
        return None

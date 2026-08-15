"""US-007: codified promotion gate — pure comparison, parity-enforced.

Pure-half tests build MetricBundles by hand; the loader tests build a real
fixture workspace (qlib_res.csv / ret.pkl fixtures from tests/test_summary.py,
conf from tests/test_signal.py write_conf).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ops.promotion_gate import (
    FAIL_MARK,
    PASS_MARK,
    GateConfig,
    GateConfigError,
    MetricBundle,
    evaluate_gate,
    hash_instruments,
    load_gate_config,
    load_metric_bundle,
)
from tests.test_signal import write_conf
from tests.test_summary import write_qlib_res_csv, write_ret_pkl

WINDOW = ("2017-01-03", "2026-07-10")
COSTS = {"open_cost": 0.0005, "close_cost": 0.0005, "min_cost": 0.0}


def bundle(
    workspace: str = "runs/candidate1234",
    ic: float | None = 0.05,
    ir: float | None = 1.30,
    mdd: float | None = -0.10,
    **over: object,
) -> MetricBundle:
    metrics = {
        key: value
        for key, value in {"IC": ic, "IR": ir, "MDD": mdd, "ARR": 0.25}.items()
        if value is not None
    }
    fields: dict = dict(
        workspace=workspace,
        metrics=metrics,
        window=WINDOW,
        market="us_liquid",
        instrument_hash="abc123def4567890",
        topk=30,
        n_drop=3,
        cost_params=COSTS,
    )
    fields.update(over)
    return MetricBundle(**fields)


def incumbent_bundle(**over: object) -> MetricBundle:
    defaults: dict = dict(workspace="runs/incumbent5678", ic=0.04, ir=1.00, mdd=-0.10)
    defaults.update(over)
    return bundle(**defaults)


# ------------------------------------------------------------------ pass/fail


def test_pass_verdict_when_all_criteria_beat_incumbent() -> None:
    verdict = evaluate_gate(bundle(), incumbent_bundle(), GateConfig())
    assert verdict.parity_ok
    assert verdict.passed
    assert not verdict.parity_mismatches
    assert all(criterion.passed for criterion in verdict.criteria)
    assert {criterion.name for criterion in verdict.criteria} == {"IR", "MDD", "IC"}


def test_ir_margin_is_strict_and_fails_alone() -> None:
    # incumbent IR 1.00 × 1.05 = 1.05 — exactly at the bar fails (strict >).
    verdict = evaluate_gate(bundle(ir=1.05), incumbent_bundle(), GateConfig())
    assert not verdict.passed
    failed = {criterion.name: criterion for criterion in verdict.criteria if not criterion.passed}
    assert set(failed) == {"IR"}
    assert "1.0500" in failed["IR"].reason and "1.0000" in failed["IR"].reason


def test_mdd_tolerance_fails_alone_and_at_limit_passes() -> None:
    # incumbent |MDD| 0.10 × 1.25 = 0.125: exactly at the limit passes...
    at_limit = evaluate_gate(bundle(mdd=-0.125), incumbent_bundle(), GateConfig())
    assert at_limit.passed
    # ...strictly beyond fails, and only the MDD criterion.
    verdict = evaluate_gate(bundle(mdd=-0.13), incumbent_bundle(), GateConfig())
    assert not verdict.passed
    failed = {criterion.name for criterion in verdict.criteria if not criterion.passed}
    assert failed == {"MDD"}


def test_ic_must_be_strictly_positive() -> None:
    verdict = evaluate_gate(bundle(ic=0.0), incumbent_bundle(), GateConfig())
    assert not verdict.passed
    failed = {criterion.name for criterion in verdict.criteria if not criterion.passed}
    assert failed == {"IC"}


def test_missing_candidate_metric_fails_that_criterion() -> None:
    verdict = evaluate_gate(bundle(ir=None), incumbent_bundle(), GateConfig())
    assert not verdict.passed
    ir = next(criterion for criterion in verdict.criteria if criterion.name == "IR")
    assert not ir.passed
    assert "unavailable" in ir.reason


def test_configurable_margins() -> None:
    config = GateConfig(ir_margin=1.5, mdd_tolerance=1.0, min_ic=0.06)
    verdict = evaluate_gate(bundle(), incumbent_bundle(), config)
    failed = {criterion.name for criterion in verdict.criteria if not criterion.passed}
    assert failed == {"IR", "IC"}  # 1.30 < 1.5; MDD equal magnitudes pass at tolerance 1.0
    assert verdict.config == config


# ------------------------------------------------------------------ parity


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"window": ("2018-01-02", "2026-07-10")}, "test window"),
        ({"market": "us_all"}, "market"),
        ({"instrument_hash": "ffff000011112222"}, "instrument list"),
        ({"topk": 50}, "topk"),
        ({"n_drop": 5}, "n_drop"),
        ({"cost_params": {**COSTS, "open_cost": 0.001}}, "cost params"),
    ],
)
def test_each_parity_mismatch_fails_and_names_the_field(override: dict, label: str) -> None:
    verdict = evaluate_gate(bundle(**override), incumbent_bundle(), GateConfig())
    assert not verdict.parity_ok
    assert not verdict.passed
    assert any(text.startswith(f"{label} mismatch") for text in verdict.parity_mismatches)


def test_missing_parity_input_fails_parity() -> None:
    # A side that cannot state its instrument list cannot certify parity.
    verdict = evaluate_gate(bundle(instrument_hash=None), incumbent_bundle(), GateConfig())
    assert not verdict.parity_ok
    assert not verdict.passed
    assert any(
        "instrument list unavailable on candidate" in text for text in verdict.parity_mismatches
    )


def test_parity_mismatch_fails_even_with_superior_metrics() -> None:
    verdict = evaluate_gate(
        bundle(ic=0.9, ir=9.0, mdd=-0.01, market="us_all"), incumbent_bundle(), GateConfig()
    )
    assert all(criterion.passed for criterion in verdict.criteria)
    assert not verdict.passed


def test_window_list_and_tuple_compare_equal() -> None:
    # workspace_window returns a JSON-friendly list; the bundle type says tuple.
    verdict = evaluate_gate(
        bundle(window=list(WINDOW)), incumbent_bundle(), GateConfig()  # type: ignore[arg-type]
    )
    assert verdict.parity_ok


# ------------------------------------------------------------------ no incumbent


def test_no_incumbent_blocks_promotion_by_default() -> None:
    verdict = evaluate_gate(bundle(), None, GateConfig())
    assert not verdict.passed
    assert verdict.parity_ok  # nothing to compare — parity is vacuous
    first = next(criterion for criterion in verdict.criteria if criterion.name == "incumbent")
    assert not first.passed
    assert "allow_first" in first.reason
    assert verdict.incumbent_workspace is None


def test_allow_first_passes_without_incumbent() -> None:
    verdict = evaluate_gate(bundle(), None, GateConfig(allow_first=True))
    assert verdict.passed
    assert {criterion.name for criterion in verdict.criteria} == {"incumbent", "IC"}


def test_allow_first_still_requires_positive_ic() -> None:
    verdict = evaluate_gate(bundle(ic=-0.01), None, GateConfig(allow_first=True))
    assert not verdict.passed


# ------------------------------------------------------------------ verdict output


def test_verdict_serializes_to_json() -> None:
    verdict = evaluate_gate(bundle(ir=1.05), incumbent_bundle(), GateConfig())
    payload = json.loads(verdict.to_json())
    assert payload["parity_ok"] is True
    assert payload["pass"] is False
    assert payload["candidate_workspace"] == "runs/candidate1234"
    assert payload["incumbent_workspace"] == "runs/incumbent5678"
    assert payload["config"]["ir_margin"] == 1.05
    by_name = {criterion["name"]: criterion for criterion in payload["criteria"]}
    assert by_name["IR"]["passed"] is False
    assert by_name["IR"]["candidate"] == 1.05
    assert by_name["IR"]["incumbent"] == 1.00


def test_slack_text_lists_every_criterion_with_both_values() -> None:
    verdict = evaluate_gate(bundle(ir=1.05), incumbent_bundle(), GateConfig())
    text = verdict.slack_text()
    assert f"{FAIL_MARK} FAIL" in text
    assert "`candidat`" in text and "`incumben`" in text  # 8-char workspace tags
    assert f"{PASS_MARK} parity" in text
    for name in ("IR", "MDD", "IC"):
        assert f" {name}: " in text
    assert "1.0500" in text and "1.0000" in text  # both IR values shown
    assert "-10.00%" in text  # incumbent MDD


def test_slack_text_shows_parity_mismatches() -> None:
    verdict = evaluate_gate(bundle(market="us_all"), incumbent_bundle(), GateConfig())
    text = verdict.slack_text()
    assert f"{FAIL_MARK} FAIL" in text
    assert "parity: market mismatch" in text
    assert "us_all" in text and "us_liquid" in text


# ------------------------------------------------------------------ config loading


def test_load_gate_config_defaults_when_section_missing(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("notion:\n  parent_page_id: abc\n")
    assert load_gate_config(path) == GateConfig()
    assert load_gate_config(tmp_path / "absent.yaml") == GateConfig()


def test_load_gate_config_reads_the_section(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "promotion_gate:\n"
        "  ir_margin: 1.10\n"
        "  mdd_tolerance: 1.5\n"
        "  min_ic: 0.01\n"
        "  allow_first: true\n"
    )
    assert load_gate_config(path) == GateConfig(
        ir_margin=1.10, mdd_tolerance=1.5, min_ic=0.01, allow_first=True
    )


def test_load_gate_config_rejects_junk_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("promotion_gate:\n  ir_margin: wide\n")
    with pytest.raises(GateConfigError, match="ir_margin"):
        load_gate_config(path)
    path.write_text("promotion_gate:\n  allow_first: 1\n")
    with pytest.raises(GateConfigError, match="allow_first"):
        load_gate_config(path)


def test_repo_config_yaml_carries_the_gate_section() -> None:
    config = load_gate_config()
    assert config == GateConfig(ir_margin=1.05, mdd_tolerance=1.25, min_ic=0.0, allow_first=False)


# ------------------------------------------------------------------ instrument hash


def test_hash_instruments_is_order_insensitive_and_deduped() -> None:
    digest = hash_instruments(["MSFT", "AAPL", " AAPL "])
    assert digest == hash_instruments(["AAPL", "MSFT"])
    assert digest != hash_instruments(["AAPL", "MSFT", "NVDA"])
    assert len(digest) == 16


def test_hash_instruments_refuses_empty_list() -> None:
    with pytest.raises(ValueError, match="empty instrument list"):
        hash_instruments(["", "  "])


# ------------------------------------------------------------------ bundle loading


def fixture_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    write_qlib_res_csv(workspace / "qlib_res.csv")
    write_ret_pkl(workspace / "ret.pkl", days=60)
    write_conf(workspace, "conf_combined_factors.yaml", topk=30, n_drop=3, costs=COSTS)
    return workspace


def test_load_metric_bundle_from_fixture_workspace(tmp_path: Path) -> None:
    workspace = fixture_workspace(tmp_path)
    loaded = load_metric_bundle(workspace, instrument_hash="abc123def4567890")
    # METRIC_SPECS labelling (tests/test_summary.FIXTURE_METRICS values):
    assert loaded.metrics["IC"] == pytest.approx(0.0432)
    assert loaded.metrics["IR"] == pytest.approx(1.02)
    assert loaded.metrics["MDD"] == pytest.approx(-0.084)
    assert "Sharpe" in loaded.metrics  # derived from ret.pkl (not in the csv)
    last_day = pd.bdate_range("2025-01-02", periods=60)[-1]
    expected_last = pd.Timestamp(last_day).date().isoformat()  # pyright: ignore[reportArgumentType]
    assert loaded.window == ("2025-01-02", expected_last)
    assert loaded.market == "us_liquid"
    assert (loaded.topk, loaded.n_drop) == (30, 3)
    assert loaded.cost_params == COSTS
    assert loaded.instrument_hash == "abc123def4567890"
    assert loaded.daily_returns is not None and len(loaded.daily_returns) == 60


def test_load_metric_bundle_degrades_field_by_field(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()
    loaded = load_metric_bundle(workspace)
    assert loaded.metrics == {}
    assert loaded.window is None
    assert loaded.market is None
    assert loaded.topk is None and loaded.n_drop is None
    assert loaded.cost_params is None
    assert loaded.instrument_hash is None
    assert loaded.daily_returns is None
    # An empty bundle against an incumbent fails parity, not crashes:
    verdict = evaluate_gate(loaded, incumbent_bundle(), GateConfig())
    assert not verdict.parity_ok and not verdict.passed


def test_loaded_bundles_round_trip_through_the_gate(tmp_path: Path) -> None:
    candidate = load_metric_bundle(fixture_workspace(tmp_path / "cand"), instrument_hash="h1")
    incumbent = load_metric_bundle(fixture_workspace(tmp_path / "inc"), instrument_hash="h1")
    # Identical artifacts: parity holds, but IR cannot beat itself × 1.05.
    verdict = evaluate_gate(candidate, incumbent, GateConfig())
    assert verdict.parity_ok
    assert not verdict.passed
    failed = {criterion.name for criterion in verdict.criteria if not criterion.passed}
    assert failed == {"IR"}

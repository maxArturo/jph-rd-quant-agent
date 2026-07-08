"""Tests for the US-017 fin_quant hooks (research/us_quant.py).

Importing research.us_quant pulls in rdagent's qlib scenario tree (seconds) —
that cost is confined to this module.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT_TEMPLATES = ("factor_template", "model_template")


@pytest.fixture(autouse=True)
def _tmp_workspace(tmp_path, monkeypatch):
    """Experiment construction writes workspaces under RD_AGENT_SETTINGS.workspace_path."""
    from rdagent.core.conf import RD_AGENT_SETTINGS

    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "ws")


def _hypothesis() -> Any:
    from rdagent.core.proposal import Hypothesis

    return Hypothesis(
        hypothesis="20-day momentum predicts next-day returns",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )


class TestEnvSeam:
    """The QLIB_QUANT_* env vars rdagent reads must resolve to our classes."""

    def test_env_var_names_resolve_us_classes(self, monkeypatch) -> None:
        from rdagent.app.qlib_rd_loop.conf import QuantBasePropSetting
        from rdagent.core.utils import import_class

        from research.us_quant import (
            USQlibFactorHypothesis2Experiment,
            USQlibModelHypothesis2Experiment,
        )

        monkeypatch.setenv(
            "QLIB_QUANT_FACTOR_HYPOTHESIS2EXPERIMENT",
            "research.us_quant.USQlibFactorHypothesis2Experiment",
        )
        monkeypatch.setenv(
            "QLIB_QUANT_MODEL_HYPOTHESIS2EXPERIMENT",
            "research.us_quant.USQlibModelHypothesis2Experiment",
        )
        setting = QuantBasePropSetting()
        assert import_class(setting.factor_hypothesis2experiment) is (
            USQlibFactorHypothesis2Experiment
        )
        assert import_class(setting.model_hypothesis2experiment) is (
            USQlibModelHypothesis2Experiment
        )

    def test_template_filenames_match_upstream(self) -> None:
        """repoint relies on total replacement: US folders must mirror upstream names."""
        from rdagent.utils.agent.tpl import PROJ_PATH

        from research.us_quant import US_TEMPLATES

        upstream = PROJ_PATH / "scenarios" / "qlib" / "experiment"
        for folder in REPO_ROOT_TEMPLATES:
            ours = {
                p.name
                for p in (US_TEMPLATES / folder).rglob("*")
                if p.suffix in (".py", ".yaml", ".md")
            }
            theirs = {
                p.name
                for p in (upstream / folder).rglob("*")
                if p.suffix in (".py", ".yaml", ".md")
            }
            assert ours == theirs, f"{folder} filenames diverge from upstream"


class TestConvertResponseRepoints:
    def test_factor_experiment_and_baseline_get_us_templates(self) -> None:
        from research.us_quant import USQlibFactorHypothesis2Experiment

        response = json.dumps(
            {
                "Momentum20": {
                    "description": "20-day price momentum",
                    "formulation": "close / Ref(close, 20) - 1",
                    "variables": {"close": "adjusted close"},
                }
            }
        )
        trace = SimpleNamespace(hist=[])
        exp = USQlibFactorHypothesis2Experiment().convert_response(
            response, _hypothesis(), trace  # type: ignore[arg-type]
        )

        assert [t.factor_name for t in exp.sub_tasks] == ["Momentum20"]
        # New experiment AND the fresh baseline workspace both carry US confs.
        workspaces = [exp.experiment_workspace, exp.based_experiments[0].experiment_workspace]
        for ws in workspaces:
            conf = ws.file_dict["conf_baseline.yaml"]
            assert "us_data" in conf
            assert "region: us" in conf
            assert "SPY" in conf
            for filename, text in ws.file_dict.items():
                assert "cn_data" not in text, f"{filename} still points at the CN store"
                assert "csi300" not in text.lower(), f"{filename} still references CSI300"

    def test_model_experiment_gets_us_templates(self) -> None:
        from research.us_quant import USQlibModelHypothesis2Experiment

        response = json.dumps(
            {
                "LSTMAlpha": {
                    "description": "LSTM on daily features",
                    "formulation": "y = LSTM(x)",
                    "architecture": "2-layer LSTM",
                    "variables": {"x": "feature matrix"},
                    "hyperparameters": {"hidden_size": 64},
                    "training_hyperparameters": {"n_epochs": "10"},
                    "model_type": "TimeSeries",
                }
            }
        )
        trace = SimpleNamespace(hist=[])
        exp = USQlibModelHypothesis2Experiment().convert_response(
            response, _hypothesis(), trace  # type: ignore[arg-type]
        )

        assert exp.experiment_workspace is not None
        conf = exp.experiment_workspace.file_dict["conf_sota_factors_model.yaml"]
        assert "us_data" in conf
        assert "region: us" in conf
        assert "cn_data" not in conf


class TestRepointHelper:
    def test_replaces_cn_files_in_place(self) -> None:
        from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

        from research.us_quant import repoint_us_templates

        exp = QlibModelExperiment(sub_tasks=[])
        assert exp.experiment_workspace is not None
        before = exp.experiment_workspace.file_dict["conf_baseline_factors_model.yaml"]
        assert "cn_data" in before  # upstream CN template injected by __init__

        returned = repoint_us_templates(exp)

        assert returned is exp
        after = exp.experiment_workspace.file_dict["conf_baseline_factors_model.yaml"]
        assert "us_data" in after
        assert "cn_data" not in after
        # The on-disk workspace copy is replaced too, not just file_dict.
        on_disk = (
            exp.experiment_workspace.workspace_path / "conf_baseline_factors_model.yaml"
        ).read_text()
        assert on_disk == after

    def test_ignores_objects_without_workspace_or_template(self) -> None:
        from research.us_quant import repoint_us_templates

        plain = SimpleNamespace(based_experiments=[object()])
        assert repoint_us_templates(plain) is plain

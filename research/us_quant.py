"""US-market hooks for ``rdagent fin_quant`` (US-017).

APP_TPL redirects prompt lookups but NOT the workspace template folders:
``QlibFactorExperiment`` / ``QlibModelExperiment`` hardcode
``Path(__file__).parent / "factor_template"`` (rdagent/scenarios/qlib/
experiment/*.py at the pinned commit), and the proposal classes construct
those experiment classes directly. The configurable seam is
``QuantBasePropSetting`` (env prefix ``QLIB_QUANT_``): point

    QLIB_QUANT_FACTOR_HYPOTHESIS2EXPERIMENT=research.us_quant.USQlibFactorHypothesis2Experiment
    QLIB_QUANT_MODEL_HYPOTHESIS2EXPERIMENT=research.us_quant.USQlibModelHypothesis2Experiment

and every experiment they emit gets the US-patched templates from
research/us_templates re-injected over the upstream CN ones. The US template
folders keep the exact upstream filenames, so ``inject_code_from_folder``
replaces every YAML wholesale (it overwrites same-named entries in both the
workspace ``file_dict`` and on disk). ops/run_us_quant.sh wires these env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from rdagent.scenarios.qlib.experiment import factor_experiment, model_experiment
from rdagent.scenarios.qlib.experiment import quant_experiment as _quant
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
)
from rdagent.scenarios.qlib.proposal.model_proposal import (
    QlibModelHypothesis2Experiment,
)

from research.us_validation import install_us_validation

if TYPE_CHECKING:
    from rdagent.components.coder.factor_coder.factor import FactorExperiment
    from rdagent.components.coder.model_coder.model import ModelExperiment
    from rdagent.core.proposal import Hypothesis, Trace

US_TEMPLATES = Path(__file__).resolve().parent / "us_templates"
US_FACTOR_TEMPLATE = US_TEMPLATES / "factor_template"
US_MODEL_TEMPLATE = US_TEMPLATES / "model_template"

# Per-run custom universe override (US-023): the run env (gpu_worker.sh run
# exports) sets this to the universe's rendered template copy
# (~/rdq-data/templates/<name>) before the loop starts, so every experiment
# backtests `market: <name>` instead of the us_liquid default.
TEMPLATES_ENV_VAR = "RDQ_UNIVERSE_TEMPLATES"


def _template_root() -> Path:
    """The template folder for this run: env override or the us_liquid default.

    Read at call time (not import) — the fork inherits the env, and tests
    monkeypatch it. A set-but-broken override raises instead of silently
    falling back: a run on the wrong universe is worse than a dead run
    (2026-08-05 incident).
    """
    override = os.environ.get(TEMPLATES_ENV_VAR, "").strip()
    if not override:
        return US_TEMPLATES
    root = Path(override).expanduser()
    missing = [sub for sub in ("factor_template", "model_template") if not (root / sub).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{TEMPLATES_ENV_VAR}={root} lacks {', '.join(missing)} — re-confirm the"
            " universe so orchestrator/universe.py re-renders its template copy"
        )
    return root

# NOTE: components' FactorExperiment/ModelExperiment are bare aliases of
# Experiment (no distinct type), so template choice must match the concrete
# Qlib classes. quant_experiment.py defines its own same-named pair — cover both.
_FACTOR_EXPERIMENT_CLASSES = (
    factor_experiment.QlibFactorExperiment,
    _quant.QlibFactorExperiment,
)
_MODEL_EXPERIMENT_CLASSES = (
    model_experiment.QlibModelExperiment,
    _quant.QlibModelExperiment,
)

ExpT = TypeVar("ExpT")


def _template_for(exp: object) -> Path | None:
    if isinstance(exp, _FACTOR_EXPERIMENT_CLASSES):
        return _template_root() / "factor_template"
    if isinstance(exp, _MODEL_EXPERIMENT_CLASSES):
        return _template_root() / "model_template"
    return None


def repoint_us_templates(exp: ExpT) -> ExpT:
    """Re-inject the US template files over ``exp`` and its based_experiments.

    Covers the freshly built experiment AND the baseline
    ``QlibFactorExperiment(sub_tasks=[])`` that factor_proposal creates inside
    ``convert_response`` (the workspace the baseline backtest runs in).
    Experiments pulled from ``trace.hist`` were already patched when they were
    created; re-injecting them is a harmless rewrite of identical files.
    """
    for e in (exp, *getattr(exp, "based_experiments", ())):
        template = _template_for(e)
        workspace = getattr(e, "experiment_workspace", None)
        if template is not None and workspace is not None:
            workspace.inject_code_from_folder(template)
    return exp


# Resolving the QLIB_QUANT_* class paths imports this module inside every
# fin_quant process during loop construction — before the rd_loop interaction
# gate runs — so this import side effect is the seam that makes the run's
# feature validation and factor env US-correct (see research/us_validation.py).
install_us_validation()


class USQlibFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    def convert_response(
        self, response: str, hypothesis: Hypothesis, trace: Trace
    ) -> FactorExperiment:
        return repoint_us_templates(super().convert_response(response, hypothesis, trace))


class USQlibModelHypothesis2Experiment(QlibModelHypothesis2Experiment):
    def convert_response(
        self, response: str, hypothesis: Hypothesis, trace: Trace
    ) -> ModelExperiment:
        return repoint_us_templates(super().convert_response(response, hypothesis, trace))

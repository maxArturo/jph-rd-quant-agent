"""US-016: US market template copies and APP_TPL prompt overrides.

Covers:
- every us_templates conf_*.yaml renders (jinja stubbed) with the US values
  (provider_uri us_data, region us, market us_liquid, SPY benchmark, no
  A-share limit_threshold, US costs);
- the copies stay drop-in compatible with the upstream template folders;
- the app_tpl override files carry exactly the intended keys, are picked up
  through rdagent's real APP_TPL resolution, and non-overridden keys fall
  through to upstream;
- no file under the pinned rdagent install differs from what pip installed
  from the pinned commit (upstream untouched).
"""

from __future__ import annotations

import base64
import datetime
import hashlib
from importlib.metadata import distribution
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined

REPO = Path(__file__).resolve().parent.parent
US_TEMPLATES = REPO / "research" / "us_templates"
APP_TPL = REPO / "research" / "app_tpl"

FACTOR_YAMLS = [
    "factor_template/conf_baseline.yaml",
    "factor_template/conf_combined_factors.yaml",
    "factor_template/conf_combined_factors_sota_model.yaml",
]
MODEL_YAMLS = [
    "model_template/conf_baseline_factors_model.yaml",
    "model_template/conf_sota_factors_model.yaml",
]
ALL_YAMLS = FACTOR_YAMLS + MODEL_YAMLS

# Every jinja variable used across the five templates, stubbed with plausible
# values ({% if %} vars set to None so optional blocks drop out).
STUB_CONTEXT: dict[str, object] = {
    "train_start": "2015-01-01",
    "train_end": "2020-12-31",
    "valid_start": "2021-01-01",
    "valid_end": "2021-12-31",
    "test_start": "2022-01-01",
    "test_end": "2023-12-31",
    "feature_expressions": '["$close/$open"]',
    "feature_names": '["FEAT0"]',
    "n_epochs": 10,
    "lr": 0.001,
    "early_stop": 5,
    "batch_size": 256,
    "weight_decay": 0.0,
    "num_features": 20,
    "num_timesteps": None,
    "step_len": None,
}


def render(path: Path) -> dict:
    env = Environment(undefined=StrictUndefined)
    text = env.from_string(path.read_text()).render(**STUB_CONTEXT)
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def upstream_qlib_dir() -> Path:
    # rdagent.scenarios.qlib.experiment is a namespace package (__file__ is
    # None) — resolve it from the package root instead.
    from rdagent.utils.agent.tpl import PROJ_PATH

    return PROJ_PATH / "scenarios" / "qlib" / "experiment"


@pytest.mark.parametrize("rel", ALL_YAMLS)
def test_template_renders_with_us_values(rel: str) -> None:
    conf = render(US_TEMPLATES / rel)
    assert conf["qlib_init"]["provider_uri"] == "~/.qlib/qlib_data/us_data"
    assert conf["qlib_init"]["region"] == "us"
    assert conf["market"] == "us_liquid"
    assert conf["benchmark"] == "SPY"
    # anchors must propagate into the sections qlib actually reads
    assert conf["data_handler_config"]["instruments"] == "us_liquid"
    backtest = conf["port_analysis_config"]["backtest"]
    assert backtest["benchmark"] == "SPY"
    exchange = backtest["exchange_kwargs"]
    assert "limit_threshold" not in exchange  # A-share ±10% limit removed
    assert exchange["deal_price"] == "close"
    assert exchange["open_cost"] == 0.0005
    assert exchange["close_cost"] == 0.0005
    assert exchange["min_cost"] == 0


@pytest.mark.parametrize("rel", ALL_YAMLS)
def test_template_keeps_upstream_task_structure(rel: str) -> None:
    conf = render(US_TEMPLATES / rel)
    task = conf["task"]
    # yaml.safe_load parses unquoted ISO dates into datetime.date
    assert task["dataset"]["kwargs"]["segments"]["train"] == [
        datetime.date(2015, 1, 1),
        datetime.date(2020, 12, 31),
    ]
    record_classes = [r["class"] for r in task["record"]]
    assert record_classes == ["SignalRecord", "SigAnaRecord", "PortAnaRecord"]


def test_no_ashare_language_in_research_overrides() -> None:
    for base in (US_TEMPLATES, APP_TPL):
        for path in base.rglob("*"):
            if path.is_dir() or path.name == "README.md":
                continue
            text = path.read_text()
            for needle in ("cn_data", "csi300", "CSI300", "SH000300", "A-share", "China"):
                assert needle not in text, f"{needle!r} left in {path}"


@pytest.mark.parametrize("folder", ["factor_template", "model_template"])
def test_copies_are_dropin_for_upstream_folders(folder: str) -> None:
    upstream = upstream_qlib_dir() / folder
    ours = US_TEMPLATES / folder
    upstream_files = {p.name for p in upstream.iterdir() if p.name != "__pycache__"}
    our_files = {p.name for p in ours.iterdir()}
    assert our_files == upstream_files
    # only the YAMLs are patched; support files stay byte-identical
    for name in upstream_files:
        if not name.endswith(".yaml"):
            assert (ours / name).read_bytes() == (upstream / name).read_bytes()


def test_app_tpl_override_files_hold_only_intended_keys() -> None:
    exp = yaml.safe_load((APP_TPL / "scenarios/qlib/experiment/prompts.yaml").read_text())
    assert set(exp) == {"qlib_factor_experiment_setting", "qlib_model_experiment_setting"}
    loader = yaml.safe_load(
        (APP_TPL / "scenarios/qlib/factor_experiment_loader/prompts.yaml").read_text()
    )
    assert set(loader) == {
        "factor_viability_system",
        "factor_relevance_system",
        "factor_duplicate_system",
    }
    for text in loader.values():
        assert "US equity market" in text


def test_app_tpl_resolution_through_rdagent() -> None:
    """Drive rdagent's real loader with APP_TPL set: overridden keys come from
    research/app_tpl, everything else falls through to the pinned upstream."""
    from rdagent.core.conf import RD_AGENT_SETTINGS
    from rdagent.utils.agent.tpl import load_content

    exp_dir = upstream_qlib_dir()
    loader_dir = exp_dir.parent / "factor_experiment_loader"
    saved = RD_AGENT_SETTINGS.app_tpl
    RD_AGENT_SETTINGS.app_tpl = str(APP_TPL)
    try:
        setting = load_content(".prompts:qlib_factor_experiment_setting", caller_dir=exp_dir)
        assert "US stocks (us_liquid universe)" in setting and "CSI300" not in setting
        model_setting = load_content(".prompts:qlib_model_experiment_setting", caller_dir=exp_dir)
        assert "US stocks (us_liquid universe)" in model_setting

        loader_keys = (
            "factor_viability_system",
            "factor_relevance_system",
            "factor_duplicate_system",
        )
        for key in loader_keys:
            text = load_content(f".prompts:{key}", caller_dir=loader_dir)
            assert "US equity market" in text and "A-share" not in text

        # key absent from the override file -> upstream content
        background = load_content(".prompts:qlib_quant_background", caller_dir=exp_dir)
        assert "Wall Street" in background
    finally:
        RD_AGENT_SETTINGS.app_tpl = saved

    # with app_tpl unset the upstream text is untouched
    original = load_content(".prompts:qlib_factor_experiment_setting", caller_dir=exp_dir)
    assert "CSI300" in original


def test_pinned_rdagent_install_unmodified() -> None:
    """Every rdagent file pip installed from the pinned commit still hashes to
    its RECORD entry — i.e. nothing customized the upstream tree in place."""
    dist = distribution("rdagent")
    files = dist.files
    assert files is not None
    checked = 0
    mismatched = []
    for f in files:
        if not str(f).startswith("rdagent/") or f.hash is None:
            continue  # .pyc caches carry no hash and are legitimately new
        assert f.hash.mode == "sha256"
        digest = hashlib.sha256(Path(str(dist.locate_file(f))).read_bytes()).digest()
        actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if actual != f.hash.value:
            mismatched.append(str(f))
        checked += 1
    assert checked > 100, "suspiciously few hashed files — RECORD parsing broken?"
    assert not mismatched, f"pinned rdagent files modified in place: {mismatched}"

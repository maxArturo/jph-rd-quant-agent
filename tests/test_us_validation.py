"""Tests for the US-043 execution-environment shims (research/us_validation.py).

The validator tests are offline (subprocess.run monkeypatched). The
install/binding tests import rdagent's qlib scenario tree (seconds) — that
cost is confined to the TestInstall class, mirroring test_us_quant_hooks.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import us_validation


@pytest.fixture()
def captured_run(monkeypatch):
    """Stub subprocess.run recording the call; returncode settable per test."""
    calls: list[dict] = []
    result = SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return result

    monkeypatch.setattr(us_validation.subprocess, "run", fake_run)
    monkeypatch.setattr(us_validation, "_log_failure", lambda msg: calls.append({"log": msg}))
    return calls, result


class TestValidator:
    def test_probe_runs_in_this_interpreter(self, captured_run) -> None:
        calls, _ = captured_run
        assert us_validation.validate_us_features(["$close/Ref($close, 1)"]) is True
        (call,) = calls
        assert call["cmd"][0] == sys.executable
        assert call["cmd"][1] == "-c"

    def test_probe_targets_us_store(self, captured_run, monkeypatch) -> None:
        calls, _ = captured_run
        monkeypatch.setenv("RDQ_QLIB_STORE", "/data/us_store")
        monkeypatch.setenv("RDQ_VALIDATION_INSTRUMENT", "AAPL")
        us_validation.validate_us_features(["$close"])
        script = calls[0]["cmd"][2]
        assert "'/data/us_store'" in script
        assert "region=\"us\"" in script or "region='us'" in script
        assert "'AAPL'" in script
        assert "'$close'" in script

    def test_nonzero_exit_is_invalid_and_logged(self, captured_run) -> None:
        calls, result = captured_run
        result.returncode = 1
        result.stderr = "qlib.data raised: field not found"
        assert us_validation.validate_us_features(["$nope"]) is False
        assert any("field not found" in c.get("log", "") for c in calls)

    def test_timeout_is_invalid(self, monkeypatch) -> None:
        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

        monkeypatch.setattr(us_validation.subprocess, "run", raise_timeout)
        monkeypatch.setattr(us_validation, "_log_failure", lambda msg: None)
        assert us_validation.validate_us_features(["$close"]) is False

    def test_empty_expressions_skip_the_probe(self, monkeypatch) -> None:
        def boom(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("no subprocess for an empty expression list")

        monkeypatch.setattr(us_validation.subprocess, "run", boom)
        assert us_validation.validate_us_features([]) is True


class TestInstall:
    """Binding swaps in the pinned tree (imports rdagent — seconds)."""

    def test_all_upstream_bindings_swapped(self) -> None:
        us_validation.install_us_validation()

        import rdagent.components.workflow.rd_loop as rd_loop
        import rdagent.utils.qlib as rd_qlib
        from rdagent.components.coder.factor_coder import config as factor_config
        from rdagent.components.coder.model_coder import conf as model_config
        from rdagent.scenarios.qlib.experiment import (
            factor_experiment,
            model_experiment,
            quant_experiment,
        )

        assert rd_qlib.validate_qlib_features is us_validation.validate_us_features
        # rd_loop from-imports the name; its own binding is the one the
        # base-feature gate calls (rd_loop.py _interact_init_params).
        assert rd_loop.validate_qlib_features is us_validation.validate_us_features
        for module in (factor_config, factor_experiment, quant_experiment):
            assert module.get_factor_env is us_validation.get_us_factor_env
        for module in (model_config, model_experiment, quant_experiment):
            assert module.get_model_env is us_validation.get_us_model_env

    def test_us_quant_import_installs_the_shim(self) -> None:
        """The QLIB_QUANT_* class paths import research.us_quant inside every
        fin_quant process — that import alone must make validation US-correct."""
        import importlib

        import rdagent.components.workflow.rd_loop as rd_loop

        importlib.import_module("research.us_quant")
        assert rd_loop.validate_qlib_features is us_validation.validate_us_features

    def test_qt_docker_prepare_skips_cn_download(self, monkeypatch) -> None:
        """Class patch: prepare = image build/pull only. Upstream would raise
        StopIteration on empty extra_volumes, then auto-download the CN
        dataset this box never uses."""
        us_validation.install_us_validation()

        from rdagent.utils.env import DockerEnv, QTDockerEnv

        assert QTDockerEnv.prepare.__name__ == "us_qt_docker_prepare"
        calls: list[object] = []
        monkeypatch.setattr(DockerEnv, "prepare", lambda self, *a, **k: calls.append(self))
        # Fresh conf: mutating QTDockerEnv()'s default conf would poison the
        # process-wide mount config — the exact bug get_us_model_env fixes.
        from rdagent.utils.env import QlibDockerConf

        env = QTDockerEnv(conf=QlibDockerConf())
        env.conf.extra_volumes = {}  # the upstream get_model_env(extra_volumes={}) shape
        env.prepare()
        assert calls == [env]

    def test_get_us_model_env_never_touches_the_shared_conf(self, monkeypatch) -> None:
        """Upstream get_model_env mutates QTDockerEnv's class-default conf
        (mutable default argument), deleting the ~/.qlib mount for every
        later backtest container — the 'instrument ... does not contain data
        for day' failure. The US env must build a fresh conf and merge the
        caller's volumes over the defaults."""
        import inspect

        from rdagent.utils.env import QTDockerEnv

        monkeypatch.setenv("MODEL_CoSTEER_ENV_TYPE", "docker")
        monkeypatch.setattr(QTDockerEnv, "prepare", lambda self, *a, **k: None)
        shared_conf = inspect.signature(QTDockerEnv.__init__).parameters["conf"].default
        qlib_mount = next(iter(shared_conf.extra_volumes))
        before = dict(shared_conf.extra_volumes)

        env = us_validation.get_us_model_env(extra_volumes={"/data": "/mnt/data"})

        assert env.conf is not shared_conf
        assert qlib_mount in env.conf.extra_volumes  # store mount survives the merge
        assert env.conf.extra_volumes["/data"] == "/mnt/data"
        assert env.conf.running_timeout_period == 600
        # The shared default (used by QlibFBWorkspace.execute backtests) is intact.
        assert shared_conf.extra_volumes == before
        assert shared_conf.running_timeout_period == 3600

    def test_get_us_model_env_default_volumes_keep_the_store_mount(self, monkeypatch) -> None:
        """The poisoning call shape: get_model_env() with no volumes (both
        get_runtime_environment call sites). The env must still carry ~/.qlib."""
        from rdagent.utils.env import QTDockerEnv

        monkeypatch.setenv("MODEL_CoSTEER_ENV_TYPE", "docker")
        monkeypatch.setattr(QTDockerEnv, "prepare", lambda self, *a, **k: None)

        env = us_validation.get_us_model_env()

        assert any(".qlib" in host for host in env.conf.extra_volumes)

    def test_get_us_factor_env_wraps_this_venv(self) -> None:
        env = us_validation.get_us_factor_env(
            extra_volumes={"/host": "/mnt"}, running_timeout_period=123
        )
        assert env.conf.bin_path == str(Path(sys.executable).parent)
        assert env.conf.extra_volumes == {"/host": "/mnt"}
        assert env.conf.running_timeout_period == 123

"""US-market runtime shims for the pinned rdagent tree's conda/CN hardcodes.

Several upstream assumptions cannot hold on this box (docker, no conda,
US-only qlib store):

1. ``rdagent.utils.qlib.validate_qlib_features`` — the base-feature gate in
   ``rd_loop._interact_init_params`` — runs its probe inside ``QlibCondaEnv``
   (needs a conda binary + an ``rdagent4qlib`` env) against qlib's DEFAULT
   provider (CN data, instrument SH600000, 2008-2020). Without conda the
   probe's PATH collapses to /bin:/usr/bin (no ``python``), and even with
   conda the CN store is absent — validation can never pass, so the gate
   re-asks for a revised base-feature config forever and every fin_quant run
   stalls at startup.
2. ``rdagent.components.coder.factor_coder.config.get_factor_env`` builds
   ``CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))``; without
   conda that is ``None`` and pydantic raises at scenario prompt render time
   (``get_runtime_environment``, factor_experiment.py:102).
3. ``QTDockerEnv.prepare`` auto-downloads the CN dataset whenever
   ``<first extra_volume>/qlib_data/cn_data`` is missing (it always is — this
   box holds only us_data), and crashes with ``StopIteration`` first when a
   caller replaced ``extra_volumes`` with the empty default
   (``get_model_env(extra_volumes={})``, the quant scenario's
   ``get_runtime_environment``). Patched at class level to image build/pull
   only, covering every consumer (``QlibFBWorkspace.execute`` backtests,
   ``get_model_env``, ``generate_data_folder_from_qlib``).
4. ``QTDockerEnv.__init__`` takes ``conf: DockerConf = QlibDockerConf()`` — a
   mutable default evaluated ONCE at class definition, so every
   ``QTDockerEnv()`` in the process shares that one conf object. Upstream
   ``get_model_env`` then does ``env.conf.extra_volumes = extra_volumes.copy()``
   (``{}`` at both quant/model ``get_runtime_environment`` call sites, hit
   while rendering the scenario prompt) and ``running_timeout_period = 600``
   — silently DELETING the ``~/.qlib -> /root/.qlib`` mount (and shrinking
   the 3600s backtest budget) for every later qlib container in the run.
   Each subsequent ``QlibFBWorkspace.execute`` backtest then sees an empty
   ``/root/.qlib`` and dies in instrument loading with ``ValueError:
   instrument ... does not contain data for day`` before any factor code
   runs. Upstream never trips this because its default model env is conda;
   ``MODEL_CoSTEER_ENV_TYPE=docker`` (US-043) put this box on the docker
   path. Replaced by ``get_us_model_env``: a FRESH ``QlibDockerConf`` per
   call, caller volumes merged OVER the defaults (the qlib store mount
   survives), and the shared class default is never touched.

All are replaced at RUNTIME by assignment (the US-024 pattern — the pinned
tree on disk stays untouched; tests/test_us_templates.py hashes it): the
validator becomes a subprocess of THIS interpreter (the repo venv ships qlib)
probing the US store, the factor env becomes a plain ``LocalEnv`` whose
``bin_path`` is this venv's bin/ — matching FACTOR_CoSTEER_PYTHON_BIN, which
ops points at the same interpreter — and the model env keeps its docker
mounts per call (#4). Model training and qlib backtests still follow
``MODEL_CoSTEER_ENV_TYPE=docker`` (QTDockerEnv, local_qlib:latest), wired in
ops/rdq-research.service and run_us_quant.sh.

``install_us_validation()`` runs at ``research.us_quant`` import — every
fin_quant process resolves the QLIB_QUANT_*_HYPOTHESIS2EXPERIMENT class paths
(and therefore imports us_quant) while the loop object is constructed, before
the interaction gate — and again from ``research.server_ui.main()`` so
UI-forked children inherit patched modules even if the class-path env vars
are ever unset. Assignment must cover the ``from x import y`` binding in each
consuming module, not just the defining module; keep the target list in sync
with upstream call sites when bumping research/PINNED_COMMIT.

Knobs (read at call time so tests and ops can override): RDQ_QLIB_STORE,
RDQ_VALIDATION_INSTRUMENT (default SPY — run_us_quant.sh --check requires it
in the store), RDQ_VALIDATION_START/END, RDQ_VALIDATION_TIMEOUT (seconds).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

_PROBE_TEMPLATE = """
import qlib
from qlib.data import D

qlib.init(provider_uri={store!r}, region="us")
df = D.features([{instrument!r}], {expressions!r}, start_time={start!r}, end_time={end!r})
assert not df.dropna(how="all").empty, "expressions produced no data"
"""


def _probe_script(expressions: list[str]) -> str:
    store = os.environ.get(
        "RDQ_QLIB_STORE", str(Path("~/.qlib/qlib_data/us_data").expanduser())
    )
    return textwrap.dedent(
        _PROBE_TEMPLATE.format(
            store=store,
            instrument=os.environ.get("RDQ_VALIDATION_INSTRUMENT", "SPY"),
            expressions=list(expressions),
            start=os.environ.get("RDQ_VALIDATION_START", "2016-01-01"),
            end=os.environ.get("RDQ_VALIDATION_END", "2024-12-31"),
        )
    )


def validate_us_features(expressions: list[str]) -> bool:
    """Drop-in for ``rdagent.utils.qlib.validate_qlib_features`` (same contract:
    True iff every qlib expression evaluates), probing the US store with the
    interpreter running this process instead of a conda env."""
    if not expressions:
        return True
    try:
        res = subprocess.run(
            [sys.executable, "-c", _probe_script(expressions)],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("RDQ_VALIDATION_TIMEOUT", "600")),
        )
    except subprocess.TimeoutExpired:
        _log_failure("feature validation probe timed out")
        return False
    if res.returncode != 0:
        # Upstream swallows the reason; keep the tail where operators look.
        _log_failure(f"feature validation probe failed:\n{res.stderr.strip()[-2000:]}")
    return res.returncode == 0


def _log_failure(message: str) -> None:
    try:
        from rdagent.log import rdagent_logger

        rdagent_logger.warning(message)
    except Exception:  # pragma: no cover - logging must never mask the verdict
        print(message, file=sys.stderr)


def get_us_factor_env(
    conf_type: str | None = None,
    extra_volumes: dict | None = None,
    running_timeout_period: int = 600,
    enable_cache: Any = None,
) -> Any:
    """Drop-in for ``get_factor_env``: the repo venv as a LocalEnv.

    ``enable_cache`` is accepted for signature parity and ignored —
    ``LocalConf`` has no such field (neither does upstream's ``CondaConf``).
    """
    from rdagent.utils.env import LocalConf, LocalEnv

    env = LocalEnv(
        conf=LocalConf(
            default_entry="python main.py",
            bin_path=str(Path(sys.executable).parent),
        )
    )
    env.conf.extra_volumes = dict(extra_volumes or {})
    env.conf.running_timeout_period = running_timeout_period
    env.prepare()
    return env


def get_us_model_env(
    conf_type: str | None = None,
    extra_volumes: dict | None = None,
    running_timeout_period: int = 600,
    enable_cache: Any = None,
) -> Any:
    """Drop-in for ``get_model_env`` that never poisons the shared conf.

    Upstream mutates ``QTDockerEnv()``'s class-default ``QlibDockerConf``
    (see module docstring #4), wiping the ``~/.qlib`` mount for every later
    backtest container in the process. Build a FRESH conf per call instead,
    and merge the caller's ``extra_volumes`` OVER the defaults so the qlib
    store mount survives.
    """
    from rdagent.components.coder.model_coder.conf import ModelCoSTEERSettings
    from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QlibDockerConf, QTDockerEnv

    settings = ModelCoSTEERSettings()
    if settings.env_type == "docker":
        conf = QlibDockerConf()  # fresh instance — NOT QTDockerEnv's shared default
        conf.extra_volumes = {**conf.extra_volumes, **(extra_volumes or {})}
        env: Any = QTDockerEnv(conf=conf)
    elif settings.env_type == "conda":
        env = QlibCondaEnv(conf=QlibCondaConf())
        env.conf.extra_volumes = dict(extra_volumes or {})
    else:
        raise ValueError(f"Unknown env type: {settings.env_type}")
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


def install_us_validation() -> None:
    """Point every upstream binding at the US implementations (idempotent)."""
    import rdagent.components.workflow.rd_loop as rd_loop
    import rdagent.utils.qlib as rd_qlib
    from rdagent.components.coder.factor_coder import config as factor_config
    from rdagent.components.coder.model_coder import conf as model_config
    from rdagent.scenarios.qlib.experiment import (
        factor_experiment,
        model_experiment,
        quant_experiment,
    )
    from rdagent.utils.env import DockerEnv, QTDockerEnv

    def us_qt_docker_prepare(self: Any, *args: Any, **kwargs: Any) -> None:
        """Image build/pull only — no CN-dataset auto-download (see module
        docstring #3). The US store lives on the host and reaches containers
        through QlibDockerConf's ~/.qlib mount; templates point provider_uri
        at us_data."""
        DockerEnv.prepare(self)

    rd_qlib.validate_qlib_features = validate_us_features
    rd_loop.validate_qlib_features = validate_us_features
    factor_config.get_factor_env = get_us_factor_env
    factor_experiment.get_factor_env = get_us_factor_env
    quant_experiment.get_factor_env = get_us_factor_env
    model_config.get_model_env = get_us_model_env
    model_experiment.get_model_env = get_us_model_env
    quant_experiment.get_model_env = get_us_model_env
    QTDockerEnv.prepare = us_qt_docker_prepare

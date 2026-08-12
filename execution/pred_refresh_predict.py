"""Exact-weights re-predict for the promoted strategy (US-049).

Copied into the promoted workspace by execution/pred_refresh.py on every
refresh (so code deploys propagate without re-snapshotting) and executed
INSIDE the local_qlib container with cwd=/workspace/qlib_workspace. Only
in-container dependencies (qlib, torch, ruamel.yaml) may be imported — lazily,
inside functions — and never anything from this repo.

Why not qrun: a qrun refresh RE-FITS the model, and with early stopping the
fit length is stochastic — the promoted GeneralPTNN stopped after epoch 8
(~52 min) when trained, but the 2026-08-07 morning refresh was still finding
improvements at epoch 14 when the 70-min budget killed it (worst case is
n_epochs=50 ≈ 4.5 h). No pre-open timeout can contain that. Re-predicting
from the promoted weights is deterministic (~10-15 min, dominated by the
dataset build) AND trades exactly the backtested model instead of a fresh
stochastic re-fit.

The flow mirrors qrun's own entrypoint (qlib.cli.run.workflow) minus
task_train:

1. render conf_pred_refresh.yaml with qlib's render_template (jinja context
   from the environment — docker -e populated it from pred_refresh.env, with
   test_end overridden to today);
2. qlib.init with the mlflow URI defaulted to ./mlruns, exactly like qrun;
3. build the dataset from the rendered task config (the handler re-fits its
   infer processors on the same train window, so inference normalization
   matches training);
4. CPU-unpickle pred_refresh_params.pkl — the promoted run's trained model,
   snapshotted at promote time. A CUDA-trained pickle (the GPU workers are
   the research backend) loads through a map_location shim and is forced to
   CPU;
5. inside an R.start run: save params.pkl, then SignalRecord.generate()
   (pred.pkl + label.pkl) — the same artifact set a qrun refresh produced,
   where execution/signal.py's newest-mtime rule finds it.

Any exception propagates: python exits nonzero, the traceback lands in
logs/pred_refresh_<date>.log, and the caller raises PredRefreshError.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

# Duplicated from execution/pred_refresh.py on purpose (this file must not
# import the repo); tests/test_pred_refresh.py asserts they stay equal.
CONF_NAME = "conf_pred_refresh.yaml"
PARAMS_NAME = "pred_refresh_params.pkl"


def load_model_cpu(path: Path):
    """Unpickle a trained qlib model, mapping any CUDA tensors to CPU.

    A plain pickle.load of a CUDA-trained model raises on a CPU-only box;
    torch tensors reduce through torch.storage._load_from_bytes, so routing
    that one symbol through torch.load(map_location="cpu") is sufficient.
    """
    import torch  # pyright: ignore[reportMissingImports] — in-container dep, not in the venv

    class _CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
            return super().find_class(module, name)

    with path.open("rb") as fh:
        model = _CPUUnpickler(fh).load()

    if not torch.cuda.is_available():
        # The storages are on CPU now, but the model's own bookkeeping may
        # still say cuda (GeneralPTNN moves batches with `.to(self.device)`).
        if hasattr(model, "device"):
            model.device = torch.device("cpu")
        for value in vars(model).values():
            if isinstance(value, torch.nn.Module):
                value.to("cpu")
    return model


def main() -> None:
    import qlib
    from qlib.cli.run import render_template, sys_config
    from qlib.config import C
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord
    from ruamel.yaml import YAML

    rendered = render_template(CONF_NAME)
    config = YAML(typ="safe", pure=True).load(rendered)
    sys_config(config, CONF_NAME)

    # qrun parity: honor an explicit exp_manager, else file:./mlruns.
    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    dataset = init_instance_by_config(config["task"]["dataset"])
    model = load_model_cpu(Path(PARAMS_NAME))

    experiment_name = config.get("experiment_name", "workflow")
    with R.start(experiment_name=experiment_name):
        recorder = R.get_recorder()
        R.save_objects(**{"params.pkl": model})
        SignalRecord(model=model, dataset=dataset, recorder=recorder).generate()


if __name__ == "__main__":
    main()

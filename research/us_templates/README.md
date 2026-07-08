# US-market template copies (US-016)

US-patched copies of the pinned RD-Agent qlib workspace templates. The pinned
upstream tree (see `research/PINNED_COMMIT`) is never edited; these folders are
drop-in replacements for the `template_folder_path` that `QlibFBWorkspace`
injects into every experiment workspace.

- `factor_template/` — copy of `rdagent/scenarios/qlib/experiment/factor_template/`
- `model_template/`  — copy of `rdagent/scenarios/qlib/experiment/model_template/`

`read_exp_res.py` and the per-folder `README.md` are byte-identical upstream
copies (the workspace runs `python read_exp_res.py`); only the `conf_*.yaml`
files are patched.

## Patch (applied to every conf_*.yaml)

| upstream (China A-share)              | here (US)                              |
|---------------------------------------|----------------------------------------|
| `provider_uri: ~/.qlib/qlib_data/cn_data` | `provider_uri: ~/.qlib/qlib_data/us_data` |
| `region: cn`                          | `region: us`                           |
| `market: &market csi300`              | `market: &market us_liquid`            |
| `benchmark: &benchmark SH000300`      | `benchmark: &benchmark SPY`            |
| `limit_threshold: 0.095`              | removed (no US daily price limit; qlib `region: us` default is `None`) |
| `close_cost: 0.0015`, `min_cost: 5`   | `close_cost: 0.0005`, `min_cost: 0`    |

Rationale for benchmark/cost choices: see `docs/decisions.md` (US-016 entry).
The `market:` value is the instruments filename in the store; US-023 renders
per-run copies with `market: <custom universe>`.

Prompt-text overrides (A-share language in upstream `prompts.yaml`) live
separately in `research/app_tpl/`, loaded via the `APP_TPL` env var.

## Rebasing onto a new upstream pin

1. Re-copy the two upstream folders over these (minus `__pycache__`).
2. Re-apply the table above (a plain `sed` per file; the US-016 entry in
   `docs/decisions.md` records the exact substitutions).
3. Run `pytest tests/test_us_templates.py` — it renders every YAML and asserts
   the patched values, and re-audits for leftover A-share strings.

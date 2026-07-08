# research/ — RD-Agent + LiteLLM conventions

- Anything that talks to an LLM provider must run through the proxy:
  `onecli run --agent rdq-research -- <cmd>`. It injects `HTTPS_PROXY` and
  CA-bundle env vars (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, ...) that
  LiteLLM/httpx honor automatically.
- API keys in code/env are placeholders only — the proxy overrides auth
  headers. Set `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY` to any non-empty value to
  satisfy LiteLLM's client-side presence checks (see `.env.example`).
- LiteLLM (1.91.0, pulled in by the rdagent pin) rejects `temperature` != 1
  for `claude-sonnet-5` with `UnsupportedParamsError`. Omit temperature or set
  `litellm.drop_params = True`.
- Anthropic JSON mode (`response_format={"type": "json_object"}`) may still
  return ```json-fenced output — parse with `probe_llm.extract_json_object`,
  and budget max_tokens generously (fences + prose-y JSON values).
- `import litellm` takes seconds; keep it lazy (inside functions) so offline
  unit tests and pyright stay fast.
- The stack is OpenAI-free: chat = `anthropic/...`, embeddings = `voyage/...`.
  Never introduce `OPENAI_*` variables; tests/test_probe_llm.py enforces this
  for `.env.example`.
- rdagent's dep tree is partly unpinned: `pydantic-ai-slim` 2.x breaks the
  `rdagent` CLI import at our pinned commit, so `install.sh` pins
  `pydantic-ai-slim[mcp,openai,prefect]==1.107.0` right after installing
  rdagent. Re-run `research/install.sh` after any pip operation that touches
  pydantic-ai; `tests/test_run_vanilla_factor.py::test_rdagent_cli_importable`
  guards the regression.

## US-market customization (never edit the pinned rdagent tree)

- `us_templates/` = patched copies of the two upstream workspace template
  folders. Only `conf_*.yaml` differ; `read_exp_res.py`/`README.md` must stay
  byte-identical to upstream (tests enforce it) — the dirs are excluded from
  ruff/pyright in pyproject.toml for that reason. Never run autofixers there.
- `app_tpl/` = partial prompt overrides loaded via the `APP_TPL` env var
  (`RDAgentSettings.app_tpl`, no env prefix; an absolute path works). Override
  files hold ONLY the overridden keys — rdagent's `load_content` falls through
  to upstream on missing keys. Mirror the upstream path under `app_tpl/`
  (e.g. `app_tpl/scenarios/qlib/experiment/prompts.yaml`).
- `APP_TPL` does NOT redirect the workspace template folders — upstream
  hardcodes `Path(__file__).parent / "factor_template"` in the experiment
  classes. The supported injection point is the env-configurable class paths
  in `rdagent/app/qlib_rd_loop/conf.py` (`QLIB_QUANT_*` etc.).
- `tests/test_us_templates.py::test_pinned_rdagent_install_unmodified` hashes
  every installed rdagent file against pip's RECORD — any in-place tweak to
  the upstream tree fails `make check`.

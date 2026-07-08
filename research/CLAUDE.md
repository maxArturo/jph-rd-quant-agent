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

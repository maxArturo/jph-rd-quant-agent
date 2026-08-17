# rd-agent-q

RD-Agent(Q) Slack-driven quant research and paper trading system: a Slack
orchestrator (Claude) launches RD-Agent(Q) research runs on disposable GPU
droplets (`ops/gpu_pipeline` — the only research backend), injects a digest of
prior runs into each new run's instruction, gates every candidate against the
promoted incumbent (parity-enforced criteria plus a reserved confirmation
window; auto-promotes on pass), records everything to Notion, and a
deterministic nightly rebalancer trades an Alpaca **paper** account through
the OneCLI gateway, with a post-close realized-vs-backtest divergence tracker
that can halt trading. Live trading is out of scope.

## Standalone constraint

This repo is standalone — it has **no dependency on nanoclaw** (code, paths,
or services). The only external service it talks to directly is the OneCLI
gateway at `http://127.0.0.1:10254`, which proxies and injects credentials for
all outbound API calls (Slack, Notion, FMP, Alpaca paper, LLM). No API keys
live in this repo or its environment files.

## Documents

- [PLAN.md](PLAN.md) — architecture, port table, identity/credential scoping
- [tasks/prd-rdagent-q-trading.md](tasks/prd-rdagent-q-trading.md) — source PRD

## Layout

```
orchestrator/   Slack bot, state store, LLM router, GPU run backend, run memory
execution/      Alpaca paper client, signal -> orders pipeline, safety gates,
                divergence tracker
research/       RD-Agent pin, US templates, prompt overrides, LLM probe
data/           FMP client, adjustment factors, Qlib store + universe builders
ops/            GPU pipeline + promotion gate, setup/verify scripts,
                systemd units, runbook
docs/reference/ schemas and reference docs
tests/          pytest suite
```

## Development

Requires Python >= 3.10.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make check           # ruff + pyright + pytest
research/install.sh  # install RD-Agent at the pinned commit (research/PINNED_COMMIT)
```

RD-Agent is installed from a pinned upstream commit rather than declared in
`pyproject.toml`; see `docs/decisions.md` for the pin and rationale.

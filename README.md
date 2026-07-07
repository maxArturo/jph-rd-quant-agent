# rd-agent-q

RD-Agent(Q) Slack-driven quant research and paper trading system: a Slack
orchestrator (Claude) drives RD-Agent(Q) research loops via `server_ui`,
records everything to Notion, and a deterministic nightly rebalancer trades an
Alpaca **paper** account through the OneCLI gateway. Live trading is out of
scope.

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
orchestrator/   Slack bot, state store, LLM router, RD-Agent control client
execution/      Alpaca paper client, signal -> orders pipeline, safety gates
research/       RD-Agent pin, US templates, prompt overrides, LLM probe
data/           FMP client, adjustment factors, Qlib store + universe builders
ops/            setup/verify scripts, systemd units, runbook
docs/reference/ schemas and reference docs
tests/          pytest suite
```

## Development

Requires Python >= 3.10.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make check   # ruff + pyright + pytest
```

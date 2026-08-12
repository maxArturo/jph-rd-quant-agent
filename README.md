# rd-agent-q

RD-Agent(Q) Slack-driven quant research and trading system: a Slack
orchestrator (Claude) drives RD-Agent(Q) research loops via `server_ui`,
records everything to Notion, and a deterministic pre-open rebalancer trades
an Alpaca **paper** account — and, once armed, a real-money Alpaca **live**
account — through the OneCLI gateway.

Live trading (2026-08-10 decisions, superseding 2026-08-05's
paper-first/two-step draft — see `docs/decisions.md`): the live channel
(#live-trading-quant-research) is a full research peer of #quant-research;
any promotable run can be promoted **direct to live** with a single Slack
message and **no confirmation step**; guardrails are conservative and
independent of paper's (10% equity allocation, $500/order, $5k daily
notional, 5% drawdown). Live orders flow only under the dedicated
`rdq-exec-live` OneCLI identity; with `SLACK_LIVE_CHANNEL_ID` unset the
system is paper-only, byte-for-byte as before.

## Standalone constraint

This repo is standalone — it has **no dependency on nanoclaw** (code, paths,
or services). The only external service it talks to directly is the OneCLI
gateway at `http://127.0.0.1:10254`, which proxies and injects credentials for
all outbound API calls (Slack, Notion, FMP, Alpaca paper and live, LLM). No
API keys live in this repo or its environment files.

## Documents

- [PLAN.md](PLAN.md) — architecture, port table, identity/credential scoping
- [tasks/prd-rdagent-q-trading.md](tasks/prd-rdagent-q-trading.md) — source PRD
- [tasks/prd-live-trading.md](tasks/prd-live-trading.md) — live-trading PRD
  (US-050..US-060); go-live checklist in [ops/runbook.md](ops/runbook.md) §8

## Layout

```
orchestrator/   Slack bot, state store, LLM router, RD-Agent control client
execution/      Alpaca client (paper default, live opt-in), signal -> orders
                pipeline, safety gates (paper + live guardrail configs)
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
make check           # ruff + pyright + pytest
research/install.sh  # install RD-Agent at the pinned commit (research/PINNED_COMMIT)
```

RD-Agent is installed from a pinned upstream commit rather than declared in
`pyproject.toml`; see `docs/decisions.md` for the pin and rationale.

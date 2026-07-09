# Operator runbook — rd-agent-q emergency procedures

Audience: the human operator of this box. Everything here is **paper
trading** (there is no live identity, and `execution/alpaca_client.py`
refuses the live host), but treat the procedures as if it were real money —
that is the point of the paper milestone.

For a full emergency stop, run the sections in order: **halt trading first**
(it is one file write and stops the next rebalance instantly), then pause
research, then flatten if the book itself must go to zero. Record what you
did and why in the Decision Log (the Slack tools do this automatically;
manual actions need a manual note).

## 1. Halt the rebalancer (stop new orders)

Preferred — in Slack (#quant-research), any thread:

> halt trading, reason: <why>

The `halt_trading` tool writes the breaker halt file, confirms in-thread,
and writes a Decision Log row. While the file exists, `execution/rebalance.py`
exits 0 with a "halted" notice and submits nothing; every daily summary shows
`breaker: HALTED — <reason>`.

Manual fallback (Slack down):

```sh
echo "manual halt: <why>" > ~/rdq-data/breaker/halt
```

Belt-and-braces (also stops the timer from even starting the pipeline):

```sh
XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user disable --now rdq-rebalance.timer
```

Resume later with the `resume_trading` Slack tool (removes the file, logs the
decision) or `rm ~/rdq-data/breaker/halt`, and re-enable the timer if you
disabled it.

## 2. Pause the research loop

Preferred — in Slack, in the thread that owns the run:

> stop the research run

The `stop_run` tool sends `POST /control stop` to server_ui, cancels pending
hypothesis prompts, and marks the run stopped (resumable later with
`resume_run` from the same thread).

Manual fallback — stop the control plane outright (kills server_ui **and**
its child research subprocesses, same cgroup):

```sh
XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user stop rdq-research.service
```

Research runs cost LLM tokens and disk, not money — pausing them is never
urgent the way halting trading is.

## 3. Flatten positions (go to zero)

Cancels every open order, liquidates every position, and confirms
`GET /v2/positions` is empty:

```sh
onecli run --agent rdq-exec-paper -- .venv/bin/python -m ops.flatten
```

- Exit 0: account confirmed flat.
- Exit 1: liquidations submitted but positions not yet empty — almost always
  a closed market (market orders fill at the next open). Rerun after the
  open to confirm; the halt file from step 1 keeps the rebalancer from
  re-entering positions in the meantime.
- Exit 2: the flatten could not run (auth/HTTP) — check
  `ops/check_onecli.sh`.

Expected follow-ups: liquidation orders have no Trade Ledger rows, so
`ops/reconcile.py` will flag them as missing ledger rows for the flatten
date — that is the audit trail working. Note the flatten in the Decision
Log.

## 4. Rotate keys via OneCLI

Vaulted secrets (Alpaca paper key+secret, Anthropic, Voyage, FMP): rotate at
the provider first (e.g. regenerate the paper keys in the Alpaca dashboard),
then update the vault **in place** — assignments are by secret id and
survive an update:

```sh
onecli secrets list                              # find the secret id
onecli secrets update --id <id> --value <new>    # both Alpaca secrets: key id AND secret key
ops/check_onecli.sh                              # every identity/service must PASS again
```

Never `secrets delete` + `secrets create` for rotation — that drops the
per-agent assignments (and `agents set-secrets` replaces the whole list).

Not in the vault:

- **Notion** auth is an app connection granted per agent in the OneCLI web
  UI (`https://nanoclaw-prod.tail05c9bf.ts.net/`) — re-grant there if Notion
  calls start returning 401 (docs/decisions.md 2026-07-09).
- **Slack** tokens live in the repo-root `.env` (sanctioned exception, never
  proxied): regenerate in the Slack app config, update
  `SLACK_OAUTH_TOKEN`/`SLACK_SOCKET_TOKEN`, then
  `systemctl --user restart rdq-orchestrator.service`.

## 5. Audit Tailscale exposure

`tailscale serve status` is the source of truth for what this box exposes.
Audit it against the PLAN.md §1 port table:

```sh
tailscale serve status
```

- Every mapping must say **(tailnet only)**; `tailscale funnel` output must
  never appear. If a funnel exists: `tailscale funnel reset`.
- Allowed from this repo: at most `https=19900 -> http://127.0.0.1:19900`
  (rdagent trace viewer, only while research monitoring is wanted).
  Pre-existing box mappings (`:443 -> 127.0.0.1:10254` OneCLI UI,
  `:3100 -> 127.0.0.1:3001`) are not ours to change.
- `rdagent server_ui` (:19899) must **never** be served — it is
  localhost-only by design (known flask-cors advisories).
- Remove an unexpected mapping with
  `tailscale serve --https=<port> off`, then find what added it.

Cross-check nothing repo-owned listens beyond loopback:

```sh
ss -tlnp | grep -vE '127\.0\.0\.1|\[::1\]'
```

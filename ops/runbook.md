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

Scripted audit (also checks every rdq-* unit and that nothing repo-owned
listens beyond loopback) — exit 0 healthy, nonzero naming each failing check:

```sh
ops/health.sh
```

To expose the rdagent trace viewer to the tailnet (the only mapping this
repo is allowed to add — tailnet-only, per the PLAN.md §1 port table):

```sh
ops/expose_traces.sh                  # tailscale serve --bg --https=19900 http://127.0.0.1:19900
tailscale serve --https=19900 off     # remove when monitoring is done
```

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

## 6. Routine monitoring & triage

Not emergencies — the checks for "is it alive and what is it doing".
From non-login shells, prefix every `systemctl --user` / `journalctl --user`
with `XDG_RUNTIME_DIR=/run/user/$(id -u)`.

### One-shot health check

```sh
ops/health.sh    # unit states + loopback audit + tailscale exposure; exit 0 = healthy
```

### Logs

```sh
journalctl --user -u rdq-orchestrator.service -f     # Slack bot, live
journalctl --user -u rdq-research.service -f         # server_ui control plane
journalctl --user -u rdq-rebalance.service -n 100    # last rebalance run
journalctl --user -u rdq-data-refresh.service -n 50  # last data refresh
journalctl --user -u rdq-sweep.service -n 50         # last retention sweep
```

The orchestrator is quiet by design for plain conversation: it logs tool
actions (`saved directive`, `started research run`, `trading halted`, ...)
and exceptions, **not** every message. "No log lines" after a chat message
is normal; the reply in the Slack thread is the signal. A missing reply
with no logged exception means the message never reached the bot — see the
deafness check below.

### Per-subsystem probes

```sh
# research control plane up?
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:19899/test   # want 200

# research run internals (hypotheses, Co-STEER attempts, backtest logs):
# start the viewer (transient unit; `rdagent ui` shells out to bare
# `streamlit`, so the venv must be on PATH), then map it tailnet-only:
systemd-run --user --unit=rdq-trace-viewer \
  -p WorkingDirectory=$HOME/rd-agent-q \
  -E PATH=$HOME/rd-agent-q/.venv/bin:/usr/local/bin:/usr/bin:/bin \
  -E STREAMLIT_SERVER_ADDRESS=127.0.0.1 -E STREAMLIT_SERVER_HEADLESS=true \
  $HOME/rd-agent-q/.venv/bin/rdagent ui --port 19900 \
  --log-dir $HOME/rdq-runs/server_ui/traces
ops/expose_traces.sh    # then open https://<tailnet-host>:19900
# stop when monitoring is done:
#   systemctl --user stop rdq-trace-viewer; tailscale serve --https=19900 off

# research run stuck? tail its trace log — if the newest line is
# "Requesting base feature configuration from user." and the file mtime is
# stale, the base-feature gate is failing validation on every submit
# (rd_loop retries forever). The probe's stderr is logged as
# "feature validation probe failed" (research/us_validation.py); a missing
# shim (e.g. QLIB_QUANT_* class paths unset) reverts to upstream's
# conda/CN-data probe, which can NEVER pass on this box — see
# docs/decisions.md US-043. Restart the service and relaunch the run.
tail -n 5 ~/rdq-runs/server_ui/traces/*/*.log

# orchestrator state (read-only peek; directives/runs/pending_interactions/
# promoted_strategy):
.venv/bin/python -c "import sqlite3; con=sqlite3.connect('orchestrator/state.sqlite'); \
  [print(t, con.execute(f'select count(*) from {t}').fetchone()[0]) for t in \
  ('directives','runs','pending_interactions','promoted_strategy')]"

# ledger vs broker (read-only both sides):
onecli run --agent rdq-exec-paper -- .venv/bin/python -m ops.reconcile
```

### Trading-day monitoring

The daily Slack summary (weekdays ~08:00 ET) is itself the monitor: equity,
orders, fills, gate/breaker rejections, and always a `breaker:` state line.
**A missing summary is a finding** — the rebalancer posts one on every day
it reaches the gate, including no-trade and rejection days. If it hasn't
appeared by ~08:10 ET, check `journalctl --user -u rdq-rebalance.service`.

### Slack-bot deafness check

The Socket Mode websocket can die without crashing the process (so
`Restart=always` never fires) — the failure mode is a healthy-looking
service that answers nothing. Check that the bot process holds a direct
connection to Slack on :443:

```sh
MAINPID=$(XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user show -p MainPID --value rdq-orchestrator.service)
ss -tnp | grep "pid=$MAINPID" | grep ':443 '   # want one ESTAB line to a public IP
```

No `:443` line (or a `CLOSE-WAIT` to `127.0.0.1:10254`) = deaf; restart the
service. Root cause of the known instance (2026-07-09): slack_sdk reads
`HTTPS_PROXY` but ignores `NO_PROXY`, so under `onecli run` the websocket
tunneled through the OneCLI proxy, which drops long-lived connections.
`orchestrator/app.py` now forces `proxy = None` on both Slack clients; if
deafness recurs, verify that override is still in place before hunting
elsewhere. Messages sent while the bot was deaf are **not replayed** —
resend them.

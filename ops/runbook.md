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

### Clearing a divergence auto-halt (US-016)

`rdq-divergence.timer` (weekdays 16:30 ET) compares realized returns since
promotion against a haircut backtest expectation (mean × 0.5). A trailing
z-score below −2 is a Slack warning only; z below −3 **or** drawdown since
promotion beyond backtest MDD × 1.25 writes the same breaker halt file with
a :rotating_light: notice naming the trigger. To clear one:

1. Read the trigger: `cat ~/rdq-data/breaker/halt` (and the Slack notice —
   realized vs expected numbers, drawdown since promotion).
2. Decide: is the strategy broken (consider rolling back the promotion,
   §7) or is the breach explainable (regime move, single-name event)?
3. Clear only after recording the decision: `resume trading` in Slack
   (writes the Decision Log row for you), or `rm ~/rdq-data/breaker/halt`
   plus a manual Decision Log note.

The tracker runs again at the next weekday close — if the breach condition
still holds it halts again, so clearing without addressing the cause buys
exactly one trading day.

## 2. Pause the research loop

Research runs execute on a disposable GPU droplet driven by
`ops/gpu_pipeline` (transient unit `rdq-gpu-run-<thread>`).

Preferred — in Slack, in the thread that owns the run:

> stop the research run

The `stop_run` tool cancels the run on the worker (kills its tmux session);
the pipeline then fetches partial results, posts the final summary, and
destroys the droplet — results up to the last completed loop stay
promotable. Runs are **not resumable**, and only the owning thread can stop
one (global run lock, US-020).

Manual fallback (Slack down):

```sh
ops/gpu_worker/gpu_worker.sh ssh tmux kill-session -t rdq-run
```

The droplet bills until destroyed — the pipeline's teardown handles that,
with the hourly `rdq-gpu-watchdog` as the 24h backstop. Confirm with
`ops/gpu_worker/gpu_worker.sh status`.

Research runs cost LLM tokens and GPU-droplet dollars (~$0.76/hr at the
default size), not brokerage money — halting trading is always the more
urgent action.

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
- Port 19899 (the retired server_ui control plane, decommissioned US-026)
  must **never** be served or listened on again — `ops/health.sh` fails the
  audit if it reappears.
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
journalctl --user -u 'rdq-gpu-run-*' -f              # active GPU research run (transient unit)
journalctl --user -u rdq-rebalance.service -n 100    # last rebalance run
journalctl --user -u rdq-data-refresh.service -n 50  # last data refresh
journalctl --user -u rdq-pred-refresh.service -n 50  # last prediction refresh (US-049)
journalctl --user -u rdq-divergence.service -n 50    # last divergence check (US-016/017)
journalctl --user -u rdq-reconcile.service -n 50     # last ledger reconcile (US-019)
journalctl --user -u rdq-sweep.service -n 50         # last retention sweep
```

### "predictions stale" rebalance abort

The 04:45 ET `rdq-pred-refresh.service` re-predicts the promoted workspace's
predictions every trading morning from the promoted run's exact weights
(docker run of the workspace copy of `execution/pred_refresh_predict.py`,
~10-15 min, US-049 — no re-fit; container output in
`<workspace>/logs/pred_refresh_<date>.log`). If a rebalance still aborts
stale, check that unit's journal first, kill any stuck
`rdq-pred-refresh-<date>` container (`docker ps`), then rerun supervised and
re-trade:

```sh
.venv/bin/python -m execution.pred_refresh --no-slack   # exit 0 = fresh again
systemctl --user start rdq-rebalance.service            # recover the trading day
```

If the failure says the snapshot is missing (`conf_pred_refresh.yaml` /
`pred_refresh.env` / `pred_refresh_params.pkl`), re-promote the strategy
from its Slack thread — the promotion flow writes the snapshot — or run
`execution.pred_refresh.snapshot_pred_refresh` by hand (it needs the
workspace's backtested mlflow run: params.pkl + portfolio_analysis/).

The orchestrator is quiet by design for plain conversation: it logs tool
actions (`saved directive`, `started research run`, `trading halted`, ...)
and exceptions, **not** every message. "No log lines" after a chat message
is normal; the reply in the Slack thread is the signal. A missing reply
with no logged exception means the message never reached the bot — see the
deafness check below.

### Per-subsystem probes

```sh
# GPU research run alive? droplet / tunnel / tmux / GPU util / log tail:
ops/gpu_worker/gpu_worker.sh status
# live pipeline stage, loop progress, test/confirmation windows, gate verdict:
cat ~/rdq-runs/gpu_worker/pipeline_status.json

# research run stuck? tail the run log on the worker — if the newest trace
# line is "Requesting base feature configuration from user." and old, the
# base-feature gate is failing validation on every submit (rd_loop retries
# forever; a missing shim reverts to upstream's conda/CN-data probe, which
# can NEVER pass — docs/decisions.md US-043). Stop the run (§2) and relaunch.
ops/gpu_worker/gpu_worker.sh ssh tail -n 20 /root/rdq-runs/gpu-run.log

# legacy on-box trace internals (pre-GPU-era runs only; fetched GPU traces
# do NOT appear here — use the Slack digests / Notion write-up instead):
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

# orchestrator state (read-only peek; directives/runs/promoted_strategy/
# promotion_history):
.venv/bin/python -c "import sqlite3; con=sqlite3.connect('orchestrator/state.sqlite'); \
  [print(t, con.execute(f'select count(*) from {t}').fetchone()[0]) for t in \
  ('directives','runs','promoted_strategy','promotion_history')]"

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

### GPU base snapshots (US-022)

Research runs boot from the newest `rdq-gpu-base-<hash>-<ts>` image whose
worker-inputs hash **and** region match (selection/prune logic:
`ops/gpu_snapshot.py`; full mechanics in ops/gpu_worker/README.md). When
inputs drift, the run full-bootstraps and rebakes automatically at teardown —
no operator action. Manual overrides on `python -m ops.gpu_pipeline`:
`--snapshot bake` forces a rebake (use after rebuilding `local_qlib:latest`,
which the hash does not cover); `--no-snapshot` ignores snapshots entirely.
Inspect state with `.venv/bin/python -m ops.gpu_snapshot hash` and
`... select --region tor1 [--hash <hash>]`; a bake failure is a Slack
warning only — the next run bootstraps in full and retries.

## 7. Roll back a promotion

Every promotion (auto-gate, Slack thread, CLI) appends an append-only
`promotion_history` row, and `ops/sweep.py` protects the workspaces of the
last 3 entries — so the previous strategy is on disk and one command away.
Run from the deployed checkout (`~/rd-agent-q`) so it targets the real
orchestrator/state.sqlite:

```sh
.venv/bin/python -m ops.rollback_promotion                  # dry-run: shows current vs previous
.venv/bin/python -m ops.rollback_promotion --yes            # re-promote the previous entry
.venv/bin/python -m ops.rollback_promotion --to <ws> --yes  # or a named workspace from history
```

- Restores the history entry's RECORDED config (what was actually traded),
  never a re-derivation from the workspace conf.
- Re-runs the pred-refresh snapshot BEFORE flipping the pointer — a
  snapshot failure leaves the current promotion untouched.
- `--keep-snapshot` preserves an operator-pinned `conf_pred_refresh.yaml`
  (required when the target workspace is pinned to a frozen `*_promoted_*`
  universe — the tool prints a reminder).
- Refuses when the target workspace directory no longer exists (swept).
- The rollback is itself a new history row (source `cli`, with a rollback
  marker in gate_verdict), so it is auditable and reversible.

Afterwards, verify the refresh path produces fresh predictions for the
restored strategy:

```sh
.venv/bin/python -m execution.pred_refresh --no-slack   # want exit 0
```

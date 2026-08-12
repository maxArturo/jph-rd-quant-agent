# Operator runbook — rd-agent-q emergency procedures

Audience: the human operator of this box. Sections 1–6 cover the **paper**
account (`rdq-exec-paper`, #quant-research). Section 7 covers the
**real-money LIVE** account (`rdq-exec-live`,
#live-trading-quant-research); section 8 is the go-live checklist.

**Paper and live are fully disjoint — a paper incident must not touch live
(and vice versa).** Halt files, breaker state, limits/breaker configs,
promotion slots, OneCLI identities, Slack channels, and Notion databases
are all separate. Halting, flattening, or demoting one account does nothing
to the other; only act on both when the incident genuinely spans both.

For a full emergency stop, run the sections in order: **halt trading first**
(it is one file write and stops the next rebalance instantly), then pause
research, then flatten if the book itself must go to zero. Record what you
did and why in the Decision Log (the Slack tools do this automatically;
manual actions need a manual note).

## 1. Halt the rebalancer (paper — stop new orders)

For the live account, see §7.1 — this section halts **paper only**.

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

## 3. Flatten positions (paper — go to zero)

For the live account, see §7.2 — this section flattens **paper only**.

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
journalctl --user -u rdq-pred-refresh.service -n 50  # last prediction refresh (US-048)
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
  ('directives','runs','pending_interactions','promoted_strategy','promoted_strategy_live')]"

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

## 7. LIVE account — real-money emergency procedures

Everything in this section moves (or stops) **real money**. The live
rebalancer (`rdq-rebalance-live.timer`, Mon–Fri 08:10 America/New_York)
trades `live_equity_allocation_pct` of live equity under
`execution/limits.live.json` and `execution/breaker.live.json`. All live
controls are independent of paper's (see the note at the top): a paper
incident must not touch live, and a live incident needs the live
procedures below — the paper ones (§1–§3) have no effect on the live
account.

### 7.1 Halt LIVE trading (stop new live orders)

Preferred — in Slack (**#live-trading-quant-research**, any thread):

> halt live trading, reason: <why>

The `halt_live_trading` tool writes the LIVE breaker halt file, confirms
in-thread (the reply says LIVE unmistakably), and writes a Decision Log
row. While the file exists, `execution/rebalance.py --live` exits 0 with a
halted notice and submits nothing. The tool is refused from
#quant-research — live controls only work in the live channel (and vice
versa for the paper tools).

Manual fallback (Slack down) — exact live halt file path:

```sh
echo "manual halt: <why>" > ~/rdq-data/breaker-live/halt
```

Belt-and-braces (also stops the timer from even starting the pipeline):

```sh
XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user disable --now rdq-rebalance-live.timer
```

Resume with `resume_live_trading` in the live channel (removes the file,
logs the decision) or `rm ~/rdq-data/breaker-live/halt`, and re-enable the
timer if you disabled it. A present live (or paper) halt file shows as a
`WARN breaker` line in `ops/health.sh` — deliberate operator state, never
a failed check.

### 7.2 Flatten LIVE positions (go to zero)

One-liner, under the live identity:

```sh
onecli run --agent rdq-exec-live -- .venv/bin/python -m ops.flatten --live
```

Exit codes are the same contract as paper (§3): 0 confirmed flat, 1
submitted-but-not-confirmed (usually a closed market — rerun after the
open; halt live first so the 08:10 rebalance cannot re-enter), 2
operational failure (check `ops/check_onecli.sh`). Liquidation orders have
no Trade Ledger (Live) rows, so `ops/reconcile.py --live` will flag them
for the flatten date — that is the audit trail working. Note the flatten
in the Decision Log.

### 7.3 Demote the live strategy (stop trading it, keep the account)

In the live channel:

> demote live

The `demote_live` tool clears the live promotion slot (a single message,
no confirmation — same contract as promote) and posts what was demoted.
The next live rebalance aborts with "no promoted strategy" and trades
nothing; existing positions stay on the book (flatten separately if the
book itself must go to zero). The paper slot is never touched.

### 7.4 Rotate LIVE keys in OneCLI

Same procedure as §4, live edition: regenerate the **live** API keys in
the Alpaca dashboard (top-left toggle on Live), then update both vaulted
live secrets **in place** — never delete + recreate (that drops the
per-agent assignment):

```sh
onecli secrets list                              # find the live key id + secret ids
onecli secrets update --id <id> --value <new>    # both live secrets
ops/check_onecli.sh                              # rdq-exec-live -> live host must PASS again
```

`ops/check_onecli.sh` proves the four isolation directions on every run:
only `rdq-exec-live` authenticates to `api.alpaca.markets`, and it cannot
reach the paper host.

## 8. Go-live checklist — the operator's FINAL manual step

All code for live trading ships and verifies against fakes **before** any
of this; funding the account and vaulting live keys is deliberately the
last thing that happens. Run the order of operations exactly:

1. **Deploy all live-trading stories** (merge to main, pull on the box).
2. **Install + restart** — deployed code is not running code:

   ```sh
   ops/install_services.sh          # links rdq-rebalance-live.{service,timer}
   XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user daemon-reload
   XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user restart rdq-orchestrator.service
   XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user enable --now rdq-rebalance-live.timer
   ```

   (`install_services.sh` links but never enables — the timer must be
   enabled explicitly.)
3. **Operator tasks A–D** below.
4. **Review the live guardrail configs**: `execution/limits.live.json`
   ($500/order, 10% position, 60 orders/day, 60 positions),
   `execution/breaker.live.json` ($5,000 daily notional, 5% drawdown),
   `execution/allocation.live.json` (`live_equity_allocation_pct`: 10).
5. **Send the one promotion message** in #live-trading-quant-research
   (e.g. "promote to live" for the paper-promoted strategy, or name a
   run). There is no confirmation step — the armed summary it posts IS the
   after-the-fact confirmation; read it.
6. **Observe the first live rebalance** (08:10 ET): fill summary in the
   live channel, rows in Trade Ledger (Live).
7. **Reconcile**:

   ```sh
   onecli run --agent rdq-exec-live -- .venv/bin/python -m ops.reconcile --live
   ```

   Exit 0 = ledger matches the live account.

### Operator tasks (from tasks/prd-live-trading.md §6)

**A. Alpaca (live account)**

1. Log in at https://app.alpaca.markets, switch the top-left toggle from
   Paper to **Live**, and complete live brokerage onboarding (identity,
   funding source) if not already done.
2. Fund the account. Only `live_equity_allocation_pct` (10%) of equity is
   traded; the breaker caps daily notional at $5,000.
3. Wait for account status **ACTIVE** (dashboard header; verifiable via
   `GET /v2/account` once task B is done).
4. With the Live toggle selected, generate **live** API keys (right-hand
   panel → API Keys). They are distinct from paper keys; the secret shows
   once — paste both straight into OneCLI (task B) and store them nowhere
   else.
5. Recommended account configuration (settable via
   `PATCH /v2/account/configurations` after task B): `no_shorting: true`,
   `max_margin_multiplier: "1"` (cash-like), fractional trading OFF (the
   pipeline trades whole shares).
6. Optional but recommended: enable Alpaca's email trade confirmations as
   an independent audit stream.

**B. OneCLI (live identity)**

1. Run `ops/setup_onecli.sh` — it creates `rdq-exec-live` with host
   allowlist `api.alpaca.markets api.notion.com financialmodelingprep.com`
   (no paper host; no other identity may hold the live host).
2. Vault the live key id + secret in the OneCLI web UI
   (`https://nanoclaw-prod.tail05c9bf.ts.net/`), host pattern
   `api.alpaca.markets`, mirroring the paper assignment shape — then
   **rerun `ops/setup_onecli.sh`** so the new secrets are assigned to
   `rdq-exec-live`.
3. Run `ops/check_onecli.sh`: the live-auth probe flips from
   SKIP-with-WARN to PASS, and all three must-fail directions
   (paper→live, orchestrator→live, live→paper) must PASS as refusals.

**C. Slack**

1. Invite the bot to **#live-trading-quant-research**: `/invite @<bot>`
   (channel already exists).
2. Copy the channel id (channel details → About → bottom) into the
   repo-root `.env` as `SLACK_LIVE_CHANNEL_ID` — setting it is what arms
   the live features.
3. After deploy: `systemctl --user restart rdq-orchestrator` — deployed
   code is not running code.

**D. Notion**

1. Ensure the integration used by the orchestrator can create a sibling of
   "Automated AI Quant Investment" — or create an empty page titled exactly
   "Automated AI Quant Investment — LIVE 🔴" and share it with the
   integration; `ops/bootstrap_notion.py --live` adopts it by exact title.
2. Grant the Notion app connection to `rdq-exec-live` in the gateway web
   UI (per-agent grant, no CLI) — the live rebalancer writes the live
   ledger, and `ops/reconcile.py --live` reads it, under that identity.

**E. First week**

1. Be around for the first 08:10 ET rebalance (order-of-operations steps
   6–7 above).
2. Deliberately halt live from Slack once (§7.1), confirm the next
   rebalance exits 0 without trading, then resume.

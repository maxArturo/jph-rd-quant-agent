# GPU burst worker — hypothesis runs on a disposable DO GPU droplet

Why this exists: broad-universe hypothesis runs (us_liquid, 581 names) die on
the control box — GRU training runs at ~5–6 min/epoch on 3 shared CPU threads
and hits the 3600s qrun wall-clock on nearly every hypothesis (2026-08-05
`average-urn` run). On a GPU those epochs take seconds; the timeout stops
binding and the research agent stops wasting hypotheses on "fit the compute
budget".

Everything is driven **from the control box** by `ops/gpu_worker/gpu_worker.sh`.
The worker is disposable and holds **no credentials**: LLM/embedding traffic
goes through an SSH reverse tunnel back to the control box's OneCLI proxy
(`127.0.0.1:10255`), authenticated with the per-agent proxy token (written
only to the worker's `/root/rdq-proxy.env`, mode 600) and TLS-verified against
the proxy's MITM CA bundle. The worker never appears on the tailnet and needs
no inbound access to the control box — the control box dials out.

Why the stack is GPU-ready with zero code changes:

- `local_qlib:latest` ships torch 2.2.1 + CUDA 12.1 + cuDNN (verified).
- rdagent's `QlibDockerConf.enable_gpu` defaults to true and `_gpu_kwargs`
  probes the NVIDIA runtime, passing `--gpus`-equivalent device requests only
  when present (falls back to CPU cleanly).
- The model templates already set `GPU: 0` (qlib device index) — cuda:0 is
  used whenever `torch.cuda.is_available()`.

## Prerequisites (one-time)

- `doctl` on the control box, authenticated with a DO API token:
  `doctl auth init` (or `export DIGITALOCEAN_ACCESS_TOKEN=...`). Minimum
  custom scopes: **droplet create/read/delete** and **ssh_key create/read**
  (the AI/ML image disables root passwords, so droplet create 422s without a
  registered account SSH key — cloud-init injection does not satisfy it).
  Custom-scoped tokens 403 on `/v2/account`; the script never calls it.
- `local_qlib:latest` present locally (it is), `research/.env` present,
  us_data store + factor source built (all true on this box).

## Cost

| Size slug | GPU | vCPU/RAM | $/hr |
|---|---|---|---|
| `gpu-4000adax1-20gb` (default) | RTX 4000 Ada 20GB | 8 / 32GB | ~$0.76 |
| `gpu-6000adax1-48gb` | RTX 6000 Ada 48GB | | ~$1.57 |
| `gpu-l40sx1-48gb` | L40S 48GB | | ~$1.57 |
| `gpu-h100x1-80gb` | H100 80GB | | ~$3.39 |

Billing is per-second and runs **until `destroy`** — destroying the droplet is
part of finishing a run, not cleanup you can defer. `status` shows what is
still alive.

GPU **stock comes and goes per region** (2026-08-06: the 4000 Ada and L40S
were sold out everywhere; the 6000 Ada was tor1-only). `provision` pre-flights
availability and fails with the live table; the pipeline additionally walks
`RDQ_GPU_SIZE_PLAN` (size:region fallbacks) until one has stock. Check with
`gpu_worker.sh sizes` and override, e.g.:

```sh
RDQ_GPU_SIZE=gpu-6000adax1-48gb RDQ_GPU_REGION=tor1 ops/gpu_worker/gpu_worker.sh provision
```

**Base snapshots (fast boots):** run the pipeline once with `--snapshot` (or
`gpu_worker.sh snapshot` on a bootstrapped worker) to bake an
`rdq-gpu-base-<ts>` image — provision auto-boots the newest one, cutting the
~20 min image-ship bootstrap to a ~3 min delta sync. Snapshots are regional
(usable where they were taken), cost ~$0.06/GiB/mo, and only the newest is
kept. Re-snapshot after `local_qlib:latest` or venv-dependency changes.

## Slack-first flow (the normal path)

All research runs execute on GPU workers (2026-08-06 decision) — this box is
the control plane. In #quant-research: refine the idea (save_directive),
optionally build a universe (set_universe → confirm_universe), then "start
the research". The bot launches the pipeline as a transient user unit
(`rdq-gpu-run-<thread>`), and everything lands back in the SAME thread:
per-loop digests, the final summary with the promotion candidate, a
**candidate-vs-promoted equity chart**, and a link to a **plain-language
Notion write-up**. Mid-run: ask "how's the run going?"
(check_research_status) or "cancel the run" (stop_run — partial results stay
promotable). Promotion is the usual two-yes thread flow; the fetched GPU
trace is located via orchestrator/gpu_backend.locate_run_artifacts (last
SOTA loop wins, not the last loop).

Notes: GPU runs are autonomous only (no supervised mode), cannot be resumed,
and the run's directive IS seeded into the loop (research/quant_runner.py) —
what you agree on in the thread steers the hypotheses.

## The one-command pipeline (manual/CLI path)

```sh
.venv/bin/python -m ops.gpu_pipeline --loop_n 10 \
    [--universe <name>] [--instruction "..."] [--snapshot] [--no-notion]
```

Runs the WHOLE lifecycle: provision (falling through `RDQ_GPU_SIZE_PLAN`
until a size has stock) → bootstrap → tunnel → check → run → **per-loop Slack
digests** (hypothesis, SOTA verdict, IC/ARR/MDD, posted as a thread in
#quant-research) → fetch → final summary naming the promotion candidate and
the exact promote command → **destroy**. Teardown also happens on pipeline
failure and on the `--max-hours` abort (default 24h); `--keep-worker` opts
out. `--no-slack` sends every digest to stderr instead. The tunnel is
self-healing: each poll restarts `rdq-gpu-tunnel` if it dropped.

Stop a run early (results up to that loop are still fetched + reported):

```sh
ops/gpu_worker/gpu_worker.sh ssh tmux kill-session -t rdq-run
```

Billing guard: `rdq-gpu-watchdog.timer` (hourly, installed via
`ops/install_services.sh`) warns to Slack when a worker sits idle with no
run, and fetch+destroys any worker older than 24h — a crashed pipeline
cannot leak a droplet for more than a day.

## Manual lifecycle (what the pipeline drives)

```sh
ops/gpu_worker/gpu_worker.sh provision      # create droplet (~2 min; billing starts)
ops/gpu_worker/gpu_worker.sh bootstrap      # apt + image ship (10-30 min first time) + data + venv
ops/gpu_worker/gpu_worker.sh tunnel         # reverse tunnel + proxy env; probes Anthropic == 200
ops/gpu_worker/gpu_worker.sh check          # remote run_us_quant.sh --check (must end "OK: environment ready")
ops/gpu_worker/gpu_worker.sh run --loop_n 10
ops/gpu_worker/gpu_worker.sh status         # droplet / tunnel / tmux / GPU util / log tail
ops/gpu_worker/gpu_worker.sh ssh <cmd>      # arbitrary remote command (worker SSH opts)
ops/gpu_worker/gpu_worker.sh fetch          # rsync traces + workspaces back
ops/gpu_worker/gpu_worker.sh destroy        # DELETE the droplet (stops billing)
```

`bootstrap` and `tunnel` are idempotent — rerun after interruptions. The
tunnel runs as transient user unit `rdq-gpu-tunnel` (`Restart=always`), so it
survives network blips during multi-day runs; check it with
`systemctl --user status rdq-gpu-tunnel`.

The run executes `ops/run_us_quant.sh` with `RDQ_LAUNCHER=direct` inside tmux
session `rdq-run` on the worker — the same wrapper, env wiring, and date knobs
as control-box runs (`RDQ_TEST_END` etc. pass through: set them in the
environment of `gpu_worker.sh run`... they are NOT forwarded automatically;
for non-default dates, edit `/root/rdq-launch.sh` via a manual tmux launch or
export them in the heredoc — see Troubleshooting).

## Where results land

`fetch` rsyncs the worker's `/root/rdq-runs/` into
`~/rdq-runs/gpu_worker/results/` on the control box:

- trace logs: `results/us_quant/log/<timestamp>/`
- workspaces (qlib_res.csv, mlruns/pred.pkl): `results/us_quant/workspace/`
- combined stdout: `results/gpu-run.log`

Judge a run by the usual completion criterion (non-NaN IC in qlib_res.csv,
sane ARR/MDD, US tickers in pred.pkl — see run_us_quant.sh header).

**Promotion caveat:** these runs live outside the orchestrator/server_ui flow,
so Slack-thread promotion does not see them (no `runs` row, and the trace's
runner-result pickles carry the WORKER's absolute workspace paths). Promote a
fetched workspace with the dedicated command instead — it writes the same
records as a thread promotion (pred-refresh snapshot into the workspace +
the `promoted_strategy` row with the conf-derived universe):

```sh
.venv/bin/python -m ops.promote_fetched --workspace ~/rdq-runs/gpu_worker/results/us_quant/workspace/<hash>          # dry-run
.venv/bin/python -m ops.promote_fetched --workspace ... --yes                                                        # write
.venv/bin/python -m execution.pred_refresh --no-slack                                                                # verify
```

It refuses on unusable confs/missing artifacts, shows the currently promoted
strategy before replacing it, and reminds you that (a) an operator-pinned
market in the OLD snapshot (e.g. a frozen `*_promoted_*` universe) must be
re-applied to the new one, and (b) the Notion Decision Log entry is manual
(no thread exists for the run). Run it from the deployed checkout
(~/rd-agent-q) so it targets the real orchestrator/state.sqlite.

**Trace visibility & retention:** fetched runs don't appear in the :19900
trace viewer, and `ops/sweep.py` neither protects nor prunes
`~/rdq-runs/gpu_worker/results/**` (it is outside the sweep's roots) — prune
old fetched runs by hand when disk matters.

## Security notes

- The worker is root-SSH-only with a dedicated keypair
  (`~/.ssh/rdq_gpu_worker`); nothing else is exposed by us. It is a public
  droplet — optionally attach a DO cloud firewall restricting :22 to this
  box's egress IP (`curl -s ifconfig.me`).
- The proxy token on the worker (`/root/rdq-proxy.env`, mode 600) is scoped
  to the rdq-research agent and dies with the droplet. Rotate via OneCLI if
  a worker is ever compromised mid-run.
- Never point the worker at server_ui (:19899) or the tailnet — it doesn't
  need either.

## Troubleshooting

- **tunnel probe != 200**: `systemctl --user status rdq-gpu-tunnel` (is ssh
  alive?); `onecli run --agent rdq-research -- curl ...` locally (is the
  proxy itself healthy?); re-run `tunnel` (it re-fetches the token).
- **CUDA assert in bootstrap**: wrong image slug — single-GPU workers must use
  the AI/ML-ready `gpu-h100x1-base` image (yes, for non-H100 sizes too).
- **run dies immediately**: `ssh root@<ip> cat /root/rdq-runs/gpu-run.log`;
  a "OneCLI proxy not reachable" FAIL means the tunnel dropped — restart it
  and relaunch (`run` refuses while the tmux session lingers: `ssh root@<ip>
  tmux kill-session -t rdq-run` first).
- **custom dates / universe**: launch manually on the worker instead of
  `gpu_worker.sh run`:
  `ssh root@<ip>` then
  `tmux new -s rdq-run` and inside:
  `set -a; . /root/rdq-proxy.env; set +a; RDQ_LAUNCHER=direct RDQ_TEST_END=2026-07-31 /root/rd-agent-q/ops/run_us_quant.sh --loop_n 10`
  (remember: TEST_END must stay short of the store end).
- **qrun timeout on huge models even on GPU**: raise it for the run with
  `QLIB_DOCKER_RUNNING_TIMEOUT_PERIOD=7200` (or `None` to disable) in the
  launch env.

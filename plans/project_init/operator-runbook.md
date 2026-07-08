# RD-Agent(Q) Trading — Operator Runbook (out-of-loop steps)

The ralph loop (`scripts/ralph/prd.json`, 42 stories) owns everything **deterministic
and code-shaped**. The steps below need a human: credential grants, third-party app
creation, long-running runs, and multi-day observation that a headless `claude --print`
iteration can't do. Ralph will attempt the affected stories regardless — when a
prerequisite below is missing it should record the actionable gap in `progress.txt`
and move on, so unblocking these early keeps the loop from stalling.

Sources of truth: `tasks/prd-rdagent-q-trading.md` (acceptance criteria),
`PLAN.md` (architecture, OneCLI mechanics, Tailscale port table).

---

## BEFORE launching ralph (blocking)

### 0. Initial git commit  (blocks everything)
Ralph needs `main` to exist so it can create `ralph/rdagent-q-trading`.
```bash
cd /home/nanoclaw/rd-agent-q
git add -A && git commit -m "initial: plan, PRD, ralph setup"
```

### 1. OneCLI secret assignment  (blocks US-003, US-004, US-011, US-025, US-028)
`ops/setup_onecli.sh` (US-003) registers the identities, but **new OneCLI agents start
in `selective` mode with zero secrets** — assignment is manual:

```bash
onecli agents list                 # find the three rdq-* agent ids after US-003 runs
onecli secrets list                # find secret ids
onecli agents set-secrets --id <agent-id> --secret-ids <id1>,<id2>,...
# or per agent: onecli agents set-secret-mode --id <agent-id> --mode all  (broader than needed)
```

| Identity | Needs secrets for |
|---|---|
| `rdq-orchestrator` | Anthropic (`api.anthropic.com`), Notion (`api.notion.com`), Alpaca **paper** (`paper-api.alpaca.markets`), Slack (`slack.com`, if OneCLI injection works — see step 3) |
| `rdq-research` | Anthropic, Voyage **embedding** key (`api.voyageai.com`), FMP (`financialmodelingprep.com`) |
| `rdq-exec-paper` | Alpaca **paper** only. Deliberately NOT live keys, NOT anything else. |

- [ ] Vault the Voyage embedding key (`VOYAGE_API_KEY`, host `api.voyageai.com`) if not already present (Anthropic has no embeddings API; Voyage decided 2026-07-07 — see docs/decisions.md).
- [ ] Confirm **no** `rdq-exec-live` identity exists and no live Alpaca secret is assigned anywhere.
- [ ] Confirm the Anthropic org has **≥30-day data retention** (Fable 5 400s under ZDR).
- [ ] After assignment, `ops/check_onecli.sh` should exit 0 — including the check that
      `rdq-exec-paper` gets 401/403 from `api.alpaca.markets` (isolation proof).

### 2. FMP plan-tier check  (blocks US-011–US-013 design hardening)
- [ ] Verify the FMP tier includes `/stable/historical-price-eod/full` plus the
      **splits** and **dividends** endpoints (needed to compute adjustment factors),
      and note the request-rate limit for a ~1000-ticker × 10-year backfill.
- [ ] Record findings in `docs/decisions.md` (US-011's iteration will look for them).

### 3. Slack app  (blocks US-006)
- [ ] Create a Slack app (Socket Mode ON) with bot scopes: `chat:write`,
      `channels:history`, `reactions:write`; install to the workspace.
- [ ] Create `#quant-research` and invite the bot.
- [ ] Collect `xoxb-` (bot) and `xapp-` (app-level, `connections:write`) tokens.
- [ ] Decide token path: try vaulting both in OneCLI with host pattern `slack.com`
      first; if injection fails for Socket Mode, put them in the repo-local `.env`
      (chat tokens only) — either way the US-006 iteration documents the outcome
      in `docs/decisions.md`.

### 4. Notion parent page  (blocks US-026)
- [ ] Create (or pick) a parent page for the five databases and **share it with the
      Notion integration** whose token is vaulted in OneCLI.
- [ ] Put the parent page ID where US-026 can read it: `orchestrator/config.yaml`
      under `notion.parent_page_id` (create the file with just that key if it
      doesn't exist yet).

### 5. Alpaca paper account baseline  (blocks US-028 smoke, US-040)
- [ ] Reset the Alpaca **paper** account in the dashboard.
- [ ] Record starting equity here → **STARTING_EQUITY = ____ USD (as of ____)**
      (US-031's drawdown high-water mark and the Phase-5 milestone both key off a
      known baseline).

### 6. Host prerequisites  (blocks US-005, US-013, US-017)
- [ ] Docker runs sudo-less for this user (`docker ps` works) — `rdagent health_check`
      requires it.
- [ ] Disk headroom: cn_data + us_data stores, `local_qlib:latest` image, and
      RD-Agent workspaces/mlruns comfortably fit (budget ≥50 GB free).
- [ ] `loginctl enable-linger nanoclaw` so systemd **user** units survive logout
      (US-010, US-018, US-036, US-041).
- [ ] GPU check: factor-only loops are CPU-fine; if no CUDA GPU, expect `fin_quant`
      model-evolution iterations to be slow/skipped — acceptable for v1.

Then launch:
```bash
scripts/ralph/ralph.sh --tool claude <max_iterations>
```

---

## DURING the loop (async / long-running — run in parallel with ralph)

### A. Vanilla RD-Agent full run  (US-005's full-run criterion)
Ralph verifies `--check` mode only. The real proof is yours:
```bash
ops/run_vanilla_factor.sh        # first run: downloads cn_data + builds local_qlib:latest (hours)
```
- [ ] Completed with `qlib_res.csv` (non-null IC) in the run workspace.
- [ ] Note the wall-clock time — it calibrates expectations for US-017.

### B. Full US data backfill  (after US-013 lands)
```bash
onecli run --agent rdq-research -- python data/build_store.py --tickers <broad-list>
```
- [ ] Backfill completes (it's checkpointed — safe to interrupt/resume).
- [ ] Spot-check a known split (NVDA around 2024-06-10) shows no price cliff.

### C. US fin_quant milestone run  (US-017's full-run criterion — the Phase-2 gate)
```bash
ops/run_us_quant.sh --loop_n 2
```
- [ ] Completes with plausible metrics (IC non-NaN, |ARR| < 200%, MDD < 0) and
      `pred.pkl` indexed by US tickers. Paste the metrics into `progress.txt`.

### D. Service verification after reboot  (US-010 / US-018 / US-036 / US-041 criteria)
- [ ] Reboot the box once after the units land; `ops/health.sh` exits 0.

---

## AFTER ralph finishes (operator-only milestones)

### E. End-to-end research flow  (PRD Phase-3 milestone)
- [ ] In `#quant-research`: post an idea → confirm the proposed universe → approve/edit
      a hypothesis via buttons → receive the metrics summary + equity-curve chart.
- [ ] Reconstruct the same run from Notion alone (idea → hypotheses → results chain)
      — PRD US-016 audit.

### F. First supervised rebalances  (before unattended operation)
- [ ] Promote a strategy via the Slack button (Decision Log row appears in Notion).
- [ ] Run `execution/rebalance.py --dry-run` and sanity-check the printed order list.
- [ ] Watch the first 2–3 live-timer paper rebalances end-to-end (fills in the
      Trade Ledger, daily summary in Slack).

### G. Ten-day unattended paper milestone  (PRD US-020 gate; blocks any live-trading PRD)
- [ ] ≥10 consecutive trading days with zero manual intervention.
- [ ] `ops/reconcile.py` exits 0 for the full window.
- [ ] Halt/resume drill: `halt_trading` in Slack → next rebalance exits 0 with the
      "halted" notice → `resume_trading` → trading resumes next cycle.
- [ ] Flatten drill (once, deliberately): `ops/flatten.py` → positions empty →
      re-promote/resume.

### H. Exposure audit  (recurring, monthly)
- [ ] `ops/health.sh` exits 0.
- [ ] `tailscale serve status` matches the PLAN.md port table — trace viewer :19900
      tailnet-only at most; **never** `funnel`; OneCLI :10254 stays as-is.
- [ ] `onecli agents secrets --id <rdq-exec-paper-id>` still shows paper-only.

---

## Standing rules

- **Live trading is out of scope.** No live keys in the vault for rdq identities, no
  `rdq-exec-live`, no approval rules needed for paper hosts. Going live requires its
  own PRD and the Phase-6 gating in `PLAN.md`.
- Emergencies: `ops/runbook.md` (halt → flatten → rotate keys → audit exposure)
  once US-040 lands; until then, the halt file + Alpaca dashboard are the levers.
- If ralph stalls on a story repeatedly, check `progress.txt` for the recorded gap —
  it's usually one of the checkboxes above.

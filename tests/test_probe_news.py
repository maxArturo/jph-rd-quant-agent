"""Offline tests for ops/probe_news.sh (US-071 / prd US-011).

Same pattern as tests/test_probe_market_series.py: the probe shells out to
`onecli run --agent rdq-research -- curl ...`, so the end-to-end tests run
the REAL script with a stub `onecli` first on PATH that emulates the FMP
news endpoints — dense/sparse tickers, pagination at limit=250, and the
live wire's ET timestamps — without any live HTTP.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "probe_news.sh"

START = "2025-01-02"
END = "2025-02-28"  # 58 calendar days
NOW_ET = "2025-03-03 10:00:00"

TICKERS = "AAPL:mega NVDA:mega EXPE:mid DECK:mid ASAN:thin"

# Articles per calendar day per ticker (ASAN is special-cased: one article
# only on the 15th — genuinely thin, like the real name).
PER_DAY = {"AAPL": 6, "NVDA": 5, "EXPE": 2, "DECK": 1}

# Emulates: onecli run --agent <id> -- curl -s -o <file> -w %{http_code} <url>
# /news/stock serves deterministic newest-first articles (PER_DAY per
# calendar day, sliced by page*limit); /news/stock-latest serves a wire whose
# newest article sits 6 minutes behind PROBE_STUB_NOW_ET. In healthy mode
# AAPL's page 0 returns 249 rows with more history behind it — the live
# endpoint's page-size jitter — so the probe's empty-page-only termination
# rule is always exercised. Modes:
#   http_404 / zero / minute_res / depth_capped (with PROBE_STUB_BAD_SYMBOL),
#   sparse_all (nobody needs page 1), utc_wire (+240m), stale_wire (-400m).
STUB_ONECLI = """#!/usr/bin/env python3
import json, os, sys
from datetime import date, datetime, timedelta
from urllib.parse import urlparse, parse_qs

args = sys.argv[1:]
out_path = args[args.index("-o") + 1]
url = args[-1]
parsed = urlparse(url)
q = parse_qs(parsed.query)
mode = os.environ.get("PROBE_STUB_MODE", "healthy")
bad = os.environ.get("PROBE_STUB_BAD_SYMBOL", "")
now_et = datetime.strptime(os.environ["PROBE_STUB_NOW_ET"], "%Y-%m-%d %H:%M:%S")

PER_DAY = {"AAPL": 6, "NVDA": 5, "EXPE": 2, "DECK": 1}


def respond(code, body):
    with open(out_path, "w") as fh:
        fh.write(body)
    sys.stdout.write(code)
    sys.exit(0)


def articles(sym, start, end, seconds=True):
    per_day = 1 if mode == "sparse_all" else PER_DAY.get(sym, 0)
    d = date.fromisoformat(end)
    first = date.fromisoformat(start)
    if mode == "depth_capped" and sym == bad:
        first = max(first, d - timedelta(days=10))
    out = []
    while d >= first:
        n = (1 if d.day == 15 else 0) if sym == "ASAN" else per_day
        for k in range(n - 1, -1, -1):
            ts = "%s 09:%02d" % (d.isoformat(), k)
            if seconds:
                ts += ":33"
            out.append({"symbol": sym, "publishedDate": ts,
                        "publisher": "Wire", "title": "t", "url": "u"})
        d -= timedelta(days=1)
    return out


if parsed.path.endswith("/news/stock"):
    sym = q["symbols"][0]
    page = int(q["page"][0])
    limit = int(q["limit"][0])
    if mode == "http_404" and sym == bad:
        respond("404", '{"error": "not found"}')
    if mode == "zero" and sym == bad:
        respond("200", "[]")
    seconds = not (mode == "minute_res" and sym == bad)
    rows = articles(sym, q["from"][0], q["to"][0], seconds=seconds)
    if mode == "healthy" and sym == "AAPL":  # page-size jitter: page 0 is 249
        start = 0 if page == 0 else (limit - 1) + (page - 1) * limit
        end = start + (limit - 1 if page == 0 else limit)
    else:
        start, end = page * limit, (page + 1) * limit
    respond("200", json.dumps(rows[start:end]))
elif parsed.path.endswith("/news/stock-latest"):
    offset = {"utc_wire": 240, "stale_wire": -400}.get(mode, -6)
    newest = now_et + timedelta(minutes=offset)
    rows = [
        {"symbol": "WIRE", "publishedDate":
            (newest - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
         "publisher": "Wire", "title": "t", "url": "u"}
        for i in range(3)
    ]
    respond("200", json.dumps(rows))
else:
    respond("404", "{}")
"""


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def run_probe(
    tmp_path: Path, mode: str = "healthy", **extra_env: str
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "onecli"
    stub.write_text(STUB_ONECLI)
    stub.chmod(0o755)
    universe = tmp_path / "us_liquid.txt"
    syms = [f"SYM{i}" for i in range(10)]
    rows = [f"{s}\t2025-01-02\t2026-08-21" for s in syms]
    rows.append("SYM0\t2020-01-02\t2024-06-30")  # PIT dup: 10 distinct symbols
    universe.write_text("\n".join(rows) + "\n")
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:{env['PATH']}"
    env.update(
        {
            "RDQ_PROBE_START": START,
            "RDQ_PROBE_END": END,
            "RDQ_PROBE_NOW_ET": NOW_ET,
            "RDQ_PROBE_UNIVERSE": str(universe),
            "RDQ_PROBE_TICKERS": TICKERS,
            "PROBE_STUB_MODE": mode,
            "PROBE_STUB_NOW_ET": NOW_ET,
            **extra_env,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env
    )


class TestShellcheck:
    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shellcheck_clean(self) -> None:
        proc = subprocess.run(
            ["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


class TestHealthyPath:
    def test_all_checks_pass(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        days = _days(START, END)
        # AAPL: 6/day * 58 days = 348 articles; jittered page 0 (249) means
        # pages of 249 + 99 + the empty end-of-history page = 3 requests.
        assert f"PASS  AAPL (mega)  articles={6 * days} requests=3" in proc.stdout
        assert f"PASS  NVDA (mega)  articles={5 * days} requests=3" in proc.stdout
        assert f"PASS  EXPE (mid)  articles={2 * days} requests=2" in proc.stdout
        # ASAN: one article on Jan 15 + Feb 15 only — thin, still a PASS.
        assert "PASS  ASAN (thin)  articles=2 requests=2" in proc.stdout
        # AAPL's short page 0 with history behind it must be surfaced, not
        # treated as end-of-history (the walk continued to page 1).
        assert "NOTE  1 short non-final page(s) observed" in proc.stdout
        assert "PASS  pagination verified at limit=250" in proc.stdout
        assert "gap=6m — publishedDate is US/Eastern" in proc.stdout
        assert "FAILURE" not in proc.stdout

    def test_distribution_and_budget_lines(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path)
        days = _days(START, END)
        # AAPL has exactly 6 articles every day of the window.
        assert (
            f"avg/day=6.0 active_days={days}/{days} p50/day=6 p90/day=6 max/day=6"
            in proc.stdout
        )
        # 3+3+2+2+2 = 12 sampled requests over 5 tickers; 10-symbol universe
        # (the PIT duplicate row must not double-count) -> ceil(12*10/5) = 24.
        assert (
            "BUDGET  sampled 12 page-requests over 5 tickers (mean 2.4 requests/ticker"
            in proc.stdout
        )
        assert "10 tickers, 2025-01-02 ->" in proc.stdout
        assert "~= 24 requests; at 3 req/s" in proc.stdout


class TestFailurePaths:
    def test_http_error_is_a_failure_not_a_skip(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="http_404", PROBE_STUB_BAD_SYMBOL="AAPL")
        assert proc.returncode == 1
        assert "FAILURE  AAPL  HTTP 404" in proc.stdout
        # The other samples still get probed and reported.
        assert "PASS  NVDA (mega)" in proc.stdout

    def test_zero_articles_is_a_failure(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="zero", PROBE_STUB_BAD_SYMBOL="ASAN")
        assert proc.returncode == 1
        assert "FAILURE  ASAN (thin)  zero articles" in proc.stdout

    def test_minute_resolution_timestamps_fail(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="minute_res", PROBE_STUB_BAD_SYMBOL="EXPE")
        assert proc.returncode == 1
        assert "FAILURE  EXPE" in proc.stdout
        assert "not at second resolution" in proc.stdout

    def test_plan_capped_history_fails_for_mega(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="depth_capped", PROBE_STUB_BAD_SYMBOL="NVDA")
        assert proc.returncode == 1
        assert "FAILURE  NVDA (mega)  history looks plan-capped" in proc.stdout
        assert "oldest article 2025-02-18" in proc.stdout  # END - 10 days

    def test_unproven_pagination_fails(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="sparse_all")
        assert proc.returncode == 1
        assert "FAILURE  pagination not demonstrated" in proc.stdout

    def test_utc_looking_wire_fails_loud(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="utc_wire")
        assert proc.returncode == 1
        assert "min in the FUTURE vs now-ET" in proc.stdout
        assert "looks like UTC" in proc.stdout

    def test_stale_wire_fails(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="stale_wire")
        assert proc.returncode == 1
        assert "min stale vs now-ET" in proc.stdout

    def test_missing_universe_dies_loud(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, RDQ_PROBE_UNIVERSE=str(tmp_path / "absent.txt"))
        assert proc.returncode == 1
        assert "universe file not found" in proc.stderr

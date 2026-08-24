"""Offline tests for ops/probe_market_series.sh (US-064 / prd US-004).

The probe shells out to `onecli run --agent rdq-research -- curl ...`, so the
end-to-end tests run the REAL script with a stub `onecli` on PATH that
emulates the FMP responses — including the treasury endpoint's ~90-calendar-
day silent truncation — without any live HTTP.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "ops" / "probe_market_series.sh"

START = "2025-01-02"
END = "2025-06-30"  # 180 calendar days -> 2 treasury chunks, cap demonstrable

SYMBOLS = ["BZUSD", "CLUSD", "HOUSD", "NGUSD", "RBUSD", "GCUSD", "DXUSD"]

# Emulates: onecli run --agent <id> -- curl -s -o <file> -w %{http_code} <url>
# Serves weekday-daily rows; treasury requests wider than 90 calendar days are
# truncated to the trailing 90 days (the real FMP cap) unless
# PROBE_STUB_MODE=no_cap. PROBE_STUB_BAD_SYMBOL drives per-symbol failures.
STUB_ONECLI = """#!/usr/bin/env python3
import json, os, sys
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qs

args = sys.argv[1:]
out_path = args[args.index("-o") + 1]
url = args[-1]
parsed = urlparse(url)
q = parse_qs(parsed.query)
mode = os.environ.get("PROBE_STUB_MODE", "healthy")
bad = os.environ.get("PROBE_STUB_BAD_SYMBOL", "")


def weekdays(start, end):
    d, e, out = date.fromisoformat(start), date.fromisoformat(end), []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def respond(code, body):
    with open(out_path, "w") as fh:
        fh.write(body)
    sys.stdout.write(code)
    sys.exit(0)


if "historical-price-eod" in parsed.path:
    sym = q["symbol"][0]
    if mode == "http_404" and sym == bad:
        respond("404", '{"error": "not found"}')
    if mode == "empty_symbol" and sym == bad:
        respond("200", "[]")
    rows = [
        {"symbol": sym, "date": d, "price": 75.0, "volume": 1000}
        for d in weekdays(q["from"][0], q["to"][0])
    ]
    respond("200", json.dumps(rows))
elif "treasury-rates" in parsed.path:
    frm, to = q["from"][0], q["to"][0]
    span = (date.fromisoformat(to) - date.fromisoformat(frm)).days
    if span > 90 and mode != "no_cap":
        frm = (date.fromisoformat(to) - timedelta(days=89)).isoformat()
    rows = [
        {"date": d, "month1": 4.1, "year10": 4.25, "year30": 4.5}
        for d in weekdays(frm, to)
    ]
    respond("200", json.dumps(rows))
else:
    respond("404", "{}")
"""


def _weekdays(start: str, end: str) -> list[str]:
    d, e, out = date.fromisoformat(start), date.fromisoformat(end), []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def run_probe(
    tmp_path: Path, mode: str = "healthy", **extra_env: str
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "onecli"
    stub.write_text(STUB_ONECLI)
    stub.chmod(0o755)
    calendar = tmp_path / "day.txt"
    calendar.write_text("\n".join(_weekdays(START, END)) + "\n")
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}:{env['PATH']}"
    env.update(
        {
            "RDQ_PROBE_START": START,
            "RDQ_PROBE_END": END,
            "RDQ_PROBE_CALENDAR": str(calendar),
            "PROBE_STUB_MODE": mode,
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
    def test_all_series_pass_and_cap_demonstrated(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for sym in SYMBOLS:
            assert f"PASS  {sym}  rows=" in proc.stdout
        assert "PASS  treasury-rates  rows=" in proc.stdout
        assert "2 chunked requests" in proc.stdout
        assert "PASS  treasury cap" in proc.stdout
        assert "FAILURE" not in proc.stdout

    def test_counts_reflect_the_cap(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path)
        chunked = len(_weekdays(START, END))
        capped_from = (date.fromisoformat(END) - timedelta(days=89)).isoformat()
        single = len(_weekdays(capped_from, END))
        assert f"rows={chunked}" in proc.stdout
        assert f"returned {single} rows < {chunked} chunked" in proc.stdout


class TestFailurePaths:
    def test_http_error_is_a_failure_not_a_skip(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="http_404", PROBE_STUB_BAD_SYMBOL="NGUSD")
        assert proc.returncode == 1
        assert "FAILURE  NGUSD  HTTP 404" in proc.stdout
        # The other series still get probed and reported.
        assert "PASS  BZUSD" in proc.stdout
        assert "1 FAILURE(s)" in proc.stdout

    def test_empty_response_is_a_failure(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="empty_symbol", PROBE_STUB_BAD_SYMBOL="HOUSD")
        assert proc.returncode == 1
        assert "FAILURE  HOUSD  empty response" in proc.stdout

    def test_missing_cap_fails_the_probe(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, mode="no_cap")
        assert proc.returncode == 1
        assert "cap not demonstrated" in proc.stdout

    def test_missing_calendar_dies_loud(self, tmp_path: Path) -> None:
        proc = run_probe(tmp_path, RDQ_PROBE_CALENDAR=str(tmp_path / "absent.txt"))
        assert proc.returncode == 1
        assert "store calendar not found" in proc.stderr

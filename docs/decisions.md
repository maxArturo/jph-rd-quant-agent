# Decision Log

Running log of technical decisions with rationale. Newest entries at the bottom.

## 2026-07-07 — RD-Agent pinned commit (US-002)

**Decision:** Pin microsoft/RD-Agent to commit
`4f9ecb005881cddc08df0124a2e894c018007679` (main HEAD as of 2026-05-06,
"Document FT-Agent ICML release (#1406)").

**Why:**
- Upstream activity slowed in 2026 (PLAN.md finding #6); this commit has been
  the unmoved tip of `main` for ~2 months at pin time — the de facto stable.
- It is 28 commits ahead of the last tagged release `v0.8.0` (2025-11-03) and
  includes the post-release fixes; setuptools-scm reports it as `0.8.1.dev28`.
- Pinning a full SHA (not a branch or tag) makes installs reproducible and
  makes our template/prompt patch surface (US-016) diffable against a fixed
  upstream tree.

**Install path:** `research/PINNED_COMMIT` holds the SHA;
`research/install.sh` installs `rdagent @ git+https://github.com/microsoft/RD-Agent@<SHA>`
into the project venv (`.venv`) and verifies the import. RD-Agent is
deliberately NOT a `pyproject.toml` dependency: its heavy dependency tree
(litellm, streamlit, docker, ...) is only needed by the research engine, and a
direct-URL git pin in `pyproject` would slow every editable reinstall.

**Verification:** `pip check` clean; `python -c 'import rdagent'` exits 0;
pip's `direct_url.json` records `commit_id = 4f9ecb00...` (exact pin).

**Rebase procedure (future upstream update):** update `research/PINNED_COMMIT`,
re-run `research/install.sh`, re-run the upstream-pristine test (US-016) and
re-diff `research/us_templates/` + `research/app_tpl/` against the new tree.

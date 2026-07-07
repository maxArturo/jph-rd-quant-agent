"""Scaffold sanity checks: the repo layout US-001 promises actually exists."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_DIRS = [
    "orchestrator",
    "execution",
    "research",
    "data",
    "ops",
    "docs/reference",
    "tests",
]


def test_expected_directories_exist() -> None:
    for rel in EXPECTED_DIRS:
        assert (REPO_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_readme_states_standalone_constraint() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    assert "nanoclaw" in readme
    assert "http://127.0.0.1:10254" in readme
    assert "PLAN.md" in readme
    assert "tasks/prd-rdagent-q-trading.md" in readme

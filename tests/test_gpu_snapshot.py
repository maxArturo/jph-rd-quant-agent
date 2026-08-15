"""Offline tests for GPU base-snapshot bookkeeping (ops/gpu_snapshot.py, US-022)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ops.gpu_snapshot import (
    HASH_LEN,
    KEEP_SNAPSHOTS,
    BaseImage,
    _makefile_venv_targets,
    list_base_images,
    main,
    newest_in_region,
    prune_snapshots,
    select_snapshot,
    worker_inputs_hash,
)

DEFAULT_MAKEFILE = (
    "VENV := .venv\n"
    "BIN := $(VENV)/bin\n"
    "\n"
    ".PHONY: check lint venv\n"
    "\n"
    "check: lint\n"
    "\n"
    "lint:\n"
    "\t$(BIN)/ruff check .\n"
    "\n"
    "venv:\n"
    "\tpython3 -m venv $(VENV)\n"
    "\t$(BIN)/pip install -e '.[dev]'\n"
)


def make_repo(
    tmp_path: Path,
    *,
    pinned: str = "abc123",
    worker: str = "#!/usr/bin/env bash\necho v1\n",
    makefile: str = DEFAULT_MAKEFILE,
) -> Path:
    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True, exist_ok=True)
    (repo / "ops" / "gpu_worker").mkdir(parents=True, exist_ok=True)
    (repo / "research" / "PINNED_COMMIT").write_text(pinned + "\n")
    (repo / "ops" / "gpu_worker" / "gpu_worker.sh").write_text(worker)
    (repo / "Makefile").write_text(makefile)
    return repo


def make_store(
    tmp_path: Path,
    *,
    fields: tuple[str, ...] = ("close.day.bin", "volume.day.bin"),
    calendars: tuple[str, ...] = ("day.txt",),
    calendar_body: str = "2026-01-02\n",
) -> Path:
    store = tmp_path / "store"
    (store / "features" / "aapl").mkdir(parents=True, exist_ok=True)
    (store / "calendars").mkdir(exist_ok=True)
    for name in fields:
        (store / "features" / "aapl" / name).write_bytes(b"")
    for name in calendars:
        (store / "calendars" / name).write_text(calendar_body)
    return store


class TestWorkerInputsHash:
    def test_deterministic_short_hex(self, tmp_path: Path) -> None:
        repo, store = make_repo(tmp_path), make_store(tmp_path)
        first = worker_inputs_hash(repo, store)
        assert first == worker_inputs_hash(repo, store)
        assert len(first) == HASH_LEN
        assert all(c in "0123456789abcdef" for c in first)

    def test_changes_on_pinned_commit(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        before = worker_inputs_hash(make_repo(tmp_path), store)
        after = worker_inputs_hash(make_repo(tmp_path, pinned="def456"), store)
        assert before != after

    def test_changes_on_worker_script(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        before = worker_inputs_hash(make_repo(tmp_path), store)
        after = worker_inputs_hash(make_repo(tmp_path, worker="#!/bin/bash\necho v2\n"), store)
        assert before != after

    def test_changes_on_venv_target(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        before = worker_inputs_hash(make_repo(tmp_path), store)
        changed = DEFAULT_MAKEFILE.replace("pip install -e '.[dev]'", "pip install -e .")
        after = worker_inputs_hash(make_repo(tmp_path, makefile=changed), store)
        assert before != after

    def test_unchanged_by_non_venv_target(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        before = worker_inputs_hash(make_repo(tmp_path), store)
        changed = DEFAULT_MAKEFILE.replace("ruff check .", "ruff check . --fix")
        after = worker_inputs_hash(make_repo(tmp_path, makefile=changed), store)
        assert before == after

    def test_unchanged_by_store_data_refresh(self, tmp_path: Path) -> None:
        """The store's CONTENT rolls daily and rsyncs on every bootstrap —
        only layout changes may force a rebake."""
        repo = make_repo(tmp_path)
        before = worker_inputs_hash(repo, make_store(tmp_path))
        refreshed = make_store(tmp_path, calendar_body="2026-01-02\n2026-01-03\n")
        assert worker_inputs_hash(repo, refreshed) == before

    def test_changes_when_feature_field_added(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        before = worker_inputs_hash(repo, make_store(tmp_path))
        wider = make_store(tmp_path, fields=("close.day.bin", "volume.day.bin", "vwap.day.bin"))
        assert worker_inputs_hash(repo, wider) != before

    def test_missing_pinned_commit_fails_loud(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        (repo / "research" / "PINNED_COMMIT").unlink()
        with pytest.raises(OSError):
            worker_inputs_hash(repo, make_store(tmp_path))

    def test_cli_hash_prints_short_hex(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["hash"]) == 0
        out = capsys.readouterr().out.strip()
        assert len(out) == HASH_LEN


class TestMakefileVenvSlice:
    def test_keeps_assignments_and_venv_recipe_only(self) -> None:
        kept = _makefile_venv_targets(DEFAULT_MAKEFILE)
        assert "VENV := .venv" in kept
        assert "pip install -e '.[dev]'" in kept
        assert "ruff check" not in kept
        assert "check: lint" not in kept


def image(
    id_: str, name: str, regions: tuple[str, ...] = ("tor1",), created_at: str = ""
) -> BaseImage:
    return BaseImage(id=id_, name=name, regions=regions, created_at=created_at)


HASH_A = "a" * HASH_LEN
HASH_B = "b" * HASH_LEN


class TestSelection:
    def test_matches_hash_and_region(self) -> None:
        images = [image("1", f"rdq-gpu-base-{HASH_A}-20260810-1200")]
        selected = select_snapshot(images, HASH_A, "tor1")
        assert selected is not None and selected.id == "1"

    def test_hash_mismatch_never_matches(self) -> None:
        images = [image("1", f"rdq-gpu-base-{HASH_A}-20260810-1200")]
        assert select_snapshot(images, HASH_B, "tor1") is None

    def test_region_aware(self) -> None:
        """A hash match in another region must not be selected — the image
        isn't there for a size-plan fallback provision."""
        images = [image("1", f"rdq-gpu-base-{HASH_A}-20260810-1200", regions=("nyc2",))]
        assert select_snapshot(images, HASH_A, "tor1") is None
        selected = select_snapshot(images, HASH_A, "nyc2")
        assert selected is not None and selected.id == "1"

    def test_newest_match_wins(self) -> None:
        images = [
            image("old", f"rdq-gpu-base-{HASH_A}-20260801-0900", created_at="2026-08-01T09:00Z"),
            image("new", f"rdq-gpu-base-{HASH_A}-20260810-1200", created_at="2026-08-10T12:00Z"),
        ]
        selected = select_snapshot(images, HASH_A, "tor1")
        assert selected is not None and selected.id == "new"

    def test_legacy_unhashed_name_never_matches(self) -> None:
        legacy = image("1", "rdq-gpu-base-20260810-1200")
        assert legacy.inputs_hash is None
        assert select_snapshot([legacy], HASH_A, "tor1") is None

    def test_newest_in_region_ignores_hash(self) -> None:
        images = [
            image("a", f"rdq-gpu-base-{HASH_A}-20260801-0900", created_at="2026-08-01T09:00Z"),
            image("b", "rdq-gpu-base-20260810-1200", created_at="2026-08-10T12:00Z"),
            image("c", f"rdq-gpu-base-{HASH_B}-20260812-1200", regions=("nyc2",),
                  created_at="2026-08-12T12:00Z"),
        ]
        selected = newest_in_region(images, "tor1")
        assert selected is not None and selected.id == "b"
        assert newest_in_region(images, "atl1") is None


class FakeDoctl:
    """Records doctl invocations; serves a canned image list, accepts deletes."""

    def __init__(self, images: list[dict], fail_delete_ids: set[str] | None = None) -> None:
        self._images = images
        self._fail_delete = fail_delete_ids or set()
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(list(cmd))
        if cmd[:4] == ["doctl", "compute", "image", "list"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(self._images), stderr="")
        if cmd[:4] == ["doctl", "compute", "image", "delete"]:
            if cmd[-1] in self._fail_delete:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="denied")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected doctl call: {cmd}")


def doctl_image(id_: int, name: str, regions: list[str], created_at: str) -> dict:
    return {"id": id_, "name": name, "regions": regions, "created_at": created_at}


class TestDoctlListAndPrune:
    def test_list_filters_prefix_and_parses(self) -> None:
        fake = FakeDoctl(
            [
                doctl_image(1, f"rdq-gpu-base-{HASH_A}-20260810-1200", ["tor1"], "2026-08-10"),
                doctl_image(2, "ubuntu-24-04-x64", ["tor1"], "2026-01-01"),
            ]
        )
        images = list_base_images(fake)
        assert [i.id for i in images] == ["1"]
        assert images[0].regions == ("tor1",)
        assert images[0].inputs_hash == HASH_A

    def test_list_raises_on_doctl_failure(self) -> None:
        def broken(cmd, **kwargs):  # noqa: ANN001, ANN003
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="401 unauthorized")

        with pytest.raises(RuntimeError, match="401"):
            list_base_images(broken)

    def test_prune_keeps_newest_two(self) -> None:
        fake = FakeDoctl(
            [
                doctl_image(1, f"rdq-gpu-base-{HASH_A}-20260801-0900", ["tor1"], "2026-08-01"),
                doctl_image(2, f"rdq-gpu-base-{HASH_A}-20260805-0900", ["tor1"], "2026-08-05"),
                doctl_image(3, f"rdq-gpu-base-{HASH_B}-20260810-0900", ["tor1"], "2026-08-10"),
                doctl_image(4, "ubuntu-24-04-x64", ["tor1"], "2026-01-01"),
            ]
        )
        deleted = prune_snapshots(fake, keep=KEEP_SNAPSHOTS)
        assert [i.id for i in deleted] == ["1"]
        delete_calls = [c for c in fake.calls if c[:4] == ["doctl", "compute", "image", "delete"]]
        assert delete_calls == [["doctl", "compute", "image", "delete", "-f", "1"]]

    def test_prune_failed_delete_is_skipped_not_raised(self) -> None:
        fake = FakeDoctl(
            [
                doctl_image(1, f"rdq-gpu-base-{HASH_A}-20260801-0900", ["tor1"], "2026-08-01"),
                doctl_image(2, f"rdq-gpu-base-{HASH_A}-20260803-0900", ["tor1"], "2026-08-03"),
                doctl_image(3, f"rdq-gpu-base-{HASH_A}-20260805-0900", ["tor1"], "2026-08-05"),
                doctl_image(4, f"rdq-gpu-base-{HASH_B}-20260810-0900", ["tor1"], "2026-08-10"),
            ],
            fail_delete_ids={"1"},
        )
        deleted = prune_snapshots(fake)
        assert [i.id for i in deleted] == ["2"]

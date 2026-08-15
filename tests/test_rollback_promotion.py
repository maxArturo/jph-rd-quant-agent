"""Offline tests for the promotion rollback CLI (ops/rollback_promotion.py, US-006).

snapshot_pred_refresh is monkeypatched (it needs a full fetched workspace with
docker logs — tests/test_pred_refresh.py owns that machinery); these tests
assert on the pointer flip, the appended history row, and the refusal paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import ops.rollback_promotion as rollback_promotion
from execution.pred_refresh import PredRefreshError
from ops.rollback_promotion import RollbackError, select_target
from orchestrator.state import StateStore


def promote(store: StateStore, workspace: Path, **config: object) -> None:
    store.set_promoted_strategy(str(workspace), {"universe": "us_liquid", **config})


@pytest.fixture()
def snapshot_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record snapshot_pred_refresh calls; return plausible snapshot paths."""
    calls: list[Path] = []

    def fake_snapshot(workspace: Path) -> tuple[Path, Path, Path]:
        calls.append(workspace)
        return (
            workspace / "conf_pred_refresh.yaml",
            workspace / "pred_refresh.env",
            workspace / "pred_refresh_params.pkl",
        )

    monkeypatch.setattr(rollback_promotion, "snapshot_pred_refresh", fake_snapshot)
    return calls


@pytest.fixture()
def box(tmp_path: Path) -> dict[str, Path]:
    """A state DB with two past promotions: ws_a (older) then ws_b (current)."""
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    db = tmp_path / "state.sqlite"
    store = StateStore(db)
    promote(store, ws_a, topk=20)
    promote(store, ws_b, topk=30)
    return {"db": db, "ws_a": ws_a, "ws_b": ws_b}


def run(box: dict[str, Path], *extra: str) -> int:
    return rollback_promotion.main(["--db", str(box["db"]), "--no-slack", *extra])


class TestSelectTarget:
    def test_default_is_newest_entry_not_currently_promoted(self, box: dict[str, Path]) -> None:
        store = StateStore(box["db"])
        entry = select_target(store.list_promotion_history(), str(box["ws_b"]), None)
        assert entry.workspace_path == str(box["ws_a"])

    def test_named_target_picks_its_newest_entry(self, tmp_path: Path) -> None:
        ws_a, ws_b, ws_c = (tmp_path / name for name in ("ws_a", "ws_b", "ws_c"))
        store = StateStore(tmp_path / "state.sqlite")
        for workspace, topk in ((ws_a, 10), (ws_b, 20), (ws_a, 15), (ws_c, 30)):
            promote(store, workspace, topk=topk)
        entry = select_target(store.list_promotion_history(), str(ws_c), ws_a)
        assert entry.workspace_path == str(ws_a)
        assert entry.config["topk"] == 15  # the LATER ws_a promotion wins

    def test_refuses_target_equal_to_current(self, box: dict[str, Path]) -> None:
        store = StateStore(box["db"])
        with pytest.raises(RollbackError, match="already the promoted strategy"):
            select_target(store.list_promotion_history(), str(box["ws_b"]), box["ws_b"])

    def test_refuses_workspace_never_promoted(self, box: dict[str, Path], tmp_path: Path) -> None:
        store = StateStore(box["db"])
        with pytest.raises(RollbackError, match="never appears in promotion history"):
            select_target(store.list_promotion_history(), str(box["ws_b"]), tmp_path / "ws_x")

    def test_refuses_empty_history(self) -> None:
        with pytest.raises(RollbackError, match="history is empty"):
            select_target([], None, None)

    def test_refuses_when_only_the_current_workspace_exists(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws_only"
        store = StateStore(tmp_path / "state.sqlite")
        promote(store, workspace)
        promote(store, workspace)  # re-promoted twice, still nothing else
        with pytest.raises(RollbackError, match="nothing to roll back to"):
            select_target(store.list_promotion_history(), str(workspace), None)


class TestMain:
    def test_rollback_restores_previous_pointer_and_appends_history(
        self, box: dict[str, Path], snapshot_calls: list[Path]
    ) -> None:
        assert run(box, "--yes") == 0
        store = StateStore(box["db"])
        promoted = store.get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_a"])
        assert promoted.config["topk"] == 20  # ws_a's own recorded config
        assert snapshot_calls == [box["ws_a"].resolve()]

        history = store.list_promotion_history()
        assert len(history) == 3  # two seeds + the rollback row
        newest = history[0]
        assert newest.workspace_path == str(box["ws_a"])
        assert newest.source == "cli"
        assert newest.replaced_workspace == str(box["ws_b"])
        assert newest.gate_verdict is not None
        assert newest.gate_verdict["action"] == "rollback"
        assert newest.gate_verdict["rolled_back_from"] == str(box["ws_b"])

    def test_to_flag_targets_a_named_entry(
        self, box: dict[str, Path], snapshot_calls: list[Path]
    ) -> None:
        assert run(box, "--to", str(box["ws_a"]), "--yes") == 0
        promoted = StateStore(box["db"]).get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_a"])

    def test_refuses_missing_workspace_dir(
        self, box: dict[str, Path], snapshot_calls: list[Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        shutil.rmtree(box["ws_a"])
        assert run(box, "--yes") == 1
        assert "no longer exists" in capsys.readouterr().err
        store = StateStore(box["db"])
        promoted = store.get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_b"])  # pointer untouched
        assert len(store.list_promotion_history()) == 2  # no row appended
        assert snapshot_calls == []

    def test_dry_run_writes_nothing(
        self, box: dict[str, Path], snapshot_calls: list[Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run(box) == 0
        assert "dry-run" in capsys.readouterr().out
        promoted = StateStore(box["db"]).get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_b"])
        assert snapshot_calls == []

    def test_snapshot_failure_aborts_before_pointer_flip(
        self, box: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        def broken(workspace: Path) -> tuple[Path, Path, Path]:
            raise PredRefreshError("no docker log")

        monkeypatch.setattr(rollback_promotion, "snapshot_pred_refresh", broken)
        assert run(box, "--yes") == 1
        assert "snapshot failed" in capsys.readouterr().err
        store = StateStore(box["db"])
        promoted = store.get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_b"])
        assert len(store.list_promotion_history()) == 2

    def test_keep_snapshot_skips_the_re_snapshot(
        self, box: dict[str, Path], snapshot_calls: list[Path]
    ) -> None:
        (box["ws_a"] / "conf_pred_refresh.yaml").write_text("market: pinned_universe\n")
        assert run(box, "--yes", "--keep-snapshot") == 0
        assert snapshot_calls == []
        assert (box["ws_a"] / "conf_pred_refresh.yaml").read_text() == (
            "market: pinned_universe\n"
        )
        promoted = StateStore(box["db"]).get_promoted_strategy()
        assert promoted is not None
        assert promoted.workspace_path == str(box["ws_a"])

    def test_refuses_missing_db(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = rollback_promotion.main(
            ["--db", str(tmp_path / "absent.sqlite"), "--no-slack", "--yes"]
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err
        assert not (tmp_path / "absent.sqlite").exists()  # never creates the DB

    def test_refuses_empty_history(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = tmp_path / "state.sqlite"
        StateStore(db)
        rc = rollback_promotion.main(["--db", str(db), "--no-slack", "--yes"])
        assert rc == 1
        assert "history is empty" in capsys.readouterr().err

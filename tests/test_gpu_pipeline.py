"""Offline tests for the GPU pipeline driver (ops/gpu_pipeline.py)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ops.gpu_pipeline as gpu_pipeline
from ops.gpu_pipeline import (
    ACCOUNT_CONTEXT,
    PipelineOptions,
    SlackThread,
    StatusFile,
    format_final_summary,
    format_loop_digest,
    notion_writeup,
    parse_size_plan,
    post_comparison_chart,
    promoted_slots,
    reportable,
    worker_sh,
)
from orchestrator.config import SlackConfig

CONFIG = SlackConfig(bot_token="xoxb-1", app_token="xapp-1", channel_id="C_PAPER")


class RecordingWebClient:
    """slack_sdk.WebClient stand-in recording posts/uploads (US-017 routing)."""

    instances: list[RecordingWebClient] = []

    def __init__(self, token: str) -> None:
        self.token = token
        self.proxy: str | None = "preset"
        self.posts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        RecordingWebClient.instances.append(self)

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        self.posts.append(kwargs)
        return {"ts": "1755000000.000100"}

    def files_upload_v2(self, **kwargs: Any) -> None:  # noqa: N802
        self.uploads.append(kwargs)


@pytest.fixture
def slack_env(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingWebClient.instances = []
    monkeypatch.setattr("slack_sdk.WebClient", RecordingWebClient)
    monkeypatch.setattr("orchestrator.config.load_slack_config", lambda: CONFIG)


class StubSlack:
    """SlackThread stand-in for chart tests: records posts and uploads."""

    def __init__(self) -> None:
        self.posts: list[str] = []
        self.uploads: list[dict[str, Any]] = []

    def post(self, text: str) -> None:
        self.posts.append(text)

    def upload(self, png: bytes, *, filename: str, title: str) -> None:
        self.uploads.append({"png": png, "filename": filename, "title": title})


class TestSizePlan:
    def test_parses_ordered_pairs(self) -> None:
        plan = parse_size_plan("gpu-4000adax1-20gb:tor1, gpu-6000adax1-48gb:nyc2")
        assert plan == [("gpu-4000adax1-20gb", "tor1"), ("gpu-6000adax1-48gb", "nyc2")]

    def test_rejects_missing_region(self) -> None:
        with pytest.raises(ValueError, match="SIZE:REGION"):
            parse_size_plan("gpu-4000adax1-20gb")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_size_plan(" , ")


class TestFormatting:
    def test_sota_loop_digest(self) -> None:
        digest = format_loop_digest(
            {
                "loop": 5,
                "action": "factor",
                "decision": True,
                "hypothesis": "vol-normalized momentum",
                "workspace": "/x/ws/0bf074144a98499a8ddb31fc3df65fa8",
                "metrics": {"IC": 0.0186, "ARR": 0.7128, "MDD": -0.14},
            }
        )
        assert "Loop 5" in digest
        assert "new SOTA" in digest
        assert "IC 0.0186" in digest
        assert "`0bf07414`" in digest

    def test_pending_loop_digest_degrades(self) -> None:
        digest = format_loop_digest({"loop": 1, "decision": None})
        assert "no verdict" in digest
        assert "no backtest artifacts" in digest
        assert "n/a" in digest

    def test_reportable_requires_verdict(self) -> None:
        assert reportable({"decision": True})
        assert reportable({"decision": False})
        assert not reportable({"decision": None})
        assert not reportable({})

    def test_final_summary_with_candidate(self) -> None:
        status = {
            "loops": [
                {"loop": 0, "decision": True, "workspace": "/x/aa", "metrics": {"IC": 0.01}},
                {"loop": 1, "decision": False, "workspace": "/x/bb", "metrics": {"IC": 0.02}},
            ],
            "candidate_loop": 0,
        }
        text = format_final_summary(status, 0, 1.5, "gpu-6000adax1-48gb")
        assert "exit 0" in text
        assert "2 loops, 1 SOTA" in text
        assert "ops.promote_fetched" in text
        assert "$2.3" in text  # 1.5h * $1.57/hr ≈ $2.35

    def test_final_summary_without_candidate(self) -> None:
        text = format_final_summary({"loops": [], "candidate_loop": None}, 1, 0.5, "unknown-size")
        assert "nothing to promote" in text


class TestSlackFallback:
    def test_disabled_thread_prints_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        thread = SlackThread(enabled=False)
        thread.post("hello world")
        assert "hello world" in capsys.readouterr().err


class TestWorkerSh:
    def test_failure_raises_with_stderr_tail(self) -> None:
        with pytest.raises(RuntimeError, match="unknown subcommand"):
            worker_sh("frobnicate")


class TestSlackThreadChannel:
    def test_posts_to_owning_channel(self, slack_env: None) -> None:
        thread = SlackThread(enabled=True, thread_ts="1.2", channel="C_LIVE")
        thread.post("hello")
        client = RecordingWebClient.instances[0]
        assert client.posts == [{"channel": "C_LIVE", "text": "hello", "thread_ts": "1.2"}]
        assert client.proxy is None  # Slack must never ride the onecli proxy

    def test_defaults_to_paper_channel(self, slack_env: None) -> None:
        SlackThread(enabled=True, thread_ts="1.2").post("hi")
        assert RecordingWebClient.instances[0].posts[0]["channel"] == "C_PAPER"

    def test_upload_goes_to_owning_channel(self, slack_env: None) -> None:
        thread = SlackThread(enabled=True, thread_ts="1.2", channel="C_LIVE")
        thread.upload(b"png", filename="f.png", title="t")
        assert RecordingWebClient.instances[0].uploads[0]["channel"] == "C_LIVE"


class TestStatusFileChannel:
    def test_records_owning_channel(self, tmp_path: Path) -> None:
        StatusFile(tmp_path / "s.json", "1.2", "C_LIVE")
        data = json.loads((tmp_path / "s.json").read_text())
        assert data["thread_ts"] == "1.2"
        assert data["channel"] == "C_LIVE"

    def test_channel_defaults_to_none(self, tmp_path: Path) -> None:
        StatusFile(tmp_path / "s.json", "1.2")
        assert json.loads((tmp_path / "s.json").read_text())["channel"] is None


def make_workspace(tmp_path: Path, name: str, *, with_ret: bool = True) -> Path:
    workspace = tmp_path / name
    workspace.mkdir()
    if with_ret:
        (workspace / "ret.pkl").write_bytes(b"\x00")
    return workspace


def pin_slots(
    monkeypatch: pytest.MonkeyPatch, paper: Path | None, live: Path | None
) -> None:
    """Point the promoted-slot loaders at fixed workspaces (None = empty slot)."""
    from execution.promoted import NoPromotedStrategyError

    def loader(workspace: Path | None):  # noqa: ANN202
        def load(db_path: Path | None = None):  # noqa: ANN202, ARG001
            if workspace is None:
                raise NoPromotedStrategyError("slot is empty")
            return SimpleNamespace(workspace_path=str(workspace))

        return load

    monkeypatch.setattr("execution.promoted.load_promoted_strategy", loader(paper))
    monkeypatch.setattr("execution.promoted.load_promoted_strategy_live", loader(live))


class StubSlackThread(SlackThread):
    """Disabled SlackThread whose posts/uploads are captured for assertions."""

    def __init__(self) -> None:
        super().__init__(enabled=False)
        self.posts: list[str] = []
        self.uploads: list[dict[str, Any]] = []

    def post(self, text: str) -> None:
        self.posts.append(text)

    def upload(self, png: bytes, *, filename: str, title: str) -> None:
        self.uploads.append({"png": png, "filename": filename, "title": title})


@pytest.fixture
def render_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def render(
        candidate_ret: Path,
        promoted_ret: Path | None,
        *,
        candidate_label: str,
        promoted_label: str,
        title: str = "Candidate vs promoted — cumulative return",
    ) -> bytes:
        calls.append(
            {
                "candidate_ret": candidate_ret,
                "promoted_ret": promoted_ret,
                "candidate_label": candidate_label,
                "promoted_label": promoted_label,
                "title": title,
            }
        )
        return b"png"

    monkeypatch.setattr("orchestrator.summary.render_comparison_curve", render)
    return calls


class TestPromotedSlots:
    def test_lists_pinned_slots_paper_first(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        paper = make_workspace(tmp_path, "aaaa1111aaaa")
        live = make_workspace(tmp_path, "bbbb2222bbbb")
        pin_slots(monkeypatch, paper, live)
        assert promoted_slots() == [("paper", paper), ("live", live)]

    def test_empty_slots_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pin_slots(monkeypatch, None, None)
        assert promoted_slots() == []


class TestComparisonChart:
    def test_differing_slots_chart_candidate_against_both(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        render_calls: list[dict[str, Any]],
    ) -> None:
        """US-017: paper and live slots pin different strategies -> two charts,
        each naming the slot it compares against."""
        paper = make_workspace(tmp_path, "aaaa1111aaaa")
        live = make_workspace(tmp_path, "bbbb2222bbbb")
        candidate = make_workspace(tmp_path, "cccc3333cccc")
        pin_slots(monkeypatch, paper, live)
        slack = StubSlackThread()

        post_comparison_chart(slack, str(candidate))

        assert [u["title"] for u in slack.uploads] == [
            "Candidate vs promoted (paper slot)",
            "Candidate vs promoted (live slot)",
        ]
        assert [u["filename"] for u in slack.uploads] == [
            "candidate_vs_promoted_paper.png",
            "candidate_vs_promoted_live.png",
        ]
        assert [c["promoted_label"] for c in render_calls] == [
            "promoted (aaaa1111, paper slot)",
            "promoted (bbbb2222, live slot)",
        ]
        assert [c["promoted_ret"] for c in render_calls] == [
            paper / "ret.pkl",
            live / "ret.pkl",
        ]

    def test_shared_workspace_gets_one_chart_naming_both_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        render_calls: list[dict[str, Any]],
    ) -> None:
        shared = make_workspace(tmp_path, "aaaa1111aaaa")
        candidate = make_workspace(tmp_path, "cccc3333cccc")
        pin_slots(monkeypatch, shared, shared)
        slack = StubSlackThread()

        post_comparison_chart(slack, str(candidate))

        (upload,) = slack.uploads
        assert upload["title"] == "Candidate vs promoted (paper+live slots)"
        assert upload["filename"] == "candidate_vs_promoted_paper_live.png"
        (call,) = render_calls
        assert call["promoted_label"] == "promoted (aaaa1111, paper+live slots)"
        assert "live paper" not in call["promoted_label"]  # old mislabel is gone

    def test_no_promoted_slots_degrades_to_candidate_alone(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        render_calls: list[dict[str, Any]],
    ) -> None:
        candidate = make_workspace(tmp_path, "cccc3333cccc")
        pin_slots(monkeypatch, None, None)
        slack = StubSlackThread()

        post_comparison_chart(slack, str(candidate))

        (upload,) = slack.uploads
        assert upload["filename"] == "candidate_vs_promoted.png"
        assert upload["title"] == "Candidate vs promoted"
        (call,) = render_calls
        assert call["promoted_ret"] is None
        assert call["promoted_label"] == "promoted"

    def test_missing_candidate_ret_skips_all_charts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        render_calls: list[dict[str, Any]],
    ) -> None:
        candidate = make_workspace(tmp_path, "cccc3333cccc", with_ret=False)
        pin_slots(monkeypatch, None, None)
        slack = StubSlackThread()

        post_comparison_chart(slack, str(candidate))

        assert slack.uploads == []
        assert render_calls == []
        assert "no ret.pkl" in slack.posts[0]


class TestMainChannelWiring:
    def test_channel_flag_reaches_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def run_pipeline(options: PipelineOptions) -> int:
            captured["options"] = options
            return 0

        monkeypatch.setattr(gpu_pipeline, "run_pipeline", run_pipeline)
        rc = gpu_pipeline.main(["--channel", "C_LIVE", "--thread-ts", "1.2"])
        assert rc == 0
        assert captured["options"].channel == "C_LIVE"
        assert captured["options"].thread_ts == "1.2"

    def test_channel_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            gpu_pipeline, "run_pipeline", lambda options: captured.update(options=options) or 0
        )
        gpu_pipeline.main([])
        assert captured["options"].channel is None


class TestNotionWriteupContext:
    def test_context_states_the_account_context(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """US-017: the write-up receives the slot semantics instead of the old
        'trades a paper account' assertion."""
        monkeypatch.setenv("HOME", str(tmp_path))
        recorded: dict[str, Any] = {}

        def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            recorded["cmd"] = list(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="https://notion.so/x\n", stderr="")

        monkeypatch.setattr(gpu_pipeline.subprocess, "run", run)
        url = notion_writeup(PipelineOptions(), {"loops": []}, {"loop": 1})

        assert url == "https://notion.so/x"
        context_path = tmp_path / "rdq-runs" / "gpu_worker" / "notion_context.json"
        context = json.loads(context_path.read_text())
        assert context["account_context"] == ACCOUNT_CONTEXT
        assert "real-money account" in context["account_context"]
        assert "paper" in context["account_context"]

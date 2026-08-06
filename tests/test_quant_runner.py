"""Offline tests for the directive-seeding fin_quant wrapper (research/quant_runner.py)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import research.quant_runner as quant_runner


class StubLoop:
    def __init__(self, setting) -> None:  # noqa: ANN001
        self.setting = setting
        self.plan: dict = {}
        self.ran: dict | None = None

    async def run(self, *, step_n=None, loop_n=None, all_duration=None):  # noqa: ANN001
        self.ran = {"step_n": step_n, "loop_n": loop_n, "all_duration": all_duration}


def install_stub_quant_module(monkeypatch) -> ModuleType:  # noqa: ANN001
    module = ModuleType("rdagent.app.qlib_rd_loop.quant")
    module.__dict__["QUANT_PROP_SETTING"] = SimpleNamespace(name="stub-setting")
    module.__dict__["QuantRDLoop"] = StubLoop
    monkeypatch.setitem(sys.modules, "rdagent.app.qlib_rd_loop.quant", module)
    return module


class TestBuildLoop:
    def test_seeds_user_instruction_into_plan(self, monkeypatch) -> None:
        install_stub_quant_module(monkeypatch)
        loop = quant_runner.build_loop("focus on volume factors")
        assert loop.plan == {"user_instruction": "focus on volume factors"}

    def test_no_instruction_leaves_plan_untouched(self, monkeypatch) -> None:
        install_stub_quant_module(monkeypatch)
        assert quant_runner.build_loop(None).plan == {}


class TestMain:
    def test_passes_budget_through_to_the_loop(self, monkeypatch) -> None:
        install_stub_quant_module(monkeypatch)
        captured: dict = {}

        def fake_build(instruction):  # noqa: ANN001
            loop = StubLoop(None)
            captured["loop"] = loop
            captured["instruction"] = instruction
            return loop

        monkeypatch.setattr(quant_runner, "build_loop", fake_build)
        rc = quant_runner.main(
            ["--loop-n", "10", "--all-duration", "12h", "--user-instruction", "steer me"]
        )
        assert rc == 0
        assert captured["instruction"] == "steer me"
        assert captured["loop"].ran == {"step_n": None, "loop_n": 10, "all_duration": "12h"}

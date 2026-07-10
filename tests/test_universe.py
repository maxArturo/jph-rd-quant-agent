"""US-023: set_universe — proposal validation (min-size warning, all-US
refusal), gap reporting on materialize, template render with a custom market,
and the conversational set_universe/confirm_universe/start_research wiring.

Service tests run against real fixture Qlib stores (reusing the builders from
tests/test_make_universe.py); conversation tests use FakeClient (mocked
Anthropic) + a stubbed UniverseManager — no network anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from orchestrator.conversation import (
    ConversationCore,
    format_universe_proposal,
    format_universe_ready,
)
from orchestrator.llm import ModelRouter
from orchestrator.state import StateStore
from orchestrator.universe import (
    DEFAULT_US_TEMPLATES,
    MARKET_LINE,
    MaterializedUniverse,
    TemplateRenderError,
    UniverseGapError,
    UniverseProposal,
    UniverseRefusalError,
    UniverseService,
    normalize_tickers,
    render_universe_templates,
)
from tests.test_build_store import FakeFmp
from tests.test_conversation import (
    THREAD,
    RecordingSay,
    StubLauncher,
    start_research_script,
)
from tests.test_llm import FakeClient, message, text_block, tool_use_block
from tests.test_make_universe import build_fixture_store, make_bars

STORE_SPECS = {
    "AAPL": (100.0, 1_000_000.0),
    "MSFT": (200.0, 500_000.0),
    "NVDA": (150.0, 800_000.0),
    "SPY": (400.0, 2_000_000.0),
}


def make_service(
    tmp_path: Path, store: Path, min_size: int = 2, fmp: FakeFmp | None = None
) -> UniverseService:
    return UniverseService(
        store=store,
        factor_source_root=tmp_path / "factor_source",
        templates_root=tmp_path / "templates",
        min_size=min_size,
        fmp_client=fmp,
    )


def write_us_liquid(store: Path, symbols: list[str]) -> None:
    """Fixture us_liquid instruments file with spans copied from all.txt."""
    spans = {
        line.split("\t")[0]: line
        for line in (store / "instruments" / "all.txt").read_text().splitlines()
        if line.strip()
    }
    (store / "instruments" / "us_liquid.txt").write_text(
        "".join(spans[s] + "\n" for s in symbols)
    )


# ---------------------------------------------------------------------------
# propose(): normalization, warnings, refusals


def test_propose_normalizes_and_dedups(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    proposal = make_service(tmp_path, store).propose(
        " AI_Semis ", ["nvda", " amd ", "NVDA", ""]
    )
    assert proposal.name == "ai_semis"
    assert proposal.tickers == ("NVDA", "AMD")  # upper-cased, deduped, order kept


def test_propose_warns_below_min_size_suggesting_peer_padding(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store, min_size=3)
    proposal = service.propose("ai_pair", ["NVDA", "AAPL"])
    assert len(proposal.warnings) == 1
    warning = proposal.warnings[0].lower()
    assert "2 tickers" in warning and "3" in warning
    assert "padding" in warning and "peers" in warning

    # configurable: at/above the threshold there is no warning
    assert service.propose("ai_trio", ["NVDA", "AAPL", "MSFT"]).warnings == ()


def test_propose_warns_on_store_gaps_promising_backfill(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store)
    proposal = service.propose("ai_semis", ["NVDA", "AMD", "AVGO"])
    gap_warnings = [w for w in proposal.warnings if "backfill" in w]
    assert len(gap_warnings) == 1
    assert "2 ticker(s)" in gap_warnings[0]
    assert "AMD" in gap_warnings[0] and "AVGO" in gap_warnings[0]
    assert "FMP" in gap_warnings[0]


def test_propose_refuses_builtin_and_reserved_names(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store)
    for name in ("us_liquid", "sp500", "all", "US_Liquid"):
        with pytest.raises(UniverseRefusalError, match="built-in|reserved"):
            service.propose(name, ["NVDA", "AMD"])


def test_propose_refuses_invalid_names_and_tickers(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store)
    with pytest.raises(UniverseRefusalError, match="invalid universe name"):
        service.propose("9lives", ["NVDA"])
    with pytest.raises(UniverseRefusalError, match="invalid ticker"):
        service.propose("ok_name", ["NV DA"])
    with pytest.raises(UniverseRefusalError, match="no tickers"):
        service.propose("ok_name", ["", "  "])


def test_propose_refuses_all_us_pointing_at_us_liquid(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    write_us_liquid(store, ["AAPL", "MSFT", "NVDA"])
    service = make_service(tmp_path, store)
    # Superset of the us_liquid default == an all-US universe in disguise.
    with pytest.raises(UniverseRefusalError, match="us_liquid"):
        service.propose("megacaps", ["AAPL", "MSFT", "NVDA", "TSLA"])
    # A genuine subset is fine.
    assert service.propose("pair", ["AAPL", "NVDA"]).tickers == ("AAPL", "NVDA")


def test_propose_refuses_whole_store_coverage_without_us_liquid_file(
    tmp_path: Path,
) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store)
    with pytest.raises(UniverseRefusalError, match="all-US"):
        service.propose("everything", list(STORE_SPECS))


def test_normalize_tickers_rejects_empty_list() -> None:
    with pytest.raises(UniverseRefusalError):
        normalize_tickers([])


# ---------------------------------------------------------------------------
# materialize(): gap reporting + data work


def test_materialize_reports_fmp_missing_tickers_and_builds_no_universe(
    tmp_path: Path,
) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    fmp = FakeFmp(bars={"FAKE1": (), "FAKE2": ()})
    service = make_service(tmp_path, store, fmp=fmp)
    with pytest.raises(UniverseGapError) as excinfo:
        service.materialize("ai_semis", ["AAPL", "FAKE1", "FAKE2"])
    text = str(excinfo.value)
    assert "FAKE1" in text and "FAKE2" in text
    assert "re-propose" in text  # actionable
    assert not (store / "instruments" / "ai_semis.txt").exists()
    assert not (tmp_path / "factor_source" / "ai_semis").exists()
    assert not (tmp_path / "templates" / "ai_semis").exists()


def test_materialize_backfills_store_gaps_from_fmp(tmp_path: Path) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    fmp = FakeFmp(bars={"AMD": make_bars("AMD", close=120.0, volume=900.0)})
    service = make_service(tmp_path, store, fmp=fmp)
    result = service.materialize("ai_semis", ["NVDA", "AAPL", "AMD"])

    # The new ticker landed in the store and in the universe file.
    all_symbols = {
        line.split("\t")[0]
        for line in (store / "instruments" / "all.txt").read_text().splitlines()
        if line.strip()
    }
    assert "AMD" in all_symbols
    rows = result.instruments_path.read_text().splitlines()
    assert [r.split("\t")[0] for r in rows] == ["AAPL", "AMD", "NVDA"]
    assert (result.factor_source / "data_folder" / "daily_pv.h5").exists()

    # Idempotent: a re-materialize finds no gaps and fetches nothing more.
    fetched_before = list(fmp.fetched)
    service.materialize("ai_semis", ["NVDA", "AAPL", "AMD"])
    assert fmp.fetched == fetched_before


def test_materialize_writes_instruments_factor_source_and_templates(
    tmp_path: Path,
) -> None:
    store = build_fixture_store(tmp_path, STORE_SPECS)
    service = make_service(tmp_path, store)
    result = service.materialize("ai_pair", ["NVDA", "AAPL"])

    assert result.name == "ai_pair"
    # instruments file: sorted SYMBOL\tstart\tend rows with spans from all.txt
    rows = result.instruments_path.read_text().splitlines()
    assert [r.split("\t")[0] for r in rows] == ["AAPL", "NVDA"]
    all_rows = set((store / "instruments" / "all.txt").read_text().splitlines())
    assert set(rows) <= all_rows

    # factor source: consumable folders for FACTOR_CoSTEER_DATA_FOLDER(_DEBUG)
    assert result.factor_source == tmp_path / "factor_source" / "ai_pair"
    assert (result.factor_source / "data_folder" / "daily_pv.h5").exists()
    assert (result.factor_source / "data_folder_debug" / "daily_pv.h5").exists()

    # template copy rendered with market: ai_pair (checked in detail below)
    assert result.templates_dir == tmp_path / "templates" / "ai_pair"
    assert (result.templates_dir / "factor_template").is_dir()
    assert (result.templates_dir / "model_template").is_dir()


# ---------------------------------------------------------------------------
# template render with custom market (acceptance test)


def test_render_templates_sets_custom_market_in_every_conf(tmp_path: Path) -> None:
    dest = render_universe_templates("ai_semis", tmp_path / "tpl" / "ai_semis")
    conf_files = sorted(dest.glob("*/conf_*.yaml"))
    source_confs = sorted(DEFAULT_US_TEMPLATES.glob("*/conf_*.yaml"))
    assert [p.name for p in conf_files] == [p.name for p in source_confs]
    assert len(conf_files) == 5
    for conf in conf_files:
        text = conf.read_text()
        assert "market: &market ai_semis" in text
        assert "us_liquid" not in text
        assert "benchmark: &benchmark SPY" in text  # benchmark untouched


def test_render_templates_copies_non_conf_files_byte_identical(tmp_path: Path) -> None:
    dest = render_universe_templates("ai_semis", tmp_path / "tpl" / "ai_semis")
    for sub in ("factor_template", "model_template"):
        for name in ("read_exp_res.py", "README.md"):
            assert (dest / sub / name).read_bytes() == (
                DEFAULT_US_TEMPLATES / sub / name
            ).read_bytes()


def test_render_templates_refuses_on_marker_drift(tmp_path: Path) -> None:
    source = tmp_path / "bad_templates"
    (source / "factor_template").mkdir(parents=True)
    (source / "factor_template" / "conf_baseline.yaml").write_text("market: csi300\n")
    with pytest.raises(TemplateRenderError, match="lacks the expected line"):
        render_universe_templates("x", tmp_path / "out", source)


def test_render_templates_regenerates_cleanly(tmp_path: Path) -> None:
    dest = tmp_path / "tpl" / "ai_semis"
    render_universe_templates("ai_semis", dest)
    stale = dest / "factor_template" / "stale.yaml"
    stale.write_text("leftover")
    render_universe_templates("ai_semis", dest)
    assert not stale.exists()
    assert MARKET_LINE.replace("us_liquid", "ai_semis") in (
        dest / "factor_template" / "conf_baseline.yaml"
    ).read_text()


# ---------------------------------------------------------------------------
# conversational tools (stubbed UniverseManager, mocked Anthropic + Slack)


class StubUniverseService:
    """Records propose/materialize calls; injectable errors and warnings."""

    def __init__(
        self,
        warnings: tuple[str, ...] = (),
        propose_error: Exception | None = None,
        materialize_error: Exception | None = None,
    ) -> None:
        self.warnings = warnings
        self.propose_error = propose_error
        self.materialize_error = materialize_error
        self.proposals: list[tuple[str, list[str]]] = []
        self.materialized: list[tuple[str, list[str]]] = []

    def propose(self, name: str, tickers: Any) -> UniverseProposal:
        if self.propose_error is not None:
            raise self.propose_error
        self.proposals.append((name, list(tickers)))
        return UniverseProposal(
            name=name.strip().lower(),
            tickers=tuple(str(t).strip().upper() for t in tickers if str(t).strip()),
            warnings=self.warnings,
        )

    def materialize(self, name: str, tickers: Any) -> MaterializedUniverse:
        if self.materialize_error is not None:
            raise self.materialize_error
        self.materialized.append((name, list(tickers)))
        return MaterializedUniverse(
            name=name,
            tickers=tuple(tickers),
            instruments_path=Path(f"/store/instruments/{name}.txt"),
            factor_source=Path(f"/data/factor_source/{name}"),
            templates_dir=Path(f"/data/templates/{name}"),
        )


def make_core(
    tmp_path: Path,
    client: FakeClient,
    service: StubUniverseService,
    launcher: StubLauncher | None = None,
) -> tuple[ConversationCore, StateStore]:
    store = StateStore(db_path=tmp_path / "state.sqlite")
    core = ConversationCore(
        store=store,
        router=ModelRouter(client=client),
        rdagent=launcher if launcher is not None else StubLauncher(),
        universes=service,
    )
    return core, store


def set_universe_script(final_reply: str = "Proposed — please confirm.") -> list[Any]:
    return [
        message(
            "tool_use",
            [
                tool_use_block(
                    "tu_su", "set_universe", {"name": "ai_semis", "tickers": ["nvda", "amd"]}
                )
            ],
        ),
        message("end_turn", [text_block(final_reply)]),
    ]


def confirm_universe_script(final_reply: str = "Universe built.") -> list[Any]:
    return [
        message("tool_use", [tool_use_block("tu_cu", "confirm_universe", {})]),
        message("end_turn", [text_block(final_reply)]),
    ]


def tool_result_of(client: FakeClient) -> dict[str, Any]:
    return client.stream_calls[1]["messages"][2]["content"][0]


def test_set_universe_posts_proposal_and_persists_before_any_data_work(
    tmp_path: Path,
) -> None:
    warning = "only 2 tickers (recommended minimum 30): consider padding with peers"
    service = StubUniverseService(warnings=(warning,))
    client = FakeClient(judgment_messages=set_universe_script())
    core, store = make_core(tmp_path, client, service)
    say = RecordingSay()

    core.handle_message(THREAD, "research NVDA and AMD", say)

    # proposal recorded, nothing materialized (confirmation gates data work)
    assert service.proposals == [("ai_semis", ["nvda", "amd"])]
    assert service.materialized == []
    record = store.get_thread_universe(THREAD)
    assert record is not None
    assert record.status == "proposed"
    assert record.name == "ai_semis"
    assert record.tickers == ("NVDA", "AMD")

    # proposal (incl. the min-size warning) posted in-thread for confirmation
    posted = say.calls[0]["text"]
    assert posted == format_universe_proposal(
        UniverseProposal("ai_semis", ("NVDA", "AMD"), (warning,))
    )
    assert "ai_semis" in posted and "NVDA, AMD" in posted
    assert warning in posted
    assert "Confirm" in posted

    # the tool told the model to wait for explicit confirmation
    result = tool_result_of(client)
    assert result.get("is_error") is None
    assert "confirm" in result["content"].lower()


def test_set_universe_refusal_is_error_and_persists_nothing(tmp_path: Path) -> None:
    service = StubUniverseService(
        propose_error=UniverseRefusalError(
            "this ticker list covers every name in 'us_liquid' — use us_liquid"
        )
    )
    client = FakeClient(judgment_messages=set_universe_script("Use us_liquid instead."))
    core, store = make_core(tmp_path, client, service)
    say = RecordingSay()

    core.handle_message(THREAD, "make a universe of the whole market", say)

    result = tool_result_of(client)
    assert result["is_error"] is True
    assert "us_liquid" in result["content"]
    assert store.get_thread_universe(THREAD) is None
    assert [c["text"] for c in say.calls] == ["Use us_liquid instead."]


def test_set_universe_rejected_when_thread_already_has_a_run(tmp_path: Path) -> None:
    service = StubUniverseService()
    client = FakeClient(judgment_messages=set_universe_script("A run is active."))
    core, store = make_core(tmp_path, client, service)
    store.create_run(THREAD, "/stub-traces/existing", universe="us_liquid")

    core.handle_message(THREAD, "switch universe", RecordingSay())

    result = tool_result_of(client)
    assert result["is_error"] is True
    assert "already has a research run" in result["content"]
    assert service.proposals == []
    assert store.get_thread_universe(THREAD) is None


def test_confirm_universe_materializes_and_flips_status(tmp_path: Path) -> None:
    service = StubUniverseService()
    client = FakeClient(judgment_messages=confirm_universe_script())
    core, store = make_core(tmp_path, client, service)
    store.propose_thread_universe(THREAD, "ai_semis", ["NVDA", "AMD"])
    say = RecordingSay()

    core.handle_message(THREAD, "yes, confirmed", say)

    assert service.materialized == [("ai_semis", ["NVDA", "AMD"])]
    record = store.get_thread_universe(THREAD)
    assert record is not None and record.status == "confirmed"
    assert say.calls[0]["text"] == format_universe_ready(
        MaterializedUniverse(
            name="ai_semis",
            tickers=("NVDA", "AMD"),
            instruments_path=Path("/store/instruments/ai_semis.txt"),
            factor_source=Path("/data/factor_source/ai_semis"),
            templates_dir=Path("/data/templates/ai_semis"),
        )
    )
    assert "market: ai_semis" in say.calls[0]["text"]


def test_confirm_universe_without_proposal_is_error(tmp_path: Path) -> None:
    service = StubUniverseService()
    client = FakeClient(judgment_messages=confirm_universe_script("Nothing to confirm."))
    core, _ = make_core(tmp_path, client, service)

    core.handle_message(THREAD, "confirm", RecordingSay())

    result = tool_result_of(client)
    assert result["is_error"] is True
    assert "set_universe" in result["content"]
    assert service.materialized == []


def test_confirm_universe_gap_error_keeps_proposal_pending(tmp_path: Path) -> None:
    service = StubUniverseService(
        materialize_error=UniverseGapError(
            "2 ticker(s) absent from the store — backfill first: FAKE1 FAKE2"
        )
    )
    client = FakeClient(judgment_messages=confirm_universe_script("Missing tickers."))
    core, store = make_core(tmp_path, client, service)
    store.propose_thread_universe(THREAD, "ai_semis", ["AAPL", "FAKE1", "FAKE2"])

    core.handle_message(THREAD, "confirm", RecordingSay())

    result = tool_result_of(client)
    assert result["is_error"] is True
    assert "FAKE1" in result["content"] and "FAKE2" in result["content"]
    record = store.get_thread_universe(THREAD)
    assert record is not None and record.status == "proposed"  # retryable


def test_start_research_uses_confirmed_universe_and_stores_tickers(
    tmp_path: Path,
) -> None:
    service = StubUniverseService()
    client = FakeClient(judgment_messages=start_research_script())
    launcher = StubLauncher()
    core, store = make_core(tmp_path, client, service, launcher)
    store.create_directive(THREAD, objective="Momentum within AI semis")
    store.propose_thread_universe(THREAD, "ai_semis", ["NVDA", "AMD"])
    store.confirm_thread_universe(THREAD)

    core.handle_message(THREAD, "start the research", RecordingSay())

    assert launcher.started == [
        {"directive": "Momentum within AI semis", "universe": "ai_semis"}
    ]
    run = store.get_run(THREAD)
    assert run is not None
    assert run.universe == "ai_semis"
    assert run.universe_tickers == ("NVDA", "AMD")


def test_start_research_rejected_while_universe_unconfirmed(tmp_path: Path) -> None:
    service = StubUniverseService()
    client = FakeClient(judgment_messages=start_research_script("Confirm it first."))
    launcher = StubLauncher()
    core, store = make_core(tmp_path, client, service, launcher)
    store.create_directive(THREAD, objective="Momentum within AI semis")
    store.propose_thread_universe(THREAD, "ai_semis", ["NVDA", "AMD"])

    core.handle_message(THREAD, "start the research", RecordingSay())

    result = tool_result_of(client)
    assert result["is_error"] is True
    assert "not confirmed" in result["content"]
    assert launcher.started == []
    assert store.get_run(THREAD) is None

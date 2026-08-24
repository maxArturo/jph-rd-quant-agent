"""Tests for data/menu.py: store introspection, curated merge, doc drift (US-061)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.build_store import MARKET_FIELDS, TickerBundle, build_store
from data.menu import (
    CURATED_FIELDS,
    DOC_PATH,
    UNDOCUMENTED_DESCRIPTION,
    MenuError,
    build_menu,
    main,
    menu_json,
    render_doc,
    render_menu,
)
from tests.test_build_store import make_bars


def fixture_store(tmp_path: Path) -> Path:
    """Tiny two-ticker store with a PIT-style universe and every $mkt_* series.

    Carries the same canonical field set as the real store (per-ticker OHLCV +
    factor plus the market broadcast series) so the doc drift test is hermetic.
    """
    store = tmp_path / "us_data"
    bars = make_bars("AAPL")
    build_store(
        [
            TickerBundle("AAPL", bars, (), ()),
            TickerBundle("MSFT", make_bars("MSFT", close=200.0), (), ()),
        ],
        store,
        extra_instruments={
            "tiny_pit": [
                ("AAPL", "2024-01-02", "2024-01-04"),
                ("AAPL", "2024-01-08", "2024-01-08"),
                ("MSFT", "2024-01-02", "2024-01-08"),
            ]
        },
        market_series={
            name: [(bar.date, 10.0 * (i + 1) + j) for j, bar in enumerate(bars)]
            for i, name in enumerate(MARKET_FIELDS)
        },
    )
    return store


def add_mystery_field(store: Path) -> None:
    """Plant a bin file for a field the curated table does not know about."""
    path = store / "features" / "aapl" / "mystery.day.bin"
    np.array([0.0, 1.0, 2.0], dtype="<f").tofile(path)


# ---------------------------------------------------------------------------
# Introspection


def test_build_menu_introspects_store(tmp_path: Path) -> None:
    menu = build_menu(fixture_store(tmp_path))
    assert menu.calendar_start == "2024-01-02"
    assert menu.calendar_end == "2024-01-08"
    # Multi-row PIT symbols dedup to distinct-symbol counts.
    assert menu.universes == {"all": 2, "tiny_pit": 2}
    assert menu.field_names() == tuple(CURATED_FIELDS)
    assert all(field.documented for field in menu.fields)


def test_unknown_store_field_marked_undocumented_not_dropped(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    add_mystery_field(store)
    menu = build_menu(store)
    assert "$mystery" in menu.field_names()
    mystery = next(field for field in menu.fields if field.name == "$mystery")
    assert not mystery.documented
    assert mystery.description == UNDOCUMENTED_DESCRIPTION
    assert mystery.kind == "unknown"
    # Documented fields keep canonical order; unknowns are appended.
    assert menu.field_names() == tuple(CURATED_FIELDS) + ("$mystery",)


def test_build_menu_missing_store_raises(tmp_path: Path) -> None:
    with pytest.raises(MenuError):
        build_menu(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Renderings


def test_menu_json_stable_schema(tmp_path: Path) -> None:
    payload = menu_json(build_menu(fixture_store(tmp_path)))
    assert payload["schema_version"] == 1
    assert payload["date_range"] == {"start": "2024-01-02", "end": "2024-01-08"}
    assert payload["universes"] == {"all": 2, "tiny_pit": 2}
    fields = payload["fields"]
    assert isinstance(fields, list)
    for entry in fields:
        assert set(entry) == {"field", "kind", "description", "pit_note", "documented"}
    assert [entry["field"] for entry in fields] == list(CURATED_FIELDS)


def test_render_menu_lists_fields_and_universes(tmp_path: Path) -> None:
    text = render_menu(build_menu(fixture_store(tmp_path)))
    for name in CURATED_FIELDS:
        assert name in text
    assert "- all: 2" in text
    assert "- tiny_pit: 2" in text
    assert "2024-01-02 -> 2024-01-08" in text


# ---------------------------------------------------------------------------
# Doc drift: the checked-in doc must match the module's schema-only rendering.
# Hermetic — the fixture store carries the same canonical field set as the real
# store, and render_doc deliberately embeds no volatile store facts.


def test_checked_in_doc_matches_module_output(tmp_path: Path) -> None:
    rendered = render_doc(build_menu(fixture_store(tmp_path)))
    assert DOC_PATH.exists(), "run: python -m data.menu --write-doc"
    assert DOC_PATH.read_text() == rendered, (
        "docs/reference/data-menu.md drifted from data/menu.py — "
        "regenerate with: python -m data.menu --write-doc"
    )


def test_render_doc_is_schema_only(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    doc = render_doc(build_menu(store))
    assert "2024-01-02" not in doc  # no calendar dates
    assert "tiny_pit" not in doc  # no universes
    assert str(store) not in doc  # no store path
    for name in CURATED_FIELDS:
        assert f"`{name}`" in doc


def test_render_doc_surfaces_undocumented_marker(tmp_path: Path) -> None:
    store = fixture_store(tmp_path)
    add_mystery_field(store)
    doc = render_doc(build_menu(store))
    assert "$mystery" in doc
    assert UNDOCUMENTED_DESCRIPTION in doc


# ---------------------------------------------------------------------------
# CLI


def test_cli_prints_human_menu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--store", str(fixture_store(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "Qlib store data menu" in out
    assert "$close" in out


def test_cli_json_roundtrips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--store", str(fixture_store(tmp_path)), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert [entry["field"] for entry in payload["fields"]] == list(CURATED_FIELDS)


def test_cli_write_doc(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = fixture_store(tmp_path)
    doc_path = tmp_path / "doc" / "data-menu.md"
    assert main(["--store", str(store), "--write-doc", "--doc-path", str(doc_path)]) == 0
    assert doc_path.read_text() == render_doc(build_menu(store))


def test_cli_missing_store_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--store", str(tmp_path / "nope")]) == 1
    assert "ERROR" in capsys.readouterr().err

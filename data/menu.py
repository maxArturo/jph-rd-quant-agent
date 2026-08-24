"""Data menu: machine-readable inventory of what the Qlib store can express (US-061).

Single source of truth for "what data exists" — the directive-crafting prompt
(orchestrator/prompts.py) and the directive pre-flight check both consume this
module instead of hand-copied field lists.

Introspection (fields, trading-day range, universes) comes from the store on
disk; semantics (descriptions, PIT caveats, market-level vs per-ticker) come
from the curated table below. A store field missing from the curated table is
surfaced with an "undocumented" marker, never dropped — the drift test on
docs/reference/data-menu.md turns that marker into a failing check.

CLI:
  python -m data.menu               human-readable menu (live store facts)
  python -m data.menu --json        machine-readable menu (stable schema)
  python -m data.menu --write-doc   regenerate docs/reference/data-menu.md

The written doc is deliberately schema-only (no calendar dates, no universe
counts): those change with every daily refresh and would make the checked-in
doc drift daily. Run the CLI for live coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.build_store import DEFAULT_STORE_PATH, FREQ

SCHEMA_VERSION = 1

KIND_PER_TICKER = "per-ticker"
KIND_MARKET = "market-level"
KIND_UNKNOWN = "unknown"

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "reference" / "data-menu.md"

UNDOCUMENTED_DESCRIPTION = (
    "UNDOCUMENTED — present in the store but missing from CURATED_FIELDS in "
    "data/menu.py; add a curated entry before building factors on it"
)
UNDOCUMENTED_PIT_NOTE = "unknown — treat as unusable until documented"


class MenuError(RuntimeError):
    """Raised when the store cannot be introspected."""


@dataclass(frozen=True)
class CuratedField:
    kind: str
    description: str
    pit_note: str


# Curated semantics, in canonical display order. Every field the store is
# EXPECTED to carry needs an entry here; new series (e.g. $mkt_*) must be added
# when they land or the menu flags them undocumented and the doc drift test
# fails until the doc is regenerated.
CURATED_FIELDS: dict[str, CuratedField] = {
    "$open": CuratedField(
        KIND_PER_TICKER,
        "Adjusted open price (raw open * $factor).",
        "Backward-adjusted: $factor embeds split/dividend knowledge from after this "
        "date. Returns and ratios are PIT-safe; absolute price levels are not.",
    ),
    "$high": CuratedField(
        KIND_PER_TICKER,
        "Adjusted intraday high (raw high * $factor).",
        "Same backward-adjustment caveat as $open.",
    ),
    "$low": CuratedField(
        KIND_PER_TICKER,
        "Adjusted intraday low (raw low * $factor).",
        "Same backward-adjustment caveat as $open.",
    ),
    "$close": CuratedField(
        KIND_PER_TICKER,
        "Adjusted close price (raw close * $factor).",
        "Same backward-adjustment caveat as $open. Raw close = $close / $factor.",
    ),
    "$volume": CuratedField(
        KIND_PER_TICKER,
        "Share volume divided by $factor (Qlib convention). $close * $volume is the "
        "RAW daily dollar volume — the factors cancel.",
        "Volume is known at that day's close; the $factor scaling shares $open's "
        "backward-adjustment caveat.",
    ),
    "$factor": CuratedField(
        KIND_PER_TICKER,
        "Cumulative backward price-adjustment factor; 1.0 on the store window's last "
        "bar. raw price = stored / $factor, raw volume = stored * $factor.",
        "NOT point-in-time: computed from the full split/dividend history known at "
        "store-build time. Never use $factor itself as a signal.",
    ),
}


@dataclass(frozen=True)
class MenuField:
    name: str
    kind: str
    description: str
    pit_note: str
    documented: bool


@dataclass(frozen=True)
class DataMenu:
    store: str
    calendar_start: str
    calendar_end: str
    universes: dict[str, int]  # instruments file name -> distinct symbol count
    fields: tuple[MenuField, ...]

    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


def _store_field_names(store: Path) -> set[str]:
    """Union of feature bin field names across all instruments, '$'-prefixed."""
    features = store / "features"
    if not features.is_dir():
        raise MenuError(f"store has no features directory at {features}")
    suffix = f".{FREQ}.bin"
    names: set[str] = set()
    for instrument_dir in features.iterdir():
        if not instrument_dir.is_dir():
            continue
        for bin_file in instrument_dir.iterdir():
            if bin_file.name.endswith(suffix):
                names.add("$" + bin_file.name[: -len(suffix)])
    if not names:
        raise MenuError(f"no feature bins found under {features}")
    return names


def _calendar_range(store: Path) -> tuple[str, str]:
    path = store / "calendars" / f"{FREQ}.txt"
    if not path.exists():
        raise MenuError(f"store has no calendar at {path}")
    days = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not days:
        raise MenuError(f"calendar {path} is empty")
    return days[0], days[-1]


def _universes(store: Path) -> dict[str, int]:
    instruments = store / "instruments"
    if not instruments.is_dir():
        raise MenuError(f"store has no instruments directory at {instruments}")
    out: dict[str, int] = {}
    for path in sorted(instruments.glob("*.txt")):
        # PIT universes carry multiple membership-span rows per symbol — dedup.
        symbols = {line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()}
        out[path.stem] = len(symbols)
    if not out:
        raise MenuError(f"no universe files found under {instruments}")
    return out


def _merge_fields(present: set[str]) -> tuple[MenuField, ...]:
    fields: list[MenuField] = []
    for name, curated in CURATED_FIELDS.items():
        if name in present:
            fields.append(
                MenuField(name, curated.kind, curated.description, curated.pit_note, True)
            )
    for name in sorted(present - set(CURATED_FIELDS)):
        fields.append(
            MenuField(name, KIND_UNKNOWN, UNDOCUMENTED_DESCRIPTION, UNDOCUMENTED_PIT_NOTE, False)
        )
    return tuple(fields)


def build_menu(store: Path) -> DataMenu:
    """Introspect the store and merge the curated table. Raises MenuError."""
    store = store.expanduser()
    start, end = _calendar_range(store)
    return DataMenu(
        store=str(store),
        calendar_start=start,
        calendar_end=end,
        universes=_universes(store),
        fields=_merge_fields(_store_field_names(store)),
    )


def menu_json(menu: DataMenu) -> dict[str, object]:
    """Stable machine-readable schema (bump SCHEMA_VERSION on breaking change)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "store": menu.store,
        "date_range": {"start": menu.calendar_start, "end": menu.calendar_end},
        "universes": dict(menu.universes),
        "fields": [
            {
                "field": field.name,
                "kind": field.kind,
                "description": field.description,
                "pit_note": field.pit_note,
                "documented": field.documented,
            }
            for field in menu.fields
        ],
    }


def _field_lines(menu: DataMenu) -> list[str]:
    lines: list[str] = []
    for field in menu.fields:
        lines.append(f"- {field.name} [{field.kind}]: {field.description}")
        lines.append(f"  PIT: {field.pit_note}")
    return lines


def render_menu(menu: DataMenu) -> str:
    """Human-readable menu with live store facts (also injected into prompts)."""
    lines = [
        "Qlib store data menu",
        f"Store: {menu.store}",
        f"Trading days: {menu.calendar_start} -> {menu.calendar_end}",
        "",
        "Universes (distinct symbols):",
    ]
    lines.extend(f"- {name}: {count}" for name, count in menu.universes.items())
    lines.append("")
    lines.append("Fields:")
    lines.extend(_field_lines(menu))
    return "\n".join(lines) + "\n"


def render_doc(menu: DataMenu) -> str:
    """docs/reference/data-menu.md body — schema-only, so it only changes when
    the store's field set or the curated table changes (never on daily refresh)."""
    lines = [
        "# Data menu — Qlib store field reference",
        "",
        "<!-- GENERATED by `python -m data.menu --write-doc`. Do not hand-edit: -->",
        "<!-- a pytest (tests/test_menu.py) fails when this file drifts from the -->",
        "<!-- module's output. Edit CURATED_FIELDS in data/menu.py instead.      -->",
        "",
        "Live coverage (trading-day range, universes) is intentionally omitted —",
        "it changes with every daily refresh. Run `python -m data.menu` for it.",
        "",
        "## Fields",
        "",
        "| Field | Kind | Description | PIT note |",
        "| ----- | ---- | ----------- | -------- |",
    ]
    for field in menu.fields:
        lines.append(
            f"| `{field.name}` | {field.kind} | {field.description} | {field.pit_note} |"
        )
    lines.extend(
        [
            "",
            "## Ground rules",
            "",
            "- A hypothesis may only reference fields listed above; anything else must be",
            "  parked for ingestion first (directive pre-flight enforces this).",
            "- `per-ticker` fields vary across instruments; `market-level` fields carry the",
            "  same value on every instrument for a given date (broadcast series).",
            "- Fields marked UNDOCUMENTED exist in the store but have no curated semantics",
            "  yet — do not build factors on them until they are documented here.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the Qlib store data menu (US-061)."
    )
    parser.add_argument(
        "--store", default=DEFAULT_STORE_PATH, help=f"Qlib store dir (default {DEFAULT_STORE_PATH})"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--write-doc", action="store_true", help=f"regenerate {DOC_PATH} and exit"
    )
    parser.add_argument(
        "--doc-path", default=str(DOC_PATH), help="doc target for --write-doc (for tests)"
    )
    args = parser.parse_args(argv)
    try:
        menu = build_menu(Path(args.store))
    except (MenuError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.write_doc:
        doc_path = Path(args.doc_path)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(render_doc(menu))
        print(f"wrote {doc_path}")
        return 0
    if args.json:
        print(json.dumps(menu_json(menu), indent=2))
        return 0
    print(render_menu(menu), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

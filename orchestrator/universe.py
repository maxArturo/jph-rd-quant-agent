"""Per-run custom universes (US-023).

RD-Agent(Q) ranks cross-sectionally, so a ticker/sector idea must become a
confirmed instruments universe before research starts. The conversational
core drives a two-step flow with this module:

1. ``propose(name, tickers)`` — cheap validation only (no data work):
   normalizes the ticker list, refuses built-in/reserved names and all-US
   proposals (those should use the built-in us_liquid), warns when the
   list is below ``min_size`` (cross-sectional ranking needs breadth), and
   warns which tickers are not in the store yet (they will be backfilled
   from FMP on confirm). The core posts the proposal in-thread for operator
   confirmation.
2. ``materialize(name, tickers)`` — after the operator confirms: backfills
   store gaps from FMP (data/refresh.extend_store; symbols FMP has no data
   for are a hard error listing each one), writes the instruments file
   (data/make_universe), regenerates the factor source h5s
   (data/make_factor_source), and renders a per-universe copy of the US
   workspace templates with ``market: <name>``.

Layout convention (mirrors the us_liquid wiring from US-017):
- instruments file: ``<store>/instruments/<name>.txt``
- factor source:    ``<factor_source_root>/<name>/{data_folder,data_folder_debug}``
- template copy:    ``<templates_root>/<name>/{factor_template,model_template}``

NOTE: rdq-research.service still points FACTOR_CoSTEER_DATA_FOLDER and the
us_quant hooks at us_liquid, so server-spawned runs cannot consume these
artifacts yet — per-run env wiring is future work (docs/decisions.md US-023).
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.build_store import DEFAULT_STORE_PATH, MARKET_ALL
from data.fmp import FmpClient
from data.make_universe import (
    DEFAULT_CONFIG_PATH,
    make_universe,
    read_instrument_spans,
    resolve_config,
)

DEFAULT_MIN_SIZE = 30
DEFAULT_FACTOR_SOURCE_ROOT = Path("~/rdq-data/factor_source")
DEFAULT_TEMPLATES_ROOT = Path("~/rdq-data/templates")
DEFAULT_US_TEMPLATES = Path(__file__).resolve().parent.parent / "research" / "us_templates"

# The exact anchor line every US-016 conf yaml carries; the render replaces it.
MARKET_LINE = "market: &market us_liquid"
TEMPLATE_SUBDIRS = ("factor_template", "model_template")

_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")
# Same rule data/make_universe.py enforces on instruments filenames.
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class UniverseRefusalError(ValueError):
    """The proposal itself is unacceptable (reserved name, all-US, ...)."""


class UniverseGapError(ValueError):
    """Requested tickers could not be sourced (message lists them)."""


class TemplateRenderError(RuntimeError):
    """The US template copies could not be rendered for the universe."""


@dataclass(frozen=True)
class UniverseProposal:
    name: str
    tickers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedUniverse:
    name: str
    tickers: tuple[str, ...]
    instruments_path: Path
    factor_source: Path
    templates_dir: Path


def normalize_tickers(tickers: Sequence[str]) -> list[str]:
    """Upper-cased, stripped, deduplicated (order-preserving) ticker list."""
    seen: dict[str, None] = {}
    for raw in tickers:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if not _TICKER_RE.fullmatch(symbol):
            raise UniverseRefusalError(f"invalid ticker symbol: {raw!r}")
        seen.setdefault(symbol)
    if not seen:
        raise UniverseRefusalError(
            "no tickers provided — propose an explicit ticker list for the universe"
        )
    return list(seen)


def render_universe_templates(
    name: str, dest: Path, source: Path = DEFAULT_US_TEMPLATES
) -> Path:
    """Copy the US workspace templates to ``dest`` with ``market: <name>``.

    conf_*.yaml files get the market anchor line replaced (benchmark stays
    SPY); every other file (read_exp_res.py, README.md) is copied byte-
    identical. Regenerating an existing copy replaces it wholesale.
    """
    if dest.exists():
        shutil.rmtree(dest)
    for sub in TEMPLATE_SUBDIRS:
        src_dir = source / sub
        if not src_dir.is_dir():
            raise TemplateRenderError(f"missing US template folder: {src_dir}")
        out_dir = dest / sub
        out_dir.mkdir(parents=True)
        for file in sorted(src_dir.iterdir()):
            if file.name.startswith("conf_") and file.suffix == ".yaml":
                text = file.read_text()
                if MARKET_LINE not in text:
                    raise TemplateRenderError(
                        f"{file} lacks the expected line {MARKET_LINE!r} — "
                        "research/us_templates drifted; update MARKET_LINE"
                    )
                (out_dir / file.name).write_text(
                    text.replace(MARKET_LINE, f"market: &market {name}")
                )
            else:
                shutil.copy2(file, out_dir / file.name)
    return dest


class UniverseService:
    """Validates, confirms, and materializes per-run custom universes."""

    def __init__(
        self,
        store: Path = Path(DEFAULT_STORE_PATH),
        factor_source_root: Path = DEFAULT_FACTOR_SOURCE_ROOT,
        templates_root: Path = DEFAULT_TEMPLATES_ROOT,
        us_templates: Path = DEFAULT_US_TEMPLATES,
        config_path: Path = DEFAULT_CONFIG_PATH,
        min_size: int = DEFAULT_MIN_SIZE,
        fmp_client: FmpClient | None = None,
    ) -> None:
        self.store = store
        self.factor_source_root = factor_source_root
        self.templates_root = templates_root
        self.us_templates = us_templates
        self.config_path = config_path
        self.min_size = min_size
        # Created lazily on first gap backfill; the proxy injects the FMP key
        # only when the process runs under an identity holding that secret.
        self._fmp_client = fmp_client

    def propose(self, name: str, tickers: Sequence[str]) -> UniverseProposal:
        """Validate a proposal without doing any data work."""
        clean = str(name).strip().lower()
        if not _NAME_RE.fullmatch(clean):
            raise UniverseRefusalError(
                f"invalid universe name {name!r}: use lowercase letters, digits, underscores"
            )
        if clean == MARKET_ALL or resolve_config(clean, self.config_path).builtin:
            raise UniverseRefusalError(
                f"'{clean}' is a built-in/reserved universe — custom universes need a"
                " new name; broad-market runs should just use us_liquid directly"
            )
        symbols = normalize_tickers(tickers)
        self._refuse_all_us(symbols)
        warnings: list[str] = []
        if len(symbols) < self.min_size:
            warnings.append(
                f"only {len(symbols)} tickers (recommended minimum {self.min_size}):"
                " cross-sectional ranking needs breadth — consider padding the list"
                " with liquid sector peers before confirming"
            )
        gaps = self._store_gaps(symbols)
        if gaps:
            warnings.append(
                f"{len(gaps)} ticker(s) not in the data store yet ({', '.join(gaps)}):"
                " confirming will backfill their full daily history from FMP before"
                " building the universe — expect the confirmation step to take a few"
                " minutes"
            )
        return UniverseProposal(name=clean, tickers=tuple(symbols), warnings=tuple(warnings))

    def _store_gaps(self, symbols: Sequence[str]) -> list[str]:
        spans = read_instrument_spans(self.store.expanduser())
        return sorted(s for s in symbols if s not in spans)

    def _refuse_all_us(self, symbols: Sequence[str]) -> None:
        """Refuse proposals that cover the whole store / the us_liquid default."""
        store = self.store.expanduser()
        proposed = set(symbols)
        for reference in ("us_liquid", MARKET_ALL):
            path = store / "instruments" / f"{reference}.txt"
            if not path.exists():
                continue
            ref_symbols = {
                line.split("\t")[0]
                for line in path.read_text().splitlines()
                if line.strip()
            }
            if ref_symbols and proposed >= ref_symbols:
                raise UniverseRefusalError(
                    f"this ticker list covers every name in '{reference}' — that is an"
                    " all-US universe, not a custom one; use the built-in us_liquid"
                    " universe instead (start_research defaults to it)"
                )

    def materialize(self, name: str, tickers: Sequence[str]) -> MaterializedUniverse:
        """Do the confirmed universe's data work; raises on any gap/failure."""
        proposal = self.propose(name, tickers)  # revalidate stored values
        store = self.store.expanduser()
        gaps = self._store_gaps(proposal.tickers)
        if gaps:
            # Lazy import: refresh pulls in numpy (multi-second import).
            from data.refresh import extend_store

            if self._fmp_client is None:
                self._fmp_client = FmpClient()
            extended = extend_store(store, self._fmp_client, gaps)
            problems: list[str] = []
            if extended.missing:
                problems.append(
                    f"no daily price data for {' '.join(extended.missing)}"
                    " (delisted, renamed, or non-US symbols are the usual causes)"
                )
            if extended.gapped:
                problems.append(
                    f"unusable gapped price history for {' '.join(extended.gapped)}"
                    " (mid-series holes, e.g. a listing that moved venues)"
                )
            if problems:
                raise UniverseGapError(
                    f"FMP has {'; '.join(problems)} — re-propose the universe"
                    " without these tickers"
                )
        instruments_path = make_universe(
            name=proposal.name,
            store=store,
            tickers=",".join(proposal.tickers),
            config_path=self.config_path,
        )
        # Lazy import: make_factor_source pulls in pandas (multi-second import).
        from data.make_factor_source import make_factor_source

        factor_source = self.factor_source_root.expanduser() / proposal.name
        make_factor_source(universe=proposal.name, store=store, output=factor_source)
        templates_dir = render_universe_templates(
            proposal.name,
            self.templates_root.expanduser() / proposal.name,
            self.us_templates,
        )
        return MaterializedUniverse(
            name=proposal.name,
            tickers=proposal.tickers,
            instruments_path=instruments_path,
            factor_source=factor_source,
            templates_dir=templates_dir,
        )

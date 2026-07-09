"""Trading circuit breaker for the paper rebalancer.

Three independent trips, checked in a fixed order; the first one hit is
returned as a typed :class:`BreakerTrip` (``None`` means trading may
proceed):

1. ``halt_file`` — operator kill switch. If the file exists, trading halts
   unconditionally. US-038's ``halt_trading``/``resume_trading`` Slack tools
   write/remove it via :meth:`Breaker.halt`/:meth:`Breaker.clear_halt`; the
   file's text content (if any) is surfaced in the trip message. Per US-034,
   a halt-file trip is the one abort that exits 0 ("halted" is a deliberate
   state, not an error) — distinguish it by ``trip.reason``.
2. ``max_daily_notional_usd`` — the day's already-traded notional (caller
   counts it from a fresh orders/fills snapshot) must not exceed the cap.
3. ``max_drawdown_pct`` — equity must not sit more than the configured
   percent below the persisted high-water mark.

Thresholds live in ``execution/breaker.paper.json`` (both keys required,
unknown keys refused — same strictness as ``limits.paper.json``). Committed
defaults are sized for the $100k paper account: daily notional 200000 allows
one full both-sides turnover of the book; drawdown 20 halts after a 20% loss
from the peak.

The high-water mark is file-backed JSON (``{"high_water_mark": <float>}``)
so it survives restarts. It is raised — never lowered — and only on a CLEAN
pass: a tripped check leaves the file untouched. A corrupt or unreadable
high-water-mark file raises :class:`BreakerStateError` instead of silently
resetting (a reset would disarm the drawdown kill switch).

Boundary semantics match the order gate: a value exactly AT a limit passes;
strictly over trips. Trip messages start with the violated config key (or
``halt_file``).

Like the order gate, this module does no HTTP: the caller passes fresh
``account.equity`` and the day's traded notional in.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "breaker.paper.json"

DEFAULT_STATE_DIR = Path.home() / "rdq-data" / "breaker"
DEFAULT_HALT_FILE = DEFAULT_STATE_DIR / "halt"
DEFAULT_HWM_FILE = DEFAULT_STATE_DIR / "high_water_mark.json"

_CONFIG_KEYS = ("max_daily_notional_usd", "max_drawdown_pct")
_HWM_KEY = "high_water_mark"


class BreakerError(RuntimeError):
    """Base error for breaker configuration/state problems."""


class BreakerConfigError(BreakerError):
    """execution/breaker.paper.json is missing, malformed, or incomplete."""


class BreakerStateError(BreakerError):
    """The persisted high-water-mark file is corrupt or unreadable."""


class BreakerReason(Enum):
    """Which check tripped. Values match the trip-message prefix."""

    HALT_FILE = "halt_file"
    DAILY_NOTIONAL = "max_daily_notional_usd"
    DRAWDOWN = "max_drawdown_pct"


@dataclass(frozen=True)
class BreakerTrip:
    reason: BreakerReason
    message: str


@dataclass(frozen=True)
class BreakerConfig:
    max_daily_notional_usd: float
    max_drawdown_pct: float


def load_breaker_config(path: Path | str = CONFIG_PATH) -> BreakerConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise BreakerConfigError(f"breaker config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BreakerConfigError(f"breaker config {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BreakerConfigError(f"breaker config {path} must hold a JSON object")
    unknown = sorted(set(raw) - set(_CONFIG_KEYS))
    if unknown:
        raise BreakerConfigError(f"breaker config {path} has unknown keys: {', '.join(unknown)}")
    missing = sorted(set(_CONFIG_KEYS) - set(raw))
    if missing:
        raise BreakerConfigError(f"breaker config {path} is missing keys: {', '.join(missing)}")
    for key in _CONFIG_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise BreakerConfigError(
                f"breaker config {path}: {key} must be a positive number, got {value!r}"
            )
    drawdown = float(raw["max_drawdown_pct"])
    if drawdown >= 100:
        raise BreakerConfigError(
            f"breaker config {path}: max_drawdown_pct must be below 100, got {drawdown!r}"
        )
    return BreakerConfig(
        max_daily_notional_usd=float(raw["max_daily_notional_usd"]),
        max_drawdown_pct=drawdown,
    )


class Breaker:
    """File-backed circuit breaker; construct once per rebalance run."""

    def __init__(
        self,
        config: BreakerConfig,
        halt_file: Path | str = DEFAULT_HALT_FILE,
        high_water_mark_file: Path | str = DEFAULT_HWM_FILE,
    ) -> None:
        self.config = config
        self.halt_file = Path(halt_file)
        self.high_water_mark_file = Path(high_water_mark_file)

    def check(self, equity: float, day_notional_usd: float) -> BreakerTrip | None:
        """Run all breaker checks against a fresh account snapshot.

        ``equity`` is the account's current equity; ``day_notional_usd`` is
        the notional already traded today. Returns the first trip, or None —
        and on a clean pass raises the persisted high-water mark to
        ``equity`` if it is a new peak.
        """
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError(f"equity must be a positive finite number, got {equity!r}")
        if not math.isfinite(day_notional_usd) or day_notional_usd < 0:
            raise ValueError(
                f"day_notional_usd must be a non-negative finite number, got {day_notional_usd!r}"
            )

        if self.halt_file.exists():
            note = self.halt_note
            detail = f" ({note})" if note else ""
            return BreakerTrip(
                reason=BreakerReason.HALT_FILE,
                message=f"halt_file: {self.halt_file} exists — trading halted by operator{detail}",
            )

        if day_notional_usd > self.config.max_daily_notional_usd:
            return BreakerTrip(
                reason=BreakerReason.DAILY_NOTIONAL,
                message=(
                    f"max_daily_notional_usd: ${day_notional_usd:,.2f} traded today exceeds "
                    f"the ${self.config.max_daily_notional_usd:,.2f} daily cap"
                ),
            )

        high_water_mark = self._read_high_water_mark()
        if high_water_mark is not None:
            drawdown_pct = (high_water_mark - equity) / high_water_mark * 100.0
            if drawdown_pct > self.config.max_drawdown_pct:
                return BreakerTrip(
                    reason=BreakerReason.DRAWDOWN,
                    message=(
                        f"max_drawdown_pct: equity ${equity:,.2f} is {drawdown_pct:.2f}% below "
                        f"the ${high_water_mark:,.2f} high-water mark, over the "
                        f"{self.config.max_drawdown_pct:g}% kill-switch"
                    ),
                )

        if high_water_mark is None or equity > high_water_mark:
            self._write_high_water_mark(equity)
        return None

    def halt(self, note: str = "") -> None:
        """Write the halt file (operator kill switch). Idempotent."""
        self.halt_file.parent.mkdir(parents=True, exist_ok=True)
        self.halt_file.write_text(note.strip() + "\n" if note.strip() else "")

    def clear_halt(self) -> None:
        """Remove the halt file if present."""
        self.halt_file.unlink(missing_ok=True)

    @property
    def halted(self) -> bool:
        return self.halt_file.exists()

    @property
    def halt_note(self) -> str:
        """The halt file's note text ("" when absent, empty, or unreadable)."""
        try:
            return self.halt_file.read_text().strip()
        except OSError:
            return ""

    @property
    def high_water_mark(self) -> float | None:
        """The persisted high-water mark (None before the first clean pass).

        Raises :class:`BreakerStateError` on a corrupt file, same as check().
        """
        return self._read_high_water_mark()

    def _read_high_water_mark(self) -> float | None:
        try:
            raw = json.loads(self.high_water_mark_file.read_text())
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            raise BreakerStateError(
                f"high-water-mark file {self.high_water_mark_file} is unreadable ({exc}); "
                "refusing to trade — restore or deliberately delete it to re-seed"
            ) from exc
        value = raw.get(_HWM_KEY) if isinstance(raw, dict) else None
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise BreakerStateError(
                f"high-water-mark file {self.high_water_mark_file} must hold "
                f'{{"{_HWM_KEY}": <positive number>}}, got {raw!r}; '
                "refusing to trade — restore or deliberately delete it to re-seed"
            )
        return float(value)

    def _write_high_water_mark(self, equity: float) -> None:
        self.high_water_mark_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.high_water_mark_file.with_name(self.high_water_mark_file.name + ".tmp")
        tmp.write_text(json.dumps({_HWM_KEY: equity}) + "\n")
        os.replace(tmp, self.high_water_mark_file)

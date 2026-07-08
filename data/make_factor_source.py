"""Factor source data generator: universe -> daily_pv h5 files for RD-Agent's factor coder.

Regenerates the factor coder's source data from the US Qlib store for a named
universe (an instruments/<universe>.txt written by data/make_universe.py).
Refuses to run when that instruments file does not exist.

Output layout under --output (the future FACTOR_CoSTEER_DATA_FOLDER root):
- daily_pv_all.h5    full universe frame (mirrors upstream generate.py naming)
- daily_pv_debug.h5  debug subset: last --debug-days trading days x first
                     --debug-instruments symbols (upstream uses ~2y x 100)
- data_folder/       daily_pv.h5 (= all) + README.md - point
                     FACTOR_CoSTEER_DATA_FOLDER here (US-017)
- data_folder_debug/ daily_pv.h5 (= debug) + README.md - point
                     FACTOR_CoSTEER_DATA_FOLDER_DEBUG here

RD-Agent links every file in the data folder into each factor workspace and
prompts the LLM with descriptions of the DEBUG folder's files, so both folders
must contain the SAME filename (daily_pv.h5) and the README that tells the
model how to read it (pd.read_hdf(..., key="data")).

Frame contract (byte-compatible with upstream generate.py / qlib D.features):
MultiIndex (datetime, instrument), float32 columns
$open/$close/$high/$low/$volume/$factor, rows only inside each instrument's
own data span. Prices are ADJUSTED, volume is raw/factor (store conventions,
see data/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data.build_store import DEFAULT_STORE_PATH, FREQ

ALL_H5 = "daily_pv_all.h5"
DEBUG_H5 = "daily_pv_debug.h5"
CONSUMABLE_H5 = "daily_pv.h5"
HDF_KEY = "data"
# Column order matches upstream rdagent factor_data_template/generate.py.
COLUMNS = ("$open", "$close", "$high", "$low", "$volume", "$factor")
DEFAULT_DEBUG_DAYS = 504  # ~2 trading years, like upstream's 2018-2019 window
DEFAULT_DEBUG_INSTRUMENTS = 100  # upstream caps the debug frame at 100 names

README_TEXT = """\
# How to read files.
For example, if you want to read `daily_pv.h5`
```Python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files**.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price and volume data (US market).                |

# For different data, We have some basic knowledge for them

## Daily price and volume data
Index: MultiIndex (datetime, instrument).
$open: adjusted open price of the stock on that day.
$close: adjusted close price of the stock on that day.
$high: adjusted high price of the stock on that day.
$low: adjusted low price of the stock on that day.
$volume: adjusted volume of the stock on that day (raw volume / factor).
$factor: price adjustment factor of the stock on that day.
"""


class FactorSourceError(RuntimeError):
    """Raised when the factor source data cannot be generated."""


def read_universe_symbols(store: Path, universe: str) -> list[str]:
    """Symbols listed in instruments/<universe>.txt (sorted); refuses if absent."""
    path = store / "instruments" / f"{universe}.txt"
    if not path.exists():
        raise FactorSourceError(
            f"universe '{universe}' has no instruments file at {path} - "
            "create it first with data/make_universe.py"
        )
    symbols = [line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()]
    if not symbols:
        raise FactorSourceError(f"instruments file {path} is empty")
    return sorted(symbols)


def _read_calendar(store: Path) -> pd.DatetimeIndex:
    path = store / "calendars" / f"{FREQ}.txt"
    if not path.exists():
        raise FactorSourceError(f"store has no calendar at {path}")
    days = [line for line in path.read_text().splitlines() if line.strip()]
    if not days:
        raise FactorSourceError(f"calendar {path} is empty")
    return pd.DatetimeIndex(pd.to_datetime(days), name="datetime")


def _read_feature(store: Path, symbol: str, field: str) -> tuple[int, np.ndarray]:
    """(calendar start index, float32 values) for one symbol/field bin file."""
    path = store / "features" / symbol.lower() / f"{field}.{FREQ}.bin"
    if not path.exists():
        raise FactorSourceError(f"missing feature file {path}")
    data = np.fromfile(path, dtype="<f")
    if len(data) < 2:
        raise FactorSourceError(f"feature file {path} has no values")
    return int(data[0]), data[1:]


def load_universe_frame(store: Path, universe: str) -> pd.DataFrame:
    """The universe's OHLCV+factor frame, indexed (datetime, instrument).

    Reads the store bins directly (equivalent to qlib D.features + swaplevel +
    sort_index, without the multi-second qlib import); each instrument
    contributes rows only inside its own [first, last] data span.
    """
    calendar = _read_calendar(store)
    symbols = read_universe_symbols(store, universe)
    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        columns: dict[str, np.ndarray] = {}
        span: tuple[int, int] | None = None
        for column in COLUMNS:
            start, values = _read_feature(store, symbol, column.lstrip("$"))
            if span is None:
                span = (start, len(values))
            elif span != (start, len(values)):
                raise FactorSourceError(f"feature span mismatch for {symbol} in {store}")
            columns[column] = values
        assert span is not None
        start, length = span
        if start + length > len(calendar):
            raise FactorSourceError(f"{symbol} feature span exceeds calendar length in {store}")
        index = pd.MultiIndex.from_product(
            [calendar[start : start + length], [symbol]], names=["datetime", "instrument"]
        )
        parts.append(pd.DataFrame(columns, index=index))
    return pd.concat(parts).sort_index()


def debug_subset(
    frame: pd.DataFrame,
    debug_days: int = DEFAULT_DEBUG_DAYS,
    debug_instruments: int = DEFAULT_DEBUG_INSTRUMENTS,
) -> pd.DataFrame:
    """Small frame for the coder's fast eval loop: recent days x leading symbols."""
    if debug_days < 1 or debug_instruments < 1:
        raise FactorSourceError("debug_days and debug_instruments must be >= 1")
    dates = frame.index.get_level_values("datetime").unique().sort_values()
    instruments = frame.index.get_level_values("instrument").unique().sort_values()
    mask = np.asarray(
        frame.index.get_level_values("datetime").isin(dates[-debug_days:])
    ) & np.asarray(
        frame.index.get_level_values("instrument").isin(instruments[:debug_instruments])
    )
    subset = frame.loc[mask]
    if subset.empty:
        raise FactorSourceError("debug subset is empty; widen debug_days/debug_instruments")
    return subset


def _write_h5(frame: pd.DataFrame, path: Path) -> None:
    path.unlink(missing_ok=True)  # to_hdf appends to existing files; start clean
    frame.to_hdf(path, key=HDF_KEY, mode="w")


def make_factor_source(
    universe: str,
    store: Path,
    output: Path,
    debug_days: int = DEFAULT_DEBUG_DAYS,
    debug_instruments: int = DEFAULT_DEBUG_INSTRUMENTS,
) -> tuple[Path, Path]:
    """Generate the daily_pv h5 files + consumable folders; returns (all, debug) paths."""
    store = store.expanduser()
    output = output.expanduser()
    frame = load_universe_frame(store, universe)
    output.mkdir(parents=True, exist_ok=True)

    all_path = output / ALL_H5
    debug_path = output / DEBUG_H5
    _write_h5(frame, all_path)
    _write_h5(debug_subset(frame, debug_days, debug_instruments), debug_path)

    for folder, source in (("data_folder", all_path), ("data_folder_debug", debug_path)):
        target_dir = output / folder
        target_dir.mkdir(exist_ok=True)
        target = target_dir / CONSUMABLE_H5
        target.unlink(missing_ok=True)
        shutil.copy(source, target)
        (target_dir / "README.md").write_text(README_TEXT)
    return all_path, debug_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RD-Agent factor source h5 files (daily_pv) for a universe."
    )
    parser.add_argument(
        "--universe", required=True, help="universe name (instruments/<name>.txt must exist)"
    )
    parser.add_argument("--output", required=True, help="target folder for the h5 files")
    parser.add_argument(
        "--store", default=DEFAULT_STORE_PATH, help=f"Qlib store dir (default {DEFAULT_STORE_PATH})"
    )
    parser.add_argument(
        "--debug-days",
        type=int,
        default=DEFAULT_DEBUG_DAYS,
        help=f"trading days kept in the debug h5 (default {DEFAULT_DEBUG_DAYS})",
    )
    parser.add_argument(
        "--debug-instruments",
        type=int,
        default=DEFAULT_DEBUG_INSTRUMENTS,
        help=f"instruments kept in the debug h5 (default {DEFAULT_DEBUG_INSTRUMENTS})",
    )
    args = parser.parse_args(argv)
    try:
        all_path, debug_path = make_factor_source(
            universe=args.universe,
            store=Path(args.store),
            output=Path(args.output),
            debug_days=args.debug_days,
            debug_instruments=args.debug_instruments,
        )
    except (FactorSourceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"factor source for universe '{args.universe}' written: {all_path} and {debug_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

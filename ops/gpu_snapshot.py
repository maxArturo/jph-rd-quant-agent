"""GPU base-snapshot bookkeeping: worker-inputs hash, selection, pruning (US-022).

The ~20-minute worker bootstrap (16.5GB docker image ship + venv build) is
baked into a DO image named ``rdq-gpu-base-<hash>-<YYYYmmdd-HHMM>``, where
``<hash>`` is a short digest over every worker-affecting input:

- research/PINNED_COMMIT (the rdagent version the venv installs)
- ops/gpu_worker/gpu_worker.sh (the bootstrap procedure itself)
- the Makefile's venv/install targets (what ``make venv`` builds)
- a STRUCTURAL store marker (feature field names + calendar files — the
  store's data content changes daily and rsyncs on every bootstrap anyway,
  so it must never force a rebake; only layout changes matter)
- the market-series manifest (ordered $mkt_* series names + a schema version
  string, US-068) — snapshots baked before a substrate expansion must never
  be selected for runs that expect the new fields

At launch ops/gpu_pipeline.py selects the newest image whose hash AND region
both match (snapshots are regional — a size-plan fallback into another region
must not boot an image that isn't there): match = boot from snapshot; no
match = full bootstrap this run, then bake a fresh hash-tagged image at
teardown, pruning superseded base images down to the newest KEEP_SNAPSHOTS.

CLI (ops/gpu_worker/gpu_worker.sh delegates here so selection/prune logic has
exactly one offline-testable implementation):

    python -m ops.gpu_snapshot hash
    python -m ops.gpu_snapshot select --region tor1 [--hash abc123def456]
    python -m ops.gpu_snapshot prune [--keep 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from data.build_store import MARKET_FIELDS

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep in sync with SNAPSHOT_PREFIX in ops/gpu_worker/gpu_worker.sh.
SNAPSHOT_PREFIX = "rdq-gpu-base"
# DO's AI/ML-ready image (same constant as gpu_worker.sh DEFAULT_GPU_IMAGE) —
# the boot image whenever no base snapshot matches.
DEFAULT_GPU_IMAGE = "gpu-h100x1-base"
KEEP_SNAPSHOTS = 2
HASH_LEN = 12

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def qlib_store_path() -> Path:
    return Path(os.environ.get("RDQ_QLIB_STORE", "~/.qlib/qlib_data/us_data")).expanduser()


# Bump when the market-series companion contract changes shape (not just when
# a series is added — the series list is hashed on its own); pre-bump
# snapshots stop matching and rebake on their next run.
MARKET_MANIFEST_SCHEMA_VERSION = "market-series-v1"


def market_series_manifest(
    series: Sequence[str] = MARKET_FIELDS,
    schema_version: str = MARKET_MANIFEST_SCHEMA_VERSION,
) -> str:
    """The market-series manifest hashed into the worker-inputs digest: the
    ordered series names the substrate carries plus a schema version string.
    Any change (new series, contract bump) invalidates existing snapshots, so
    a run expecting $mkt_* fields can never select a pre-expansion image."""
    return json.dumps({"schema_version": schema_version, "series": list(series)})


def _makefile_venv_targets(text: str) -> str:
    """The venv-affecting slice of the Makefile: variable assignments plus the
    venv/install target recipes. Editing e.g. the check target must NOT force
    a rebake, so the digest covers only what ``make venv`` runs."""
    keep: list[str] = []
    in_target = False
    for line in text.splitlines():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*[:?+]?=", line):
            keep.append(line)
            in_target = False
            continue
        target = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if target:
            in_target = target.group(1) in {"venv", "install"}
        if in_target and (target or line.startswith("\t")):
            keep.append(line)
    return "\n".join(keep)


def store_schema_marker(store: Path) -> str:
    """Structural fingerprint of the qlib store — layout, never content.

    The store's data rolls forward daily (and bootstrap rsyncs it fresh every
    provision regardless of boot image), so content must not churn the hash;
    what invalidates a snapshot is a schema change: different feature fields
    or calendar files.
    """
    fields: list[str] = []
    features = store / "features"
    if features.is_dir():
        for symbol in sorted(features.iterdir()):
            if symbol.is_dir():
                fields = sorted(p.name for p in symbol.iterdir())
                break
    calendars_dir = store / "calendars"
    calendars = (
        sorted(p.name for p in calendars_dir.glob("*.txt")) if calendars_dir.is_dir() else []
    )
    return json.dumps({"fields": fields, "calendars": calendars}, sort_keys=True)


def worker_inputs_hash(
    repo_root: Path = REPO_ROOT,
    store: Path | None = None,
    market_manifest: str | None = None,
) -> str:
    """Short digest over every worker-affecting input; embedded in the
    snapshot name so drift is detectable by name alone. Raises on a missing
    input file — an unhashable tree must fail loud, not bake a mislabeled
    image."""
    store = store if store is not None else qlib_store_path()
    manifest = market_manifest if market_manifest is not None else market_series_manifest()
    parts = [
        ("pinned_commit", (repo_root / "research" / "PINNED_COMMIT").read_text().strip()),
        ("install_sh", (repo_root / "research" / "install.sh").read_text()),
        ("gpu_worker_sh", (repo_root / "ops" / "gpu_worker" / "gpu_worker.sh").read_text()),
        ("makefile_venv", _makefile_venv_targets((repo_root / "Makefile").read_text())),
        ("store_schema", store_schema_marker(store)),
        ("market_manifest", manifest),
    ]
    digest = hashlib.sha256()
    for label, payload in parts:
        digest.update(label.encode())
        digest.update(b"\x00")
        digest.update(payload.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:HASH_LEN]


@dataclass(frozen=True)
class BaseImage:
    """One rdq-gpu-base-* DO image, as listed by doctl."""

    id: str
    name: str
    regions: tuple[str, ...]
    created_at: str

    @property
    def inputs_hash(self) -> str | None:
        """The hash embedded in ``rdq-gpu-base-<hash>-<ts>``; None for legacy
        unhashed names (which therefore never match a selection)."""
        pattern = rf"{re.escape(SNAPSHOT_PREFIX)}-([0-9a-f]{{{HASH_LEN}}})-\d{{8}}-\d{{4}}"
        match = re.fullmatch(pattern, self.name)
        return match.group(1) if match else None


def list_base_images(runner: Runner = subprocess.run) -> list[BaseImage]:
    """All rdq-gpu-base-* images on the account; raises when doctl fails."""
    result = runner(
        ["doctl", "compute", "image", "list", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:]
        raise RuntimeError(f"doctl image list failed: {tail[0] if tail else 'no output'}")
    try:
        raw = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"doctl image list returned unparseable JSON ({exc})") from exc
    images: list[BaseImage] = []
    for entry in raw if isinstance(raw, list) else []:
        name = str(entry.get("name") or "")
        if not name.startswith(f"{SNAPSHOT_PREFIX}-"):
            continue
        images.append(
            BaseImage(
                id=str(entry.get("id") or ""),
                name=name,
                regions=tuple(entry.get("regions") or ()),
                created_at=str(entry.get("created_at") or ""),
            )
        )
    return images


def _newest_first(images: Iterable[BaseImage]) -> list[BaseImage]:
    # created_at is ISO-8601 (sorts lexically); name's date suffix tiebreaks.
    return sorted(images, key=lambda image: (image.created_at, image.name), reverse=True)


def select_snapshot(
    images: Iterable[BaseImage], inputs_hash: str, region: str
) -> BaseImage | None:
    """Newest image matching BOTH the worker-inputs hash and the region."""
    matches = [
        image
        for image in images
        if image.inputs_hash == inputs_hash and region in image.regions
    ]
    ordered = _newest_first(matches)
    return ordered[0] if ordered else None


def newest_in_region(images: Iterable[BaseImage], region: str) -> BaseImage | None:
    """Newest base image available in the region regardless of hash — the
    manual-provision fallback (no hash known outside the pipeline)."""
    ordered = _newest_first(image for image in images if region in image.regions)
    return ordered[0] if ordered else None


def prune_snapshots(runner: Runner = subprocess.run, keep: int = KEEP_SNAPSHOTS) -> list[BaseImage]:
    """Delete all but the newest ``keep`` base images (any hash, any region —
    superseded hashes are dead weight; ~$0.06/GiB/mo). Returns what was
    deleted; a failed delete is skipped, the next bake's prune retries."""
    deleted: list[BaseImage] = []
    for image in _newest_first(list_base_images(runner))[keep:]:
        result = runner(
            ["doctl", "compute", "image", "delete", "-f", image.id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            deleted.append(image)
        else:
            print(f"prune: could not delete {image.name} ({image.id})", file=sys.stderr)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash", help="print the worker-inputs hash")
    p_select = sub.add_parser(
        "select", help="print the newest matching base-image id (empty output = no match)"
    )
    p_select.add_argument("--region", required=True)
    p_select.add_argument(
        "--hash",
        dest="inputs_hash",
        default=None,
        help="required worker-inputs hash; omitted = newest in region regardless of hash",
    )
    p_prune = sub.add_parser("prune", help="delete superseded base images")
    p_prune.add_argument("--keep", type=int, default=KEEP_SNAPSHOTS)
    args = parser.parse_args(argv)

    if args.command == "hash":
        print(worker_inputs_hash())
        return 0
    if args.command == "select":
        images = list_base_images()
        if args.inputs_hash:
            image = select_snapshot(images, args.inputs_hash, args.region)
        else:
            image = newest_in_region(images, args.region)
        if image is not None:
            print(image.id)
        return 0
    for image in prune_snapshots(keep=args.keep):
        print(f"deleted {image.name} ({image.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

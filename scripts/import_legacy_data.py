"""Import archived Brittany/Marcio data into the active ``code/data`` layout.

The import is intentionally conservative:

- existing active files are never deleted;
- differing active files are copied to ``data/_pre_import_backup/<timestamp>/``
  before replacement;
- identical files are left alone;
- every action is recorded in ``data/legacy_import_manifest.json``.

Marcio's Chafee-Infante archive contains one headerless ``train_data.csv`` and
no test split. We convert it to the active headered pair format and mirror it
to ``test.csv`` so the unified trainer has a validation loader without
generating synthetic replacement data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
ACTIVE_DATA = CODE_ROOT / "data"
ARCHIVE_ROOT = WORKSPACE_ROOT / "archive"

BRITTANY_DATASETS: tuple[tuple[Path, Path], ...] = (
    (
        ARCHIVE_ROOT / "brittany" / "data" / "coral",
        ACTIVE_DATA / "coral",
    ),
    (
        ARCHIVE_ROOT / "brittany" / "data" / "Leslie_3D_larger_domain_tail_only",
        ACTIVE_DATA / "Leslie_3D_larger_domain_tail_only",
    ),
)

MARCIO_SOURCE = ARCHIVE_ROOT / "marcio" / "scripts" / "train_data.csv"
MARCIO_TARGET_DIR = ACTIVE_DATA / "chafee_infante"
MANIFEST_PATH = ACTIVE_DATA / "legacy_import_manifest.json"


@dataclass(frozen=True)
class ImportEntry:
    source: str
    target: str
    action: str
    source_sha256: str | None = None
    target_sha256: str | None = None
    backup: str | None = None
    bytes: int | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_existing(target: Path, backup_root: Path) -> Path | None:
    if not target.exists():
        return None
    backup = backup_root / target.relative_to(CODE_ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup)
    return backup


def _copy_file(source: Path, target: Path, backup_root: Path, *, dry_run: bool) -> ImportEntry:
    if not source.exists():
        raise FileNotFoundError(source)

    source_hash = _sha256(source)
    if target.exists():
        target_hash = _sha256(target)
        if source_hash == target_hash:
            return ImportEntry(
                source=str(source.relative_to(WORKSPACE_ROOT)),
                target=str(target.relative_to(CODE_ROOT)),
                action="identical",
                source_sha256=source_hash,
                target_sha256=target_hash,
                bytes=source.stat().st_size,
            )
    else:
        target_hash = None

    backup = None if dry_run else _backup_existing(target, backup_root)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return ImportEntry(
        source=str(source.relative_to(WORKSPACE_ROOT)),
        target=str(target.relative_to(CODE_ROOT)),
        action="would_copy" if dry_run else "copied",
        source_sha256=source_hash,
        target_sha256=source_hash if not dry_run else target_hash,
        backup=str(backup.relative_to(CODE_ROOT)) if backup is not None else None,
        bytes=source.stat().st_size,
    )


def _copy_tree(source_dir: Path, target_dir: Path, backup_root: Path, *, dry_run: bool) -> list[ImportEntry]:
    entries: list[ImportEntry] = []
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        if source.suffix != ".csv" and not source.name.endswith("_metadata.json"):
            continue
        target = target_dir / source.relative_to(source_dir)
        entries.append(_copy_file(source, target, backup_root, dry_run=dry_run))
    return entries


def _marcio_header() -> list[str]:
    return [f"x{i}" for i in range(64)] + [f"y{i}" for i in range(64)]


class _HashingTextSink(io.TextIOBase):
    def __init__(self) -> None:
        self.hash = hashlib.sha256()
        self.nbytes = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        data = s.encode()
        self.hash.update(data)
        self.nbytes += len(data)
        return len(s)


def _marcio_converted_digest(source: Path) -> tuple[str, int, int]:
    sink = _HashingTextSink()
    writer = csv.writer(sink)
    row_count = 0
    writer.writerow(_marcio_header())
    with source.open(newline="") as src:
        reader = csv.reader(src)
        for row in reader:
            if len(row) != 128:
                raise ValueError(f"expected 128 columns in {source}, got {len(row)}")
            writer.writerow(row)
            row_count += 1
    return sink.hash.hexdigest(), sink.nbytes, row_count


def _convert_marcio_csv(source: Path, target: Path, backup_root: Path, *, dry_run: bool) -> ImportEntry:
    if not source.exists():
        raise FileNotFoundError(source)

    source_hash = _sha256(source)
    converted_hash, converted_bytes, _row_count = _marcio_converted_digest(source)
    current_hash = _sha256(target) if target.exists() else None
    if current_hash == converted_hash:
        return ImportEntry(
            source=str(source.relative_to(WORKSPACE_ROOT)),
            target=str(target.relative_to(CODE_ROOT)),
            action="identical",
            source_sha256=source_hash,
            target_sha256=current_hash,
            bytes=target.stat().st_size,
        )

    backup = None if dry_run else _backup_existing(target, backup_root)

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(newline="") as src, target.open("w", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            writer.writerow(_marcio_header())
            for row in reader:
                if len(row) != 128:
                    raise ValueError(f"expected 128 columns in {source}, got {len(row)}")
                writer.writerow(row)

    return ImportEntry(
        source=str(source.relative_to(WORKSPACE_ROOT)),
        target=str(target.relative_to(CODE_ROOT)),
        action="would_convert" if dry_run else "converted",
        source_sha256=source_hash,
        target_sha256=converted_hash if not dry_run else current_hash,
        backup=str(backup.relative_to(CODE_ROOT)) if backup is not None else None,
        bytes=converted_bytes,
    )


def _write_json(path: Path, payload: dict, backup_root: Path, *, dry_run: bool) -> ImportEntry:
    text = json.dumps(payload, indent=2) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    current_hash = _sha256(path) if path.exists() else None
    if current_hash == digest:
        return ImportEntry(
            source="generated",
            target=str(path.relative_to(CODE_ROOT)),
            action="identical",
            source_sha256=digest,
            target_sha256=current_hash,
            bytes=path.stat().st_size,
        )

    backup = None if dry_run else _backup_existing(path, backup_root)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    return ImportEntry(
        source="generated",
        target=str(path.relative_to(CODE_ROOT)),
        action="would_write" if dry_run else "wrote",
        source_sha256=digest,
        target_sha256=_sha256(path) if path.exists() and not dry_run else current_hash,
        backup=str(backup.relative_to(CODE_ROOT)) if backup is not None else None,
        bytes=len(text.encode()),
    )


def _marcio_metadata(dataset_name: str, role: str) -> dict:
    return {
        "dataset_name": dataset_name,
        "role": role,
        "source": "archive/marcio/scripts/train_data.csv",
        "source_lineage": "marcio_chafee_infante_spectral",
        "system": "ChafeeInfante",
        "dimension": 64,
        "n_samples": 1000,
        "n_iterations": 30,
        "skip_initial_steps": 0,
        "original_has_header": False,
        "original_has_test_split": False,
        "model_params": {
            "N": 64,
            "alpha": 28.0,
            "tau": 0.1,
            "amplitude": 2.0,
            "decay": 0.5,
            "random_seed": 7206,
        },
    }


def import_legacy_data(*, dry_run: bool = False) -> dict:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_root = ACTIVE_DATA / "_pre_import_backup" / timestamp
    entries: list[ImportEntry] = []

    for source_dir, target_dir in BRITTANY_DATASETS:
        entries.extend(_copy_tree(source_dir, target_dir, backup_root, dry_run=dry_run))

    train_target = MARCIO_TARGET_DIR / "train.csv"
    test_target = MARCIO_TARGET_DIR / "test.csv"
    entries.append(_convert_marcio_csv(MARCIO_SOURCE, train_target, backup_root, dry_run=dry_run))
    entries.append(_convert_marcio_csv(MARCIO_SOURCE, test_target, backup_root, dry_run=dry_run))
    entries.append(
        _write_json(
            MARCIO_TARGET_DIR / "train_metadata.json",
            _marcio_metadata("train", "train"),
            backup_root,
            dry_run=dry_run,
        )
    )
    entries.append(
        _write_json(
            MARCIO_TARGET_DIR / "test_metadata.json",
            _marcio_metadata("test", "test_mirror_of_train"),
            backup_root,
            dry_run=dry_run,
        )
    )

    manifest = {
        "timestamp_utc": timestamp,
        "dry_run": dry_run,
        "backup_root": str(backup_root.relative_to(CODE_ROOT)),
        "entries": [asdict(entry) for entry in entries],
    }
    if not dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report actions without copying files")
    args = parser.parse_args(argv)

    manifest = import_legacy_data(dry_run=args.dry_run)
    json.dump(manifest, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

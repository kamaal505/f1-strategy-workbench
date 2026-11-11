"""Content-addressed Arrow snapshots; mutable caches are never replay inputs."""
from __future__ import annotations
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import MappingProxyType
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
from ..contracts import Snapshot, SnapshotManifest, SnapshotRef, WorkbenchError, FileEntry, Availability
from ..identity import canonical, digest, file_digest, check_id, confined
from .schemas import SCHEMAS


def identity_document(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k not in {"snapshot_id", "acquired_utc"}}


def publish_snapshot(*, root: Path, metadata: dict, tables: dict[str, pa.Table]) -> SnapshotRef:
    catalog = root / "snapshots"
    if root.is_symlink() or catalog.is_symlink():
        raise WorkbenchError("unsafe_path", "Symlink snapshot root forbidden")
    catalog.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=catalog))
    try:
        entries = []
        availability = dict(metadata.get("availability", {}))
        for name, table in sorted(tables.items()):
            member = f"{name}.parquet"
            path = confined(staging, member, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="zstd")
            entries.append(FileEntry(path=member, sha256=file_digest(path), rows=table.num_rows).model_dump(mode="json"))
            availability[name] = Availability(status="available" if table.num_rows else "empty", rows=table.num_rows).model_dump(mode="json")
        for required in ("laps", "track_status", "session_status"):
            if required not in tables or tables[required].num_rows == 0:
                raise WorkbenchError("missing_stream", f"Required stream {required} missing or empty")
        payload = dict(metadata, files=entries, availability=availability)
        # Normalize through the model before computing identity (defaults are semantic).
        payload["snapshot_id"] = "0" * 64
        manifest = SnapshotManifest.model_validate_json(canonical(payload))
        payload = manifest.model_dump(mode="json")
        identifier = digest(identity_document(payload))
        payload["snapshot_id"] = identifier
        (staging / "manifest.json").write_bytes(canonical(payload))
        # Validate the complete private publication before it becomes visible.
        _open_directory(staging, identifier)
        destination = confined(catalog, identifier, must_exist=False)
        if destination.exists():
            open_snapshot(root, identifier)
        else:
            os.replace(staging, destination)
        return SnapshotRef(snapshot_id=identifier, session_id=manifest.session_id, path=destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _open_directory(path: Path, identifier: str) -> Snapshot:
    manifest_path = confined(path, "manifest.json")
    try:
        raw = json.loads(manifest_path.read_bytes())
        manifest = SnapshotManifest.model_validate_json(canonical(raw))
    except (ValueError, ValidationError):
        raise WorkbenchError("corrupt_snapshot", "Invalid snapshot manifest") from None
    if manifest.snapshot_id != identifier or digest(identity_document(manifest.model_dump(mode="json"))) != identifier:
        raise WorkbenchError("corrupt_snapshot", "Snapshot identity mismatch")
    if digest(manifest.event_metadata) != manifest.event_metadata_digest:
        raise WorkbenchError("corrupt_snapshot", "Event metadata mismatch")
    tables = {}
    seen = set()
    for entry in manifest.files:
        if entry.path in seen or not entry.path.endswith(".parquet"):
            raise WorkbenchError("corrupt_snapshot", "Duplicate or invalid member")
        seen.add(entry.path)
        member = confined(path, entry.path)
        if file_digest(member) != entry.sha256:
            raise WorkbenchError("corrupt_snapshot", "Snapshot member checksum mismatch")
        try:
            table = pq.ParquetFile(member).read()
        except (pa.ArrowException, OSError):
            raise WorkbenchError("corrupt_snapshot", "Unreadable Parquet member") from None
        name = entry.path[:-8]
        if table.num_rows != entry.rows:
            raise WorkbenchError("corrupt_snapshot", "Snapshot row count mismatch")
        if name in SCHEMAS and table.schema != SCHEMAS[name]:
            raise WorkbenchError("corrupt_snapshot", "Canonical Arrow schema mismatch")
        tables[name] = table
        availability = manifest.availability.get(name)
        if availability is None or availability.rows != table.num_rows or availability.status != ("available" if table.num_rows else "empty"):
            raise WorkbenchError("corrupt_snapshot", "Stream availability disagrees with member")
    for required in ("laps", "track_status", "session_status"):
        if required not in tables or not tables[required].num_rows:
            raise WorkbenchError("missing_stream", "Required canonical stream unavailable")
    for name, availability in manifest.availability.items():
        if name not in tables and name not in ("car_samples", "position_samples") and availability.status != "unavailable":
            raise WorkbenchError("corrupt_snapshot", "Available stream has no member")
    return Snapshot(SnapshotRef(snapshot_id=identifier, session_id=manifest.session_id, path=path), manifest, MappingProxyType(tables))


def open_snapshot(root: Path, snapshot_id: str) -> Snapshot:
    check_id(snapshot_id)
    path = confined(root, f"snapshots/{snapshot_id}")
    return _open_directory(path, snapshot_id)


def list_snapshots(root: Path) -> list[dict]:
    catalog = root / "snapshots"
    if not catalog.exists():
        return []
    results = []
    for path in sorted(catalog.iterdir()):
        if path.name.startswith("."):
            continue
        snap = open_snapshot(root, path.name)
        results.append({"snapshot_id": snap.ref.snapshot_id, "session_id": snap.ref.session_id, "synthetic": snap.manifest.synthetic})
    return results

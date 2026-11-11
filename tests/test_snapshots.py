from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pytest

from f1_strategy.contracts import IngestRequest, WorkbenchError
from f1_strategy.data.acquire import acquire_session
from f1_strategy.data.normalize import normalize_streams, quality_report, source_table
from f1_strategy.data.query import prefix_table
from f1_strategy.data.snapshots import open_snapshot, publish_snapshot, identity_document
from f1_strategy.identity import canonical, digest
from f1_strategy.synthetic import generate_snapshot


def test_offline_roundtrip_and_content_identity(tmp_path, socket_disabled):
    ref = generate_snapshot(tmp_path, 1)
    snapshot = open_snapshot(tmp_path, ref.snapshot_id)
    assert snapshot.manifest.synthetic
    assert snapshot.table("source/laps").equals(snapshot.table("laps"))
    assert snapshot.manifest.availability["car_samples"].status == "unavailable"
    assert generate_snapshot(tmp_path, 1).snapshot_id == ref.snapshot_id
    metadata = snapshot.manifest.model_dump(mode="python", exclude={"snapshot_id", "files"})
    tables = dict(snapshot.tables)
    changed = tables["laps"].to_pylist()
    changed[0]["lap_time_s"] += 1
    tables["laps"] = pa.Table.from_pylist(changed, schema=tables["laps"].schema)
    new = publish_snapshot(root=tmp_path, metadata=metadata, tables=tables)
    assert new.snapshot_id != ref.snapshot_id
    cutoff = snapshot.table("laps").column("release_ns")[4].as_py()
    prefix = prefix_table(snapshot, "laps", cutoff)
    assert all(value.as_py() <= cutoff for value in prefix.column("release_ns"))


def test_checksum_and_path_traversal_rejected(tmp_path):
    ref = generate_snapshot(tmp_path, 2)
    path = ref.path / "laps.parquet"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(WorkbenchError, match="checksum"):
        open_snapshot(tmp_path, ref.snapshot_id)
    ref = generate_snapshot(tmp_path, 3)
    manifest_path = ref.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["files"][0]["path"] = "../escape.parquet"
    identifier = digest(identity_document(payload))
    payload["snapshot_id"] = identifier
    target = ref.path.parent / identifier
    ref.path.rename(target)
    (target / "manifest.json").write_bytes(canonical(payload))
    with pytest.raises(WorkbenchError) as exc:
        open_snapshot(tmp_path, identifier)
    assert exc.value.code == "unsafe_path"


def test_missing_required_and_symlink_rejected(tmp_path):
    ref = generate_snapshot(tmp_path, 4)
    snapshot = open_snapshot(tmp_path, ref.snapshot_id)
    metadata = snapshot.manifest.model_dump(mode="python", exclude={"snapshot_id", "files"})
    tables = dict(snapshot.tables)
    del tables["track_status"]
    with pytest.raises(WorkbenchError) as exc:
        publish_snapshot(root=tmp_path, metadata=metadata, tables=tables)
    assert exc.value.code == "missing_stream"
    member = ref.path / "laps.parquet"
    target = tmp_path / "external.parquet"
    member.rename(target)
    member.symlink_to(target)
    with pytest.raises(WorkbenchError) as exc:
        open_snapshot(tmp_path, ref.snapshot_id)
    assert exc.value.code == "unsafe_path"


def test_normalization_retains_source_and_independent_pit_events():
    frame = pd.DataFrame({"DriverNumber": ["44", "44"], "Driver": ["HAM", "HAM"], "LapNumber": [1., 2.],
                          "Time": pd.to_timedelta([90, 185], unit="s"), "LapStartTime": pd.to_timedelta([0, 90], unit="s"),
                          "LapTime": pd.to_timedelta([90, 95], unit="s"), "PitInTime": pd.to_timedelta([None, 180], unit="s"),
                          "PitOutTime": pd.to_timedelta([None, 190], unit="s"), "Compound": [None, "MEDIUM"],
                          "TyreLife": [None, 2.], "IsAccurate": [True, None], "Deleted": [False, True]})
    source = source_table(frame)
    assert source.column_names == list(frame.columns)
    assert source.column("Compound").to_pylist() == [None, "MEDIUM"]
    assert source.column("IsAccurate").to_pylist() == [True, None]
    tables = normalize_streams("2023-07-R", pd.Timestamp("2023-01-01", tz="UTC"), {"laps": frame})
    assert tables["laps"].column("compound").to_pylist() == ["UNKNOWN", "MEDIUM"]
    pits = tables["pit_events"].to_pylist()
    assert [(p["kind"], p["release_ns"]) for p in pits] == [("entry", 180_000_000_000), ("exit", 190_000_000_000)]
    assert quality_report(tables)["missing_mandatory_time"]["count"] == 0


def test_acquisition_requires_explicit_operator_permission(tmp_path, monkeypatch, socket_disabled):
    monkeypatch.delenv("F1_SOURCE_ACCESS_CONFIRMED", raising=False)
    monkeypatch.delenv("F1_SNAPSHOT_RETENTION_CONFIRMED", raising=False)
    with pytest.raises(WorkbenchError) as exc:
        acquire_session(IngestRequest(year=2023, round_number=7), root=tmp_path)
    assert exc.value.code == "authorization_required"

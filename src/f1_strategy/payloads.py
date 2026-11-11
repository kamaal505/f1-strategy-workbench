"""Saved analyses are renderable without acquisition, snapshots or model loading."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from .contracts import DecisionResult, RaceState, REPLAY_NOTICE, WorkbenchError
from .identity import canonical, confined, verify_payload


def frame_records(frame: pd.DataFrame) -> list[dict]:
    # Pandas' JSON encoder emits nulls for unknown source values, never NaN literals.
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="ns"))


def prefix_payload(state: RaceState) -> dict:
    return {
        "snapshot_id": state.snapshot_id, "session_id": state.session_id, "synthetic": state.synthetic,
        "driver_number": state.driver_number, "completed_lap": state.completed_lap,
        "cutoff_ns": state.cutoff_ns, "cutoff_utc": state.cutoff_utc.isoformat(),
        "max_feature_release_ns": state.max_feature_release_ns, "scheduled_laps": state.scheduled_laps,
        "replay_notice": REPLAY_NOTICE, "quality_flags": list(state.flags),
        "drivers": {key: dict(asdict(value), used_compounds=sorted(value.used_compounds)) for key, value in sorted(state.drivers.items())},
        "laps": frame_records(state.laps), "weather": frame_records(state.weather),
        "track_status": frame_records(state.track_status), "session_status": frame_records(state.session_status),
        "messages": frame_records(state.messages), "pit_events": frame_records(state.pit_events),
        "unknown_value_policy": "Source nulls are unavailable observations, not zero.",
        "source_lineage": list(state.source_lineage),
    }


def load_analysis(path: Path) -> DecisionResult:
    if path.is_symlink():
        raise WorkbenchError("unsafe_path", "Analysis symlinks are forbidden")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise WorkbenchError("capacity_exceeded", "Saved analysis exceeds 64 MiB")
    try:
        payload = json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        verify_payload(payload)
        return DecisionResult.model_validate_json(canonical(payload))
    except WorkbenchError:
        raise
    except (ValueError, KeyError, TypeError):
        raise WorkbenchError("corrupt_analysis", "Invalid saved decision payload") from None


def save_analysis(result: DecisionResult, *, output_dir: Path) -> Path:
    payload = result.model_dump(mode="json")
    verify_payload(payload)
    directory = confined(output_dir, result.analysis_id, must_exist=False)
    directory.mkdir(parents=True, exist_ok=True)
    destination = confined(directory, "analysis.json", must_exist=False)
    content = canonical(payload)
    if destination.exists():
        existing = load_analysis(destination)
        if existing.payload_checksum != result.payload_checksum:
            raise WorkbenchError("corrupt_analysis", "Existing analysis differs; refusing overwrite")
        return destination
    fd, temporary = tempfile.mkstemp(prefix=".analysis-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
        # Hard-link publication refuses races that would overwrite an existing payload.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if load_analysis(destination).payload_checksum != result.payload_checksum:
                raise WorkbenchError("corrupt_analysis", "Concurrent analysis differs") from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination

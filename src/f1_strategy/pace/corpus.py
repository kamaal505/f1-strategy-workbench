"""Offline schedule corpus and verified, event-time snapshot resolution.

The schedule records membership, not permission to acquire or retain timing data.
Resolution never contacts FastF1 and does not invent snapshot identities.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath

import pandas as pd

from ..contracts import TrainingSpec, WorkbenchError
from ..data.snapshots import open_snapshot


def _asset(name: str):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts) or path.parts[0] != "evaluation":
        raise WorkbenchError("invalid_corpus", "Corpus assets must be under evaluation/")
    return files("f1_strategy").joinpath("assets", *path.parts)


def load_corpus(asset: str = "evaluation/corpus.json") -> dict:
    """Load a bundled corpus and verify each retained upstream source byte-for-byte."""
    try:
        corpus = json.loads(_asset(asset).read_bytes())
        if corpus["schema_version"] != 1:
            raise ValueError("unsupported schema")
        for source in corpus["sources"]:
            actual = hashlib.sha256(_asset(source["asset"]).read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise ValueError("source digest mismatch")
        ids = [event["session_id"] for event in corpus["events"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate session")
        for event in corpus["events"]:
            if event["session_id"] != f'{event["year"]}-{event["round_number"]:02d}-R' or event["partition"] not in ("initialization", "development", "calibration", "holdout"):
                raise ValueError("invalid event")
        return corpus
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise WorkbenchError("invalid_corpus", f"Invalid corpus or source provenance: {exc}") from exc


def _last_release(snapshot):
    latest = None
    for name in ("laps", "pit_events", "weather", "track_status", "session_status", "messages"):
        table = snapshot.tables.get(name)
        if table is None or not table.num_rows:
            continue
        values = table.column("release_ns").to_pandas().dropna()
        if not values.empty:
            value = int(values.max())
            latest = value if latest is None else max(latest, value)
    if latest is None:
        return None
    return pd.Timestamp(snapshot.manifest.stream_origin_utc) + pd.Timedelta(latest, unit="ns")


def resolve_corpus(corpus: dict, *, root: Path, as_of_utc: datetime | None = None) -> list[dict]:
    """Resolve each race to one verified local snapshot, optionally strictly before cutoff.

    As-of concerns event release time, not acquisition time: FastF1 data are
    reconstructed retrospectively and original receipt/correction times are unknown.
    Multiple eligible publications are ambiguous rather than silently selected.
    """
    if as_of_utc is not None and (as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None):
        raise WorkbenchError("invalid_corpus", "As-of cutoff must be timezone-aware")
    cutoff = pd.Timestamp(as_of_utc.astimezone(timezone.utc)) if as_of_utc is not None else None
    catalog = root / "snapshots"
    matches = {}
    if catalog.exists():
        for path in sorted(catalog.iterdir()):
            if path.name.startswith("."):
                continue
            snapshot = open_snapshot(root, path.name)
            manifest = snapshot.manifest
            if manifest.synthetic:
                continue
            latest = _last_release(snapshot)
            matches.setdefault(manifest.session_id, []).append((snapshot, latest))
    result = []
    for event in corpus["events"]:
        row = dict(event, snapshot_id=None)
        candidates = [(snap, latest) for snap, latest in matches.get(event["session_id"], [])
                      if snap.manifest.year == event["year"] and snap.manifest.round_number == event["round_number"]]
        eligible = [(snap, latest) for snap, latest in candidates
                    if latest is not None and (cutoff is None or latest < cutoff)]
        if len(eligible) == 1:
            snapshot, latest = eligible[0]
            row["snapshot_id"] = snapshot.ref.snapshot_id
            row["max_release_utc"] = latest.isoformat()
        elif len(eligible) > 1:
            row["reason"] = "multiple verified snapshots; selection is ambiguous"
        elif candidates:
            row["reason"] = "no verified snapshot with timed releases strictly before cutoff" if cutoff is not None else "snapshot has no timed releases"
        elif not event.get("f1_api_support", True):
            row["reason"] = "FastF1 schedule reports timing API unsupported; no verified snapshot"
        else:
            row["reason"] = "no verified local snapshot"
        result.append(row)
    return result


def resolve_asof_training(asset: str, *, root: Path) -> TrainingSpec:
    """Resolve an unresolved showcase descriptor into an immutable training spec.

    Earlier initialization, development and calibration races form the candidate
    history. The latest 20% of earlier eligible events, at least two, are
    reserved for calibration; holdout races never enter this history. Every candidate
    must resolve unambiguously, rather than silently changing the training set.
    """
    try:
        descriptor = json.loads(_asset(asset).read_bytes())
        if descriptor["schema_version"] != 1 or descriptor["status"] != "unresolved" or descriptor["cutoff"] != "scheduled_race_start_utc":
            raise ValueError("invalid as-of descriptor")
        import math
        corpus = load_corpus(descriptor["corpus_asset"])
        target = next(event for event in corpus["events"] if event["session_id"] == descriptor["target_session_id"])
        cutoff = datetime.fromisoformat(target["race_start_utc"])
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        raise WorkbenchError("invalid_corpus", f"Invalid as-of descriptor: {exc}") from exc
    rows = resolve_corpus(corpus, root=root, as_of_utc=cutoff)
    history = [row for row in rows if row["partition"] != "holdout"
               and row["race_start_utc"] is not None
               and datetime.fromisoformat(row["race_start_utc"]) < cutoff]
    history.sort(key=lambda row: (row["race_start_utc"], row["session_id"]))
    missing = [row["session_id"] for row in history if row["snapshot_id"] is None]
    if missing:
        raise WorkbenchError("insufficient_evidence", "Unresolved earlier snapshots: " + ", ".join(missing))
    count = max(2, math.ceil(len(history) * .2))
    if len(history) - count < 5:
        raise WorkbenchError("insufficient_evidence", "At least five fit events and two calibration events are required")
    fitted, calibration = history[:-count], history[-count:]
    first_cal = open_snapshot(root, calibration[0]["snapshot_id"])
    calibration_start = pd.Timestamp(first_cal.manifest.stream_origin_utc) + pd.Timedelta(first_cal.manifest.race_start_ns, unit="ns")
    if any(pd.Timestamp(row["max_release_utc"]) >= calibration_start for row in fitted):
        raise WorkbenchError("future_artifact", "Fit source releases overlap the calibration period")
    try:
        return TrainingSpec(
            training_snapshot_ids=tuple(row["snapshot_id"] for row in fitted),
            calibration_snapshot_ids=tuple(row["snapshot_id"] for row in calibration),
            data_cutoff_utc=cutoff,
            model_kind=descriptor["model_kind"],
            seed=descriptor["seed"],
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise WorkbenchError("invalid_corpus", f"Invalid as-of training settings: {exc}") from exc

"""Deterministic, explicitly synthetic observations using the acquisition schemas."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from .contracts import TrainingSpec, DEFAULT_GRID
from .data.schemas import table_from_rows
from .data.snapshots import publish_snapshot
from .identity import canonical, digest, lock_digest

DRIVERS = ("44", "1", "14", "16")


def generate_snapshot(root: Path, event: int, *, target: bool = False):
    rng = np.random.default_rng(2026 + event)
    year = 2023 if target else 2022
    session_id = f"SYN-{year}-{event:02d}-R"
    origin = datetime(year, 3, 1, 12, tzinfo=timezone.utc) + timedelta(days=event * 8)
    start = 300_000_000_000
    def common(kind, index, ns):
        return {"session_id": session_id, "source_row_id": f"{session_id}:{kind}:{index}", "event_ns": ns,
                "release_ns": ns, "time_basis": "estimated_event"}
    laps, pits = [], []
    for di, driver in enumerate(DRIVERS):
        clock = start + int(di * 2e9)
        stop = 9 + (event + di) % 7
        initial_age = (event * 3 + di * 4) % 17
        initial_compound = ("MEDIUM", "HARD", "SOFT")[(event + di) % 3]
        next_compound = ("MEDIUM", "HARD", "SOFT")[(event + di + 1) % 3]
        if target:
            initial_compound, next_compound, initial_age, stop = "MEDIUM", "HARD", 4, 17
        for lap in range(1, 25):
            fresh = lap > stop
            compound = next_compound if fresh else initial_compound
            age = lap - stop if fresh else initial_age + lap
            if target and lap <= 3:
                compound, age = "SOFT", initial_age + lap
            elif target and not fresh:
                age = lap - 3  # fresh medium set: target finish stays within prior age support
            slope = {"SOFT": .16, "MEDIUM": .10, "HARD": .07}[compound]
            contrast = {"SOFT": -.7, "MEDIUM": 0., "HARD": .45}[compound]
            warmup = .45 if lap == stop + 2 else .2 if lap == stop + 3 else 0.
            if target and lap in (5, 6):
                warmup = .45 if lap == 5 else .2
            duration = 90 + di * .3 + contrast + slope * max(age - 2, 0) - .06 * (lap - 1) + warmup
            duration += float(rng.normal(0, .10))
            if lap == 7 and di == 2:
                duration += 3.0  # robust fitting retains a legitimate slow outlier
            if lap == stop + 1 or (target and lap == 4):
                duration += 22.0
            end = clock + round(duration * 1e9)
            row = common("lap", f"{driver}:{lap}", end)
            row.update(driver_number=driver, driver_code=f"S{di}", lap_number=lap, lap_start_ns=clock,
                       lap_end_ns=end, lap_time_s=duration, sector1_s=duration / 3, sector2_s=duration / 3,
                       sector3_s=duration / 3, source_stint=(3 if fresh else 2 if lap > 3 else 1) if target else (2 if fresh else 1), compound=compound,
                       tyre_age_laps=age, fresh_tyre=fresh, source_track_status="1", is_accurate=True,
                       fastf1_generated=False, final_deleted=False, final_deleted_reason=None, source_position=di + 1)
            laps.append(row)
            if lap == stop or (target and lap == 3):
                pit_key = f"{driver}:{lap}" if target else driver
                entry = common("pit", f"{pit_key}:entry", end - 4_000_000_000)
                entry.update(driver_number=driver, kind="entry", source_lap_number=lap)
                pits.append(entry)
                exit_row = common("pit", f"{pit_key}:exit", end + 23_000_000_000)
                exit_row.update(driver_number=driver, kind="exit", source_lap_number=lap + 1)
                pits.append(exit_row)
            clock = end
    weather = []
    for i in range(42):
        row = common("weather", i, start + i * 60_000_000_000)
        row.update(air_temp_c=23., track_temp_c=32. + i * .02, humidity_pct=45., pressure_mbar=1010.,
                   rainfall=False, wind_direction_deg=180., wind_speed_ms=1.5)
        weather.append(row)
    status = dict(common("track", 0, start), source_status="1", status="GREEN", message="Synthetic green")
    session_status = dict(common("session", 0, start), status="Started")
    messages = []
    focal_lap6 = next(x for x in laps if x["driver_number"] == "44" and x["lap_number"] == 6)
    for i, (offset, text) in enumerate(((100, "CAR 44 LAP 6 TIME DELETED"), (200, "CAR 44 LAP 6 TIME REINSTATED"))):
        ns = focal_lap6["lap_end_ns"] + offset * 1_000_000_000
        row = common("message", i, ns)
        row.update(time_basis="message_timestamp", utc=origin + timedelta(microseconds=ns / 1000),
                   category="Other", message=text, status=None, flag=None, scope="Driver", sector=None,
                   driver_number="44", lap_number=6)
        messages.append(row)
    tables = {name: table_from_rows(name, rows) for name, rows in {
        "laps": laps, "pit_events": pits, "weather": weather, "track_status": [status],
        "session_status": [session_status], "messages": messages}.items()}
    # These stand in for returned source streams, not raw live feed claims.
    tables.update({f"source/{name}": table for name, table in list(tables.items())})
    event_metadata = {"layout_id": "synthetic-layout", "spec_era_by_compound": {c: f"synthetic:{c}" for c in ("SOFT", "MEDIUM", "HARD")},
                      "rules_provenance": "assumed", "nomination_provenance": "assumed", "race_date": origin.date().isoformat(),
                      "source_reference": "Packaged synthetic specification", "synthetic_truth": {"fuel_gain": .06, "medium_age_slope": .10}}
    metadata = dict(schema_version=1, session_id=session_id, synthetic=True, year=year, round_number=event, session="R",
                    stream_origin_utc=origin, race_start_ns=start, scheduled_laps=24,
                    entries=tuple({"driver_number": d, "driver_code": f"S{i}"} for i, d in enumerate(DRIVERS)),
                    event_metadata=event_metadata, event_metadata_digest=digest(event_metadata), acquired_utc=origin + timedelta(days=1),
                    source_ids=(f"synthetic:v1:{session_id}",), source_endpoints=(), versions={"generator": "1", "schema": "1"},
                    lock_digest=lock_digest(), load_options={"laps": True, "weather": True, "messages": True, "telemetry": False},
                    availability={"car_samples": {"status": "unavailable", "reason": "Not generated", "rows": 0},
                                  "position_samples": {"status": "unavailable", "reason": "Not generated", "rows": 0}}, quality={})
    return publish_snapshot(root=root, metadata=metadata, tables=tables)


def generate_demo(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    refs = [generate_snapshot(root, i) for i in range(1, 8)]
    target = generate_snapshot(root, 8, target=True)
    training = TrainingSpec(training_snapshot_ids=tuple(r.snapshot_id for r in refs[:5]),
                            calibration_snapshot_ids=tuple(r.snapshot_id for r in refs[5:]),
                            data_cutoff_utc=datetime(2023, 1, 1, tzinfo=timezone.utc), model_kind="hierarchical")
    training_path = root / "training.json"
    training_path.write_bytes(canonical(training))
    scenario = {"schema_version": 1, "name": "Synthetic assumed green", "inventory_by_driver": {},
                "inventory_provenance": "assumed: one unused set per dry compound per driver", "dry_condition_assumption": "dry",
                "rules_override": {"source_reference": "Synthetic conditional rules", "provenance": "assumed"}}
    scenario_path = root / "scenario.json"
    scenario_path.write_bytes(canonical(scenario))
    from .pace.hierarchical import fit_pace
    model_ref = fit_pace(training, root=root)
    index = {"target_snapshot_id": target.snapshot_id, "training_spec_path": str(training_path.resolve()),
             "scenario_path": str(scenario_path.resolve()), "driver_number": "44", "completed_lap": 10,
             "expected_direction": "stay_out", "synthetic": True, "training_snapshot_ids": [r.snapshot_id for r in refs[:5]],
             "calibration_snapshot_ids": [r.snapshot_id for r in refs[5:]], "model_id": model_ref.model_id}
    (root / "index.json").write_bytes(canonical(index))
    return {"index_path": str((root / "index.json").resolve())}

"""Event-time reconstructed, strictly prefix-bounded race state."""
from __future__ import annotations

import re
import pandas as pd
import pyarrow.compute as pc

from .contracts import DriverState, RaceState, Snapshot, WorkbenchError
from .data.schemas import SCHEMAS


def _prefix(snapshot: Snapshot, name: str, cutoff: int) -> pd.DataFrame:
    table = snapshot.tables.get(name)
    if table is None:
        return pd.DataFrame(columns=SCHEMAS[name].names)
    table = table.filter(pc.fill_null(pc.less_equal(table["release_ns"], cutoff), False))
    frame = table.to_pandas()
    return frame.sort_values(["release_ns", "source_row_id"], kind="stable", na_position="last").reset_index(drop=True)


def _deletions(laps: pd.DataFrame, messages: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    states: dict[tuple[str, int], bool] = {}
    unresolved = False
    for row in messages.itertuples(index=False):
        text = str(row.message or "").upper()
        if not re.search(r"\b(DELETED|REINSTATED)\b", text):
            continue
        car = re.findall(r"\bCAR\s+(\d+)\b", text)
        lap = re.findall(r"\bLAP\s+(\d+)\b", text)
        # Explicit structured identifiers may supplement a uniquely parsed message.
        driver = row.driver_number if pd.notna(row.driver_number) else (car[0] if len(car) == 1 else None)
        number = row.lap_number if pd.notna(row.lap_number) else (int(lap[0]) if len(lap) == 1 else None)
        if len(set(car)) > 1 or len(set(lap)) > 1 or driver is None or number is None:
            unresolved = True
            continue
        if car and str(driver) != car[0] or lap and int(number) != int(lap[0]):
            unresolved = True
            continue
        states[(str(driver), int(number))] = "REINSTATED" not in text
    laps = laps.copy()
    laps["deleted_as_of"] = [states.get((str(d), int(n)), False) if pd.notna(n) else False
                             for d, n in zip(laps.driver_number, laps.lap_number)]
    return laps, unresolved


def build_race_state(snapshot: Snapshot, *, driver_number: str, completed_lap: int) -> RaceState:
    """Reconstruct what the released rows support just after the focal crossing."""
    if not isinstance(driver_number, str) or not driver_number or isinstance(completed_lap, bool) or not isinstance(completed_lap, int) or completed_lap < 1:
        raise WorkbenchError("invalid_input", "A driver number and positive completed lap are required")
    manifest = snapshot.manifest
    table = snapshot.table("laps")
    selected = table.filter(pc.fill_null(pc.and_(pc.equal(table["driver_number"], driver_number),
                                                   pc.equal(table["lap_number"], completed_lap)), False)).to_pylist()
    if len(selected) != 1 or selected[0]["lap_end_ns"] is None or selected[0]["release_ns"] != selected[0]["lap_end_ns"]:
        raise WorkbenchError("missing_focal_completion", "The selected driver has no uniquely timed completion")
    focal = selected[0]
    if focal["fastf1_generated"] is True:
        raise WorkbenchError("generated_focal_completion", "The focal completion is generated timing")
    cutoff = int(focal["lap_end_ns"])
    if cutoff <= manifest.race_start_ns:
        raise WorkbenchError("race_not_started", "The selected crossing precedes race start")
    frames = {name: _prefix(snapshot, name, cutoff) for name in
              ("laps", "weather", "track_status", "session_status", "messages", "pit_events")}
    statuses = frames["session_status"]
    if statuses.empty or not statuses.status.fillna("").astype(str).str.upper().eq("STARTED").any():
        raise WorkbenchError("race_not_started", "No released race-start status precedes the selected crossing")
    laps = frames["laps"]
    # Even malformed release timestamps cannot reveal a completion before its end.
    laps = laps[laps.lap_end_ns.notna() & (laps.lap_end_ns <= cutoff)].copy()
    # These final audit fields and sector completion times are not contemporaneous features.
    laps = laps.drop(columns=["final_deleted", "final_deleted_reason", "source_position", "sector1_s", "sector2_s", "sector3_s"], errors="ignore")
    laps, unresolved = _deletions(laps, frames["messages"])
    frames["laps"] = laps
    flags = []
    if unresolved:
        flags.append("unresolved_deletion_message")
    drivers = {}
    for number, group in laps.groupby("driver_number", sort=True):
        if pd.isna(number):
            continue
        group = group[group.lap_number.notna()].sort_values(["lap_end_ns", "source_row_id"], kind="stable")
        if group.empty:
            continue
        last = group.iloc[-1]
        number = str(number)
        events = frames["pit_events"]
        events = events[events.driver_number == number]
        in_pit = False
        entry = None
        last_exit = None
        for event in events.itertuples(index=False):
            if event.kind == "entry":
                in_pit, entry = True, int(event.release_ns)
            elif event.kind == "exit":
                in_pit, entry, last_exit = False, None, int(event.release_ns)
        valid = group[(group.lap_time_s.notna()) & (group.lap_time_s > 0) & ~group.deleted_as_of]
        typical = float(valid.lap_time_s.tail(5).median()) if not valid.empty else None
        elapsed = max(0., (cutoff - int(last.lap_end_ns)) / 1e9)
        stale = typical is None or elapsed > 1.5 * typical
        progress = float(last.lap_number) + (min(elapsed / typical, 0.999999) if typical and not stale and not in_pit else 0.)
        compound = str(last.compound).upper() if pd.notna(last.compound) else None
        age = int(last.tyre_age_laps) if pd.notna(last.tyre_age_laps) else None
        driver_flags = []
        if last_exit is not None and last_exit > int(last.lap_end_ns):
            compound, age = None, None
            driver_flags.append("mounted_tyre_unobserved_after_exit")
        if compound not in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
            compound = None
        if age is not None and age < 0:
            age = None
        if compound is None or age is None:
            driver_flags.append("unknown_mounted_tyre")
        if stale:
            driver_flags.append("stale_progress")
        stint = last.source_stint if pd.notna(last.source_stint) else "unknown"
        used = frozenset(str(x).upper() for x in group.compound.dropna() if str(x).upper() in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"})
        drivers[number] = DriverState(number, int(last.lap_number), int(last.lap_end_ns), progress,
                                      compound, age, f"mounted:{number}:{stint}", used, in_pit,
                                      entry, None, stale, typical, tuple(driver_flags))
    # An entrant without a released crossing is unknown, not absent/retired.
    # Keeping it explicit prevents a field simulation from silently shrinking.
    for entry_metadata in manifest.entries:
        number = str(entry_metadata["driver_number"])
        if number not in drivers:
            drivers[number] = DriverState(
                number, 0, manifest.race_start_ns, 0.0, None, None,
                f"mounted:{number}:unknown", frozenset(), stale=True,
                flags=("no_released_crossing", "unknown_mounted_tyre", "stale_progress"),
            )
    if driver_number not in drivers:
        raise WorkbenchError("missing_focal_completion", "Focal completion is unavailable")
    if drivers[driver_number].in_pit:
        raise WorkbenchError("focal_in_pit", "The focal car is already in the pit lane")
    # Progress-derived gaps are approximate, not official timing gaps.
    leader = max(drivers.values(), key=lambda d: (d.progress_laps, -d.last_crossing_ns))
    from dataclasses import replace
    for number, driver in drivers.items():
        if not driver.stale and not leader.stale and leader.typical_lap_s:
            drivers[number] = replace(driver, gap_s=max(0., (leader.progress_laps - driver.progress_laps) * leader.typical_lap_s))
    max_release = max((int(frame.release_ns.max()) for frame in frames.values() if not frame.empty and frame.release_ns.notna().any()), default=manifest.race_start_ns)
    return RaceState(snapshot.ref.snapshot_id, manifest.session_id, driver_number, completed_lap, cutoff,
                     max_release, manifest.scheduled_laps, manifest.race_start_ns, manifest.stream_origin_utc,
                     manifest.year, manifest.round_number, manifest.synthetic, laps, frames["weather"],
                     frames["track_status"], frames["session_status"], frames["messages"], frames["pit_events"],
                     drivers, tuple(flags), dict(manifest.event_metadata),
                     ({"snapshot_id": snapshot.ref.snapshot_id, "session_id": manifest.session_id,
                       "source_ids": list(manifest.source_ids), "synthetic": manifest.synthetic},))

from dataclasses import replace
import pyarrow.compute as pc
from f1_strategy.data.snapshots import open_snapshot
from f1_strategy.replay import build_race_state
from f1_strategy.synthetic import generate_snapshot


def test_entry_without_released_crossing_is_not_silently_retired(tmp_path):
    ref = generate_snapshot(tmp_path, 8, target=True)
    snapshot = open_snapshot(tmp_path, ref.snapshot_id)
    tables = dict(snapshot.tables)
    tables["laps"] = tables["laps"].filter(pc.not_equal(tables["laps"]["driver_number"], "1"))
    state = build_race_state(replace(snapshot, tables=tables), driver_number="44", completed_lap=10)
    unknown = state.drivers["1"]
    assert unknown.completed_laps == 0
    assert unknown.compound is None and unknown.tyre_age_laps is None
    assert unknown.stale and unknown.gap_s is None
    assert "no_released_crossing" in unknown.flags

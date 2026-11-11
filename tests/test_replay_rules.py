from dataclasses import replace

import pandas as pd
import pytest

from f1_strategy.contracts import PlannedStop, RuleOverride, ScenarioSpec, StrategyPlan, TyreSet, WorkbenchError
from f1_strategy.data.schemas import table_from_rows
from f1_strategy.data.snapshots import open_snapshot
from f1_strategy.replay import build_race_state
from f1_strategy.rules import check_state, plan_legality
from f1_strategy.synthetic import generate_snapshot


@pytest.fixture
def snapshot(tmp_path):
    ref = generate_snapshot(tmp_path, 7, target=True)
    return open_snapshot(tmp_path, ref.snapshot_id)


def scenario(state, **changes):
    values = dict(name="test", inventory_by_driver={state.driver_number: (
        TyreSet(set_id="hard", compound="HARD", age_laps=0, provenance="assumed"),
        TyreSet(set_id="soft", compound="SOFT", age_laps=0, provenance="assumed"),
    )}, inventory_provenance="assumed")
    values.update(changes)
    return ScenarioSpec(**values)


def with_rows(snapshot, name, rows):
    tables = dict(snapshot.tables)
    tables[name] = table_from_rows(name, rows)
    return replace(snapshot, tables=tables)


def test_prefix_mutation_and_released_pit_events(snapshot):
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    assert state.drivers["1"].completed_laps == 9
    assert state.max_feature_release_ns <= state.cutoff_ns
    changed = snapshot
    for name in ("laps", "weather", "track_status", "session_status", "messages", "pit_events"):
        rows = snapshot.table(name).to_pylist()
        for row in rows:
            if name == "laps":
                row["final_deleted"] = True
                row["source_position"] = 99
            if row["release_ns"] is not None and row["release_ns"] > state.cutoff_ns:
                if name == "laps":
                    row["lap_time_s"] = 999.0
                    row["compound"] = "WET"
                elif name == "weather": row["rainfall"] = True
                elif name == "track_status": row["status"] = "RED"
                elif name == "pit_events": row["kind"] = "entry"
                elif name == "messages": row["message"] = "CAR 44 LAP 10 TIME DELETED"
        changed = with_rows(changed, name, rows)
    other = build_race_state(changed, driver_number="44", completed_lap=10)
    pd.testing.assert_frame_equal(state.laps, other.laps)
    assert state.drivers == other.drivers
    assert check_state(state, scenario(state)) == check_state(other, scenario(other))
    assert state.laps.lap_time_s.tail(5).mean() == other.laps.lap_time_s.tail(5).mean()


def test_focal_in_pit_and_opponent_delayed_exit(snapshot):
    with pytest.raises(WorkbenchError, match="already in the pit lane") as exc:
        build_race_state(snapshot, driver_number="44", completed_lap=17)
    assert exc.value.code == "focal_in_pit"
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    rows = snapshot.table("pit_events").to_pylist()
    entry = next(x for x in rows if x["driver_number"] == "1" and x["kind"] == "entry")
    exit_ = next(x for x in rows if x["driver_number"] == "1" and x["kind"] == "exit")
    entry["event_ns"] = entry["release_ns"] = state.cutoff_ns - 2
    exit_["event_ns"] = exit_["release_ns"] = state.cutoff_ns + 1
    state = build_race_state(with_rows(snapshot, "pit_events", rows), driver_number="44", completed_lap=10)
    assert state.drivers["1"].in_pit
    assert state.drivers["1"].compound == "MEDIUM"
    exit_["event_ns"] = exit_["release_ns"] = state.cutoff_ns - 1
    state = build_race_state(with_rows(snapshot, "pit_events", rows), driver_number="44", completed_lap=10)
    assert not state.drivers["1"].in_pit
    assert state.drivers["1"].compound is None


def test_deletion_only_after_release_and_reinstatement(snapshot):
    six = build_race_state(snapshot, driver_number="44", completed_lap=6)
    eight = build_race_state(snapshot, driver_number="44", completed_lap=8)
    nine = build_race_state(snapshot, driver_number="44", completed_lap=9)
    def deleted(state):
        return bool(state.laps[(state.laps.driver_number == "44") & (state.laps.lap_number == 6)].iloc[0].deleted_as_of)
    assert not deleted(six)
    assert deleted(eight)
    assert not deleted(nine)


def test_dry_red_and_unknown_domain(snapshot):
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    spec = scenario(state)
    assert "conditional_rules" in check_state(state, spec)
    unknown = replace(state, weather=state.weather.iloc[0:0])
    assert "unknown_dry_conditions" in check_state(unknown, spec)
    assert "unknown_dry_conditions" not in check_state(unknown, scenario(state, dry_condition_assumption="dry"))
    rain = state.weather.copy()
    rain.loc[rain.index[0], "rainfall"] = True
    assert "unsupported_wet" in check_state(replace(state, weather=rain), scenario(state, dry_condition_assumption="dry"))
    red = state.track_status.copy()
    red.loc[red.index[0], "status"] = "RED"
    assert "unsupported_red_flag" in check_state(replace(state, track_status=red), spec)
    assert "unsupported_season" in check_state(replace(state, year=2025), spec)


def test_compounds_inventory_and_life(snapshot):
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    spec = scenario(state)
    hard = StrategyPlan(plan_id="hard", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id="hard"),))
    assert plan_legality(state, spec, hard) == ()
    soft = StrategyPlan(plan_id="soft", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id="soft"),))
    assert plan_legality(state, spec, soft) == ()
    absent = StrategyPlan(plan_id="absent", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id="missing"),))
    assert "unavailable_tyre_set" in plan_legality(state, spec, absent)
    override = RuleOverride(max_set_life_laps=8, source_reference="test", provenance="assumed")
    assert "set_life_exceeded" in plan_legality(state, scenario(state, rules_override=override), hard)
    only_soft = replace(state.drivers["44"], compound="SOFT", used_compounds=frozenset({"SOFT"}))
    modified = replace(state, drivers={**state.drivers, "44": only_soft})
    no_stop = StrategyPlan(plan_id="none", branch="stay_out", stops=())
    assert {"insufficient_dry_compounds", "missing_mandatory_compound"} <= set(plan_legality(modified, scenario(modified), no_stop))


def test_qatar_requires_counting_evidence_and_enforces_total_life(snapshot):
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    state = replace(state, synthetic=False, year=2023, round_number=18)
    assert "unsupported_directive" in check_state(state, scenario(state))
    old = replace(state.drivers["44"], tyre_age_laps=18)
    state = replace(state, drivers={**state.drivers, "44": old})
    plan = StrategyPlan(plan_id="pit", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id="hard"),))
    assert "set_life_exceeded" in plan_legality(state, scenario(state), plan)


def test_released_closure_and_routing(snapshot):
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    messages = state.messages.copy()
    message = messages.iloc[-1].copy()
    message["message"] = "PIT ENTRY CLOSED"
    messages = pd.concat([messages, message.to_frame().T], ignore_index=True)
    closed = replace(state, messages=messages)
    assert "pit_entry_closed" in check_state(closed, scenario(closed))
    plan = StrategyPlan(plan_id="pit", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id="hard"),))
    assert "pit_entry_closed" in plan_legality(closed, scenario(closed), plan)
    messages.loc[messages.index[-1], "message"] = "SAFETY CAR THROUGH THE PIT LANE"
    assert "unsupported_directive" in check_state(replace(state, messages=messages), scenario(state))

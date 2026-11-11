from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from f1_strategy.contracts import (DriverState, PaceAssumptions, PitAssumptions, PlannedStop,
                                   RaceState, StrategyPlan, TyreSet, WorkbenchError)
from f1_strategy.strategy.plans import default_scenario, load_scenario
from f1_strategy.strategy.simulate import _draw_plan, _finish_lap, _recommend


def state():
    drivers = {
        number: DriverState(driver_number=number, completed_laps=10, last_crossing_ns=1,
                            progress_laps=10.0, compound="MEDIUM", tyre_age_laps=10,
                            mounted_set_id=f"mounted:{number}", used_compounds=frozenset({"MEDIUM"}))
        for number in ("44", "1")
    }
    empty = pd.DataFrame()
    return RaceState(snapshot_id="a" * 64, session_id="SYN-2023-01-R", driver_number="44",
                     completed_lap=10, cutoff_ns=1, max_feature_release_ns=1,
                     scheduled_laps=24, race_start_ns=0,
                     stream_origin_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
                     year=2023, round_number=1, synthetic=True, laps=empty, weather=empty,
                     track_status=empty, session_status=empty, messages=empty, pit_events=empty,
                     drivers=drivers)


def test_virtual_inventory_expands_for_every_driver_and_sets_are_distinct():
    race = state()
    scenario = load_scenario({"name": "assumed", "inventory_by_driver": {},
                              "inventory_provenance": "assumed"}, race)
    assert scenario == default_scenario(race).model_copy(update={"name": "assumed", "inventory_provenance": "assumed"})
    sets = [tyre for inventory in scenario.inventory_by_driver.values() for tyre in inventory]
    assert len(sets) == len({tyre.set_id for tyre in sets}) == 6
    assert all(tyre.age_laps == 0 and tyre.provenance == "assumed" for tyre in sets)


def test_explicit_empty_inventory_is_not_replenished_and_unknown_driver_is_rejected():
    race = state()
    scenario = load_scenario({"name": "none", "inventory_by_driver": {"44": []},
                              "inventory_provenance": "documented"}, race)
    assert scenario.inventory_by_driver["44"] == ()
    assert len(scenario.inventory_by_driver["1"]) == 3
    with pytest.raises(WorkbenchError) as exc:
        load_scenario({"name": "bad", "inventory_by_driver": {"999": []},
                       "inventory_provenance": "assumed"}, race)
    assert exc.value.code == "invalid_scenario"


class FixedPace:
    model_id = "b" * 64
    data_cutoff_utc = datetime(2022, 1, 1, tzinfo=timezone.utc)
    feature_names = ()
    n_parameter_replicates = 1
    assumption_grid = tuple((g, d) for g in (.04, .06, .08) for d in (-.02, 0., .02))
    support = {c: {"laps": 40, "stints": 4, "events": 3, "min_age": 0, "max_age": 100,
                   "spec_eras": {f"synthetic:{c}": {"one_event_qualified": True, "min_age": 0, "max_age": 100}}}
               for c in ("SOFT", "MEDIUM", "HARD")}

    def predict_laps(self, features, *, assumptions, replicate_index=None):
        if "lap_time_s" in features:
            return features.lap_time_s.to_numpy(dtype=float)
        return np.where(features.compound == "HARD", 90.0, 92.0)

    def residual_blocks(self, compound, *, assumptions):
        return np.zeros((1, 3))


def test_fixed_distance_paired_timing_and_large_pit_loss(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], compound="MEDIUM", tyre_age_laps=12,
                    used_compounds=frozenset({"MEDIUM", "HARD"}))
    race = replace(race, drivers={**race.drivers, "44": focal})
    pit = PitAssumptions(net_transit_loss_s=9., net_transit_sd_s=0., service_s=3., service_sd_s=0.,
                         outlap_warmup_s=2., outlap_warmup_sd_s=0.)
    scenario = default_scenario(race).model_copy(update={"pit": pit, "max_additional_stops": 1})
    hard = next(t.set_id for t in scenario.inventory_by_driver["44"] if t.compound == "HARD")
    now = StrategyPlan(plan_id="now", branch="pit_now", stops=(PlannedStop(pit_lap=13, new_set_id=hard),))
    later = StrategyPlan(plan_id="later", branch="stay_out", stops=(PlannedStop(pit_lap=14, new_set_id=hard),))
    none = StrategyPlan(plan_id="none", branch="stay_out", stops=())
    a, _, _ = _draw_plan(race, FixedPace(), scenario, now, 100, 1)
    b, _, _ = _draw_plan(race, FixedPace(), scenario, later, 100, 1)
    assert np.all(a == 1096.)
    assert np.all(b == 1098.)
    costly = scenario.model_copy(update={"pit": pit.model_copy(update={"net_transit_loss_s": 47.})})
    a, _, _ = _draw_plan(race, FixedPace(), costly, now, 100, 1)
    b, _, _ = _draw_plan(race, FixedPace(), costly, none, 100, 1)
    assert np.all(a == 1134.)
    assert np.all(b == 1104.)


def test_timed_flag_and_recommendation_require_mean_and_probability():
    assert _finish_lap([7100., 7200., 7290.], 10) == 3
    assert _finish_lap([7100., 7201., 7290.], 2) == 2
    assert _recommend(-1., .9, [{"mean_delta_s": -1.}]) == "no_clear_advantage"
    assert _recommend(1., .9, [{"mean_delta_s": -1.}]) == "no_clear_advantage"

def test_neutralisation_scales_net_loss_but_not_service(tmp_path):
    from f1_strategy.contracts import NeutralisationSpec
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    scenario = default_scenario(race)
    hard = next(t.set_id for t in scenario.inventory_by_driver["44"] if t.compound == "HARD")
    plan = StrategyPlan(plan_id="now", branch="pit_now", stops=(PlannedStop(pit_lap=13, new_set_id=hard),))
    pit = PitAssumptions(net_transit_loss_s=10., net_transit_sd_s=0., service_s=3., service_sd_s=0.,
                         outlap_warmup_s=0., outlap_warmup_sd_s=0., sc_net_transit_factor=.5)
    scenario = scenario.model_copy(update={"pit": pit, "max_additional_stops": 1,
                                           "neutralisations": (NeutralisationSpec(kind="SC", start_after_leader_laps=0,
                                                                                 duration_laps=3),)})
    base, _, _ = _draw_plan(race, FixedPace(), scenario, plan, 100, 2)
    increased = scenario.model_copy(update={"pit": pit.model_copy(update={"service_s": 5.})})
    altered, _, _ = _draw_plan(race, FixedPace(), increased, plan, 100, 2)
    assert np.allclose(altered - base, 2.)


def test_single_car_comparison_preserves_paired_delta_and_seed(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.simulate import compare_strategies

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], compound="MEDIUM", tyre_age_laps=12,
                    used_compounds=frozenset({"MEDIUM", "HARD"}))
    race = replace(race, drivers={"44": focal})
    pit = PitAssumptions(net_transit_loss_s=9., net_transit_sd_s=0., service_s=3., service_sd_s=0.,
                         outlap_warmup_s=2., outlap_warmup_sd_s=0.)
    scenario = default_scenario(race).model_copy(update={"pit": pit, "max_additional_stops": 1,
                                                         "dry_condition_assumption": "dry"})
    first = compare_strategies(race, FixedPace(), scenario, n_draws=100, seed=3)
    second = compare_strategies(race, FixedPace(), scenario, n_draws=100, seed=3)
    assert first.delta_stay_minus_pit_s == pytest.approx(2.)
    assert first.prob_pit_faster == 1.
    assert first.recommendation == "pit_now"
    assert first.analysis_id == second.analysis_id
    assert first.simulation_series == second.simulation_series
    from f1_strategy.identity import verify_payload
    verify_payload(first.model_dump(mode="json"))


def test_absolute_lane_occupancy_is_conditioned_and_not_net_loss():
    from f1_strategy.strategy.occupancy import pit_occupancy_projection

    race = state()
    rival = replace(race.drivers["1"], in_pit=True, pit_entry_ns=10_000_000_000)
    race = replace(race, cutoff_ns=20_000_000_000, drivers={**race.drivers, "1": rival})
    pit = PitAssumptions(net_transit_loss_s=9., absolute_lane_duration_s=30., absolute_lane_duration_sd_s=0.)
    scenario = default_scenario(race).model_copy(update={"pit": pit})
    result = pit_occupancy_projection(race, scenario, 100, 1)["1"]
    assert result["remaining_occupancy_mean_s"] == 20.
    assert result["remaining_occupancy_interval80_s"] == [20., 20.]
    unknown = scenario.model_copy(update={"pit": pit.model_copy(update={"absolute_lane_duration_s": None})})
    assert not pit_occupancy_projection(race, unknown, 100, 1)["1"]["available"]


def test_field_clocks_use_equal_distance_and_preserve_lap_deficit():
    from f1_strategy.strategy.field import run_field

    race = state()
    rival = replace(race.drivers["1"], completed_laps=9, progress_laps=9.)
    race = replace(race, scheduled_laps=12, drivers={**race.drivers, "1": rival})
    scenario = default_scenario(race)
    plan = StrategyPlan(plan_id="stay", branch="stay_out", stops=())
    focal = np.array([[90., 90.]])
    slow = {"1": {"paths": np.array([[90., 90., 90.]]), "completed_laps": 9, "stops": ()}}
    times, ranks, _ = run_field(race, scenario, plan, focal, slow)
    assert times[0] == 180.
    assert ranks[0] == 1
    fast = {"1": {"paths": np.array([[40., 40., 40.]]), "completed_laps": 9, "stops": ()}}
    times, ranks, _ = run_field(race, scenario, plan, focal, fast)
    assert times[0] == 180.
    assert ranks[0] == 2


def test_simulated_closure_abstains_instead_of_forcing_a_branch(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.simulate import compare_strategies

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], used_compounds=frozenset({"MEDIUM", "HARD"}))
    race = replace(race, drivers={"44": focal})
    from f1_strategy.contracts import NeutralisationSpec
    scenario = default_scenario(race).model_copy(update={
        "max_additional_stops": 1, "dry_condition_assumption": "dry",
        "neutralisations": (NeutralisationSpec(kind="SC", start_after_leader_laps=1,
                                                duration_laps=1, pit_entry_open=False),),
    })
    result = compare_strategies(race, FixedPace(), scenario, n_draws=100, seed=4)
    assert result.status == "insufficient_evidence"
    assert result.recommendation == "unavailable"


def test_no_inventory_directly_proves_pit_now_unavailable(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.simulate import compare_strategies

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], used_compounds=frozenset({"MEDIUM", "HARD"}))
    race = replace(race, drivers={"44": focal})
    scenario = default_scenario(race).model_copy(update={"inventory_by_driver": {"44": ()},
                                                         "max_additional_stops": 0,
                                                         "dry_condition_assumption": "dry"})
    result = compare_strategies(race, FixedPace(), scenario, n_draws=100, seed=4)
    assert result.status == "ok"
    assert result.recommendation == "stay_out"
    assert "direct_pit_now_infeasible" in result.quality_flags


def test_comparison_rejects_a_post_cutoff_feature():
    from f1_strategy.strategy.simulate import compare_strategies

    race = replace(state(), laps=pd.DataFrame([{"release_ns": 2}]))
    with pytest.raises(WorkbenchError) as exc:
        compare_strategies(race, FixedPace(), default_scenario(race), n_draws=100, seed=1)
    assert exc.value.code == "future_feature"


def test_plan_search_uses_next_inlap_and_consumes_declared_set(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.plans import enumerate_plans

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], compound="MEDIUM", tyre_age_laps=12,
                    used_compounds=frozenset({"MEDIUM", "HARD"}))
    race = replace(race, drivers={"44": focal})
    tyre = TyreSet(set_id="one-hard", compound="HARD", age_laps=0, provenance="assumed")
    scenario = default_scenario(race).model_copy(update={"inventory_by_driver": {"44": (tyre,)},
                                                         "max_additional_stops": 1})
    plans = enumerate_plans(race, scenario, FixedPace())
    assert any(p.branch == "pit_now" and p.stops[0].pit_lap == 13 for p in plans)
    assert all(p.stops[0].pit_lap >= 14 for p in plans if p.branch == "stay_out" and p.stops)
    assert all(len({s.new_set_id for s in p.stops}) == len(p.stops) for p in plans)


def test_flag_classification_is_separate_from_equal_distance():
    from f1_strategy.strategy.field import run_field

    race = state()
    race = replace(race, cutoff_ns=7_100_000_000_000, scheduled_laps=13,
                   drivers={"44": race.drivers["44"]})
    plan = StrategyPlan(plan_id="stay", branch="stay_out", stops=())
    times, ranks, events = run_field(race, default_scenario(race), plan,
                                     np.array([[100., 90., 90.]]), {})
    assert events["flag_laps"] == [12]
    assert events["flag_times_s"] == [190.]
    assert times.tolist() == [280.]
    assert ranks.tolist() == [1]


def test_field_conditions_in_pit_car_on_absolute_occupancy(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.field import prepare_field

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    rival = replace(race.drivers["1"], in_pit=True, pit_entry_ns=race.cutoff_ns - 10_000_000_000,
                    used_compounds=frozenset({"SOFT", "MEDIUM"}))
    race = replace(race, drivers={"44": race.drivers["44"], "1": rival})
    scenario = default_scenario(race).model_copy(update={
        "pit": PitAssumptions(absolute_lane_duration_s=30., absolute_lane_duration_sd_s=0.),
        "max_additional_stops": 0,
    })
    field = prepare_field(race, FixedPace(), scenario, 100, 7)
    assert np.all(field["1"]["paths"][:, 0] == 20.)
    missing = scenario.model_copy(update={"pit": scenario.pit.model_copy(update={"absolute_lane_duration_s": None})})
    with pytest.raises(WorkbenchError) as exc:
        prepare_field(race, FixedPace(), missing, 100, 7)
    assert exc.value.code == "field_unavailable"


def test_direct_tyre_obligation_impossibility_is_infeasible(tmp_path):
    from f1_strategy.synthetic import generate_snapshot
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.simulate import compare_strategies

    ref = generate_snapshot(tmp_path, 8, target=True)
    race = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=12)
    focal = replace(race.drivers["44"], compound="MEDIUM", used_compounds=frozenset({"MEDIUM"}))
    race = replace(race, drivers={"44": focal})
    scenario = default_scenario(race).model_copy(update={"inventory_by_driver": {"44": ()},
                                                         "max_additional_stops": 0,
                                                         "dry_condition_assumption": "dry"})
    result = compare_strategies(race, FixedPace(), scenario, n_draws=100, seed=4)
    assert result.status == "infeasible"
    assert "direct_constraints_infeasible" in result.quality_flags


def test_active_neutralisation_is_not_erased_by_green_preset():
    race = replace(state(), track_status=pd.DataFrame([{"status": "SC"}]))
    scenario = load_scenario({"name": "green", "inventory_by_driver": {},
                              "inventory_provenance": "assumed", "neutralisations": []}, race)
    assert len(scenario.neutralisations) == 1
    assert scenario.neutralisations[0].kind == "SC"
    assert scenario.neutralisations[0].start_after_leader_laps == 0
    assert scenario.neutralisations[0].duration_laps == 3


def test_lapped_rival_uses_leader_crossing_for_vsc_transition():
    from f1_strategy.contracts import NeutralisationSpec
    from f1_strategy.strategy.field import run_field

    race = state()
    rival = replace(race.drivers["1"], completed_laps=9, progress_laps=9.)
    race = replace(race, scheduled_laps=12, drivers={**race.drivers, "1": rival})
    scenario = default_scenario(race).model_copy(update={
        "neutralisations": (NeutralisationSpec(kind="VSC", start_after_leader_laps=1, duration_laps=1),),
        "traffic_penalty_s": 0.0})
    plan = StrategyPlan(plan_id="stay", branch="stay_out", stops=())
    focal = np.array([[100., 100.]])
    opponents = {"1": {"paths": np.array([[50., 100., 100.]]), "completed_laps": 9, "stops": ()}}
    times, _, details = run_field(race, scenario, plan, focal, opponents)
    assert times[0] == pytest.approx(240.)
    assert details["flag_times_s"][0] == pytest.approx(240.)
    closed = scenario.model_copy(update={
        "neutralisations": (NeutralisationSpec(kind="VSC", start_after_leader_laps=1, duration_laps=1,
                                               pit_exit_open=False),)})
    opponents["1"]["stops"] = (10,)
    with pytest.raises(WorkbenchError) as exc:
        run_field(race, closed, plan, focal, opponents)
    assert exc.value.code == "field_unavailable"


def test_pit_evidence_uses_real_challenger_base_predictor(monkeypatch):
    from f1_strategy.pace.challenger import ChallengerPaceModel
    from f1_strategy.strategy.pit import completed_pit_evidence
    import f1_strategy.pace.features as pace_features

    race = state()
    laps = []
    events = []
    for visit, inlap in enumerate((5, 10, 15)):
        for lap in (inlap - 3, inlap - 2, inlap - 1, inlap, inlap + 1):
            laps.append(dict(driver_number="44", lap_number=lap, lap_time_s=97. if lap >= inlap else 92.,
                             eligible=lap < inlap, traffic_unknown=False,
                             exclusion_reasons=("pit_lap",) if lap >= inlap else (),
                             track_temperature_c=30., lap_start_ns=lap * 100,
                             lap_end_ns=lap * 100 + 90))
        events.extend((dict(driver_number="44", kind="entry", release_ns=inlap * 100 + 10,
                            source_row_id=f"{visit}a", source_lap_number=inlap),
                       dict(driver_number="44", kind="exit", release_ns=inlap * 100 + 30,
                            source_row_id=f"{visit}b", source_lap_number=inlap)))
    race = replace(race, cutoff_ns=2000, pit_events=pd.DataFrame(events),
                   weather=pd.DataFrame([dict(release_ns=0, track_temp_c=30.)]))
    monkeypatch.setattr(pace_features, "eligible_laps", lambda *args, **kwargs: pd.DataFrame(laps))
    class CleanBase(FixedPace):
        def predict_laps(self, features, *, assumptions, replicate_index=None):
            return np.full(len(features), 90.)

    challenger = object.__new__(ChallengerPaceModel)
    challenger.base_model = CleanBase()
    evidence = completed_pit_evidence(race, challenger, default_scenario(race))
    assert evidence["loss_samples"] == 3
    assert evidence["clean_predictor"] == "hierarchical base model"


def test_conditional_quantile_width_changes_paired_strategy_uncertainty(monkeypatch):
    from types import SimpleNamespace
    import f1_strategy.pace.hierarchical as hierarchical

    race = state()
    race = replace(race, scheduled_laps=13, drivers={"44": race.drivers["44"]})
    scenario = default_scenario(race).model_copy(update={"max_additional_stops": 1})
    hard = next(t.set_id for t in scenario.inventory_by_driver["44"] if t.compound == "HARD")
    now = StrategyPlan(plan_id="now", branch="pit_now", stops=(PlannedStop(pit_lap=11, new_set_id=hard),))
    stay = StrategyPlan(plan_id="stay", branch="stay_out", stops=())

    def forecast(width):
        def make(state, model, *, driver_number, compound, race_laps, tyre_ages, assumptions, outlap_race_lap=None):
            n = len(race_laps)
            q = np.tile(np.array([-2., -1., 1., 2.]) * (width if compound == "HARD" else 1.), (n, 1))
            return SimpleNamespace(race_laps=race_laps, tyre_ages=tyre_ages, compound=compound,
                                   mean_s=np.full(n, 90.), parameter_mean_s=np.full((1, n), 90.),
                                   residual_blocks_s=np.array([[-2., -1., 0.], [1., 2., 0.]]),
                                   residual_quantiles_s=q, ml_residual_mask=np.ones(n, dtype=bool),
                                   interval80_s=np.zeros((n, 2)), interval95_s=np.zeros((n, 2)),
                                   support={"advice_supported": True}, flags=())
        return make

    spreads = []
    for width in (1., 5.):
        monkeypatch.setattr(hierarchical, "forecast_laps", forecast(width))
        pit, _, segments = _draw_plan(race, FixedPace(), scenario, now, 1000, 42)
        continuing, _, _ = _draw_plan(race, FixedPace(), scenario, stay, 1000, 42)
        spreads.append(np.std(continuing - pit))
        assert segments[-1]["residual_quantiles_s"] is not None
        assert all(segments[-1]["ml_residual_mask"])
    assert spreads[1] > spreads[0] + 1.

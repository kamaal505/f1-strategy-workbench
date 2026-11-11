from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_strategy.contracts import WorkbenchError
from f1_strategy.pace.artifacts import load_model
from f1_strategy.contracts import PaceAssumptions
from f1_strategy.pace.baseline import rolling_forecast
from f1_strategy.pace.bundle import publish_bundle, read_bundle
from f1_strategy.pace.calibration import conformal_adjustment
from f1_strategy.pace.design import design_matrix, design_spec, fit_robust
from f1_strategy.pace.features import eligible_laps
from f1_strategy.pace.residuals import build_residual_blocks


def test_calibration_order_statistic_and_sparse_events():
    lower = np.zeros(40)
    upper = np.ones(40)
    observed = np.arange(40, dtype=float) + 1
    events = np.array(["a"] * 20 + ["b"] * 20)
    result = conformal_adjustment(lower, upper, observed, events, .8)
    assert result["adjustment_s"] == 32
    assert result["status"] == "calibrated"
    assert conformal_adjustment(lower, upper, observed, ["a"] * 40, .8)["status"] == "uncalibrated"
    assert conformal_adjustment(lower[:29], upper[:29], observed[:29], events[:29], .8)["adjustment_s"] is None


def test_design_unknown_group_is_zero_and_penalty_strength():
    row = dict(driver_number="44", session_id="race", driver_session="race:44", layout_id="layout",
               spec_era_id="spec", compound="MEDIUM", tyre_age_laps=10, warmup_first=0.,
               warmup_second=0., track_temperature_delta_c=0., lap_number=10)
    train = pd.DataFrame([row])
    spec = design_spec(train)
    test = pd.DataFrame([{**row, "driver_number": "new"}])
    matrix, names, penalties = design_matrix(test, spec)
    assert matrix.toarray()[0, names.index("driver_number:44")] == 0
    assert matrix.toarray()[0, names.index("age:MEDIUM")] == 8
    assert penalties[names.index("spec_age:spec")] > penalties[names.index("age:MEDIUM")]


def test_features_do_not_release_future_pit_or_weather():
    laps = pd.DataFrame([dict(driver_number="44", lap_number=i, release_ns=i*100,
                              lap_start_ns=(i-1)*100, lap_end_ns=i*100, lap_time_s=90.,
                              compound="MEDIUM", tyre_age_laps=i, fastf1_generated=False,
                              final_deleted=False) for i in range(1, 5)])
    pit = pd.DataFrame([dict(driver_number="44", source_lap_number=3, kind="exit", release_ns=350)])
    status = pd.DataFrame([dict(release_ns=0, status="GREEN")])
    weather = pd.DataFrame([dict(release_ns=0, track_temp_c=25., rainfall=False), dict(release_ns=350, track_temp_c=40., rainfall=False)])
    prefix = eligible_laps(laps, pit, status, weather, cutoff_ns=300, event_metadata={}, session_id="s")
    assert prefix.lap_number.tolist() == [1, 2, 3]
    assert prefix.track_temperature_c.tolist() == [25., 25., 25.]
    assert not prefix.warmup_first.any()
    assert prefix.loc[prefix.lap_number == 2, "eligible"].iloc[0]


def test_residual_blocks_do_not_join_stints_or_lap_gaps():
    records = []
    for stint in range(3):
        for lap in range(1, 13):
            if lap == 7:
                continue
            records.append(dict(session_id="event", driver_number="44", stint_id=stint,
                                lap_number=lap, spec_era_id="spec", compound="MEDIUM",
                                residual_s=float(lap)))
    blocks, level = build_residual_blocks(pd.DataFrame(records), spec_era_id="spec", compound="MEDIUM")
    assert level == "spec_era"
    assert blocks.shape == (21, 3)
    assert np.all(np.diff(blocks, axis=1) == 1.)
    _, level = build_residual_blocks(pd.DataFrame(records[:10]), spec_era_id="spec", compound="MEDIUM")
    assert level == "insufficient_residuals"


def test_numeric_bundle_identity_and_corruption(tmp_path):
    arrays = {"coefficients": np.array([1., 2.])}
    identifier, path = publish_bundle(tmp_path, {"kind": "example"}, arrays)
    again, _ = publish_bundle(tmp_path, {"kind": "example"}, arrays)
    assert identifier == again
    metadata, decoded = read_bundle(tmp_path, identifier)
    assert metadata["kind"] == "example"
    np.testing.assert_array_equal(decoded["coefficients"], arrays["coefficients"])
    (path / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(WorkbenchError) as error:
        read_bundle(tmp_path, identifier)
    assert error.value.code == "corrupt_artifact"


def test_fuel_assumptions_change_identifiable_conditional_slope():
    rng = np.random.default_rng(17)
    rows = []
    for event in range(5):
        for driver in range(4):
            initial_age = event * 2 + driver
            for lap in range(2, 20):
                age = initial_age + lap
                rows.append(dict(driver_number=str(driver), session_id=str(event),
                    driver_session=f"{event}:{driver}", layout_id="layout", spec_era_id="era",
                    compound="MEDIUM", tyre_age_laps=age, warmup_first=0., warmup_second=0.,
                    track_temperature_delta_c=0., lap_number=lap,
                    lap_time_s=90 + .10 * max(age - 2, 0) - .06 * (lap - 1) + rng.normal(0, .01)))
    frame = pd.DataFrame(rows)
    spec = design_spec(frame)
    _, correct, names = fit_robust(frame, spec, .06)
    _, alternative, _ = fit_robust(frame, spec, .04)
    assert abs(correct[names.index("age:MEDIUM")] - .10) < .02
    assert abs(correct[names.index("age:MEDIUM")] - alternative[names.index("age:MEDIUM")]) > .005


def test_model_with_future_cutoff_is_rejected(tmp_path):
    from datetime import datetime, timezone
    import platform
    from f1_strategy.identity import lock_digest
    from f1_strategy.pace.bundle import _runtime_versions
    metadata = {"model_kind": "hierarchical", "data_cutoff_utc": "2024-01-01T00:00:00+00:00",
                "lineage": [], "design": {}, "feature_names": [], "support": {},
                "lock_digest": lock_digest(), "versions": {**_runtime_versions(), "platform": platform.platform()},
                "training_spec": {"training_snapshot_ids": [], "tuning_snapshot_ids": [],
                                  "selection_snapshot_ids": [], "calibration_snapshot_ids": []},
                "assumption_grid": [[.06, 0.]], "variant_indices": [0]}
    identifier, _ = publish_bundle(tmp_path, metadata, {"fit_0": np.array([90.]),
                                                        "bootstrap_0": np.zeros((200, 1))})
    with pytest.raises(WorkbenchError) as error:
        load_model(tmp_path, identifier, target_start_utc=datetime(2023, 1, 1, tzinfo=timezone.utc))
    assert error.value.code == "future_artifact"


def test_rolling_baseline_inverts_assumed_fuel_correction():
    rows = pd.DataFrame({"lap_number": [2, 3, 4, 5, 6],
                         "lap_time_s": [90 - .06 * (lap - 1) for lap in range(2, 7)]})
    forecast = rolling_forecast(rows, race_laps=np.array([7, 8]), assumptions=PaceAssumptions())
    np.testing.assert_allclose(forecast["mean_s"], [89.64, 89.58])
    assert "no_compound_switch_advice" in forecast["flags"]


def test_only_two_complete_post_outlap_laps_get_learned_warmup():
    laps = pd.DataFrame([dict(driver_number="44", lap_number=i, release_ns=i * 100,
                              lap_start_ns=(i - 1) * 100, lap_end_ns=i * 100,
                              lap_time_s=90., compound="MEDIUM", tyre_age_laps=i,
                              fastf1_generated=False, final_deleted=False) for i in range(1, 7)])
    pits = pd.DataFrame([dict(driver_number="44", source_lap_number=3, kind="exit", release_ns=250)])
    status = pd.DataFrame([dict(release_ns=0, status="GREEN")])
    weather = pd.DataFrame([dict(release_ns=0, track_temp_c=25.)])
    features = eligible_laps(laps, pits, status, weather, cutoff_ns=600,
                             event_metadata={}, session_id="s")
    assert features.loc[features.warmup_first == 1, "lap_number"].tolist() == [4]
    assert features.loc[features.warmup_second == 1, "lap_number"].tolist() == [5]
    assert not features.loc[features.lap_number == 3, "eligible"].iloc[0]


def test_rolling_bundle_forecasts_released_prefix(tmp_path):
    from datetime import datetime, timezone
    from f1_strategy.contracts import TrainingSpec
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.pace.hierarchical import fit_pace, forecast_stint
    from f1_strategy.replay import build_race_state
    from f1_strategy.synthetic import generate_snapshot

    earlier = generate_snapshot(tmp_path, 1)
    target = generate_snapshot(tmp_path, 8, target=True)
    spec = TrainingSpec(training_snapshot_ids=(earlier.snapshot_id,), calibration_snapshot_ids=(),
                        data_cutoff_utc=datetime(2023, 1, 1, tzinfo=timezone.utc), model_kind="rolling")
    ref = fit_pace(spec, root=tmp_path)
    snapshot = open_snapshot(tmp_path, target.snapshot_id)
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    model = load_model(tmp_path, ref.model_id, target_start_utc=state.race_start_utc)
    forecast = forecast_stint(state, model, driver_number="44", compound=state.drivers["44"].compound,
                              tyre_age_laps=state.drivers["44"].tyre_age_laps, horizon_laps=3,
                              assumptions=PaceAssumptions())
    assert forecast.race_laps.tolist() == [11, 12, 13]
    assert forecast.support["advice_supported"] is False
    assert "no_compound_switch_advice" in forecast.flags
    assert forecast.residual_blocks_s.shape[1] == 3


def test_heldout_issued_bounds_ignore_future_temperature():
    from f1_strategy.pace.heldout import issued_parameter_paths

    frame = pd.DataFrame([
        dict(driver_number="44", session_id="e", driver_session="e:44", layout_id="l",
             spec_era_id="s", compound="MEDIUM", tyre_age_laps=i, lap_number=i,
             source_stint=1, warmup_first=0., warmup_second=0.,
             track_temperature_delta_c=0., cutoff_temperature_delta_c=0., lap_time_s=90.)
        for i in range(1, 9)
    ])
    spec = design_spec(frame)
    _, names, _ = design_matrix(frame, spec)
    fit = np.zeros(len(names) + 1)
    fit[0] = 90.
    fit[1 + names.index("temperature_delta")] = 3.
    bootstrap = np.repeat(fit[None, :], 200, axis=0)
    bootstrap[:, 0] += np.linspace(-.1, .1, 200)
    altered = frame.copy()
    altered.loc[3:, "track_temperature_delta_c"] = 50.
    altered.loc[3:, "cutoff_temperature_delta_c"] = 50.
    before = issued_parameter_paths(frame.iloc[:3], 3, spec, fit, bootstrap, .06)
    after = issued_parameter_paths(altered.iloc[:3], 3, spec, fit, bootstrap, .06)
    np.testing.assert_array_equal(np.quantile(before, [.1, .9], axis=0),
                                  np.quantile(after, [.1, .9], axis=0))


def test_prefix_adaptation_anchors_each_parameter_replicate():
    from f1_strategy.pace.adaptation import adapt_prefix
    prefix = pd.DataFrame({"lap_time_s": [90., 91., 92.], "source_stint": [1, 1, 1],
                           "tyre_age_laps": [3, 4, 5]})
    central = np.array([89., 90., 91.])
    replicates = np.array([[89., 90., 91.], [87., 88., 89.]])
    anchor, anchors, slope, adapted = adapt_prefix(prefix, central, replicates, current_age=5, same_stint=True)
    assert anchor == 1.
    np.testing.assert_array_equal(anchors, [1., 3.])
    assert slope == 0. and not adapted


def _issued_event_fixture():
    from types import SimpleNamespace
    import pyarrow as pa
    laps = pd.DataFrame([dict(session_id="session", driver_number="44", lap_number=i, release_ns=i * 100,
        lap_start_ns=(i - 1) * 100, lap_end_ns=i * 100, lap_time_s=90. + i,
        compound="MEDIUM", tyre_age_laps=i, fastf1_generated=False, final_deleted=(i == 3),
        source_stint=1) for i in range(1, 7)])
    messages = pd.DataFrame([dict(release_ns=550, message="CAR 44 LAP 3 DELETED",
                                  driver_number="44", lap_number=3)])
    frames = {"laps": laps, "messages": messages,
              "pit_events": pd.DataFrame(columns=["driver_number", "source_lap_number", "kind", "release_ns"]),
              "track_status": pd.DataFrame([dict(release_ns=0, status="GREEN")]),
              "weather": pd.DataFrame([dict(release_ns=0, track_temp_c=25., rainfall=False)])}
    snapshot = SimpleNamespace(tables={name: pa.Table.from_pandas(frame) for name, frame in frames.items()},
        manifest=SimpleNamespace(session_id="session", event_metadata={}))
    return {"snapshot": snapshot}


def test_issued_prefix_uses_deletion_messages_only_when_released():
    from f1_strategy.pace.training import issued_prefix
    event = _issued_event_fixture()
    before = issued_prefix(event, 500, "44")
    after = issued_prefix(event, 600, "44")
    assert 3 in before.lap_number.tolist()
    assert 3 not in after.lap_number.tolist()
    assert before.lap_time_s.tail(5).median() != after.lap_time_s.tail(5).median()


def test_rain_and_unknown_weather_do_not_supply_dry_evidence():
    from f1_strategy.pace.training import issued_prefix
    import pyarrow as pa
    event = _issued_event_fixture()
    event["snapshot"].tables["weather"] = pa.Table.from_pandas(pd.DataFrame([
        dict(release_ns=0, track_temp_c=25., rainfall=False),
        dict(release_ns=250, track_temp_c=25., rainfall=True),
        dict(release_ns=450, track_temp_c=25., rainfall=None)]))
    prefix = issued_prefix(event, 500, "44")
    assert prefix.lap_number.tolist() == [2]


def test_spec_pool_selector_matches_serving_fallback():
    from f1_strategy.pace.heldout import select_residual_pool
    compound = np.ones((2, 3))
    spec = np.full((2, 3), 7.)
    assert select_residual_pool("MEDIUM", "era", {"MEDIUM": compound},
                                {("era", "MEDIUM"): spec}) is spec
    assert select_residual_pool("MEDIUM", "other", {"MEDIUM": compound},
                                {("era", "MEDIUM"): spec}) is compound


def test_duplicate_session_acquisitions_are_rejected_across_roles(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from f1_strategy.pace.training import admitted_events
    import f1_strategy.pace.training as training_module
    event = _issued_event_fixture()
    snapshot = event["snapshot"]
    snapshot.table = lambda name: snapshot.tables[name]
    snapshot.manifest.stream_origin_utc = datetime(2020, 1, 1, tzinfo=timezone.utc)
    snapshot.manifest.race_start_ns = 0
    snapshot.manifest.source_ids = []
    snapshot.manifest.synthetic = True
    monkeypatch.setattr(training_module, "open_snapshot", lambda root, identifier: snapshot)
    spec = SimpleNamespace(training_snapshot_ids=("first",), calibration_snapshot_ids=(),
                           tuning_snapshot_ids=("corrected",), selection_snapshot_ids=(),
                           data_cutoff_utc=datetime(2023, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(WorkbenchError) as error:
        admitted_events(spec, root=None)
    assert error.value.code == "invalid_training"


def test_switch_compound_does_not_apply_continuation_calibration(tmp_path):
    from datetime import datetime, timezone
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.pace.hierarchical import forecast_stint
    from f1_strategy.replay import build_race_state
    from f1_strategy.synthetic import generate_snapshot

    ref = generate_snapshot(tmp_path, 8, target=True)
    state = build_race_state(open_snapshot(tmp_path, ref.snapshot_id), driver_number="44", completed_lap=10)
    class Model:
        data_cutoff_utc = datetime(2020, 1, 1, tzinfo=timezone.utc)
        lineage = ()
        n_parameter_replicates = 200
        support = {}
        arrays = {}
        metadata = {"calibration": {"0": {f"{h}:{kind}:{coverage}": {"adjustment_s": 20.}
                    for h in (1, 3) for kind in ("lap", "cumulative") for coverage in (80, 95)}}}
        def _variant(self, assumptions):
            return 0
        def predict_laps(self, frame, *, assumptions, replicate_index=None):
            return np.full(len(frame), 90.)
        def residual_blocks(self, compound, *, assumptions):
            return np.zeros((30, 3))
    driver = state.drivers["44"]
    switched = next(c for c in ("SOFT", "MEDIUM", "HARD") if c != driver.compound)
    forecast = forecast_stint(state, Model(), driver_number="44", compound=switched,
                              tyre_age_laps=0, horizon_laps=3, assumptions=PaceAssumptions())
    assert forecast.cumulative_intervals["1"]["calibration"] == "uncalibrated"
    np.testing.assert_allclose(forecast.interval80_s[:, 0], forecast.interval80_s[:, 1])

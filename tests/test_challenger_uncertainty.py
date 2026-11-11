from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from f1_strategy.contracts import PaceAssumptions, StintForecast, WorkbenchError
from f1_strategy.pace import challenger, hierarchical
from f1_strategy.pace.validation import _validate_holdout_lineage


def test_conditional_paths_future_stint_and_switch_calibration(monkeypatch):
    assumptions = PaceAssumptions()
    model = challenger.ChallengerPaceModel.__new__(challenger.ChallengerPaceModel)
    model.data_cutoff_utc = datetime(2022, 1, 1, tzinfo=timezone.utc)
    model.lineage = []
    model.base_model = object()
    model.arrays = {}
    model.metadata = {'mappings': {'0': {}}, 'calibration': {'0': {
        '1:lap:80': {'adjustment_s': 100.}, '1:lap:95': {'adjustment_s': 100.},
        '1:cumulative:80': {'adjustment_s': 100.}, '1:cumulative:95': {'adjustment_s': 100.}}}}
    model._variant = lambda _: 0
    prefix = pd.DataFrame({'lap_number': [8, 9, 10], 'source_stint': [1, 1, 1], 'start_gap_s': [3., 3., 3.]})
    model._history = lambda *args: ([1., 1., 1.], prefix)
    captured = []
    monkeypatch.setattr(challenger, '_feature_matrix', lambda row, mapping: captured.append(row.copy()) or np.zeros((1, 1)))
    monkeypatch.setattr(challenger, '_tree_predict', lambda arrays, name, x: np.array([{'mean': 0., 'q0': -4., 'q1': -2., 'q2': 2., 'q3': 4.}[name.split('_')[-1]]]))
    blocks = np.arange(-30., 30.).reshape(20, 3) / 10
    def base(state, model, *, driver_number, compound, race_laps, tyre_ages, assumptions, outlap_race_lap=None):
        n = len(race_laps)
        return StintForecast(driver_number, compound, np.asarray(race_laps), np.asarray(tyre_ages),
            np.full(n, 90.), np.full((500, n), 90.), blocks, np.zeros((n, 2)), np.zeros((n, 2)),
            {}, {}, assumptions, (), 0)
    monkeypatch.setattr(hierarchical, 'forecast_laps', base)
    state = SimpleNamespace(race_start_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
        drivers={'1': SimpleNamespace(compound='MEDIUM', completed_laps=10)},
        weather=pd.DataFrame({'release_ns': [0], 'track_temp_c': [30.]}),
        pit_events=pd.DataFrame({'driver_number': [], 'kind': [], 'release_ns': [], 'source_lap_number': []}),
        event_metadata={}, session_id='s', scheduled_laps=50)
    def forecast(compound, outlap=None):
        return model.forecast_laps(state, driver_number='1', compound=compound,
            race_laps=np.arange(11, 17), tyre_ages=np.arange(1, 7), assumptions=assumptions,
            outlap_race_lap=outlap)
    continued = forecast('MEDIUM')
    assert continued.ml_residual_mask.all()
    assert np.isfinite(continued.residual_quantiles_s).all()
    assert continued.interval80_s[0, 1] - continued.interval80_s[0, 0] > 200
    switched = forecast('HARD')
    assert not switched.ml_residual_mask[:3].any()
    assert switched.interval80_s[0, 1] - switched.interval80_s[0, 0] < 20
    assert switched.cumulative_intervals['1']['calibration'] == 'uncalibrated'
    captured.clear()
    future = forecast('HARD', outlap=11)
    assert not future.ml_residual_mask[:3].any()
    assert future.ml_residual_mask[3:].all()
    assert captured[0]['stint_age'].iloc[0] == 3
    assert future.cumulative_intervals['1']['calibration'] == 'uncalibrated'
    assert future.interval95_s[3, 1] - future.interval95_s[3, 0] > 4


def test_holdout_rejects_extra_2024_transitive_source():
    def event(identifier, partition, session, date):
        return dict(snapshot_id=identifier, partition=partition, session_id=session,
                    max_release_utc=date, race_start_utc=date)
    events = [event('dev', 'development', '2023-01-R', '2023-03-01T00:00:00+00:00'),
              event('cal', 'calibration', '2023-17-R', '2023-10-01T00:00:00+00:00'),
              event('future', 'holdout', '2024-01-R', '2024-03-01T00:00:00+00:00')]
    def source(e):
        return dict(snapshot_id=e['snapshot_id'], session_id=e['session_id'], synthetic=False,
                    max_used_utc=e['max_release_utc'])
    frozen = {'identity': {'selection_snapshot_ids': ['dev'], 'calibration_snapshot_ids': ['cal'],
                            'max_selection_utc': '2023-03-01T00:00:00+00:00'}}
    model = SimpleNamespace(lineage=[source(events[0]), source(events[1])], metadata={
        'data_cutoff_utc': '2024-03-01T00:00:00+00:00',
        'training_spec': {'training_snapshot_ids': ['dev'], 'calibration_snapshot_ids': ['cal']}})
    _validate_holdout_lineage(model, frozen, events)
    model.base_model = SimpleNamespace(lineage=[source(events[2])], metadata={
        'data_cutoff_utc': '2024-03-01T00:00:00+00:00', 'training_spec': {}})
    with pytest.raises(WorkbenchError, match='unpermitted'):
        _validate_holdout_lineage(model, frozen, events)

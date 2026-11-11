from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from f1_strategy.contracts import EvaluationSpec, WorkbenchError
from f1_strategy.pace.validation import champion_gate, metrics, paired_event_interval, run_backtest, same_stint_target, select_development, calibrate_predictions
from f1_strategy.pace.corpus import load_corpus, resolve_corpus


def test_corpus_missing_races_remain_in_denominator(tmp_path):
    corpus = load_corpus()
    rows = resolve_corpus(corpus, root=tmp_path)
    assert len([r for r in rows if r['partition'] == 'development']) == 16
    assert len([r for r in rows if r['partition'] == 'calibration']) == 6
    assert len([r for r in rows if r['partition'] == 'holdout']) == 24
    assert next(r for r in rows if r['session_id'] == '2023-07-R')['snapshot_id'] is None
    result = run_backtest(EvaluationSpec(partition='development'), root=tmp_path)
    score = json.loads(open(result['scorecard_path']).read())
    assert score['event_denominator'] == 16 and score['scored_events'] == 0
    assert result['status'] == 'unavailable'


def test_metric_macro_and_interval_math():
    rows = [dict(session_id='a', actual_s=10., prediction_s=12., cumulative_actual_s=20., cumulative_prediction_s=23., interval80_s=[9., 11.], interval95_s=[8., 12.]),
            dict(session_id='a', actual_s=10., prediction_s=12., cumulative_actual_s=20., cumulative_prediction_s=23., interval80_s=[9., 11.], interval95_s=[8., 12.]),
            dict(session_id='b', actual_s=10., prediction_s=14., cumulative_actual_s=20., cumulative_prediction_s=25., interval80_s=[11., 12.], interval95_s=[9., 11.])]
    m = metrics(rows)
    assert m['race_macro_mae_s'] == 3
    assert m['cumulative_mae_s'] == 4
    assert m['coverage80'] == .5 and m['coverage95'] == 1
    assert m['width80_s'] == 1.5 and m['width95_s'] == 3
    assert paired_event_interval({'a': 2., 'b': 4.}, {'a': 3., 'b': 5.}) == [1., 1.]


def test_champion_gate_requires_complete_development_evidence():
    baseline = {'race_macro_mae_s': 10., 'coverage80': .75}
    challenger = {'race_macro_mae_s': 9.5, 'coverage80': .76}
    assert champion_gate(baseline, challenger, {'SOFT': (10., 9.)}) == 'hgb'
    assert champion_gate(baseline, challenger, {}) == 'hierarchical'
    assert champion_gate(baseline, dict(challenger, coverage80=.7), {'SOFT': (10., 9.)}) == 'hierarchical'
    assert champion_gate(baseline, challenger, {'SOFT': (1., 2.), 'HARD': (1., 2.), 'MEDIUM': (2., 1.)}) == 'hierarchical'


def test_future_target_censoring_does_not_change_prior_decision():
    frame = pd.DataFrame({'driver_number': ['1'] * 4, 'lap_number': [10, 11, 12, 13],
                          'source_stint': [1, 1, 1, 2], 'eligible': [True] * 4})
    target, reason = same_stint_target(frame, driver_number='1', completed_lap=10, horizon=2)
    assert reason is None and target.lap_number.tolist() == [11, 12]
    assert same_stint_target(frame, driver_number='1', completed_lap=10, horizon=3)[1] == 'stint_end_censored'
    frame.loc[frame.lap_number == 12, 'eligible'] = False
    assert same_stint_target(frame, driver_number='1', completed_lap=10, horizon=2)[1] == 'future_lap_excluded'


def test_holdout_requires_valid_frozen_settings(tmp_path):
    with pytest.raises(WorkbenchError):
        run_backtest(EvaluationSpec(partition='holdout', frozen_settings_manifest='missing.json'), root=tmp_path)


def test_calibration_is_per_horizon_and_target():
    rows = [dict(session_id=str(i % 2), horizon=1, actual_s=10., cumulative_actual_s=20.,
                 interval80_s=[11., 12.], interval95_s=[9., 11.],
                 cumulative_interval80_s=[22., 23.], cumulative_interval95_s=[19., 21.])
            for i in range(30)]
    result = calibrate_predictions(rows)
    assert result['1:lap:80']['adjustment_s'] == 1.
    assert result['1:cumulative:80']['adjustment_s'] == 2.
    assert result['3:lap:80']['status'] == 'uncalibrated'


def test_selection_rejects_future_and_holdout():
    fold = dict(session_id='2023-01-R', race_start_utc='2023-03-05T15:00:00+00:00',
                fit_max_utc='2023-03-05T15:00:00+00:00', calibration_max_utc='2022-12-01T00:00:00+00:00',
                hierarchical=[], challengers={})
    with pytest.raises(WorkbenchError) as error:
        select_development([fold])
    assert error.value.code == "future_artifact"
    fold['session_id'] = '2024-01-R'
    with pytest.raises(WorkbenchError) as error:
        select_development([fold])
    assert error.value.code == "future_artifact"


def test_selection_uses_dated_development_fold_and_simple_tie():
    def row(pred):
        return dict(session_id='2023-01-R', driver_number='1', completed_lap=10, horizon=1,
                    compound='SOFT', actual_s=10., prediction_s=pred,
                    cumulative_actual_s=10., cumulative_prediction_s=pred,
                    interval80_s=[9., 11.], interval95_s=[8., 12.])
    fold = dict(session_id='2023-01-R', race_start_utc='2023-03-05T15:00:00+00:00',
                fit_max_utc='2022-10-01T00:00:00+00:00',
                calibration_max_utc='2022-12-01T00:00:00+00:00', hierarchical=[row(12.)],
                challengers={(l, m): [row(11.)] for l in (7, 15) for m in (30, 60)})
    selected = select_development([fold])
    assert selected['champion'] == 'hgb'
    assert selected['hgb_settings'] == {'max_leaf_nodes': 7, 'min_samples_leaf': 60}


def test_challenger_base_excludes_calibration_and_precedes_its_first_race():
    from types import SimpleNamespace
    from f1_strategy.contracts import TrainingSpec
    from f1_strategy.pace.challenger import _base_training_spec

    fit_ids = tuple(f'{n:064x}' for n in range(1, 7))
    calibration_ids = (f'{7:064x}', f'{8:064x}')
    cutoff = datetime(2023, 1, 1, tzinfo=timezone.utc)
    spec = TrainingSpec(training_snapshot_ids=fit_ids, calibration_snapshot_ids=calibration_ids,
                        data_cutoff_utc=cutoff, model_kind='hgb')
    starts = (datetime(2022, 10, 1, tzinfo=timezone.utc), datetime(2022, 11, 1, tzinfo=timezone.utc))
    admitted = {identifier: {'snapshot': SimpleNamespace(manifest=SimpleNamespace(
        stream_origin_utc=start, race_start_ns=0))}
        for identifier, start in zip(calibration_ids, starts)}
    base = _base_training_spec(spec, admitted)
    assert base.training_snapshot_ids == fit_ids
    assert base.calibration_snapshot_ids == ()
    assert base.data_cutoff_utc == starts[0]
    assert spec.calibration_snapshot_ids == calibration_ids and spec.data_cutoff_utc == cutoff


def test_challenger_forecast_checks_challenger_dates_not_only_base():
    from types import SimpleNamespace
    from f1_strategy.contracts import PaceAssumptions
    from f1_strategy.pace.challenger import ChallengerPaceModel

    model = ChallengerPaceModel.__new__(ChallengerPaceModel)
    model.data_cutoff_utc = datetime(2023, 1, 1, tzinfo=timezone.utc)
    model.lineage = []
    state = SimpleNamespace(race_start_utc=datetime(2022, 11, 1, tzinfo=timezone.utc))
    with pytest.raises(WorkbenchError) as error:
        model.forecast_laps(state, driver_number='1', compound='MEDIUM', race_laps=[11],
                            tyre_ages=[11], assumptions=PaceAssumptions())
    assert error.value.code == 'future_artifact'


def test_hgb_numeric_tree_round_trip_matches_estimator(tmp_path):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from f1_strategy.pace.challenger import _save_estimator, _tree_predict

    x = np.array([[float(i), float(i % 3)] for i in range(60)])
    x[::7, 1] = np.nan
    y = np.sin(x[:, 0] / 5.) + np.nan_to_num(x[:, 1], nan=-1.)
    estimator = HistGradientBoostingRegressor(max_iter=12, max_leaf_nodes=7,
                                               min_samples_leaf=3, early_stopping=False,
                                               random_state=2026).fit(x, y)
    arrays = {}
    _save_estimator(estimator, arrays, "tree")
    path = tmp_path / "trees.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        restored = {name: archive[name] for name in archive.files}
    probe = np.array([[0., np.nan], [10., 1.], [33.5, 2.], [70., np.nan]])
    np.testing.assert_allclose(_tree_predict(restored, "tree", probe), estimator.predict(probe),
                               rtol=0., atol=1e-12)

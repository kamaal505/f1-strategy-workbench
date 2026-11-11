"""Chronological, issued-forecast residual observations for ML development."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..contracts import WorkbenchError
from ..replay import build_race_state
from .artifacts import load_model
from .hierarchical import fit_pace, forecast_stint
from .training import admitted_events


_SKIPPABLE = frozenset(("missing_focal_completion", "generated_focal_completion", "race_not_started",
                        "focal_in_pit", "insufficient_evidence", "invalid_input"))


def issued_residual_rows(training, *, root, assumptions, _base_models=None):
    """Build same-stint next-lap residuals against forecasts issued at the prior crossing.

    Each event uses a separate hierarchical model fitted solely on events whose
    latest used observation precedes its race start.  The target lap is inspected
    only after issuing its forecast, and never supplies its covariates.  Rows
    include their base model and source identities so the caller can audit the
    event-chronological folds.  Events with fewer than five prior fit events do
    not supply ML observations.
    """
    admitted = admitted_events(training, root=root)
    if _base_models is None:
        _base_models = {}
    sources = []
    for identifier in training.training_snapshot_ids:
        item = admitted[identifier]
        manifest = item["snapshot"].manifest
        start = manifest.stream_origin_utc + timedelta(microseconds=manifest.race_start_ns / 1000)
        end = datetime.fromisoformat(item["lineage"]["max_used_utc"])
        sources.append((start, end, identifier, item))
    rows = []
    for start, _, identifier, item in sorted(sources, key=lambda value: (value[0], value[2])):
        earlier = [source for source in sources if source[1] < start]
        if len(earlier) < 5:
            continue
        earlier_ids = tuple(source[2] for source in earlier)
        if identifier not in _base_models:
            base_spec = training.model_copy(update={"model_kind": "hierarchical", "hyperparameters": {},
                                                   "training_snapshot_ids": earlier_ids,
                                                   "calibration_snapshot_ids": (), "tuning_snapshot_ids": (),
                                                   "selection_snapshot_ids": (), "data_cutoff_utc": start})
            reference = fit_pace(base_spec, root=root)
            base = load_model(root, reference.model_id, target_start_utc=start)
            _base_models[identifier] = (reference, base)
        reference, base = _base_models[identifier]
        frame = item["features"].query("eligible").sort_values(["driver_number", "lap_number"], kind="stable")
        for number, driver_rows in frame.groupby("driver_number", sort=True):
            driver_rows = driver_rows.reset_index(drop=True)
            history = []
            previous_stint = None
            for index in range(1, len(driver_rows)):
                previous, target = driver_rows.iloc[index - 1], driver_rows.iloc[index]
                stint = str(previous.stint_id)
                if stint != previous_stint:
                    history = []
                    previous_stint = stint
                if (int(target.lap_number) != int(previous.lap_number) + 1
                        or str(target.stint_id) != stint or target.compound != previous.compound):
                    history = []
                    continue
                try:
                    state = build_race_state(item["snapshot"], driver_number=str(number),
                                             completed_lap=int(previous.lap_number))
                    driver = state.drivers[str(number)]
                    if driver.compound != previous.compound or driver.tyre_age_laps is None:
                        history = []
                        continue
                    forecast = forecast_stint(state, base, driver_number=str(number), compound=driver.compound,
                                              tyre_age_laps=int(driver.tyre_age_laps), horizon_laps=1,
                                              assumptions=assumptions)
                except WorkbenchError as error:
                    if error.code not in _SKIPPABLE:
                        raise
                    history = []
                    continue
                value = float(target.lap_time_s) - float(forecast.mean_s[0])
                rows.append({"snapshot_id": identifier, "session_id": item["snapshot"].manifest.session_id,
                             "driver_number": str(number), "stint_id": stint,
                             "lap_number": int(previous.lap_number) + 1,
                             "compound": driver.compound, "spec_era_id": str(previous.spec_era_id),
                             "layout_id": str(previous.layout_id), "tyre_age_laps": int(driver.tyre_age_laps) + 1,
                             "stint_age": int(previous.lap_number) - int(driver_rows.loc[driver_rows.stint_id == previous.stint_id, "lap_number"].min()) + 1,
                             "track_temperature_c": float(previous.track_temperature_c),
                             "start_gap_s": float(previous.start_gap_s), "history_count": len(history),
                             "residual_mean_3": float(np.mean(history[-3:])) if history else np.nan,
                             "residual_mean_5": float(np.mean(history[-5:])) if history else np.nan,
                             "warmup": bool(target.warmup_first or target.warmup_second),
                             "residual_s": value, "base_model_id": reference.model_id,
                             "base_training_snapshot_ids": list(earlier_ids), "base_cutoff_utc": start.isoformat()})
                history.append(value)
    return pd.DataFrame(rows)


_CATEGORY = ("compound", "spec_era_id", "layout_id")
_NUMERIC = ("tyre_age_laps", "stint_age", "lap_fraction", "track_temperature_c", "start_gap_s",
            "temperature_missing", "gap_missing", "residual_mean_3", "residual_mean_5", "history_count")
_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")
_QUANTILES = (.025, .1, .9, .975)


def _feature_matrix(frame, mappings):
    columns = []
    for name in _CATEGORY:
        mapping = mappings[name]
        columns.append(np.asarray([mapping.get(str(value), -1) for value in frame[name]], dtype=float))
    for name in _NUMERIC:
        columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
    return np.column_stack(columns)


def _save_estimator(estimator, arrays, name):
    offsets, nodes = [0], []
    for predictors in estimator._predictors:
        records = predictors[0].nodes
        if np.any(records["is_categorical"]):
            raise WorkbenchError("invalid_model", "Unsupported categorical tree split")
        nodes.append(np.column_stack([records[field] for field in _FIELDS]).astype(float))
        offsets.append(offsets[-1] + len(records))
    arrays[name + "_nodes"] = np.concatenate(nodes)
    arrays[name + "_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays[name + "_baseline"] = np.asarray(estimator._baseline_prediction, dtype=float).reshape(1)


def _tree_predict(arrays, name, x):
    nodes, offsets = arrays[name + "_nodes"], arrays[name + "_offsets"]
    result = np.full(len(x), arrays[name + "_baseline"][0])
    rows = np.arange(len(x))
    for start, end in zip(offsets[:-1], offsets[1:]):
        tree = nodes[start:end]
        indices = np.zeros(len(x), dtype=np.int64)
        active = np.ones(len(x), dtype=bool)
        while active.any():
            selected = rows[active]
            current = tree[indices[selected]]
            leaves = current[:, 6] == 1
            done = selected[leaves]
            result[done] += current[leaves, 0]
            active[done] = False
            selected = selected[~leaves]
            if len(selected):
                current = current[~leaves]
                values = x[selected, current[:, 1].astype(np.int64)]
                left = np.where(np.isnan(values), current[:, 3] == 1, values <= current[:, 2])
                indices[selected] = np.where(left, current[:, 4], current[:, 5]).astype(np.int64)
    return result


def _base_training_spec(training, admitted):
    """Keep the base fit and its information cutoff before ML calibration."""
    starts = [admitted[identifier]["snapshot"].manifest.stream_origin_utc
              + timedelta(microseconds=admitted[identifier]["snapshot"].manifest.race_start_ns / 1000)
              for identifier in training.calibration_snapshot_ids]
    cutoff = min(starts, default=training.data_cutoff_utc)
    return training.model_copy(update={"model_kind": "hierarchical", "hyperparameters": {},
                                       "calibration_snapshot_ids": (), "data_cutoff_utc": cutoff})


def fit_challenger(training, *, root):
    """Fit per-assumption corrections to chronological issued base forecasts."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from ..contracts import ModelRef
    from ..identity import lock_digest
    from .bundle import publish_bundle

    if training.model_kind != "hgb":
        raise WorkbenchError("invalid_training", "Challenger requires model_kind hgb")
    allowed = {"max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization"}
    if set(training.hyperparameters) - allowed:
        raise WorkbenchError("invalid_training", "Unsupported HGB hyperparameter")
    parameters = {"learning_rate": .05, "max_iter": 200, "max_leaf_nodes": 15,
                  "min_samples_leaf": 50, "l2_regularization": 1.}
    parameters.update(training.hyperparameters)
    for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        value = parameters[key]
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < (2 if key == "max_leaf_nodes" else 1):
            raise WorkbenchError("invalid_training", "Invalid HGB integer hyperparameter")
        parameters[key] = int(value)
    if (not np.isfinite(parameters["learning_rate"]) or parameters["learning_rate"] <= 0
            or not np.isfinite(parameters["l2_regularization"]) or parameters["l2_regularization"] < 0):
        raise WorkbenchError("invalid_training", "Invalid HGB numeric hyperparameter")
    admitted = admitted_events(training, root=root)
    durations = {identifier: int(item["snapshot"].manifest.scheduled_laps)
                 for identifier, item in admitted.items() if not item["features"].empty}
    base_spec = _base_training_spec(training, admitted)
    base_cutoff = base_spec.data_cutoff_utc
    base_reference = fit_pace(base_spec, root=root)
    base = load_model(root, base_reference.model_id)
    arrays, indices, differences, mappings, row_lineage = {}, [], [], {}, {}
    base_models = {}
    for assumptions in training.assumption_grid:
        gain = round(assumptions.fuel_gain_s_per_lap - assumptions.track_drift_s_per_lap, 12)
        variant = len(differences)
        differences.append(gain)
        indices.append(variant)
        rows = issued_residual_rows(training, root=root, assumptions=assumptions, _base_models=base_models)
        if rows.empty:
            raise WorkbenchError("insufficient_evidence", "No chronological issued-forecast residuals are available")
        rows = rows[(rows.history_count >= 3) & ~rows.warmup].copy()
        if rows.empty:
            raise WorkbenchError("insufficient_evidence", "No chronological residuals have sufficient preceding history")
        rows["lap_fraction"] = [lap / durations[snapshot] for lap, snapshot in zip(rows.lap_number, rows.snapshot_id)]
        rows["temperature_missing"] = rows.track_temperature_c.isna().astype(float)
        rows["gap_missing"] = rows.start_gap_s.isna().astype(float)
        mapping = {name: {value: i for i, value in enumerate(sorted(rows[name].astype(str).unique()))}
                   for name in _CATEGORY}
        mappings[str(variant)] = mapping
        x, y = _feature_matrix(rows, mapping), rows.residual_s.to_numpy(dtype=float)
        for label, loss, quantile in [("mean", "squared_error", None),
                                      *((f"q{i}", "quantile", q) for i, q in enumerate(_QUANTILES))]:
            estimator = HistGradientBoostingRegressor(loss=loss, quantile=quantile, early_stopping=False,
                                                      random_state=training.seed, **parameters)
            estimator.fit(x, y)
            _save_estimator(estimator, arrays, f"v{variant}_{label}")
        row_lineage[str(variant)] = rows[["snapshot_id", "lap_number", "driver_number", "stint_id",
                                          "base_model_id", "base_training_snapshot_ids", "base_cutoff_utc"]].to_dict("records")
    metadata = {"model_kind": "hgb", "base_model_id": base_reference.model_id,
                "data_cutoff_utc": training.data_cutoff_utc.isoformat(),
                "assumption_grid": [(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in training.assumption_grid],
                "variant_indices": indices, "mappings": mappings, "training_spec": training.model_dump(mode="json"),
                "lineage": [item["lineage"] for item in admitted.values()], "row_lineage": row_lineage, "seed": training.seed,
                "hyperparameters": parameters, "lock_digest": lock_digest(),
                "ml_parameter_uncertainty": "not_bootstrapped", "calibration": {}}
    from .validation import calibrate_predictions, same_stint_target
    calibration_metadata = {**metadata, "data_cutoff_utc": base_cutoff.isoformat(),
                            "lineage": base.lineage}
    challenger = ChallengerPaceModel("", calibration_metadata, arrays, base, root)
    for variant in sorted(set(indices)):
        assumptions = training.assumption_grid[indices.index(variant)]
        scores = []
        for identifier in training.calibration_snapshot_ids:
            item = admitted[identifier]
            snapshot, target_rows = item["snapshot"], item["features"]
            manifest = snapshot.manifest
            for driver_number in sorted(target_rows.driver_number.dropna().astype(str).unique()):
                for lap in range(10, manifest.scheduled_laps - 4, 5):
                    try:
                        state = build_race_state(snapshot, driver_number=driver_number, completed_lap=lap)
                        driver = state.drivers[driver_number]
                        if driver.compound not in ("SOFT", "MEDIUM", "HARD") or driver.tyre_age_laps is None:
                            continue
                        laps = np.arange(lap + 1, lap + 11, dtype=np.int64)
                        ages = driver.tyre_age_laps + np.arange(1, 11, dtype=np.int64)
                        forecast = challenger.forecast_laps(state, driver_number=driver_number,
                                                            compound=driver.compound, race_laps=laps,
                                                            tyre_ages=ages, assumptions=assumptions)
                    except WorkbenchError as error:
                        if error.code not in _SKIPPABLE:
                            raise
                        continue
                    for horizon in (1, 3, 5, 10):
                        target, reason = same_stint_target(target_rows, driver_number=driver_number,
                                                           completed_lap=lap, horizon=horizon)
                        if reason:
                            continue
                        actual = target.lap_time_s.to_numpy(dtype=float)
                        cumulative = forecast.cumulative_intervals[str(horizon)]
                        scores.append({"session_id": manifest.session_id, "horizon": horizon,
                                       "actual_s": float(actual[-1]), "cumulative_actual_s": float(actual.sum()),
                                       "interval80_s": forecast.interval80_s[horizon - 1].tolist(),
                                       "interval95_s": forecast.interval95_s[horizon - 1].tolist(),
                                       "cumulative_interval80_s": cumulative["interval80_s"],
                                       "cumulative_interval95_s": cumulative["interval95_s"]})
        metadata["calibration"][str(variant)] = calibrate_predictions(scores)
    identifier, path = publish_bundle(root, metadata, arrays)
    return ModelRef(model_id=identifier, path=path, model_kind="hgb")


class ChallengerPaceModel:
    def __init__(self, model_id, metadata, arrays, base_model, root):
        from copy import copy
        self.model_id, self.metadata, self.arrays = model_id, metadata, arrays
        self.base_model, self.root = copy(base_model), root
        self.base_model.metadata = {**base_model.metadata, "calibration": {}}
        self.data_cutoff_utc = datetime.fromisoformat(metadata["data_cutoff_utc"])
        self.lineage = metadata["lineage"]
        self.assumption_grid = tuple(tuple(pair) for pair in metadata["assumption_grid"])
        self.feature_names = _CATEGORY + _NUMERIC
        self.support = base_model.support
        self.n_parameter_replicates = base_model.n_parameter_replicates

    def _variant(self, assumptions):
        pair = (assumptions.fuel_gain_s_per_lap, assumptions.track_drift_s_per_lap)
        for index, candidate in enumerate(self.assumption_grid):
            if all(abs(a - b) < 1e-12 for a, b in zip(pair, candidate)):
                return self.metadata["variant_indices"][index]
        raise WorkbenchError("missing_model_variant", "Pace model does not contain the requested assumptions")

    def predict_laps(self, features, *, assumptions, replicate_index=None):
        base = self.base_model.predict_laps(features, assumptions=assumptions, replicate_index=replicate_index)
        if not set(self.feature_names).issubset(features.columns):
            raise WorkbenchError("insufficient_evidence", "Challenger requires issued-residual history features")
        variant = self._variant(assumptions)
        x = _feature_matrix(features, self.metadata["mappings"][str(variant)])
        return base + _tree_predict(self.arrays, f"v{variant}_mean", x)

    def residual_blocks(self, compound, *, assumptions):
        return self.base_model.residual_blocks(compound, assumptions=assumptions)

    def _history(self, state, driver_number, assumptions):
        from ..data.snapshots import open_snapshot
        from .features import eligible_laps

        prefix = eligible_laps(state.laps, state.pit_events, state.track_status, state.weather,
                               cutoff_ns=state.cutoff_ns, event_metadata=state.event_metadata,
                               session_id=state.session_id)
        prefix = prefix[(prefix.driver_number == driver_number) & prefix.eligible].sort_values("lap_number")
        snapshot = open_snapshot(self.root, state.snapshot_id)
        history, previous = [], None
        for row in prefix.itertuples():
            if previous is None or row.source_stint != previous.source_stint or row.lap_number != previous.lap_number + 1:
                history, previous = [], row
                continue
            try:
                prior = build_race_state(snapshot, driver_number=driver_number, completed_lap=int(previous.lap_number))
                driver = prior.drivers[driver_number]
                if driver.compound != previous.compound or driver.tyre_age_laps is None:
                    history, previous = [], row
                    continue
                issued = forecast_stint(prior, self.base_model, driver_number=driver_number,
                                        compound=driver.compound, tyre_age_laps=driver.tyre_age_laps,
                                        horizon_laps=1, assumptions=assumptions)
            except WorkbenchError as error:
                if error.code not in _SKIPPABLE:
                    raise
                history, previous = [], row
                continue
            history.append(float(row.lap_time_s) - float(issued.mean_s[0]))
            previous = row
        if previous is None or int(previous.lap_number) != state.drivers[driver_number].completed_laps:
            history = []
        return history, prefix

    def forecast_laps(self, state, *, driver_number, compound, race_laps, tyre_ages, assumptions,
                      outlap_race_lap=None):
        from dataclasses import replace
        from .hierarchical import forecast_laps
        from .uncertainty import map_residual_draws
        if self.data_cutoff_utc > state.race_start_utc:
            raise WorkbenchError("future_artifact", "Model cutoff is after target race start")
        if any(datetime.fromisoformat(source["max_used_utc"]) >= state.race_start_utc
               for source in self.lineage):
            raise WorkbenchError("future_artifact", "Model lineage includes data not preceding the target race")

        base = forecast_laps(state, self.base_model, driver_number=driver_number, compound=compound,
                             race_laps=race_laps, tyre_ages=tyre_ages, assumptions=assumptions,
                             outlap_race_lap=outlap_race_lap)
        variant = self._variant(assumptions)
        history, prefix = self._history(state, driver_number, assumptions)
        driver = state.drivers[driver_number]
        if outlap_race_lap is not None or driver.compound != compound:
            history = []
        start_stint = int(prefix.lap_number.iloc[0]) if len(prefix) else driver.completed_laps
        if len(prefix) and "source_stint" in prefix:
            current = prefix[prefix.source_stint == prefix.source_stint.iloc[-1]]
            if len(current):
                start_stint = int(current.lap_number.iloc[0])
        if outlap_race_lap is not None:
            start_stint = int(outlap_race_lap)
        weather = state.weather.sort_values("release_ns")
        temperatures = weather.track_temp_c.dropna()
        temperature = assumptions.track_temperature_c
        if temperature is None:
            temperature = float(temperatures.iloc[-1]) if len(temperatures) else np.nan
        gap = float(prefix.start_gap_s.iloc[-1]) if len(prefix) else np.nan
        era = state.event_metadata.get("spec_era_by_compound", {}).get(compound) or f"unknown:{state.session_id}:{compound}"
        layout = state.event_metadata.get("layout_id") or f"unknown:{state.session_id}"
        scheduled = state.scheduled_laps
        corrections = np.zeros(len(base.race_laps))
        ml_used = np.zeros(len(base.race_laps), dtype=bool)
        quantiles = np.zeros((len(base.race_laps), 4))
        current_outlap = outlap_race_lap
        if current_outlap is None:
            exits = state.pit_events
            exits = exits[(exits.driver_number == driver_number) & (exits.kind == "exit")].sort_values("release_ns")
            if len(exits) and pd.notna(exits.source_lap_number.iloc[-1]):
                current_outlap = int(exits.source_lap_number.iloc[-1])
        for index, (lap, age) in enumerate(zip(base.race_laps, base.tyre_ages)):
            warmup = current_outlap is not None and lap <= current_outlap + 2
            if len(history) >= 3 and not warmup:
                row = pd.DataFrame([{"compound": compound, "spec_era_id": era, "layout_id": layout,
                                     "tyre_age_laps": age, "stint_age": lap - start_stint,
                                     "lap_fraction": lap / scheduled, "track_temperature_c": temperature,
                                     "start_gap_s": gap, "temperature_missing": float(not np.isfinite(temperature)),
                                     "gap_missing": float(not np.isfinite(gap)),
                                     "residual_mean_3": np.mean(history[-3:]),
                                     "residual_mean_5": np.mean(history[-5:]), "history_count": len(history)}])
                x = _feature_matrix(row, self.metadata["mappings"][str(variant)])
                corrections[index] = _tree_predict(self.arrays, f"v{variant}_mean", x)[0]
                quantiles[index] = sorted(_tree_predict(self.arrays, f"v{variant}_q{i}", x)[0] for i in range(4))
                ml_used[index] = True
            history.append(float(corrections[index]))
        mean = base.mean_s + corrections
        parameters = base.parameter_mean_s + corrections[None, :]
        rng = np.random.default_rng(2026)
        residuals = np.zeros_like(parameters)
        if len(base.residual_blocks_s):
            for start in range(0, len(mean), 3):
                chosen = base.residual_blocks_s[rng.integers(len(base.residual_blocks_s), size=len(parameters))]
                length = min(3, len(mean) - start)
                residuals[:, start:start + length] = chosen[:, :length]
        paths = parameters + map_residual_draws(residuals, base.residual_blocks_s, quantiles, ml_used)
        interval80 = np.quantile(paths, (.1, .9), axis=0).T
        interval95 = np.quantile(paths, (.025, .975), axis=0).T
        cumulative = {}
        for horizon in (1, 3, 5, 10):
            if horizon <= len(mean):
                totals = paths[:, :horizon].sum(axis=1)
                cumulative[str(horizon)] = {"interval80_s": np.quantile(totals, (.1, .9)).tolist(),
                                            "interval95_s": np.quantile(totals, (.025, .975)).tolist(),
                                            "calibration": "uncalibrated"}
        calibration = self.metadata.get("calibration", {}).get(str(variant), {})
        if (outlap_race_lap is None and compound == driver.compound
                and base.race_laps[0] == driver.completed_laps + 1 and np.all(np.diff(base.race_laps) == 1)):
            for horizon in (1, 3, 5, 10):
                if str(horizon) not in cumulative:
                    continue
                calibrated = True
                for coverage, interval in ((80, interval80), (95, interval95)):
                    adjustment = calibration.get(f"{horizon}:lap:{coverage}", {}).get("adjustment_s")
                    if adjustment is not None:
                        interval[horizon - 1] += [-adjustment, adjustment]
                    record = calibration.get(f"{horizon}:cumulative:{coverage}", {})
                    adjustment = record.get("adjustment_s")
                    if adjustment is None:
                        calibrated = False
                    else:
                        bounds = np.asarray(cumulative[str(horizon)][f"interval{coverage}_s"])
                        cumulative[str(horizon)][f"interval{coverage}_s"] = (bounds + [-adjustment, adjustment]).tolist()
                cumulative[str(horizon)]["calibration"] = "calibrated" if calibrated else "uncalibrated"
        flags = tuple(dict.fromkeys((*base.flags, "ml_parameter_uncertainty_unbootstrapped",
                                     "ml_piecewise_quantile_linear_tails_zero_centered", "uncalibrated")))
        return replace(base, mean_s=mean, parameter_mean_s=parameters, interval80_s=interval80,
                       interval95_s=interval95, cumulative_intervals=cumulative, flags=flags,
                       residual_quantiles_s=quantiles, ml_residual_mask=ml_used)


def load_challenger(root, id, *, target_start_utc=None):
    from .bundle import read_bundle
    from .artifacts import _inspect_for_load

    _inspect_for_load(root, id, target_start_utc)

    metadata, arrays = read_bundle(root, id)
    try:
        if metadata["model_kind"] != "hgb":
            raise ValueError("wrong model kind")
        cutoff = datetime.fromisoformat(metadata["data_cutoff_utc"])
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("naive cutoff")
        if target_start_utc is not None:
            if target_start_utc.tzinfo is None or target_start_utc.utcoffset() is None:
                raise WorkbenchError("invalid_input", "Target start must have a timezone")
            if cutoff > target_start_utc:
                raise WorkbenchError("future_artifact", "Model cutoff is after target race start")
        lineage = metadata["lineage"]
        by_id = {entry["snapshot_id"]: entry for entry in lineage}
        if len(by_id) != len(lineage):
            raise ValueError("duplicate lineage")
        for source in lineage:
            used = datetime.fromisoformat(source["max_used_utc"])
            if used.tzinfo is None or used >= cutoff:
                raise ValueError("invalid lineage cutoff")
            if target_start_utc is not None and used >= target_start_utc:
                raise WorkbenchError("future_artifact", "Model lineage includes data not preceding target race")
        base = load_model(root, metadata["base_model_id"], target_start_utc=target_start_utc)
        training = metadata["training_spec"]
        base_roles = {"fit": set(training["training_snapshot_ids"]),
                      "tuning": set(training["tuning_snapshot_ids"]),
                      "selection": set(training["selection_snapshot_ids"])}
        if base.data_cutoff_utc > cutoff:
            raise ValueError("base provenance mismatch")
        for source in base.lineage:
            identifier = source["snapshot_id"]
            expected_roles = {role for role, identifiers in base_roles.items() if identifier in identifiers}
            if (not expected_roles or set(source.get("roles", ())) != expected_roles
                    or {key: value for key, value in source.items() if key != "roles"} != by_id.get(identifier)):
                raise ValueError("base provenance mismatch")
        calibration_ids = set(training["calibration_snapshot_ids"])
        if calibration_ids & {source["snapshot_id"] for source in base.lineage}:
            raise ValueError("base includes challenger calibration")
        if not calibration_ids <= set(by_id):
            raise ValueError("missing calibration lineage")
        if len(metadata["assumption_grid"]) != len(metadata["variant_indices"]):
            raise ValueError("variant count")
        for variant in set(metadata["variant_indices"]):
            mapping = metadata["mappings"][str(variant)]
            if any(not isinstance(mapping[name], dict) for name in _CATEGORY):
                raise ValueError("invalid mappings")
            for label in ("mean", "q0", "q1", "q2", "q3"):
                name = f"v{variant}_{label}"
                nodes, offsets, baseline = (arrays[name + suffix] for suffix in ("_nodes", "_offsets", "_baseline"))
                if (nodes.ndim != 2 or nodes.shape[1] != len(_FIELDS) or offsets.ndim != 1 or len(offsets) < 2
                        or offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != len(nodes)
                        or np.any(np.diff(offsets) <= 0) or baseline.shape != (1,)):
                    raise ValueError("invalid tree shape")
                for start, end in zip(offsets[:-1], offsets[1:]):
                    tree = nodes[start:end]
                    leaves = tree[:, 6]
                    inner = tree[leaves == 0]
                    if (np.any((leaves != 0) & (leaves != 1)) or np.any(inner[:, 1] != np.floor(inner[:, 1]))
                            or np.any(inner[:, 1] < 0) or np.any(inner[:, 1] >= len(_CATEGORY) + len(_NUMERIC))
                            or np.any((inner[:, 3] != 0) & (inner[:, 3] != 1))):
                        raise ValueError("invalid tree node")
                    for column in (4, 5):
                        children = inner[:, column]
                        if (np.any(children != np.floor(children)) or np.any(children <= np.flatnonzero(leaves == 0))
                                or np.any(children >= len(tree))):
                            raise ValueError("invalid tree links")
        return ChallengerPaceModel(id, metadata, arrays, base, root)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError):
        raise WorkbenchError("corrupt_artifact", "Invalid challenger metadata or numeric arrays") from None

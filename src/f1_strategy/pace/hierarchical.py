"""Partially pooled, assumption-conditional robust pace estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import ModelRef, WorkbenchError
from ..contracts import StintForecast
from .features import eligible_laps
from ..identity import lock_digest
from .bootstrap import bootstrap_fits
from .bundle import publish_bundle
from .design import design_matrix, design_spec, fit_robust
from .heldout import calibrate_events, select_residual_pool
from .out_of_event import event_held_out_residuals
from .residuals import build_residual_blocks
from .selection import select_penalty
from .training import admitted_events


def fit_pace(training, *, root):
    if training.model_kind == "hgb":
        from .challenger import fit_challenger
        return fit_challenger(training, root=root)
    if training.model_kind == "rolling":
        if training.hyperparameters:
            raise WorkbenchError("invalid_training", "Rolling baseline has no estimator hyperparameters")
        admitted = admitted_events(training, root=root)
        metadata = {"model_kind": "rolling", "data_cutoff_utc": training.data_cutoff_utc.isoformat(),
                    "assumption_grid": [(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in training.assumption_grid],
                    "lineage": [{**v["lineage"], "roles": v["roles"]} for v in admitted.values()],
                    "lock_digest": lock_digest()}
        model_id, path = publish_bundle(root, metadata, {})
        return ModelRef(model_id=model_id, path=path, model_kind="rolling")
    if training.model_kind != "hierarchical":
        raise WorkbenchError("unsupported_model", "This fitter supports hierarchical and rolling models only")
    hp = training.hyperparameters
    if set(hp) - {"penalty_multiplier", "n_bootstraps"} or hp.get("n_bootstraps", 200) != 200:
        raise WorkbenchError("invalid_training", "Hierarchical fitting requires exactly 200 bootstrap replicates")
    multiplier = float(hp.get("penalty_multiplier", 1.))
    admitted = admitted_events(training, root=root)
    events = [admitted[id]["features"].query("eligible").copy() for id in training.training_snapshot_ids]
    if len(events) < 5 or any(event.empty for event in events):
        raise WorkbenchError("insufficient_evidence", "At least five earlier events with eligible laps are required")
    selection = {"status": "fixed", "folds": 0}
    if "penalty_multiplier" not in hp:
        from datetime import timedelta
        source = [(admitted[identifier]["lineage"]["max_used_utc"], event)
                  for identifier, event in zip(training.training_snapshot_ids, events)]
        tuning = []
        for identifier in training.tuning_snapshot_ids:
            event = admitted[identifier]
            manifest = event["snapshot"].manifest
            start = manifest.stream_origin_utc + timedelta(microseconds=manifest.race_start_ns / 1000)
            tuning.append((start.isoformat(), event["features"].query("eligible")))
        multiplier, selection = select_penalty(source, tuning, net_gain=.06)
    frame = pd.concat(events, ignore_index=True)
    design = design_spec(frame, multiplier)
    _, feature_names, _ = design_matrix(frame, design)
    arrays = {}
    assumption_grid = [(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in training.assumption_grid]
    differences = []
    variant_indices = []
    pooling = {}
    calibration = {}
    era_indices = {era: index for index, era in enumerate(sorted(frame.spec_era_id.unique()))}
    for gain, drift in assumption_grid:
        difference = round(gain - drift, 12)
        if difference in differences:
            variant_indices.append(differences.index(difference))
            continue
        variant = len(differences)
        differences.append(difference)
        variant_indices.append(variant)
        intercept, coefficients, _ = fit_robust(frame, design, difference)
        arrays[f"fit_{variant}"] = np.concatenate(([intercept], coefficients))
        arrays[f"bootstrap_{variant}"] = bootstrap_fits(events, design, difference, seed=training.seed)
        residuals = event_held_out_residuals([admitted[id] for id in training.training_snapshot_ids], difference, multiplier)
        for compound in ("SOFT", "MEDIUM", "HARD"):
            # The public protocol takes compound only. Spec-specific selection is
            # performed by the forecast path when a spec is available.
            blocks, level = build_residual_blocks(residuals, spec_era_id="", compound=compound)
            arrays[f"residual_{variant}_{compound}"] = blocks
            pooling[f"{variant}:{compound}"] = level
            for era in sorted(residuals.loc[residuals.compound == compound, "spec_era_id"].unique()):
                spec_blocks, spec_level = build_residual_blocks(residuals, spec_era_id=era, compound=compound)
                arrays[f"spec_residual_{variant}_{era_indices[era]}_{compound}"] = spec_blocks
                pooling[f"{variant}:{era}:{compound}"] = spec_level
        calibration[str(variant)] = calibrate_events(
            [admitted[id] for id in training.calibration_snapshot_ids], design, arrays[f"fit_{variant}"], arrays[f"bootstrap_{variant}"],
            {c: arrays[f"residual_{variant}_{c}"] for c in ("SOFT", "MEDIUM", "HARD")},
            difference, seed=training.seed,
            spec_pools={(era, compound): arrays[f"spec_residual_{variant}_{index}_{compound}"]
                        for era, index in era_indices.items() for compound in ("SOFT", "MEDIUM", "HARD")
                        if f"spec_residual_{variant}_{index}_{compound}" in arrays})
    support = {}
    for compound, group in frame.groupby("compound", sort=True):
        eras = {}
        for era, subset in group.groupby("spec_era_id", sort=True):
            eras[str(era)] = {"laps": int(len(subset)),
                              "stints": int(subset.groupby(["session_id", "stint_id"]).ngroups),
                              "events": int(subset.session_id.nunique()),
                              "min_age": int(subset.tyre_age_laps.min()), "max_age": int(subset.tyre_age_laps.max()),
                              "one_event_qualified": any(len(event) >= 30 and event.stint_id.nunique() >= 3
                                                         for _, event in subset.groupby("session_id"))}
        support[compound] = {"laps": int(len(group)), "stints": int(group.groupby(["session_id", "stint_id"]).ngroups),
                             "events": int(group.session_id.nunique()), "min_age": int(group.tyre_age_laps.min()),
                             "max_age": int(group.tyre_age_laps.max()), "spec_eras": eras}
    import platform
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    metadata = {"model_kind": "hierarchical", "data_cutoff_utc": training.data_cutoff_utc.isoformat(),
                "design": design, "feature_names": list(feature_names), "assumption_grid": assumption_grid,
                "variant_indices": variant_indices, "support": support,
                "lineage": [{**v["lineage"], "roles": v["roles"]} for v in admitted.values()],
                "residual_pooling": pooling, "era_indices": era_indices, "calibration": calibration,
                "penalty_selection": selection, "seed": training.seed, "lock_digest": lock_digest(),
                "training_spec": training.model_dump(mode="json"),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__, "python": platform.python_version(),
                             "platform": platform.platform()},
                "numerical_backends": [{k: value for k, value in info.items() if k != "filepath"} for info in threadpool_info()]}
    model_id, path = publish_bundle(root, metadata, arrays)
    return ModelRef(model_id=model_id, path=path, model_kind="hierarchical")


def forecast_laps(state, model, *, driver_number, compound, race_laps, tyre_ages, assumptions, outlap_race_lap=None):
    """Forecast explicit future completed-lap numbers and tyre ages."""
    if getattr(model, "metadata", {}).get("model_kind") == "hgb":
        return model.forecast_laps(
            state, driver_number=driver_number, compound=compound,
            race_laps=race_laps, tyre_ages=tyre_ages, assumptions=assumptions,
            outlap_race_lap=outlap_race_lap,
        )
    if model.data_cutoff_utc > state.race_start_utc:
        raise WorkbenchError("future_artifact", "Model cutoff is after the target race start")
    from datetime import datetime
    if any(datetime.fromisoformat(source["max_used_utc"]) >= state.race_start_utc
           for source in getattr(model, "lineage", ()) if "max_used_utc" in source):
        raise WorkbenchError("future_artifact", "Model lineage includes data from after the target race start")
    laps = np.asarray(race_laps)
    ages = np.asarray(tyre_ages)
    driver = state.drivers.get(driver_number)
    if (driver is None or laps.ndim != 1 or ages.shape != laps.shape or not len(laps)
            or laps.dtype.kind not in "iu" or ages.dtype.kind not in "iu"
            or np.any(laps <= driver.completed_laps) or np.any(ages < 0)
            or np.any(np.diff(laps) <= 0) or compound not in ("SOFT", "MEDIUM", "HARD")):
        raise WorkbenchError("invalid_input", "Invalid future race laps, tyre ages, driver or compound")
    if driver.compound is None or driver.tyre_age_laps is None:
        raise WorkbenchError("insufficient_evidence", "Mounted tyre compound or age is not observable in this prefix")
    if outlap_race_lap is not None and (not isinstance(outlap_race_lap, int) or isinstance(outlap_race_lap, bool)
                                         or outlap_race_lap <= driver.completed_laps):
        raise WorkbenchError("invalid_input", "Future outlap must be a future integer lap")
    future_stop = outlap_race_lap is not None
    if outlap_race_lap is None:
        exits = state.pit_events
        exits = exits[(exits.driver_number == driver_number) & (exits.kind == "exit")
                      & (exits.release_ns <= state.cutoff_ns)].sort_values("release_ns")
        if len(exits) and pd.notna(exits.source_lap_number.iloc[-1]):
            outlap_race_lap = int(exits.source_lap_number.iloc[-1])
    prefix = eligible_laps(state.laps, state.pit_events, state.track_status, state.weather,
                           cutoff_ns=state.cutoff_ns, event_metadata=state.event_metadata,
                           session_id=state.session_id)
    prefix = prefix[(prefix.driver_number == driver_number) & prefix.eligible].copy()
    if getattr(model, "metadata", {}).get("model_kind") == "rolling":
        from .baseline import rolling_forecast
        setting = (assumptions.fuel_gain_s_per_lap, assumptions.track_drift_s_per_lap)
        if setting not in model.assumption_grid:
            raise WorkbenchError("missing_model_variant", "Rolling model does not contain the requested assumptions")
        if compound != driver.compound or future_stop:
            raise WorkbenchError("insufficient_evidence", "Rolling baseline cannot forecast a tyre change")
        rolling = rolling_forecast(prefix, race_laps=laps, assumptions=assumptions)
        mean = rolling["mean_s"]
        residuals = rolling["residual_s"]
        blocks = np.asarray([residuals[i:i + 3] for i in range(len(residuals) - 2)], dtype=np.float64)
        residual80 = np.quantile(residuals, (.1, .9))
        residual95 = np.quantile(residuals, (.025, .975))
        interval80 = mean[:, None] + residual80
        interval95 = mean[:, None] + residual95
        return StintForecast(driver_number, compound, laps.astype(np.int64), ages.astype(np.int64),
                             mean, mean[None, :], blocks, interval80, interval95, {},
                             {"advice_supported": False, "prefix_laps": int(len(prefix))},
                             assumptions, (*rolling["flags"], "uncalibrated"), state.cutoff_ns)
    weather = state.weather.sort_values("release_ns")
    temperatures = weather.track_temp_c.dropna()
    reference = float(temperatures.iloc[0]) if len(temperatures) else float("nan")
    temperature = assumptions.track_temperature_c
    if temperature is None:
        temperature = float(temperatures.iloc[-1]) if len(temperatures) else float("nan")
    layout = str(state.event_metadata.get("layout_id") or f"unknown:{state.session_id}")
    eras = state.event_metadata.get("spec_era_by_compound", {})
    era = eras.get(compound) or f"unknown:{state.session_id}:{compound}"
    prefix["driver_session"] = state.session_id + ":" + driver_number
    prefix["track_temperature_delta_c"] = prefix.track_temperature_c - reference
    from .adaptation import adapt_prefix
    if len(prefix):
        past_mean = model.predict_laps(prefix, assumptions=assumptions)
        if hasattr(model, "predict_parameter_laps"):
            past_parameters = model.predict_parameter_laps(prefix, assumptions=assumptions)
        else:
            past_parameters = np.stack([model.predict_laps(prefix, assumptions=assumptions, replicate_index=i)
                                        for i in range(model.n_parameter_replicates)])
    else:
        past_mean = np.empty(0)
        past_parameters = np.empty((model.n_parameter_replicates, 0))
    anchor, parameter_anchor, slope, adapted_slope = adapt_prefix(
        prefix, past_mean, past_parameters, current_age=driver.tyre_age_laps,
        same_stint=not future_stop and compound == driver.compound)
    n = len(laps)
    frame = pd.DataFrame({"driver_number": [driver_number] * n, "session_id": [state.session_id] * n,
                          "driver_session": [state.session_id + ":" + driver_number] * n,
                          "layout_id": [layout] * n, "spec_era_id": [era] * n,
                          "compound": [compound] * n, "tyre_age_laps": ages,
                          "lap_number": laps,
                          "warmup_first": (laps == outlap_race_lap + 1).astype(float) if outlap_race_lap is not None else np.zeros(n),
                          "warmup_second": (laps == outlap_race_lap + 2).astype(float) if outlap_race_lap is not None else np.zeros(n),
                          "track_temperature_delta_c": np.full(n, temperature - reference)})
    mean = model.predict_laps(frame, assumptions=assumptions) + anchor
    if hasattr(model, "predict_parameter_laps"):
        parameter = model.predict_parameter_laps(frame, assumptions=assumptions)
    else:
        parameter = np.stack([model.predict_laps(frame, assumptions=assumptions, replicate_index=i)
                              for i in range(model.n_parameter_replicates)])
    parameter += parameter_anchor[:, None]
    if adapted_slope:
        correction = slope * (ages - driver.tyre_age_laps)
        mean += correction
        parameter += correction
    compound_blocks = model.residual_blocks(compound, assumptions=assumptions)
    spec_pools = {}
    if hasattr(model, "_variant") and era in model.metadata.get("era_indices", {}):
        key = f"spec_residual_{model._variant(assumptions)}_{model.metadata['era_indices'][era]}_{compound}"
        if key in model.arrays:
            spec_pools[(era, compound)] = model.arrays[key]
    blocks = select_residual_pool(compound, era, {compound: compound_blocks}, spec_pools)
    support = dict(model.support.get(compound, {}))
    flags = ([] if adapted_slope else ["pooled_only"]) + ["uncalibrated"]
    if len(prefix) < 3:
        flags.append("insufficient_prefix")
    if support.get("laps", 0) < 30 or support.get("stints", 0) < 3 or support.get("events", 0) < 2:
        flags.append("insufficient_compound_support")
    if not len(blocks):
        flags.append("insufficient_residuals")
    era_support = support.get("spec_eras", {}).get(era)
    same_spec_prefix = prefix[prefix.spec_era_id == era]
    if not ((era_support and era_support.get("one_event_qualified", False)) or len(same_spec_prefix) >= 8):
        flags.append("transfer_only")
    age_support = era_support or support
    if age_support and (ages.min() < age_support["min_age"] - 2 or ages.max() > age_support["max_age"] + 2):
        flags.append("outside_training_support")
    # Raw parameter/block quantiles: calibration is not implied by these bounds.
    rng = np.random.default_rng(2026)
    paths = parameter.copy()
    if len(blocks):
        for start in range(0, n, 3):
            selected = blocks[rng.integers(len(blocks), size=len(paths))]
            length = min(3, n - start)
            paths[:, start:start + length] += selected[:, :length]
    interval80 = np.quantile(paths, (.1, .9), axis=0).T
    interval95 = np.quantile(paths, (.025, .975), axis=0).T
    support["advice_supported"] = not any(flag in flags for flag in
        ("insufficient_prefix", "insufficient_compound_support", "insufficient_residuals", "outside_training_support", "transfer_only"))
    variant = str(model._variant(assumptions)) if hasattr(model, "_variant") else ""
    calibration = getattr(model, "metadata", {}).get("calibration", {}).get(variant, {})
    if future_stop or compound != driver.compound or laps[0] != driver.completed_laps + 1 or np.any(np.diff(laps) != 1):
        # Held-out scores concern immediate same-stint continuations only.
        calibration = {}
    cumulative = {}
    for horizon in (1, 3, 5, 10):
        if horizon <= n:
            totals = paths[:, :horizon].sum(axis=1)
            result = {}
            calibrated = True
            for coverage, interval in ((80, interval80), (95, interval95)):
                lap_record = calibration.get(f"{horizon}:lap:{coverage}", {})
                lap_adjustment = lap_record.get("adjustment_s")
                if lap_adjustment is not None:
                    interval[horizon - 1] += np.array([-lap_adjustment, lap_adjustment])
                total_record = calibration.get(f"{horizon}:cumulative:{coverage}", {})
                adjustment = total_record.get("adjustment_s")
                alpha = (1 - coverage / 100) / 2
                bounds = np.quantile(totals, (alpha, 1 - alpha))
                if adjustment is not None:
                    bounds += np.array([-adjustment, adjustment])
                else:
                    calibrated = False
                result[f"interval{coverage}_s"] = bounds.tolist()
            result["calibration"] = "calibrated" if calibrated else "uncalibrated"
            cumulative[str(horizon)] = result
    return StintForecast(driver_number, compound, laps.astype(np.int64), ages.astype(np.int64),
                         mean, parameter, blocks, interval80, interval95, cumulative, support,
                         assumptions, tuple(flags), state.cutoff_ns)


def forecast_stint(state, model, *, driver_number, compound, tyre_age_laps, horizon_laps, assumptions):
    if (not isinstance(tyre_age_laps, int) or isinstance(tyre_age_laps, bool) or tyre_age_laps < 0
            or not isinstance(horizon_laps, int) or isinstance(horizon_laps, bool) or horizon_laps <= 0):
        raise WorkbenchError("invalid_input", "Tyre age and horizon must be valid integers")
    driver = state.drivers.get(driver_number)
    if driver is None:
        raise WorkbenchError("invalid_input", "Driver not present in replay prefix")
    offset = np.arange(1, horizon_laps + 1, dtype=np.int64)
    return forecast_laps(state, model, driver_number=driver_number, compound=compound,
                         race_laps=driver.completed_laps + offset, tyre_ages=tyre_age_laps + offset,
                         assumptions=assumptions)

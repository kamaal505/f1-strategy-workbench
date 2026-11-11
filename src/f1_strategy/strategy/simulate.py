"""Paired conditional simulations of bounded candidate strategies."""
from __future__ import annotations

from importlib.resources import files
from datetime import datetime
import hashlib

import numpy as np

from .. import rules
from ..contracts import DEFAULT_GRID, REPLAY_NOTICE, DecisionResult, WorkbenchError
from ..identity import canonical, core_fingerprint, digest, lock_digest, seal_payload
from ..payloads import prefix_payload
from .plans import enumerate_plans


def _assets_digest(directory):
    root = files("f1_strategy").joinpath("assets", directory)
    if not root.is_dir():
        return digest([])
    return digest([(item.name, hashlib.sha256(item.read_bytes()).hexdigest())
                   for item in sorted(root.iterdir(), key=lambda x: x.name) if item.is_file()])


def _result(payload):
    # Include every schema default before computing the checksum; subsequent
    # model serialization must not add fields outside the sealed document.
    staged = {"analysis_id": "0" * 64, "payload_checksum": "0" * 64, **payload}
    validated = DecisionResult.model_validate_json(canonical(staged))
    sealed = seal_payload(validated.model_dump(mode="json"))
    return DecisionResult.model_validate_json(canonical(sealed))


def _forecast_plan(state, model, scenario, plan):
    from ..pace.hierarchical import forecast_laps
    focal = state.drivers[state.driver_number]
    available = {s.set_id: s for s in scenario.inventory_by_driver.get(state.driver_number, ())}
    first = state.completed_lap + 1
    compound, age = focal.compound, focal.tyre_age_laps
    segments = []
    outlap = None
    for stop in plan.stops:
        segments.append((first, stop.pit_lap, compound, age, outlap))
        tyre = available[stop.new_set_id]
        first, compound, age = stop.pit_lap + 1, tyre.compound, tyre.age_laps
        outlap = first
    segments.append((first, state.scheduled_laps, compound, age, outlap))
    result = []
    for first, last, compound, age, outlap in segments:
        if first > last:
            continue
        if compound is None or age is None:
            raise WorkbenchError("insufficient_evidence", "Tyre compound or age is unknown")
        laps = np.arange(first, last + 1, dtype=np.int64)
        ages = np.arange(age + 1, age + 1 + len(laps), dtype=np.int64)
        forecast = forecast_laps(state, model, driver_number=state.driver_number, compound=compound,
                                 race_laps=laps, tyre_ages=ages, assumptions=scenario.pace,
                                 outlap_race_lap=outlap)
        param = np.asarray(forecast.parameter_mean_s, dtype=float)
        residual = np.asarray(forecast.residual_blocks_s, dtype=float)
        if param.ndim != 2 or param.shape[1] != len(laps) or not len(param) or not np.isfinite(param).all():
            raise WorkbenchError("insufficient_evidence", "Parameter replicates unavailable")
        if residual.ndim != 2 or residual.shape[1] != 3 or not len(residual) or not np.isfinite(residual).all():
            raise WorkbenchError("insufficient_evidence", "Residual blocks unavailable")
        result.append((forecast, param, residual))
    return result


def _draw_plan(state, model, scenario, plan, n, seed, *, lap_paths=False, forecast_cache=None, components=False):
    from ..pace.uncertainty import map_residual_draws
    rng = np.random.default_rng(seed % (1 << 128))
    horizon = state.scheduled_laps - state.completed_lap
    reps = rng.integers(0, max(1, model.n_parameter_replicates), size=n)
    residual_uniform = rng.random((n, (horizon + 2) // 3))
    net_noise = rng.standard_normal((n, scenario.max_additional_stops))
    service_noise = rng.standard_normal((n, scenario.max_additional_stops))
    warm_noise = rng.standard_normal((n, scenario.max_additional_stops))
    times = np.zeros((n, horizon))
    net_costs = np.zeros((n, horizon))
    pit_costs = np.zeros((n, horizon))
    flags, forecasts = set(), []
    key = (plan.plan_id, scenario.pace.fuel_gain_s_per_lap, scenario.pace.track_drift_s_per_lap,
           scenario.pace.track_temperature_c)
    if forecast_cache is None:
        predicted = _forecast_plan(state, model, scenario, plan)
    else:
        if key not in forecast_cache:
            forecast_cache[key] = _forecast_plan(state, model, scenario, plan)
        predicted = forecast_cache[key]
    for forecast, parameter, blocks in predicted:
        flags.update(forecast.flags)
        if not forecast.support.get("advice_supported", False):
            flags.add("insufficient_compound_support")
        indices = reps % len(parameter)
        offsets = np.asarray(forecast.race_laps, dtype=int) - state.completed_lap - 1
        block_indices = np.minimum((residual_uniform[:, offsets // 3] * len(blocks)).astype(int), len(blocks) - 1)
        raw = blocks[block_indices, offsets % 3]
        quantiles = getattr(forecast, "residual_quantiles_s", None)
        mask = getattr(forecast, "ml_residual_mask", None)
        errors = map_residual_draws(raw, blocks, quantiles, mask)
        times[:, offsets] += parameter[indices] + errors
        forecasts.append({"race_laps": forecast.race_laps.tolist(), "tyre_ages": forecast.tyre_ages.tolist(),
                          "compound": forecast.compound, "mean_s": forecast.mean_s.tolist(),
                          "interval80_s": forecast.interval80_s.tolist(), "interval95_s": forecast.interval95_s.tolist(),
                          "residual_quantiles_s": None if quantiles is None else np.asarray(quantiles).tolist(),
                          "ml_residual_mask": None if mask is None else np.asarray(mask, dtype=bool).tolist(),
                          "support": forecast.support, "flags": list(forecast.flags)})
    pit = scenario.pit
    for i, stop in enumerate(plan.stops):
        offset = stop.pit_lap - state.completed_lap
        net_costs[:, offset] = np.maximum(0, pit.net_transit_loss_s + pit.net_transit_sd_s * net_noise[:, i])
        times[:, offset] += net_costs[:, offset]
        pit_costs[:, offset] = (net_costs[:, offset]
                                + np.maximum(1, pit.service_s + pit.service_sd_s * service_noise[:, i])
                                + np.maximum(0, pit.outlap_warmup_s + pit.outlap_warmup_sd_s * warm_noise[:, i]))
        times[:, offset] += pit_costs[:, offset] - net_costs[:, offset]
    if not np.isfinite(times).all() or np.any(times <= 0):
        raise WorkbenchError("numerical_failure", "Simulated lap duration is nonpositive or nonfinite")
    result = (times if lap_paths else times.sum(axis=1)), flags, forecasts
    return (*result, net_costs, pit_costs) if components else result


def _finish_lap(crossings, scheduled_laps):
    """Two-hour expiry at a crossing still requires the following lap."""
    for lap, seconds in enumerate(crossings, 1):
        if seconds >= 7200:
            return min(scheduled_laps, lap + 1)
    return scheduled_laps


def _recommend(delta, probability, sensitivity):
    signs = [x["mean_delta_s"] for x in sensitivity]
    if delta > 0 and probability >= .8 and all(x > 0 for x in signs):
        return "pit_now"
    if delta < 0 and probability <= .2 and all(x < 0 for x in signs):
        return "stay_out"
    return "no_clear_advantage"


def compare_strategies(state, model, scenario, *, n_draws, seed):
    if not isinstance(n_draws, int) or isinstance(n_draws, bool) or not 100 <= n_draws <= 5000:
        raise WorkbenchError("invalid_input", "n_draws must be between 100 and 5000")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorkbenchError("invalid_input", "seed must be an integer")
    if set(scenario.inventory_by_driver) - set(state.drivers):
        raise WorkbenchError("invalid_scenario", "Inventory contains an unknown driver")
    if state.max_feature_release_ns > state.cutoff_ns:
        raise WorkbenchError("future_feature", "Replay contains a feature released after the cutoff")
    for frame in (state.laps, state.weather, state.track_status, state.session_status, state.messages, state.pit_events):
        if not frame.empty and ("release_ns" not in frame or frame.release_ns.isna().any()
                                or (frame.release_ns > state.cutoff_ns).any()):
            raise WorkbenchError("future_feature", "Replay contains an unreleased or post-cutoff row")
    if state.drivers[state.driver_number].in_pit:
        raise WorkbenchError("focal_in_pit", "A car already in the pit lane cannot choose a stop")
    if model.data_cutoff_utc >= state.race_start_utc:
        raise WorkbenchError("future_model", "Model data cutoff must precede race start")
    for source in getattr(model, "lineage", ()):
        timestamp = source.get("max_used_utc")
        if timestamp is not None:
            try:
                used = datetime.fromisoformat(timestamp)
                if used.tzinfo is None or used >= state.race_start_utc:
                    raise WorkbenchError("future_model", "Model lineage must precede the target race")
            except (TypeError, ValueError):
                raise WorkbenchError("corrupt_artifact", "Invalid model lineage timestamp") from None
    variants = {(p.fuel_gain_s_per_lap, p.track_drift_s_per_lap) for p in DEFAULT_GRID}
    variants.add((scenario.pace.fuel_gain_s_per_lap, scenario.pace.track_drift_s_per_lap))
    traffic_values = sorted({0.0, 0.4, 0.8} - {scenario.traffic_penalty_s})
    event_bound = sum(max(0, state.scheduled_laps - d.completed_laps) + 2 for d in state.drivers.values()) * (6 * 256 + 2 * n_draws * (len(variants) + 2 + len(traffic_values) + (2 if scenario.neutralisations else 0)))
    if event_bound > 50_000_000:
        raise WorkbenchError("capacity_exceeded", "Comparison event budget exceeded")
    identity = {"schema_version": 1, "snapshot_id": state.snapshot_id, "model_id": model.model_id,
                "driver_number": state.driver_number, "completed_lap": state.completed_lap,
                "cutoff_ns": state.cutoff_ns, "scenario": scenario, "n_draws": n_draws, "seed": seed,
                "rules_digest": _assets_digest("rules"), "events_digest": _assets_digest("events"),
                "lock_digest": lock_digest(), "core_fingerprint": core_fingerprint()}
    flags = set(rules.check_state(state, scenario)) | set(state.flags)
    hard = {"unsupported_wet", "unsupported_red_flag", "unknown_dry_conditions", "unknown_tyre_history", "unsupported_directive", "unsupported_season"}
    payload = dict(snapshot_id=state.snapshot_id, session_id=state.session_id,
                   driver_number=state.driver_number, completed_lap=state.completed_lap,
                   cutoff_ns=state.cutoff_ns, model_id=model.model_id,
                   model_data_cutoff_utc=model.data_cutoff_utc, scenario=scenario,
                   n_draws=n_draws, seed=seed, status="unsupported" if flags & hard else "insufficient_evidence",
                   recommendation="unavailable", quality_flags=sorted(flags), comparison_scope="unavailable",
                   assumptions=[REPLAY_NOTICE,
                                "Fuel gain and track drift are assumptions and are not separately identified tyre wear.",
                                "Future inventory, pit losses, traffic and rival policies are conditional assumptions.",
                                "Future neutralisations occur only as explicitly specified; no future retirement or unlapping is modelled.",
                                "Lap-boundary traffic, pit transitions and neutralisation timing are approximate.",
                                "ML residual quantiles use a centered piecewise-linear inverse CDF with linear tails; empirical block ranks supply serial dependence, not additional variance. ML parameters are not bootstrapped.",
                                "Time comparisons continue to the same scheduled distance, including after the simulated flag."],
                   evidence={"event_bound": event_bound}, identity_inputs=identity,
                   prefix_state=prefix_payload(state), source_lineage=tuple(state.source_lineage) + tuple(getattr(model, "lineage", ())))
    if flags & hard:
        payload["evidence"]["reason"] = "The released prefix is outside the supported dry-race domain."
        return _result(payload)
    from .pit import completed_pit_evidence
    pit_evidence = completed_pit_evidence(state, model, scenario)
    effective = {}
    if pit_evidence["loss_s"] is not None and pit_evidence["loss_s"] >= scenario.pit.service_s:
        effective["net_transit_loss_s"] = pit_evidence["loss_s"] - scenario.pit.service_s
        effective["net_transit_sd_s"] = 1.4826 * pit_evidence["loss_mad_s"]
        effective["provenance"] = "released prior green visits; service split assumed"
    if scenario.pit.absolute_lane_duration_s is None and pit_evidence["occupancy_s"] is not None:
        effective["absolute_lane_duration_s"] = pit_evidence["occupancy_s"]
        effective["absolute_lane_duration_sd_s"] = 1.4826 * pit_evidence["occupancy_mad_s"]
    if effective:
        scenario = scenario.model_copy(update={"pit": scenario.pit.model_copy(update=effective)})
    payload["evidence"]["pit_evidence"] = pit_evidence
    payload["evidence"]["residual_distribution"] = {
        "conditional_quantiles": "piecewise-linear inverse CDF with linear tails, centered to zero mean",
        "dependence": "shared keyed hierarchical residual-block ranks",
        "ml_parameter_uncertainty": "not bootstrapped",
    }
    payload["evidence"]["effective_pit"] = scenario.pit.model_dump(mode="json")
    def abstain(exc):
        if exc.code != "insufficient_evidence":
            raise exc
        payload.update(quality_flags=sorted(flags | {exc.code}),
                       evidence={**payload["evidence"], "reason": exc.detail})
        return _result(payload)

    try:
        plans = enumerate_plans(state, scenario, model)
    except WorkbenchError as exc:
        return abstain(exc)
    groups = {branch: [p for p in plans if p.branch == branch] for branch in ("pit_now", "stay_out")}
    if not all(groups.values()):
        mounted = state.drivers[state.driver_number]
        no_change = scenario.max_additional_stops == 0 or not any(
            tyre.set_id != mounted.mounted_set_id for tyre in scenario.inventory_by_driver.get(state.driver_number, ()))
        limit = scenario.rules_override.max_set_life_laps if scenario.rules_override else None
        forced_pit = (bool(groups["pit_now"]) and not groups["stay_out"] and limit is not None
                      and mounted.tyre_age_laps is not None and mounted.tyre_age_laps + 2 > limit)
        if no_change and groups["stay_out"]:
            flags.add("direct_pit_now_infeasible")
            payload.update(status="ok", recommendation="stay_out", quality_flags=sorted(flags),
                           comparison_scope="direct rule feasibility",
                           evidence={**payload["evidence"], "reason": "No future tyre set change is available and the no-stop plan is legal."})
        elif forced_pit:
            flags.add("direct_stay_out_infeasible")
            payload.update(status="ok", recommendation="pit_now", quality_flags=sorted(flags),
                           comparison_scope="direct rule feasibility",
                           evidence={**payload["evidence"], "reason": "The mounted set cannot reach a later pit opportunity."})
        elif not plans and no_change:
            flags.add("direct_constraints_infeasible")
            payload.update(status="infeasible", quality_flags=sorted(flags),
                           evidence={**payload["evidence"], "reason": "The no-stop plan is illegal and no further set change is possible."})
        else:
            flags.add("search_exhausted")
            payload.update(quality_flags=sorted(flags), evidence={**payload["evidence"],
                           "reason": "No completion survived the bounded search for at least one branch."})
        return _result(payload)
    from .field import prepare_field, run_field
    active_status = "" if state.track_status.empty else str(state.track_status.iloc[-1].get("status", ""))
    focal_forecasts = {}
    rival_forecasts = {}

    def field_for(setting, count, draw_seed):
        active_kind = "SC" if active_status == "SC" else "VSC"
        if active_status in ("SC", "VSC", "VSC_ENDING") and not any(
            item.kind == active_kind and item.start_after_leader_laps == 0 for item in setting.neutralisations
        ):
            return None, "Remaining duration of the active neutralisation is unspecified"
        try:
            return prepare_field(state, model, setting, count, draw_seed, forecast_cache=rival_forecasts), None
        except WorkbenchError as exc:
            if exc.code != "field_unavailable":
                raise
            return None, exc.detail

    def evaluate(setting, plan, count, draw_seed, field):
        paths, candidate_flags, forecasts, net_costs, pit_costs = _draw_plan(
            state, model, setting, plan, count, draw_seed, lap_paths=True,
            forecast_cache=focal_forecasts, components=True)
        if field is None:
            if setting.neutralisations:
                raise WorkbenchError("insufficient_evidence", "Leader timeline unavailable for the neutralisation scenario")
            return paths.sum(axis=1), candidate_flags, forecasts, None, None
        try:
            times, ranks, rejoins = run_field(state, setting, plan, paths, field,
                                              focal_net_costs=net_costs, focal_pit_costs=pit_costs)
        except WorkbenchError as exc:
            if exc.code != "field_unavailable":
                raise
            if setting.neutralisations:
                raise WorkbenchError("insufficient_evidence", exc.detail) from exc
            candidate_flags.add("field_simulation_unavailable")
            return paths.sum(axis=1), candidate_flags, forecasts, None, None
        return times, candidate_flags, forecasts, ranks, rejoins

    screen_field, screen_reason = field_for(scenario, 256, seed)
    eval_field, field_reason = field_for(scenario, n_draws, seed + 1)
    summaries = []
    chosen = {}
    for branch, candidates in groups.items():
        scored = []
        for candidate in candidates:
            try:
                draw, candidate_flags, _, _, _ = evaluate(scenario, candidate, 256, seed, screen_field)
            except WorkbenchError as exc:
                return abstain(exc)
            scored.append((float(draw.mean()), candidate.plan_id, candidate))
            summaries.append({"plan": candidate.model_dump(mode="json"), "screen_mean_s": float(draw.mean()), "flags": sorted(candidate_flags)})
        chosen[branch] = min(scored)[2]
    try:
        pit, pit_flags, pit_forecast, pit_ranks, pit_rejoin = evaluate(scenario, chosen["pit_now"], n_draws, seed + 1, eval_field)
        stay, stay_flags, stay_forecast, stay_ranks, stay_rejoin = evaluate(scenario, chosen["stay_out"], n_draws, seed + 1, eval_field)
    except WorkbenchError as exc:
        return abstain(exc)
    flags.update(pit_flags | stay_flags)
    delta = stay - pit
    sensitivities = []
    settings = [("fuel_track", scenario.model_copy(update={"pace": scenario.pace.model_copy(update={"fuel_gain_s_per_lap": g, "track_drift_s_per_lap": d})}), {"fuel_gain_s_per_lap": g, "track_drift_s_per_lap": d}) for g, d in sorted(variants)]
    for adjustment in (-3.0, 3.0):
        net = max(0.0, scenario.pit.net_transit_loss_s + adjustment)
        settings.append(("pit_loss", scenario.model_copy(update={"pit": scenario.pit.model_copy(update={"net_transit_loss_s": net})}), {"net_transit_loss_s": net}))
    for penalty in traffic_values:
        settings.append(("traffic", scenario.model_copy(update={"traffic_penalty_s": penalty}), {"traffic_penalty_s": penalty}))
    if scenario.neutralisations:
        for shift in (-1, 1):
            shifted = tuple(
                item.model_copy(update={"duration_laps": max(1, item.duration_laps + shift)})
                if item.start_after_leader_laps == 0 and active_status in ("SC", "VSC", "VSC_ENDING")
                else item.model_copy(update={"start_after_leader_laps": max(0, item.start_after_leader_laps + shift)})
                for item in scenario.neutralisations)
            settings.append(("pit_boundary", scenario.model_copy(update={"neutralisations": shifted}), {"leader_lap_shift": shift}))
    field_supported = screen_field is not None and eval_field is not None and pit_ranks is not None and stay_ranks is not None
    for kind, setting, values in settings:
        if rules.plan_legality(state, setting, chosen["pit_now"]) or rules.plan_legality(state, setting, chosen["stay_out"]):
            sensitivities.append({"kind": kind, **values, "mean_delta_s": None, "reason": "Selected plan violates a shifted pit restriction"})
            field_supported = False
            continue
        field, reason = field_for(setting, n_draws, seed + 1)
        if field is None and kind == "traffic":
            sensitivities.append({"kind": kind, **values, "mean_delta_s": None, "reason": reason})
            field_supported = False
            continue
        try:
            a, af, _, ar, _ = evaluate(setting, chosen["pit_now"], n_draws, seed + 1, field)
            b, bf, _, br, _ = evaluate(setting, chosen["stay_out"], n_draws, seed + 1, field)
        except WorkbenchError as exc:
            return abstain(exc)
        if ar is None or br is None:
            field_supported = False
        if field is None:
            field_supported = False
        flags.update(af | bf)
        sensitivities.append({"kind": kind, **values, "mean_delta_s": float((b - a).mean())})
    mean, probability = float(delta.mean()), float((delta > 0).mean())
    if not field_supported:
        flags.add("field_simulation_unavailable")
        flags.add("traffic_sensitivity_unavailable")
    evidence_flags = {"insufficient_prefix", "insufficient_compound_support", "insufficient_residuals",
                      "outside_training_support", "transfer_only"}
    can_advise = field_supported and not (flags & evidence_flags)
    recommendation = _recommend(mean, probability, sensitivities) if can_advise else "unavailable"
    positions = {"available": False, "reason": field_reason or screen_reason or "Forecast or field sensitivity evidence unavailable"}
    if can_advise:
        positions = {
            "available": True,
            "pit_now": {str(rank): float(np.mean(pit_ranks == rank)) for rank in sorted(set(pit_ranks))},
            "stay_out": {str(rank): float(np.mean(stay_ranks == rank)) for rank in sorted(set(stay_ranks))},
            "assumption": "Approximate lap-boundary classification under the assumed rival policy; no future retirements.",
        }
    from .occupancy import pit_occupancy_projection
    occupancy = pit_occupancy_projection(state, scenario, n_draws, seed + 1)
    payload.update(candidate_summaries=summaries, selected_plan_ids={k: v.plan_id for k, v in chosen.items()},
                   delta_stay_minus_pit_s=mean, prob_pit_faster=probability,
                   intervals={"delta80_s": np.quantile(delta, [.1, .9]).tolist(), "delta95_s": np.quantile(delta, [.025, .975]).tolist(),
                              "probability_mc_se": float(np.sqrt(probability * (1 - probability) / n_draws)),
                              "prob_stay_loses_over_5s": float(np.mean(delta > 5.0)),
                              "prob_pit_loses_over_5s": float(np.mean(delta < -5.0))},
                   sensitivity=sensitivities, quality_flags=sorted(flags),
                   status="ok" if can_advise else "insufficient_evidence", recommendation=recommendation,
                   comparison_scope="conditional fixed-distance field timing" if field_supported else "isolated equal-distance timing only",
                   pit_exit_projection={"available": False, "reason": "Exact pit-line rejoin geometry is unavailable",
                                        "opponent_occupancy": occupancy,
                                        "outlap_crossing_note": "Crossing clocks are relative to the decision cutoff, not pit-exit timestamps."},
                   position_distribution=positions,
                   forecast_series={"pit_now": pit_forecast, "stay_out": stay_forecast},
                   simulation_series={"pit_now_s": pit.tolist(), "stay_out_s": stay.tolist(), "paired_delta_s": delta.tolist(),
                                      "pit_now_rank": pit_ranks.tolist() if can_advise else None,
                                      "stay_out_rank": stay_ranks.tolist() if can_advise else None,
                                      "pit_now_field_events": pit_rejoin,
                                      "stay_out_field_events": stay_rejoin},
                   evidence={**payload["evidence"], "reason": (
                       "Conditional lap-boundary field approximation" if can_advise else
                       "Forecast support is insufficient for advice" if field_supported else
                       (field_reason or screen_reason or "Field sensitivity unavailable"))})
    return _result(payload)

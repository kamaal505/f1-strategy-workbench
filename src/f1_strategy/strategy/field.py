"""Lap-boundary field trajectories and conditional progress estimates."""
from __future__ import annotations

import hashlib
import heapq

import numpy as np

from ..contracts import WorkbenchError


def _seed(seed, driver):
    return int.from_bytes(hashlib.sha256(f"{seed}:{driver}".encode()).digest()[:16], "big")


def _threshold(state, driver, compound):
    laps = state.laps
    if laps.empty or "source_stint" not in laps:
        return 20, "assumed"
    rows = laps[(laps.driver_number == driver) & (laps.compound == compound)].copy()
    if rows.empty:
        return 20, "assumed"
    current = rows.sort_values("lap_number").iloc[-1].source_stint
    completed = rows[rows.source_stint != current].groupby("source_stint").size()
    if len(completed) < 3:
        return 20, "assumed"
    return max(1, int(np.ceil(np.quantile(completed.to_numpy(), .8)))), "prefix"


def prepare_field(state, model, scenario, n, seed, *, forecast_cache=None):
    """Build opponent paths using released prefix and an explicit age policy.

    Unsupported progress, tyre or pit occupancy raises a named abstention; it
    never fabricates an opponent retirement or borrows a historical future stop.
    """
    from ..pace.hierarchical import forecast_laps
    from ..pace.uncertainty import map_residual_draws
    from .. import rules
    cache = {} if forecast_cache is None else forecast_cache

    def predict(number, compound, laps, ages, outlap):
        key = (number, compound, int(laps[0]), int(laps[-1]), int(ages[0]), outlap,
               scenario.pace.fuel_gain_s_per_lap, scenario.pace.track_drift_s_per_lap,
               scenario.pace.track_temperature_c)
        if key not in cache:
            cache[key] = forecast_laps(state, model, driver_number=number, compound=compound,
                                       race_laps=laps, tyre_ages=ages, assumptions=scenario.pace,
                                       outlap_race_lap=outlap)
        return cache[key]

    closure_flags = set(rules.check_state(state, scenario))
    opponents = {}
    for number, driver in sorted(state.drivers.items()):
        if number == state.driver_number:
            continue
        if (driver.stale and not driver.in_pit) or driver.compound not in ("SOFT", "MEDIUM", "HARD") or driver.tyre_age_laps is None:
            raise WorkbenchError("field_unavailable", "A rival's progress or tyre is unavailable")
        first, end = driver.completed_laps + 1, state.scheduled_laps
        if first > end:
            raise WorkbenchError("field_unavailable", "A rival has already reached scheduled race distance")
        sets = list(scenario.inventory_by_driver.get(number, ()))
        compound, age = driver.compound, driver.tyre_age_laps
        used = set(driver.used_compounds) | {compound}
        override = scenario.rules_override
        minimum = override.min_dry_specs if override else 2
        mandatory = set(override.mandatory_race_compounds if override else ("MEDIUM", "HARD"))
        limit = override.max_set_life_laps if override else None
        started_in_pit = driver.in_pit
        if started_in_pit:
            if scenario.pit.absolute_lane_duration_s is None or driver.pit_entry_ns is None:
                raise WorkbenchError("field_unavailable", "A rival already in the pit lane requires supported absolute occupancy")
            if "pit_exit_closed" in closure_flags:
                raise WorkbenchError("field_unavailable", "The released pit exit is closed")
            if any(not neutral.pit_exit_open and neutral.start_after_leader_laps == 0 for neutral in scenario.neutralisations):
                raise WorkbenchError("field_unavailable", "The hypothetical pit exit is closed at the cutoff")
            first += 1
            if first > end or not sets:
                raise WorkbenchError("field_unavailable", "No supported post-pit rival tyre is available")
            laps = np.arange(first, end + 1, dtype=np.int64)
            options = []
            for tyre in sets:
                ages = np.arange(tyre.age_laps + 1, tyre.age_laps + 1 + len(laps), dtype=np.int64)
                if tyre.life_limit_laps is not None and ages[-1] > tyre.life_limit_laps:
                    continue
                forecast = predict(number, tyre.compound, laps, ages, first)
                if not forecast.support.get("advice_supported", False):
                    continue
                options.append((float(np.asarray(forecast.mean_s).sum()), tyre.set_id, tyre))
            if not options:
                raise WorkbenchError("field_unavailable", "No supported post-pit rival tyre is available")
            tyre = min(options)[2]
            sets.remove(tyre)
            used.add(tyre.compound)
            compound, age = tyre.compound, tyre.age_laps
            limits = [x for x in (limit, tyre.life_limit_laps) if x is not None]
            limit = min(limits) if limits else None
        segments, stops = [], []
        while first <= end:
            threshold, _ = _threshold(state, number, compound)
            if limit is not None:
                threshold = min(threshold, limit)
            needs_change = len(used) < minimum or (mandatory and not used.intersection(mandatory))
            if (needs_change and first >= end) or (limit is not None and age >= limit):
                raise WorkbenchError("field_unavailable", "Rival has no legal remaining stop opportunity")
            remaining = max(1, threshold - age)
            last = min(end, first + remaining - 1)
            if needs_change:
                last = min(last, end - 1)
            if last == end or not sets or len(stops) >= scenario.max_additional_stops:
                if needs_change or (limit is not None and age + end - first + 1 > limit):
                    raise WorkbenchError("field_unavailable", "Rival tyre policy cannot satisfy known limits")
                segments.append((first, end, compound, age, first if stops or started_in_pit else None))
                break
            if "pit_entry_closed" in closure_flags or "pit_exit_closed" in closure_flags:
                raise WorkbenchError("field_unavailable", "Rival stop conflicts with released pit closure")
            segments.append((first, last, compound, age, first if stops or started_in_pit else None))
            # Evaluate available tyres by isolated mean pace for the remaining
            # distance; each forecast is anchored to the released real prefix.
            options = []
            laps = np.arange(last + 1, end + 1, dtype=np.int64)
            for tyre in sets:
                if needs_change and (len(used | {tyre.compound}) < minimum or (mandatory and not (used | {tyre.compound}).intersection(mandatory))):
                    continue
                ages = np.arange(tyre.age_laps + 1, tyre.age_laps + 1 + len(laps), dtype=np.int64)
                if tyre.life_limit_laps is not None and ages[-1] > tyre.life_limit_laps:
                    continue
                forecast = predict(number, tyre.compound, laps, ages, last + 1)
                if not forecast.support.get("advice_supported", False):
                    continue
                options.append((float(np.asarray(forecast.mean_s).sum()), tyre.set_id, tyre))
            if not options:
                # No defensible rival policy rather than an invented set.
                raise WorkbenchError("field_unavailable", "No supported rival tyre set is available")
            tyre = min(options)[2]
            sets.remove(tyre)
            used.add(tyre.compound)
            limits = [x for x in (override.max_set_life_laps if override else None, tyre.life_limit_laps) if x is not None]
            limit = min(limits) if limits else None
            stops.append(last)
            first, compound, age = last + 1, tyre.compound, tyre.age_laps
        rng = np.random.default_rng(_seed(seed, number))
        horizon = end - driver.completed_laps
        reps = np.random.default_rng(seed % (1 << 128)).integers(0, max(1, model.n_parameter_replicates), size=n)
        block_u = rng.random((n, (horizon + 2) // 3))
        paths = np.zeros((n, horizon))
        net_costs = np.zeros((n, horizon))
        pit_costs = np.zeros((n, horizon))
        for first, last, compound, age, outlap in segments:
            laps = np.arange(first, last + 1, dtype=np.int64)
            ages = np.arange(age + 1, age + 1 + len(laps), dtype=np.int64)
            forecast = predict(number, compound, laps, ages, outlap)
            parameter = np.asarray(forecast.parameter_mean_s, dtype=float)
            blocks = np.asarray(forecast.residual_blocks_s, dtype=float)
            if (not forecast.support.get("advice_supported", False) or parameter.ndim != 2 or not len(parameter)
                    or parameter.shape[1] != len(laps) or blocks.ndim != 2 or not len(blocks)
                    or blocks.shape[1] != 3 or not np.isfinite(parameter).all() or not np.isfinite(blocks).all()):
                raise WorkbenchError("field_unavailable", "Rival pace is outside forecast support")
            if first == driver.completed_laps + 1:
                first_parameter, first_blocks, first_length = parameter, blocks, len(laps)
                first_forecast = forecast
            offsets = laps - driver.completed_laps - 1
            indices = np.minimum((block_u[:, offsets // 3] * len(blocks)).astype(int), len(blocks) - 1)
            raw = blocks[indices, offsets % 3]
            paths[:, offsets] = parameter[reps % len(parameter)] + map_residual_draws(
                raw, blocks, getattr(forecast, "residual_quantiles_s", None),
                getattr(forecast, "ml_residual_mask", None))
        if started_in_pit:
            from scipy.stats import truncnorm
            elapsed = (state.cutoff_ns - driver.pit_entry_ns) / 1e9
            mean = scenario.pit.absolute_lane_duration_s
            sd = scenario.pit.absolute_lane_duration_sd_s
            if elapsed < 0:
                raise WorkbenchError("field_unavailable", "Rival pit entry is after the decision cutoff")
            if sd is None or sd == 0:
                if mean <= elapsed:
                    raise WorkbenchError("field_unavailable", "Absolute lane duration is inconsistent with elapsed occupancy")
                paths[:, 0] = mean - elapsed
            else:
                paths[:, 0] = truncnorm.rvs((elapsed - mean) / sd, np.inf, loc=mean, scale=sd,
                                              size=n, random_state=rng) - elapsed
        else:
            elapsed = (state.cutoff_ns - driver.last_crossing_ns) / 1e9
            if elapsed < 0:
                raise WorkbenchError("field_unavailable", "Rival crossing is after the decision cutoff")
            length = min(3, first_length)
            q = getattr(first_forecast, "residual_quantiles_s", None)
            mask = getattr(first_forecast, "ml_residual_mask", None)
            mapped = map_residual_draws(first_blocks[:, :length], first_blocks,
                                        None if q is None else q[:length],
                                        None if mask is None else mask[:length])
            for draw in np.flatnonzero(paths[:, 0] <= elapsed):
                replicate = reps[draw] % len(first_parameter)
                valid = np.flatnonzero(first_parameter[replicate, 0] + mapped[:, 0] > elapsed)
                if not len(valid):
                    raise WorkbenchError("field_unavailable", "Rival in-progress lap lacks supported conditional duration")
                selected = int(rng.choice(valid))
                paths[draw, :length] = first_parameter[replicate, :length] + mapped[selected]
            paths[:, 0] -= elapsed
        if started_in_pit:
            paths[:, 1] += np.maximum(0., rng.normal(scenario.pit.outlap_warmup_s,
                                                     scenario.pit.outlap_warmup_sd_s, n))
        for stop in stops:
            index = stop - driver.completed_laps
            net_costs[:, index] = np.maximum(0., rng.normal(scenario.pit.net_transit_loss_s, scenario.pit.net_transit_sd_s, n))
            pit_costs[:, index] = (net_costs[:, index]
                                   + np.maximum(1., rng.normal(scenario.pit.service_s, scenario.pit.service_sd_s, n))
                                   + np.maximum(0., rng.normal(scenario.pit.outlap_warmup_s, scenario.pit.outlap_warmup_sd_s, n)))
            paths[:, index] += pit_costs[:, index]
        if not np.isfinite(paths).all() or np.any(paths <= 0):
            raise WorkbenchError("field_unavailable", "Rival sampled paths are nonpositive or nonfinite")
        route_stops = ([driver.completed_laps + 1] if started_in_pit else []) + stops
        opponents[number] = {"paths": paths, "completed_laps": driver.completed_laps,
                             "stops": tuple(route_stops), "net_costs": net_costs,
                             "pit_costs": pit_costs, "started_in_pit": started_in_pit}
    return opponents


def run_field(state, scenario, focal_plan, focal_paths, opponents, *, focal_net_costs=None, focal_pit_costs=None):
    """Process crossing clocks at fixed distance and report approximate ranks.

    Same-lap following is resolved at the crossing; pit routes may reorder.
    Final timing remains to the focal scheduled distance, independent of flag.
    """
    n = len(focal_paths)
    times = np.zeros(n)
    ranks = np.zeros(n, dtype=int)
    rejoin = []
    flag_laps = np.zeros(n, dtype=int)
    flag_times = np.zeros(n)
    drivers = {state.driver_number: {"paths": focal_paths, "completed_laps": state.completed_lap,
                                    "stops": tuple(s.pit_lap for s in focal_plan.stops),
                                    "net_costs": focal_net_costs, "pit_costs": focal_pit_costs}, **opponents}
    expiry_ns = state.race_start_ns + 7_200_000_000_000
    released = state.laps
    expiry_lap = None
    if not released.empty:
        leaders = released[released.lap_end_ns.notna()].groupby("lap_number", as_index=False).lap_end_ns.min()
        eligible = leaders[(leaders.lap_end_ns >= expiry_ns) & (leaders.lap_end_ns <= state.cutoff_ns)]
        if not eligible.empty:
            expiry_lap = int(eligible.sort_values("lap_end_ns").iloc[0].lap_number)
    if expiry_lap is not None and max(data["completed_laps"] for data in drivers.values()) >= min(state.scheduled_laps, expiry_lap + 1):
        raise WorkbenchError("field_unavailable", "The timed-race flag has already been reached")
    elapsed_at_cutoff = (state.cutoff_ns - state.race_start_ns) / 1e9
    for draw in range(n):
        queue = []
        clocks = {number: 0.0 for number in drivers}
        lap = {number: data["completed_laps"] for number, data in drivers.items()}
        crossing = {}
        leader_lap = max(lap.values())
        initial_leader_lap = leader_lap
        penalized = set()
        flag_lap = min(state.scheduled_laps, expiry_lap + 1) if expiry_lap is not None else state.scheduled_laps
        expiry_seen = expiry_lap is not None
        flag_time = None
        classification = {}
        sequence = 0
        scheduled = {}
        scheduled_index = {}
        scheduled_start = {}
        scheduled_pace = {}
        scheduled_fixed = {}

        def active(relative):
            return next((item for item in scenario.neutralisations
                         if item.start_after_leader_laps <= relative < item.start_after_leader_laps + item.duration_laps), None)

        def pace_factor(neutral):
            return 1.0 if neutral is None else (1.6 if neutral.kind == "SC" else 1.4)

        def schedule(car, index, instant, neutral):
            nonlocal sequence
            data = drivers[car]
            total = float(data["paths"][draw, index])
            fixed_array = data.get("pit_costs")
            net_array = data.get("net_costs")
            fixed = 0.0 if fixed_array is None else float(fixed_array[draw, index])
            net = 0.0 if net_array is None else float(net_array[draw, index])
            if neutral is not None:
                discount = scenario.pit.sc_net_transit_factor if neutral.kind == "SC" else scenario.pit.vsc_net_transit_factor
                fixed -= net * (1.0 - discount)
            pace = total - (0.0 if fixed_array is None else float(fixed_array[draw, index]))
            factor = 1.0 if data.get("started_in_pit") and index == 0 else pace_factor(neutral)
            scheduled_start[car], scheduled_pace[car], scheduled_fixed[car] = instant, pace, fixed
            scheduled_index[car] = index
            scheduled[car] = instant + pace * factor + fixed
            heapq.heappush(queue, (scheduled[car], sequence, car, index))
            sequence += 1

        def transition(instant, previous, new):
            nonlocal sequence
            for car in tuple(scheduled):
                idx = scheduled_index[car]
                if drivers[car].get("started_in_pit") and idx == 0:
                    continue
                old_factor = pace_factor(previous)
                span = scheduled[car] - scheduled_start[car]
                fraction = max(0.0, min(1.0, (scheduled[car] - instant) / span)) if span > 0 else 0.0
                pace = scheduled_pace[car] * fraction
                fixed = max(0.0, scheduled[car] - instant - pace * old_factor)
                scheduled_start[car], scheduled_pace[car], scheduled_fixed[car] = instant, pace, fixed
                scheduled[car] = instant + pace * pace_factor(new) + fixed
                heapq.heappush(queue, (scheduled[car], sequence, car, idx))
                sequence += 1

        order = sorted(drivers, key=lambda number: (-state.drivers[number].progress_laps, number))
        for number in drivers:
            schedule(number, 0, 0.0, active(0))
        while queue:
            instant, _, number, index = heapq.heappop(queue)
            if number not in scheduled or scheduled_index[number] != index or instant != scheduled[number]:
                continue
            data = drivers[number]
            current_lap = lap[number] + 1
            route_change = current_lap in data["stops"] or current_lap - 1 in data["stops"]
            neutral = active(leader_lap - initial_leader_lap)
            neutral_kind = None if neutral is None else neutral.kind
            if neutral is not None and ((current_lap in data["stops"] and not neutral.pit_entry_open)
                                        or (current_lap - 1 in data["stops"] and not neutral.pit_exit_open)):
                raise WorkbenchError("field_unavailable", "A simulated stop conflicts with a hypothetical pit closure")
            position = order.index(number)
            if position and not route_change:
                ahead = order[position - 1]
                ahead_route = current_lap in drivers[ahead]["stops"] or current_lap - 1 in drivers[ahead]["stops"]
                if lap[ahead] + 1 == current_lap and ahead in scheduled and not ahead_route:
                    advantage = scheduled[ahead] - instant
                    if 0 <= advantage and (neutral_kind is not None or advantage <= scenario.overtaking_margin_s):
                        scheduled[number] = scheduled[ahead] + scenario.following_headway_s
                        heapq.heappush(queue, (scheduled[number], sequence, number, index))
                        sequence += 1
                        continue
                    if advantage > scenario.overtaking_margin_s:
                        order[position - 1], order[position] = order[position], order[position - 1]
            lap[number] = current_lap
            scheduled.pop(number, None)
            gap = None
            if not route_change:
                preceding = [t for other, t in crossing.get(current_lap, {}).items() if other != number and
                             0 <= instant - t and (neutral_kind == "SC" or instant - t < scenario.traffic_gap_s)]
                if preceding:
                    gap = instant - max(preceding)
                    if (number, index) not in penalized:
                        penalty = scenario.traffic_penalty_s if neutral_kind is None else 0.0
                        adjusted = max(instant + penalty, max(preceding) + scenario.following_headway_s)
                        if adjusted > instant:
                            lap[number] -= 1
                            scheduled[number] = adjusted
                            heapq.heappush(queue, (adjusted, sequence, number, index))
                            sequence += 1
                            penalized.add((number, index))
                            continue
            clocks[number] = instant
            crossing.setdefault(current_lap, {})[number] = instant
            if route_change:
                order.sort(key=lambda car: (-lap[car], clocks[car], car))
            if current_lap > leader_lap:
                leader_lap = current_lap
                next_neutral = active(leader_lap - initial_leader_lap)
                if next_neutral != neutral:
                    transition(instant, neutral, next_neutral)
                if not expiry_seen and elapsed_at_cutoff + instant >= 7200.0:
                    expiry_seen = True
                    flag_lap = min(state.scheduled_laps, current_lap + 1)
                if current_lap >= flag_lap and flag_time is None:
                    flag_time = instant
            if number == state.driver_number and current_lap - 1 in data["stops"]:
                rejoin.append(instant)
            if index + 1 < data["paths"].shape[1]:
                schedule(number, index + 1, instant, active(leader_lap - initial_leader_lap))
                if neutral_kind == "SC" and gap is not None:
                    scheduled[number] = max(instant + 1.0, scheduled[number] - max(0.0, gap - 1.0) / 2)
                    heapq.heappush(queue, (scheduled[number], sequence, number, index + 1))
                    sequence += 1
        focal = state.driver_number
        times[draw] = clocks[focal]
        if flag_time is not None:
            for crossing_lap, entries in crossing.items():
                for car, instant in entries.items():
                    if instant >= flag_time and (car not in classification or instant < classification[car][1]):
                        classification[car] = (crossing_lap, instant)
        if focal not in classification or len(classification) != len(drivers):
            raise WorkbenchError("field_unavailable", "Flag-limited classification could not be resolved")
        order_at_flag = sorted(classification, key=lambda number: (-classification[number][0], classification[number][1], number))
        ranks[draw] = order_at_flag.index(focal) + 1
        flag_laps[draw] = flag_lap
        flag_times[draw] = flag_time
    return times, ranks, {"outlap_crossings_s": rejoin, "flag_laps": flag_laps.tolist(),
                          "flag_times_s": flag_times.tolist()}

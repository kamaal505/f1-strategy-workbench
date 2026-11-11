"""Scenario expansion and bounded, deterministic tyre-set plan search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pydantic import ValidationError

from ..contracts import NeutralisationSpec, PlannedStop, ScenarioSpec, StrategyPlan, TyreSet, WorkbenchError
from ..identity import digest


def default_scenario(state):
    inventory = {
        driver: tuple(TyreSet(set_id=f"virtual:{driver}:{compound}", compound=compound,
                              age_laps=0, provenance="assumed")
                      for compound in ("HARD", "MEDIUM", "SOFT"))
        for driver in sorted(state.drivers)
    }
    status = "" if state.track_status.empty else str(state.track_status.iloc[-1].get("status", ""))
    active = ()
    if status in ("SC", "VSC", "VSC_ENDING"):
        kind = "SC" if status == "SC" else "VSC"
        active = (NeutralisationSpec(kind=kind, start_after_leader_laps=0,
                                     duration_laps=3 if kind == "SC" else 2),)
    return ScenarioSpec(name="Assumed baseline", inventory_by_driver=inventory,
                        inventory_provenance="assumed: one unused set per dry compound per driver",
                        neutralisations=active)


def load_scenario(path_or_dict, state):
    try:
        if isinstance(path_or_dict, (str, Path)):
            data = json.loads(Path(path_or_dict).read_text())
        elif isinstance(path_or_dict, ScenarioSpec):
            data = path_or_dict.model_dump(mode="json")
        else:
            data = dict(path_or_dict)
        inventory = data.get("inventory_by_driver", {})
        unknown = set(inventory) - set(state.drivers)
        if unknown:
            raise WorkbenchError("invalid_scenario", "Inventory contains an unknown driver")
        # An omitted driver receives the same declared virtual inventory as the default;
        # an explicit empty list means no available future sets.
        defaults = default_scenario(state).model_dump(mode="json")["inventory_by_driver"]
        data["inventory_by_driver"] = {driver: inventory.get(driver, defaults[driver]) for driver in defaults}
        active = default_scenario(state).model_dump(mode="json")["neutralisations"]
        declared = list(data.get("neutralisations", ()))
        if active and not any(item.get("start_after_leader_laps") == 0 for item in declared):
            declared = active + declared
        data["neutralisations"] = declared
        scenario = ScenarioSpec.model_validate_json(json.dumps(data))
        ordered = sorted(scenario.neutralisations, key=lambda item: item.start_after_leader_laps)
        if any(a.start_after_leader_laps + a.duration_laps > b.start_after_leader_laps
               for a, b in zip(ordered, ordered[1:])):
            raise WorkbenchError("invalid_scenario", "Neutralisation intervals overlap")
        return scenario
    except WorkbenchError:
        raise
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise WorkbenchError("invalid_scenario", f"Invalid scenario: {exc}") from None


def _plan(branch, stops):
    key = {"branch": branch, "stops": [(s.pit_lap, s.new_set_id) for s in stops]}
    return StrategyPlan(plan_id=digest(key), branch=branch, stops=stops)


def enumerate_plans(state, scenario, model):
    """Return legal completions from a width-64, depth-at-most-three beam.

    A lack of returned plans is not proof of infeasibility. The caller must report
    search exhaustion unless a direct constraint independently proves it.
    """
    from .. import rules
    driver = state.drivers[state.driver_number]
    inventory = scenario.inventory_by_driver.get(state.driver_number, ())
    final = state.scheduled_laps
    start = state.completed_lap
    from ..pace.hierarchical import forecast_laps
    if driver.compound not in ("SOFT", "MEDIUM", "HARD") or driver.tyre_age_laps is None:
        raise WorkbenchError("insufficient_evidence", "Mounted compound or tyre age is unknown")
    if start >= final:
        return ()
    mounted = TyreSet(set_id=driver.mounted_set_id, compound=driver.compound,
                      age_laps=driver.tyre_age_laps, provenance="assumed")
    by_id = {tyre.set_id: tyre for tyre in inventory}
    cache = {}

    def curve(tyre, first):
        key = (tyre.set_id, first)
        if key not in cache:
            laps = np.arange(first, final + 1, dtype=np.int64)
            ages = np.arange(tyre.age_laps + 1, tyre.age_laps + 1 + len(laps), dtype=np.int64)
            forecast = forecast_laps(state, model, driver_number=state.driver_number,
                                     compound=tyre.compound, race_laps=laps,
                                     tyre_ages=ages, assumptions=scenario.pace,
                                     outlap_race_lap=None if tyre.set_id == mounted.set_id else first)
            values = np.asarray(forecast.mean_s, dtype=float)
            if values.shape != laps.shape or not np.isfinite(values).all():
                raise WorkbenchError("insufficient_evidence", "Invalid candidate pace forecast")
            cache[key] = values
        return cache[key]

    optimistic = curve(mounted, start + 1).copy()
    for tyre in inventory:
        for first in range(start + 2, final + 1):
            offset = first - start - 1
            optimistic[offset:] = np.minimum(optimistic[offset:], curve(tyre, first))

    def isolated(stops, *, partial):
        total = 0.0
        first = start + 1
        tyre = mounted
        for stop in stops:
            total += float(curve(tyre, first)[:stop.pit_lap - first + 1].sum())
            total += scenario.pit.net_transit_loss_s + scenario.pit.service_s + scenario.pit.outlap_warmup_s
            first = stop.pit_lap + 1
            tyre = by_id[stop.new_set_id]
        if first <= final:
            if partial:
                total += float(optimistic[first - start - 1:].sum())
            else:
                total += float(curve(tyre, first).sum())
        return total

    beam = [()]
    completed = []
    expansions = 0
    for depth in range(scenario.max_additional_stops + 1):
        next_beam = []
        for stops in beam:
            branch = "pit_now" if stops and stops[0].pit_lap == start + 1 else "stay_out"
            candidate = _plan(branch, stops)
            if not rules.plan_legality(state, scenario, candidate):
                completed.append(candidate)
            if depth == scenario.max_additional_stops:
                continue
            first_lap = stops[-1].pit_lap + 1 if stops else start + 1
            consumed = {s.new_set_id for s in stops}
            for lap in range(first_lap, final):
                for tyre in sorted(inventory, key=lambda x: x.set_id):
                    if tyre.set_id in consumed or tyre.set_id == driver.mounted_set_id:
                        continue
                    expansions += 1
                    if expansions > 50_000:
                        raise WorkbenchError("capacity_exceeded", "Plan expansion budget exceeded")
                    next_beam.append(stops + (PlannedStop(pit_lap=lap, new_set_id=tyre.set_id),))
        next_beam.sort(key=lambda seq: (isolated(seq, partial=True),
                                        tuple(s.pit_lap for s in seq),
                                        tuple(s.new_set_id for s in seq)))
        # Preserve both branches when pruning.
        immediate = [s for s in next_beam if s[0].pit_lap == start + 1][:32]
        later = [s for s in next_beam if s[0].pit_lap > start + 1][:32]
        beam = immediate + later
        retained = {tuple((stop.pit_lap, stop.new_set_id) for stop in seq) for seq in beam}
        for seq in next_beam:
            key = tuple((stop.pit_lap, stop.new_set_id) for stop in seq)
            if len(beam) == 64:
                break
            if key not in retained:
                beam.append(seq)
                retained.add(key)
    selected = []
    for branch in ("pit_now", "stay_out"):
        candidates = [plan for plan in completed if plan.branch == branch]
        candidates.sort(key=lambda plan: (isolated(plan.stops, partial=False),
                                           tuple(s.pit_lap for s in plan.stops),
                                           tuple(s.new_set_id for s in plan.stops)))
        selected.extend(candidates[:3])
    return tuple(selected)

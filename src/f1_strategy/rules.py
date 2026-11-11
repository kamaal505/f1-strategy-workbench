"""Conditional sporting constraints for the supported dry-race domain."""
from __future__ import annotations

import re
import pandas as pd
from .contracts import RaceState, ScenarioSpec, StrategyPlan

DRY = frozenset(("SOFT", "MEDIUM", "HARD"))
WET = frozenset(("INTERMEDIATE", "WET"))


def _qatar(state: RaceState) -> bool:
    return not state.synthetic and state.year == 2023 and state.round_number == 18


def _closures(state: RaceState) -> tuple[bool, bool, bool]:
    entry = exit_ = True
    routing = False
    for row in state.messages.itertuples(index=False):
        text = str(row.message or "").upper()
        if re.search(r"(?:SAFETY CAR|SC).*(?:THROUGH|VIA)\s+(?:THE\s+)?PIT\s+LANE", text):
            routing = True
        for kind in ("ENTRY", "EXIT"):
            if not re.search(r"\bPIT\s+(?:LANE\s+)?" + kind + r"\b", text):
                continue
            if re.search(r"\b(?:CLOSED|CLOSING)\b", text):
                if kind == "ENTRY": entry = False
                else: exit_ = False
            elif re.search(r"\b(?:OPEN|OPENED)\b", text):
                if kind == "ENTRY": entry = True
                else: exit_ = True
    return entry, exit_, routing


def check_state(state: RaceState, scenario: ScenarioSpec) -> tuple[str, ...]:
    flags = []
    if state.year not in (2022, 2023, 2024):
        flags.append("unsupported_season")
    if any(d.used_compounds & WET for d in state.drivers.values()):
        flags.append("unsupported_wet")
    focal_laps = state.laps[state.laps.driver_number == state.driver_number]
    if not focal_laps.empty and (~focal_laps.compound.fillna("").astype(str).str.upper().isin(DRY | WET)).any():
        flags.append("unknown_tyre_history")
    if not state.weather.empty and state.weather.rainfall.fillna(False).astype(bool).any():
        flags.append("unsupported_wet")
    if (state.weather.empty or pd.isna(state.weather.iloc[-1].rainfall)) and scenario.dry_condition_assumption != "dry":
        flags.append("unknown_dry_conditions")
    track = state.track_status
    session = state.session_status
    if (not track.empty and (track.status.fillna("").astype(str).str.upper() == "RED").any()) or (
        not session.empty and session.status.fillna("").astype(str).str.contains(r"RED|SUSPEND|ABORT", case=False, regex=True).any()
    ):
        flags.append("unsupported_red_flag")
    # Unknown event-date applicability and nomination are never promoted by season alone.
    override = scenario.rules_override
    if (not override or override.provenance != "operator_documented" or
        state.event_metadata.get("rules_provenance") != "operator_documented" or
        state.event_metadata.get("nomination_provenance") != "operator_documented"):
        flags.append("conditional_rules")
    if scenario.inventory_provenance != "operator_documented" or any(
        tyre.provenance != "operator_documented" for tyres in scenario.inventory_by_driver.values() for tyre in tyres
    ):
        flags.append("conditional_inventory")
    entry, exit_, routing = _closures(state)
    if override:
        entry = entry and override.pit_entry_open
        exit_ = exit_ and override.pit_exit_open
    if not entry: flags.append("pit_entry_closed")
    if not exit_: flags.append("pit_exit_closed")
    if routing:
        flags.append("unsupported_directive")
    if _qatar(state):
        # FIA set-specific remaining-lap criteria were communicated privately to teams;
        # ordinary FastF1 TyreLife alone does not establish those criteria.
        if not state.event_metadata.get("qatar_tyre_life_counting_documented", False):
            flags.append("unsupported_directive")
        if any(d.tyre_age_laps is None for d in state.drivers.values()):
            flags.append("unsupported_directive")
        if any(t.provenance != "operator_documented" for ts in scenario.inventory_by_driver.values() for t in ts):
            flags.append("unsupported_directive")
    return tuple(dict.fromkeys(flags))


def plan_legality(state: RaceState, scenario: ScenarioSpec, plan: StrategyPlan, driver_number: str | None = None) -> tuple[str, ...]:
    driver_number = driver_number or state.driver_number
    driver = state.drivers.get(driver_number)
    if driver is None:
        return ("unknown_driver",)
    reasons = []
    if driver.compound not in DRY or driver.tyre_age_laps is None:
        return ("unknown_mounted_tyre",)
    inventory = {tyre.set_id: tyre for tyre in scenario.inventory_by_driver.get(driver_number, ())}
    used = set(driver.used_compounds) & DRY
    used.add(driver.compound)
    override = scenario.rules_override
    min_specs = override.min_dry_specs if override else 2
    mandatory = set(override.mandatory_race_compounds if override else ("MEDIUM", "HARD"))
    cap = override.max_set_life_laps if override else None
    if _qatar(state):
        cap = min(cap, 18) if cap else 18
    current_age = driver.tyre_age_laps
    current_limit = cap
    current_lap = driver.completed_laps
    stops = plan.stops
    if len(stops) > scenario.max_additional_stops:
        reasons.append("too_many_stops")
    if plan.branch == "pit_now" and (not stops or stops[0].pit_lap != driver.completed_laps + 1):
        reasons.append("invalid_pit_now")
    if plan.branch == "stay_out" and stops and stops[0].pit_lap < driver.completed_laps + 2:
        reasons.append("invalid_stay_out")
    entry, exit_, _ = _closures(state)
    if override:
        entry = entry and override.pit_entry_open
        exit_ = exit_ and override.pit_exit_open
    for stop in stops:
        if stop.pit_lap <= current_lap or stop.pit_lap >= state.scheduled_laps:
            reasons.append("invalid_stop_lap")
        if not entry: reasons.append("pit_entry_closed")
        if not exit_: reasons.append("pit_exit_closed")
        current_age += max(0, stop.pit_lap - current_lap)
        if current_limit is not None and current_age > current_limit:
            reasons.append("set_life_exceeded")
        tyre = inventory.pop(stop.new_set_id, None)
        if tyre is None:
            reasons.append("unavailable_tyre_set")
            current_lap = stop.pit_lap
            continue
        used.add(tyre.compound)
        current_age = tyre.age_laps
        limits = [x for x in (cap, tyre.life_limit_laps) if x is not None]
        current_limit = min(limits) if limits else None
        current_lap = stop.pit_lap
    current_age += max(0, state.scheduled_laps - current_lap)
    if current_limit is not None and current_age > current_limit:
        reasons.append("set_life_exceeded")
    if len(used) < min_specs:
        reasons.append("insufficient_dry_compounds")
    if mandatory and not used.intersection(mandatory):
        reasons.append("missing_mandatory_compound")
    return tuple(dict.fromkeys(reasons))

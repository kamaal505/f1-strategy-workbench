"""Past-only pit-loss and absolute-lane occupancy evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def completed_pit_evidence(state, model, scenario):
    """Estimate total green loss and absolute occupancy from released visits.

    Each visit is an entry followed by an exit for the same car. A loss
    estimate additionally needs both completed laps, known clean start gaps,
    green status and a prior same-driver clean-lap anchor.
    """
    from ..pace.features import eligible_laps
    clean_model = getattr(model, "base_model", model)
    predictor = "hierarchical base model" if clean_model is not model else "supplied pace model"
    pits = state.pit_events
    if pits.empty:
        return {"visits": 0, "loss_samples": 0, "loss_s": None, "loss_mad_s": None,
                "occupancy_samples": 0, "occupancy_s": None, "occupancy_mad_s": None,
                "clean_predictor": predictor}
    events = pits[pits.release_ns.notna() & (pits.release_ns <= state.cutoff_ns)].sort_values(["driver_number", "release_ns", "source_row_id"])
    features = eligible_laps(state.laps, pits, state.track_status, state.weather,
                             cutoff_ns=state.cutoff_ns, event_metadata=state.event_metadata,
                             session_id=state.session_id)
    if not features.empty:
        features["driver_session"] = state.session_id + ":" + features.driver_number.astype(str)
        temperatures = state.weather.sort_values("release_ns").track_temp_c.dropna()
        reference = float(temperatures.iloc[0]) if len(temperatures) else np.nan
        features["track_temperature_delta_c"] = features.track_temperature_c - reference
    loss, occupancy = [], []
    pending = {}
    messages = state.messages
    for event in events.itertuples():
        number = str(event.driver_number)
        if event.kind == "entry":
            pending[number] = event
            continue
        if event.kind != "exit" or number not in pending:
            continue
        entry = pending.pop(number)
        if event.release_ns < entry.release_ns:
            continue
        occupancy.append((event.release_ns - entry.release_ns) / 1e9)
        if features.empty or pd.isna(entry.source_lap_number):
            continue
        inlap = int(entry.source_lap_number)
        rows = features[(features.driver_number.astype(str) == number) & features.lap_number.isin((inlap, inlap + 1))].sort_values("lap_number")
        if len(rows) != 2 or rows.lap_number.tolist() != [inlap, inlap + 1]:
            continue
        if rows.traffic_unknown.any() or any(set(reasons) - {"pit_lap"} for reasons in rows.exclusion_reasons):
            continue
        if not messages.empty:
            matching = messages[(messages.release_ns >= rows.iloc[0].lap_start_ns) &
                                (messages.release_ns <= rows.iloc[-1].lap_end_ns)]
            if matching.message.fillna("").astype(str).str.contains("PENALTY|TIME DELETED", case=False, regex=True).any():
                continue
        prior = features[(features.driver_number.astype(str) == number) & features.eligible & (features.lap_number < inlap)].tail(5)
        if len(prior) < 3:
            continue
        residual = prior.lap_time_s.to_numpy(dtype=float) - clean_model.predict_laps(prior, assumptions=scenario.pace)
        anchor = float(np.median(residual))
        clean = clean_model.predict_laps(rows, assumptions=scenario.pace) + anchor
        estimate = float(rows.lap_time_s.sum() - clean.sum() - scenario.pit.outlap_warmup_s)
        if np.isfinite(estimate) and estimate >= 0:
            loss.append(estimate)
    def summary(values):
        if len(values) < 3:
            return None, None
        median = float(np.median(values))
        return median, float(np.median(np.abs(np.asarray(values) - median)))
    total, total_mad = summary(loss)
    lane, lane_mad = summary(occupancy)
    return {"visits": len(occupancy), "loss_samples": len(loss), "loss_s": total, "loss_mad_s": total_mad,
            "occupancy_samples": len(occupancy), "occupancy_s": lane, "occupancy_mad_s": lane_mad,
            "clean_predictor": predictor}

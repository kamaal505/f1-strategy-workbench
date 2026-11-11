"""Chronological evaluation primitives and auditable corpus availability.

Only completed earlier events may supply a forecast.  Future observations are
used exclusively to decide whether a forecast has an observable clean target.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..contracts import PaceAssumptions, TrainingSpec, WorkbenchError
from ..identity import canonical, digest, confined


def metrics(rows):
    """Aggregate scored predictions, weighting races equally rather than laps."""
    if not rows:
        return {"n": 0, "events": 0, "race_macro_mae_s": None,
                "cumulative_mae_s": None, "coverage80": None, "coverage95": None,
                "width80_s": None, "width95_s": None, "pinball": None}
    events = defaultdict(list)
    for row in rows:
        events[row["session_id"]].append(row)
    def macro(fn):
        event_values = []
        for group in events.values():
            vals = [value for r in group if (value := fn(r)) is not None]
            if vals:
                event_values.append(float(np.mean(vals)))
        return float(np.mean(event_values)) if event_values else None
    def interval(r, coverage, field):
        bounds = r.get(f"interval{coverage}_s")
        if bounds is None:
            return None
        lo, hi = bounds
        if field == "coverage":
            return float(lo <= r["actual_s"] <= hi)
        return float(hi - lo)
    def pinball(r):
        scores = []
        for coverage in (80, 95):
            bounds = r.get(f"interval{coverage}_s")
            if bounds is None:
                continue
            alpha = (1 - coverage / 100) / 2
            for q, bound in ((alpha, bounds[0]), (1-alpha, bounds[1])):
                error = r["actual_s"] - bound
                scores.append(max(q * error, (q - 1) * error))
        return float(np.mean(scores)) if scores else None
    return {"n": len(rows), "events": len(events),
            "race_macro_mae_s": macro(lambda r: abs(r["prediction_s"] - r["actual_s"])),
            "cumulative_mae_s": macro(lambda r: abs(r["cumulative_prediction_s"] - r["cumulative_actual_s"]) if r.get("cumulative_prediction_s") is not None else None),
            "coverage80": macro(lambda r: interval(r, 80, "coverage")),
            "coverage95": macro(lambda r: interval(r, 95, "coverage")),
            "width80_s": macro(lambda r: interval(r, 80, "width")),
            "width95_s": macro(lambda r: interval(r, 95, "width")),
            "pinball": macro(pinball)}


def champion_gate(hierarchical, challenger, compounds):
    """Apply the development-only champion rule; missing evidence cannot win."""
    base = hierarchical.get("race_macro_mae_s")
    alt = challenger.get("race_macro_mae_s")
    cbase, calt = hierarchical.get("coverage80"), challenger.get("coverage80")
    if any(x is None for x in (base, alt, cbase, calt)) or base <= 0:
        return "hierarchical"
    pairs = [(b, a) for b, a in compounds.values() if b is not None and a is not None]
    if not pairs:
        return "hierarchical"
    worse = sum(a > b for b, a in pairs)
    return ("hgb" if alt <= .95 * base and abs(calt - .8) <= abs(cbase - .8)
            and worse <= len(pairs) / 2 else "hierarchical")


def paired_event_interval(base, challenger, *, seed=2026, draws=2000):
    """Paired event-block percentile interval for challenger minus base MAE."""
    common = sorted(set(base) & set(challenger))
    if len(common) < 2:
        return None
    diff = np.array([challenger[e] - base[e] for e in common], dtype=float)
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def permutation_importance_development(features, actual, session_ids, predict, *, seed=2026):
    """One-feature permutation increases in race-macro MAE on development only.

    ``predict`` consumes a frame and returns one prediction per row.  The caller
    supplies only already-frozen development folds; 2024 session IDs are
    rejected rather than silently used for explanation or selection.
    """
    if any(not s.startswith("2023-") or not 1 <= int(s.split("-")[1]) <= 16 for s in session_ids):
        raise WorkbenchError("future_artifact", "Permutation importance is development-only")
    actual = np.asarray(actual, dtype=float)
    sessions = np.asarray(session_ids)
    if len(features) != len(actual) or len(actual) != len(sessions) or not len(actual):
        raise WorkbenchError("invalid_evaluation", "Permutation inputs must have matching nonzero length")
    def error(frame):
        predictions = np.asarray(predict(frame), dtype=float)
        if predictions.shape != actual.shape or not np.isfinite(predictions).all():
            raise WorkbenchError("invalid_evaluation", "Invalid permutation predictions")
        absolute = np.abs(predictions - actual)
        return float(np.mean([absolute[sessions == s].mean() for s in sorted(set(sessions))]))
    baseline = error(features)
    rng = np.random.default_rng(seed)
    result = {}
    for name in features.columns:
        altered = features.copy()
        altered[name] = features[name].to_numpy()[rng.permutation(len(features))]
        result[str(name)] = error(altered) - baseline
    return result


def calibrate_predictions(rows):
    """Calculate separate lap and path-sum interval adjustments by horizon."""
    from .calibration import conformal_adjustment
    result = {}
    for horizon in (1, 3, 5, 10):
        for kind in ("lap", "cumulative"):
            actual_key = "actual_s" if kind == "lap" else "cumulative_actual_s"
            for coverage in (80, 95):
                bounds_key = ("" if kind == "lap" else "cumulative_") + f"interval{coverage}_s"
                selected = [r for r in rows if r["horizon"] == horizon
                            and r.get(bounds_key) is not None and r.get(actual_key) is not None]
                result[f"{horizon}:{kind}:{coverage}"] = conformal_adjustment(
                    [r[bounds_key][0] for r in selected], [r[bounds_key][1] for r in selected],
                    [r[actual_key] for r in selected], [r["session_id"] for r in selected],
                    coverage / 100)
    return result


def select_development(folds, *, seed=2026):
    """Select a challenger configuration from dated development predictions only.

    Each fold contains session_id, race_start_utc, fit_max_utc,
    calibration_max_utc, and predictions for hierarchical and each
    ``(max_leaf_nodes, min_samples_leaf)`` challenger setting.  Prediction
    records have the same fields as :func:`metrics`.  No holdout rows are
    accepted, even when their numeric score would not change the winner.
    """
    candidates = ((7, 30), (7, 60), (15, 30), (15, 60))
    baseline, alternatives = [], {key: [] for key in candidates}
    for fold in folds:
        session = fold["session_id"]
        if not session.startswith("2023-") or not 1 <= int(session.split("-")[1]) <= 16:
            raise WorkbenchError("future_artifact", "Selection admits only 2023 development races")
        start = datetime.fromisoformat(fold["race_start_utc"])
        for key in ("fit_max_utc", "calibration_max_utc"):
            used = datetime.fromisoformat(fold[key])
            if used.tzinfo is None or start.tzinfo is None or used >= start:
                raise WorkbenchError("future_artifact", "Selection or calibration does not precede the scored race")
        rows = fold["hierarchical"]
        if any(row["session_id"] != session for row in rows):
            raise WorkbenchError("invalid_evaluation", "Fold contains a mismatched event")
        baseline.extend(r for r in rows if r["horizon"] == 1)
        for candidate in candidates:
            rows = fold["challengers"].get(candidate, ())
            if any(row["session_id"] != session for row in rows):
                raise WorkbenchError("invalid_evaluation", "Fold contains a mismatched event")
            alternatives[candidate].extend(r for r in rows if r["horizon"] == 1)
    def keys(rows):
        return {(r["session_id"], r["driver_number"], r["completed_lap"]) for r in rows}
    baseline_keys = keys(baseline)
    valid = [(metrics(rows)["race_macro_mae_s"], candidate)
             for candidate, rows in alternatives.items()
             if rows and keys(rows) == baseline_keys and len(rows) == len(baseline)]
    if not valid or not baseline:
        return {"status": "unavailable", "reason": "insufficient_paired_development_predictions"}
    # Identical scores choose the smaller tree, then the larger minimum leaf.
    _, chosen = min(valid, key=lambda pair: (pair[0], pair[1][0], -pair[1][1]))
    alt = alternatives[chosen]
    groups = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        b = metrics([r for r in baseline if r["compound"] == compound])["race_macro_mae_s"]
        a = metrics([r for r in alt if r["compound"] == compound])["race_macro_mae_s"]
        groups[compound] = (b, a)
    winner = champion_gate(metrics(baseline), metrics(alt), groups)
    by_event = lambda rows: {s: metrics([r for r in rows if r["session_id"] == s])["race_macro_mae_s"]
                             for s in {r["session_id"] for r in rows}}
    return {"status": "ok", "champion": winner,
            "hgb_settings": {"max_leaf_nodes": chosen[0], "min_samples_leaf": chosen[1]},
            "hierarchical": metrics(baseline), "challenger": metrics(alt),
            "compound_mae_s": groups,
            "paired_event_mae_difference_ci95_s": paired_event_interval(by_event(baseline), by_event(alt), seed=seed)}


def _read_frozen(path, root):
    # Settings are operator-controlled local files, never a URL or symlink.
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise WorkbenchError("unsafe_path", "Frozen settings must be relative to the evaluation root")
    try:
        doc = json.loads(confined(root, str(p)).read_bytes())
        identity = doc["identity"]
        if doc["settings_id"] != digest(identity):
            raise ValueError
        if identity["champion"] not in ("hierarchical", "hgb"):
            raise ValueError
        if identity["partition"] != "development":
            raise ValueError
        maximum = datetime.fromisoformat(identity["max_selection_utc"])
        if maximum.tzinfo is None or maximum.year > 2023:
            raise ValueError
        if not identity["selection_snapshot_ids"]:
            raise ValueError
        from ..data.snapshots import open_snapshot
        from .corpus import _last_release
        for identifier in identity["selection_snapshot_ids"]:
            source = open_snapshot(root, identifier)
            if source.manifest.synthetic or source.manifest.year != 2023 or source.manifest.round_number > 16:
                raise ValueError
            latest = _last_release(source)
            if latest is None or latest.to_pydatetime() > maximum:
                raise ValueError
        calibration_ids = identity["calibration_snapshot_ids"]
        if not calibration_ids or set(calibration_ids) & set(identity["selection_snapshot_ids"]):
            raise ValueError
        for identifier in calibration_ids:
            source = open_snapshot(root, identifier)
            if source.manifest.synthetic or source.manifest.year != 2023 or not 17 <= source.manifest.round_number <= 22:
                raise ValueError
        return doc
    except (KeyError, ValueError, TypeError, OSError):
        raise WorkbenchError("invalid_frozen_settings", "Invalid or missing development-frozen settings") from None


def _validate_holdout_lineage(model, frozen, events):
    """Every transitive input must belong to its frozen chronological partition."""
    identity = frozen["identity"]
    by_id = {event["snapshot_id"]: event for event in events if event.get("snapshot_id")}
    selection = set(identity["selection_snapshot_ids"])
    calibration = set(identity["calibration_snapshot_ids"])
    selection_cutoff = datetime.fromisoformat(identity["max_selection_utc"])
    first_holdout = min(datetime.fromisoformat(event["race_start_utc"])
                        for event in events if event["partition"] == "holdout")
    seen = set()
    try:
        current = model
        while current is not None:
            lineage = {entry["snapshot_id"]: entry for entry in current.lineage}
            seen.update(lineage)
            cutoff = datetime.fromisoformat(current.metadata["data_cutoff_utc"])
            if cutoff.tzinfo is None or cutoff > first_holdout:
                raise ValueError
            for identifier, source in lineage.items():
                event = by_id[identifier]
                partition = event["partition"]
                maximum = datetime.fromisoformat(source["max_used_utc"])
                event_max = datetime.fromisoformat(event["max_release_utc"])
                if (partition not in ("initialization", "development", "calibration")
                        or source["session_id"] != event["session_id"] or source.get("synthetic")
                        or abs((maximum - event_max).total_seconds()) > 0.000001 or maximum >= cutoff
                        or (partition == "calibration" and identifier not in calibration)):
                    raise ValueError
            spec = current.metadata["training_spec"]
            specified = set().union(*(set(spec.get(field, ())) for field in (
                "training_snapshot_ids", "tuning_snapshot_ids", "selection_snapshot_ids", "calibration_snapshot_ids")))
            if set(lineage) != specified:
                raise ValueError
            role_fields = {"fit": "training_snapshot_ids", "tuning": "tuning_snapshot_ids",
                           "selection": "selection_snapshot_ids", "calibration": "calibration_snapshot_ids"}
            for identifier, source in lineage.items():
                if "roles" in source and set(source["roles"]) != {
                        role for role, field in role_fields.items() if identifier in spec.get(field, ())}:
                    raise ValueError
            for field, allowed in (("training_snapshot_ids", {"initialization", "development"}),
                                   ("tuning_snapshot_ids", {"initialization", "development"}),
                                   ("selection_snapshot_ids", {"development"}),
                                   ("calibration_snapshot_ids", {"calibration"})):
                for identifier in spec.get(field, ()):
                    if identifier not in lineage or by_id[identifier]["partition"] not in allowed:
                        raise ValueError
                    if field == "selection_snapshot_ids" and (identifier not in selection or
                            datetime.fromisoformat(lineage[identifier]["max_used_utc"]) > selection_cutoff):
                        raise ValueError
                    if field == "calibration_snapshot_ids" and identifier not in calibration:
                        raise ValueError
            for rows in current.metadata.get("row_lineage", {}).values():
                for row in rows:
                    if row["snapshot_id"] not in lineage:
                        raise ValueError
                    for identifier in row["base_training_snapshot_ids"]:
                        if identifier not in lineage or by_id[identifier]["partition"] not in ("initialization", "development"):
                            raise ValueError
            current = getattr(current, "base_model", None)
        if not selection | calibration <= seen:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise WorkbenchError("invalid_frozen_settings", "Frozen model contains an unpermitted source or role") from None


def same_stint_target(frame, *, driver_number, completed_lap, horizon):
    """Return an eligible future path or an explicit censoring category."""
    driver = frame[frame.driver_number.astype(str) == str(driver_number)]
    origin = driver[driver.lap_number == completed_lap]
    if len(origin) != 1:
        return None, "missing_decision_lap"
    wanted = list(range(completed_lap + 1, completed_lap + horizon + 1))
    target = driver[driver.lap_number.isin(wanted)].sort_values("lap_number")
    if target.lap_number.tolist() != wanted:
        return None, "missing_future_laps"
    if target.source_stint.isna().any() or origin.source_stint.isna().any() or any(target.source_stint != origin.source_stint.iloc[0]):
        return None, "stint_end_censored"
    if not target.eligible.all():
        return None, "future_lap_excluded"
    return target, None


def evaluate_event(snapshot, model, spec):
    """Score all scheduled decision prefixes; target eligibility is hindsight only."""
    from ..replay import build_race_state
    from .features import eligible_laps
    from .hierarchical import forecast_stint
    manifest = snapshot.manifest
    def frame(name):
        from ..data.schemas import SCHEMAS
        table = snapshot.tables.get(name)
        if table is None:
            import pandas as pd
            return pd.DataFrame(columns=SCHEMAS[name].names)
        return table.to_pandas()
    laps, pit, status, weather = (frame(n) for n in ("laps", "pit_events", "track_status", "weather"))
    timed = laps.release_ns.dropna()
    if timed.empty:
        return [], {"event_unavailable": 1}
    targets = eligible_laps(laps, pit, status, weather, cutoff_ns=int(timed.max()),
                            event_metadata=manifest.event_metadata, session_id=manifest.session_id,
                            historical=True)
    rows, counts = [], Counter()
    for driver in sorted(laps.driver_number.dropna().astype(str).unique()):
        for lap in range(spec.decision_lap_start, manifest.scheduled_laps - spec.end_margin_laps + 1, spec.decision_lap_step):
            counts["decision_prefixes"] += 1
            for horizon in spec.horizons:
                counts[f"h{horizon}:denominator"] += 1
            try:
                state = build_race_state(snapshot, driver_number=driver, completed_lap=lap)
                focal = state.drivers[driver]
                if focal.compound not in ("SOFT", "MEDIUM", "HARD") or focal.tyre_age_laps is None:
                    raise WorkbenchError("insufficient_evidence", "Unknown mounted tyre")
                forecast = forecast_stint(state, model, driver_number=driver, compound=focal.compound,
                    tyre_age_laps=int(focal.tyre_age_laps), horizon_laps=max(spec.horizons), assumptions=PaceAssumptions())
            except WorkbenchError as exc:
                for horizon in spec.horizons:
                    counts[f"h{horizon}:abstention:{exc.code}"] += 1
                continue
            for horizon in spec.horizons:
                target, reason = same_stint_target(targets, driver_number=driver, completed_lap=lap, horizon=horizon)
                if reason:
                    counts[f"h{horizon}:{reason}"] += 1
                    continue
                actual = target.lap_time_s.to_numpy(dtype=float)
                bounds80 = forecast.interval80_s[horizon - 1].tolist()
                bounds95 = forecast.interval95_s[horizon - 1].tolist()
                rows.append({"session_id": manifest.session_id, "driver_number": driver, "completed_lap": lap,
                    "horizon": horizon, "compound": focal.compound,
                    "layout": state.event_metadata.get("layout_id") or "unknown",
                    "prediction_s": float(forecast.mean_s[horizon - 1]), "actual_s": float(actual[-1]),
                    "cumulative_prediction_s": float(forecast.mean_s[:horizon].sum()),
                    "cumulative_actual_s": float(actual.sum()), "interval80_s": bounds80,
                    "interval95_s": bounds95,
                    "cumulative_interval80_s": forecast.cumulative_intervals.get(str(horizon), {}).get("interval80_s"),
                    "cumulative_interval95_s": forecast.cumulative_intervals.get(str(horizon), {}).get("interval95_s"),
                    "support": bool(forecast.support.get("advice_supported", False))})
                counts[f"h{horizon}:scored"] += 1
    return rows, dict(counts)


def run_backtest(spec, *, root: Path) -> dict:
    """Resolve the fixed corpus and emit auditable outputs, including missing races."""
    from .corpus import load_corpus, resolve_corpus
    corpus = load_corpus(spec.corpus_asset)
    events = resolve_corpus(corpus, root=root)
    frozen = _read_frozen(spec.frozen_settings_manifest, root) if spec.partition == "holdout" else None
    selected = [e for e in events if e["partition"] == spec.partition]
    from datetime import timedelta
    from ..data.snapshots import open_snapshot
    from .hierarchical import fit_pace
    from .artifacts import load_model
    audit_events, scored, development_folds = [], [], []
    for e in selected:
        item = dict(e)
        item["evaluation_status"] = "unavailable"
        item["evaluation_reason"] = item.get("reason") or "snapshot_unavailable"
        if item.get("snapshot_id"):
            try:
                snapshot = open_snapshot(root, item["snapshot_id"])
                start = snapshot.manifest.stream_origin_utc + timedelta(microseconds=snapshot.manifest.race_start_ns / 1000)
                if frozen:
                    model = load_model(root, frozen["identity"]["model_id"], target_start_utc=start)
                    if model.metadata["model_kind"] != frozen["identity"]["champion"]:
                        raise WorkbenchError("invalid_frozen_settings", "Frozen champion and model kind differ")
                    _validate_holdout_lineage(model, frozen, events)
                else:
                    prior = [p for p in events if p.get("snapshot_id") and p.get("max_release_utc")
                             and datetime.fromisoformat(p["max_release_utc"]) < start
                             and p["partition"] in ("initialization", "development")]
                    prior.sort(key=lambda p: p["max_release_utc"])
                    reserve = max(2, math.ceil(len(prior) * .2))
                    if len(prior) - reserve < 5:
                        raise WorkbenchError("insufficient_evidence", "At least five fit and two earlier calibration events are required")
                    training = TrainingSpec(training_snapshot_ids=tuple(p["snapshot_id"] for p in prior[:-reserve]),
                                            calibration_snapshot_ids=tuple(p["snapshot_id"] for p in prior[-reserve:]),
                                            data_cutoff_utc=start, model_kind="hierarchical", seed=spec.seed)
                    reference = fit_pace(training, root=root)
                    model = load_model(root, reference.model_id, target_start_utc=start)
                rows, counts = evaluate_event(snapshot, model, spec)
                if not frozen and rows:
                    from .challenger import fit_challenger, load_challenger
                    alternatives, failures = {}, {}
                    for leaves in (7, 15):
                        for minimum in (30, 60):
                            try:
                                challenger_spec = training.model_copy(update={"model_kind": "hgb",
                                    "hyperparameters": {"max_leaf_nodes": leaves, "min_samples_leaf": minimum}})
                                ref = fit_challenger(challenger_spec, root=root)
                                challenger = load_challenger(root, ref.model_id, target_start_utc=start)
                                alternatives[(leaves, minimum)], _ = evaluate_event(snapshot, challenger, spec)
                            except WorkbenchError as exc:
                                failures[f"{leaves}:{minimum}"] = exc.code
                    item["challenger_failures"] = failures
                    development_folds.append({"session_id": item["session_id"], "race_start_utc": start.isoformat(),
                        "fit_max_utc": max(p["max_release_utc"] for p in prior[:-reserve]),
                        "calibration_max_utc": max(p["max_release_utc"] for p in prior[-reserve:]),
                        "hierarchical": rows, "challengers": alternatives})
                scored.extend(rows)
                item.update(evaluation_status="scored" if rows else "unavailable",
                            evaluation_reason=None if rows else "no_eligible_targets",
                            model_id=model.model_id, counts=counts)
            except WorkbenchError as exc:
                item["evaluation_reason"] = exc.code
        audit_events.append(item)
    scored_events = sum(e["evaluation_status"] == "scored" for e in audit_events)
    selection = select_development(development_folds, seed=spec.seed) if not frozen else None
    identity = {"spec": spec.model_dump(mode="json"), "events": audit_events,
                "frozen_settings_id": frozen["settings_id"] if frozen else None}
    identifier = digest(identity)
    directory = root / "evaluation" / identifier
    if root.is_symlink() or (root / "evaluation").is_symlink() or directory.is_symlink():
        raise WorkbenchError("unsafe_path", "Symlink evaluation directory forbidden")
    directory.mkdir(parents=True, exist_ok=True)
    status = "ok" if scored_events == len(selected) and selected else ("partial" if scored_events else "unavailable")
    frozen_path = None
    if not frozen and selection["status"] == "ok":
        fitting = [e for e in events if e["partition"] in ("initialization", "development")]
        calibration = [e for e in events if e["partition"] == "calibration"]
        if all(e.get("snapshot_id") for e in fitting + calibration):
            try:
                first_holdout = min(datetime.fromisoformat(e["race_start_utc"]) for e in events if e["partition"] == "holdout")
                chosen = selection["champion"]
                final_training = TrainingSpec(training_snapshot_ids=tuple(e["snapshot_id"] for e in fitting),
                    calibration_snapshot_ids=tuple(e["snapshot_id"] for e in calibration),
                    data_cutoff_utc=first_holdout, model_kind=chosen, seed=spec.seed,
                    hyperparameters=selection["hgb_settings"] if chosen == "hgb" else {})
                if chosen == "hgb":
                    from .challenger import fit_challenger
                    final_model = fit_challenger(final_training, root=root)
                else:
                    final_model = fit_pace(final_training, root=root)
                frozen_identity = {"partition": "development", "champion": chosen,
                    "settings": selection["hgb_settings"], "model_id": final_model.model_id,
                    "selection_snapshot_ids": [e["snapshot_id"] for e in selected],
                    "calibration_snapshot_ids": [e["snapshot_id"] for e in calibration],
                    "max_selection_utc": max(e["max_release_utc"] for e in selected)}
                frozen_doc = {"settings_id": digest(frozen_identity), "identity": frozen_identity}
                frozen_path = directory / "frozen-settings.json"
                if frozen_path.is_symlink():
                    raise WorkbenchError("unsafe_path", "Symlink frozen settings forbidden")
                frozen_path.write_bytes(canonical(frozen_doc))
            except WorkbenchError as exc:
                selection["frozen_settings_unavailable_reason"] = exc.code
        else:
            selection["frozen_settings_unavailable_reason"] = "missing_fit_or_calibration_snapshot"
    scorecard = {"evaluation_id": identifier, "status": status, "partition": spec.partition,
                 "event_denominator": len(selected), "scored_events": scored_events,
                 "unavailable_events": len(selected) - scored_events,
                 "metrics_by_horizon": {str(h): metrics([r for r in scored if r["horizon"] == h]) for h in spec.horizons},
                 "compound_breakdown": {compound: metrics([r for r in scored if r["compound"] == compound]) for compound in ("SOFT", "MEDIUM", "HARD")},
                 "circuit_breakdown": {layout: metrics([r for r in scored if r["layout"] == layout]) for layout in sorted({r["layout"] for r in scored})},
                 "support_coverage": sum(r["support"] for r in scored) / len(scored) if scored else None,
                 "reason": None if scored else "No completed chronological model evaluation is available",
                 "development_selection": selection,
                 "censoring_note": "Stint-ending censoring is informative; future eligibility is not a forecast input."}
    audit = {"evaluation_id": identifier, "identity": identity, "events": audit_events}
    score_path, audit_path = directory / "scorecard.json", directory / "audit.json"
    for path, document in ((score_path, scorecard), (audit_path, audit)):
        if path.is_symlink():
            raise WorkbenchError("unsafe_path", "Symlink evaluation output forbidden")
        path.write_bytes(canonical(document))
    return {"status": status, "evaluation_id": identifier,
            "scorecard_path": str(score_path), "audit_path": str(audit_path),
            "frozen_settings_path": str(frozen_path) if frozen_path else None,
            "event_denominator": len(selected), "scored_events": scored_events}

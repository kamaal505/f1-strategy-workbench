"""Streamlit presentation over the same replay, scenario and decision core as the CLI."""
from __future__ import annotations

import base64
import json
import os
import tempfile
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .contracts import DecisionResult, REPLAY_NOTICE, WorkbenchError
from .data.snapshots import list_snapshots, open_snapshot
from .identity import canonical
from .payloads import prefix_payload
from .replay import build_race_state
from .strategy.plans import default_scenario, load_scenario


@st.cache_resource
def _coordinator(root: str):
    from .deployment import Coordinator
    return Coordinator(Path(root))


@st.cache_resource
def _packaged_demo():
    from .deployment import materialize_packaged_demo
    return materialize_packaged_demo()


def _demo():
    local = Path("data/demo")
    if (local / "index.json").is_file():
        return local, json.loads((local / "index.json").read_text(encoding="utf-8"))
    packaged = files("f1_strategy").joinpath("assets", "demo", "index.json")
    if packaged.is_file():
        _owner, root = _packaged_demo()
        return root, json.loads((root / "index.json").read_text(encoding="utf-8"))
    return None, None


def _json(value):
    st.json(value, expanded=False)


def _authorize_view(lineage):
    from .deployment import authorize_lineage
    sources = tuple(lineage)
    if not sources:
        raise WorkbenchError("authorization_required", "Source lineage is unavailable")
    authorize_lineage(sources[:1], ("public_display",))
    authorize_lineage(sources[1:], ("model_server_use",))


def _scorecards(root: Path, *, local: bool):
    if not local:
        st.info("Public evaluation scorecards are unavailable unless separately authorized for display; no accuracy is inferred from the demo.")
        return
    from .identity import confined, digest
    evaluation = root / "evaluation"
    cards = []
    if evaluation.is_dir() and not evaluation.is_symlink():
        for directory in sorted(evaluation.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or len(directory.name) != 64:
                continue
            try:
                score_path = confined(root, f"evaluation/{directory.name}/scorecard.json")
                audit_path = confined(root, f"evaluation/{directory.name}/audit.json")
                if score_path.stat().st_size > 8_000_000 or audit_path.stat().st_size > 8_000_000:
                    continue
                score = json.loads(score_path.read_bytes())
                audit = json.loads(audit_path.read_bytes())
                if score["evaluation_id"] != directory.name or audit["evaluation_id"] != directory.name or digest(audit["identity"]) != directory.name:
                    continue
                cards.append(score)
            except (OSError, ValueError, KeyError, WorkbenchError):
                continue
    if not cards:
        st.info("No completed chronological scorecard is available in this catalog.")
    for card in cards:
        with st.expander(f"{card.get('partition', 'Unknown')} scorecard · {card.get('status', 'unknown')} · {card['evaluation_id'][:12]}"):
            st.caption("Historical evaluation, including unavailable events; not a counterfactual strategy guarantee.")
            _json(card)


def _telemetry(snapshot, state):
    table = snapshot.tables.get("car_samples")
    if table is None:
        st.info("Completed-lap telemetry was not acquired; strategy analysis does not require it.")
        return
    laps = state.laps[state.laps.driver_number == state.driver_number]
    choices = [int(x) for x in laps.lap_number.dropna().tolist()]
    if not choices:
        return
    lap = st.selectbox("Completed lap diagnostic", choices, index=len(choices)-1)
    row = laps[laps.lap_number == lap].iloc[-1]
    if pd.isna(row.lap_start_ns) or pd.isna(row.lap_end_ns):
        st.info("Lap timing boundaries unavailable.")
        return
    samples = table.to_pandas()
    samples = samples[(samples.driver_number == state.driver_number) & (samples.release_ns >= row.lap_start_ns) & (samples.release_ns <= min(row.lap_end_ns, state.cutoff_ns))].copy()
    if samples.empty:
        st.info("No released car samples for this completed lap.")
        return
    samples["Elapsed seconds"] = (samples.release_ns.astype(float) - float(row.lap_start_ns)) / 1e9
    channels = [name for name in ("Speed", "Throttle", "Brake") if name in samples]
    for channel in channels:
        st.plotly_chart(px.line(samples, x="Elapsed seconds", y=channel, title=channel), use_container_width=True)
    st.caption("Car and position streams remain separate. Time-aligned diagnostic only; binary brake is not braking force. No tyre temperature, pressure, fuel or setup inference.")
    original = snapshot.table("laps").to_pandas()
    matching = original[(original.driver_number == state.driver_number) & (original.lap_number == lap)]
    if not matching.empty:
        st.write("Completed-lap sector timings (historical source)", matching[["sector1_s", "sector2_s", "sector3_s"]])


def _results(result):
    from .reporting import delta_chart, forecast_chart
    p = result.model_dump(mode="json")
    st.subheader("Conditional decision")
    cols = st.columns(3)
    cols[0].metric("Recommendation", result.recommendation.replace("_", " "))
    delta = result.delta_stay_minus_pit_s
    cols[1].metric("Stay minus pit", f"{delta:+.2f} s" if delta is not None else "Unavailable")
    prob = result.prob_pit_faster
    cols[2].metric("Pit faster", f"{prob:.1%}" if prob is not None else "Unavailable")
    st.caption(f"{result.status} · {result.comparison_scope} · analysis {result.analysis_id}")
    st.write("Positive paired delta favours pit now; times are to the same scheduled distance, not a causal race outcome.")
    st.plotly_chart(delta_chart(p), use_container_width=True)
    st.plotly_chart(forecast_chart(p), use_container_width=True, key="decision_pace")
    if result.candidate_summaries:
        rows = []
        for item in result.candidate_summaries:
            plan = item.get("plan", {})
            rows.append({"branch": plan.get("branch"), "selected": plan.get("plan_id") in result.selected_plan_ids.values(),
                         "stops": ", ".join(f"lap {s.get('pit_lap')}: {s.get('new_set_id')}" for s in plan.get("stops", [])) or "No stop",
                         "screen mean (s)": item.get("screen_mean_s"), "flags": ", ".join(item.get("flags", []))})
        st.subheader("Screened pit windows")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Screening compares a bounded set of candidates; it does not prove a global optimum.")
    for label, item in (("Evidence and rationale", result.evidence), ("Selected plans", result.selected_plan_ids),
                        ("Candidate plans", result.candidate_summaries), ("Intervals and Monte Carlo error", result.intervals),
                        ("Fixed-pair sensitivities", result.sensitivity), ("Pit exit projection", result.pit_exit_projection),
                        ("Conditional positions", result.position_distribution), ("Quality and support flags", result.quality_flags),
                        ("Assumptions", result.assumptions)):
        with st.expander(label):
            _json(item)


def main():
    st.set_page_config(page_title="F1 strategy workbench", layout="wide")
    st.title("F1 strategy workbench")
    st.caption("Historical decision replay · conditional pace and strategy research")
    st.warning(REPLAY_NOTICE)
    st.info("Dry 2022–2024 Grand Prix races only. Wet forecasting, red-flag restarts, exact pit geometry and future retirements are not modelled. Hypothetical neutralisations are not forecasts.")
    local = os.environ.get("F1_STRATEGY_MODE", "public") == "local"
    demo_root, index = _demo()
    root = demo_root
    if local and Path("data/snapshots").is_dir():
        roots = ([demo_root] if demo_root is not None else []) + [Path("data")]
        root = st.sidebar.selectbox("Local catalog", roots, format_func=str)
        if root != demo_root:
            index = {}
    index = index or {}
    if not local:
        configured = os.environ.get("F1_STRATEGY_PUBLIC_ROOT")
        if configured:
            root = Path(configured)
            index = {}
            try:
                from .identity import confined
                index = json.loads(confined(root, "index.json").read_bytes())
            except (OSError, ValueError, WorkbenchError):
                pass
        st.sidebar.caption("Public mode · operator-curated catalog")
    if root is None:
        st.warning("Synthetic demo unavailable. An operator can create it with `f1-strategy synthetic --output-root data/demo` and train its recorded spec locally; hosted training is disabled.")
        return
    try:
        coordinator = None if local else _coordinator(str(root))
        if local:
            options = list_snapshots(root)
        else:
            from .deployment import authorize_lineage, public_catalog
            options = list(public_catalog(root, coordinator=coordinator))
            missing = sorted({"2023-07-R", "2023-15-R", "2024-03-R"} - {entry["session_id"] for entry in options})
            if missing:
                st.sidebar.info("Unavailable without an authorized registry and full transitive rights: " + ", ".join(missing))
        if not options:
            st.warning("No snapshots are available.")
            return
        target = index.get("target_snapshot_id")
        default = next((i for i, entry in enumerate(options) if entry["snapshot_id"] == target), 0)
        choice = st.sidebar.selectbox("Snapshot", options, index=default, format_func=lambda x: f"{x['session_id']} {'· synthetic' if x.get('synthetic', x['session_id'].startswith('SYN-')) else ''} · {x['snapshot_id'][:10]}")
        lease = nullcontext() if local else coordinator.lease((choice["snapshot_id"],))
        if not local:
            _authorize_view(choice["source_lineage"])
        with lease:
            snapshot = open_snapshot(root, choice["snapshot_id"])
            entries = [str(e["driver_number"]) for e in snapshot.manifest.entries]
            preferred = str((index if local else choice).get("driver_number", entries[0]))
            driver = st.sidebar.selectbox("Driver number", entries, index=entries.index(preferred) if preferred in entries else 0)
            completed = st.sidebar.number_input("Completed lap", min_value=1, max_value=int(snapshot.manifest.scheduled_laps), value=min(int((index if local else choice).get("completed_lap", 10)), int(snapshot.manifest.scheduled_laps)), step=1)
            state = build_race_state(snapshot, driver_number=driver, completed_lap=int(completed))
            prefix = prefix_payload(state)
            if not local:
                # The validated catalog is rechecked on each rerun, before snapshot display.
                _authorize_view(choice["source_lineage"])
            st.sidebar.caption(f"Cutoff: {state.cutoff_utc.isoformat()} · pit now means entry opportunity on lap {state.completed_lap + 1}.")
            tabs = st.tabs(["Decision replay", "Tyres & pace", "Pit windows & scenarios", "Model evidence"])
            with tabs[0]:
                st.subheader("Released race state")
                st.caption("Only observations released by the selected crossing are included; gaps are approximate and may be stale.")
                st.dataframe(pd.DataFrame.from_dict(prefix["drivers"], orient="index"), use_container_width=True)
                for label, value in (("Weather", prefix["weather"]), ("Track status", prefix["track_status"]), ("Race-control messages", prefix["messages"]), ("Prefix quality", prefix["quality_flags"])):
                    with st.expander(label):
                        _json(value)
                if st.checkbox("Post-race evaluation · reveal hindsight", value=False):
                    st.warning("HINDSIGHT — later observations, not decision inputs. The original request and analysis are unchanged.")
                    future = snapshot.table("laps").to_pandas()
                    future = future[(future.driver_number == driver) & (future.lap_end_ns > state.cutoff_ns)]
                    st.dataframe(future, use_container_width=True)
            with tabs[1]:
                st.subheader("Tyres & pace · released laps")
                from .pace.features import eligible_laps
                eligible = eligible_laps(state.laps, state.pit_events, state.track_status, state.weather,
                                         cutoff_ns=state.cutoff_ns, event_metadata=state.event_metadata,
                                         session_id=state.session_id)
                laps = eligible[eligible.driver_number == driver]
                if not laps.empty:
                    st.plotly_chart(px.scatter(laps, x="tyre_age_laps", y="lap_time_s", color="compound", symbol="eligible",
                                               hover_data=["lap_number", "exclusion_reasons", "warmup_first", "warmup_second"],
                                               title="Released laps by tyre age and model eligibility"), use_container_width=True)
                    columns = ["lap_number", "compound", "tyre_age_laps", "lap_time_s", "eligible", "exclusion_reasons", "warmup_first", "warmup_second", "traffic_unknown"]
                    st.dataframe(laps[columns], use_container_width=True, hide_index=True)
                st.caption("Eligibility and warm-up indicators are computed by the shared pace feature builder. Slow valid laps are retained. Tyre age includes prior use; fuel, track evolution and tyre effects are not separately identified.")
                saved = st.session_state.get("decision_payload")
                if saved and saved.get("snapshot_id") == state.snapshot_id and saved.get("driver_number") == driver and saved.get("completed_lap") == int(completed):
                    if not local:
                        _authorize_view(saved["source_lineage"])
                    from .reporting import forecast_chart
                    st.subheader("Selected-plan conditional pace")
                    st.plotly_chart(forecast_chart(saved), use_container_width=True, key="tyres_pace")
                    with st.expander("Forecast support, calibration and warm-up context"):
                        _json(saved.get("forecast_series", {}))
                    st.caption("Shaded bands are 80% forecast intervals, not causal strategy coverage. Outlap warm-up is assumed separately; first and second complete post-outlap laps use model indicators.")
                else:
                    st.info("Compare strategies to see the selected-plan forecasts and their support.")
                _telemetry(snapshot, state)
            with tabs[2]:
                st.subheader("Pit windows & scenarios")
                st.caption("Inventory and losses are assumptions, not measured team data. SC/VSC presets are hypothetical; active neutralisation duration must be explicitly assumed.")
                scenario = default_scenario(state).model_dump(mode="json")
                scenario_path = index.get("scenario_path")
                if not local and choice.get("scenario") is not None:
                    scenario = load_scenario(choice["scenario"], state).model_dump(mode="json")
                elif choice["snapshot_id"] == target and scenario_path:
                    from .identity import confined
                    candidate = Path(scenario_path)
                    if not candidate.is_absolute():
                        candidate = confined(root, scenario_path)
                    elif not local:
                        candidate = confined(root, str(candidate.relative_to(root.resolve())))
                    if candidate.is_file():
                        scenario = load_scenario(candidate, state).model_dump(mode="json")
                active = "" if state.track_status.empty else str(state.track_status.iloc[-1].get("status", ""))
                initial = 3 if active == "SC" else 4 if active in ("VSC", "VSC_ENDING") else 0
                preset = st.selectbox("Future neutralisation", ["No new neutralisation", "SC after 1 leader lap for 3 laps", "VSC after 1 leader lap for 2 laps", "Active SC: assumed 3 more leader laps", "Active VSC: assumed 2 more leader laps"], index=initial)
                kind = "VSC" if "VSC" in preset else "SC"
                scenario["neutralisations"] = [] if preset.startswith("No") else [{"kind": kind, "start_after_leader_laps": 0 if preset.startswith("Active") else 1, "duration_laps": 2 if kind == "VSC" else 3, "pit_entry_open": True, "pit_exit_open": True}]
                a, b, c = st.columns(3)
                scenario["pit"]["net_transit_loss_s"] = float(a.number_input("Net pit transit loss (s)", min_value=0.0, value=float(scenario["pit"]["net_transit_loss_s"]), step=1.0))
                scenario["pit"]["service_s"] = float(b.number_input("Stationary service (s)", min_value=1.0, value=float(scenario["pit"]["service_s"]), step=0.5))
                scenario["pit"]["outlap_warmup_s"] = float(c.number_input("Outlap warm-up (s)", min_value=0.0, value=float(scenario["pit"]["outlap_warmup_s"]), step=0.5))
                a, b, c = st.columns(3)
                scenario["pace"]["fuel_gain_s_per_lap"] = float(a.selectbox("Assumed fuel gain (s/lap)", [0.04, 0.06, 0.08], index=1))
                scenario["pace"]["track_drift_s_per_lap"] = float(b.selectbox("Assumed track drift (s/lap)", [-0.02, 0.0, 0.02], index=1))
                scenario["dry_condition_assumption"] = c.selectbox("Unknown-weather assumption", ["none", "dry"], index=1 if scenario.get("dry_condition_assumption") == "dry" else 0)
                scenario["max_additional_stops"] = int(st.slider("Maximum additional stops", 0, 3, int(scenario["max_additional_stops"])))
                with st.expander("Tyre inventory and complete scenario"):
                    st.caption("Edit future set IDs, compound, age, life limit and provenance as JSON. An empty list for a driver means no available sets.")
                    inventory = st.text_area("Inventory by driver (JSON)", json.dumps(scenario["inventory_by_driver"], indent=2), height=220)
                    _json(scenario)
                draws = int(st.select_slider("Evaluation draws", options=[200, 500, 1000] if not local else [200, 500, 1000, 2000, 5000], value=200 if not local else 1000))
                model_id = index.get("model_id") if local else choice.get("model_id")
                if local and (root / "models").is_dir():
                    candidates = sorted(p.name for p in (root / "models").iterdir() if p.is_dir() and len(p.name) == 64 and not p.is_symlink())
                    if candidates:
                        selected = candidates.index(model_id) if model_id in candidates else 0
                        model_id = st.selectbox("Trusted local model", candidates, index=selected)
                if not model_id:
                    training_path = Path(index["training_spec_path"]) if index.get("training_spec_path") else None
                    if training_path is not None and not training_path.is_absolute():
                        training_path = root / training_path
                    st.warning(f"Precomputed model unavailable. Locally run `f1-strategy train --root {root} --spec {training_path or '<training spec>'}`. No model is fitted by the app.")
                if st.button("Compare strategies", type="primary", disabled=not bool(model_id)):
                    scenario["inventory_by_driver"] = json.loads(inventory)
                    spec = load_scenario(scenario, state)
                    if local:
                        from .pace.artifacts import load_model
                        from .strategy.simulate import compare_strategies
                        model = load_model(root, model_id, target_start_utc=state.race_start_utc)
                        result = compare_strategies(state, model, spec, n_draws=draws, seed=2026)
                    else:
                        lineage = tuple(choice["source_lineage"])
                        _authorize_view(lineage)
                        job = {"snapshot_id": choice["snapshot_id"], "model_id": model_id, "driver_number": driver, "completed_lap": int(completed), "scenario": spec.model_dump(mode="json"), "n_draws": draws, "seed": 2026}
                        response = coordinator.run_job("compare", job, source_lineage=lineage)
                        result = DecisionResult.model_validate_json(canonical(response.get("decision", response)))
                        _authorize_view(result.source_lineage)
                    st.session_state["decision_payload"] = result.model_dump(mode="json")
                payload = st.session_state.get("decision_payload")
                if payload:
                    result = DecisionResult.model_validate_json(canonical(payload))
                    if result.snapshot_id != state.snapshot_id or result.driver_number != driver or result.completed_lap != int(completed):
                        st.info("The saved result belongs to a different replay selection; compare again for this prefix.")
                    else:
                        if not local:
                            _authorize_view(result.source_lineage)
                        st.caption("This result reflects the scenario saved with the analysis; changing controls does not recompute it. Press Compare strategies again to apply changes.")
                        _results(result)
                        if not local:
                            authorize_lineage(tuple(result.source_lineage), ("derived_report_download",))
                        if local:
                            from .reporting import render_report
                            with tempfile.TemporaryDirectory() as directory:
                                report = render_report(result, output_dir=Path(directory))
                                report_bytes = report.path.read_bytes()
                            st.download_button("Download offline HTML report", report_bytes, file_name=f"{result.analysis_id}.html", mime="text/html")
                        else:
                            report_bytes = coordinator.report_bytes(result)
                            encoded = base64.b64encode(report_bytes).decode("ascii")
                            authorize_lineage(tuple(result.source_lineage), ("derived_report_download",))
                            # Transfer authorized bytes in this response, not an unguarded
                            # long-lived Streamlit /media URL that bypasses later expiry.
                            st.markdown(
                                f'<a download="{result.analysis_id}.html" href="data:application/octet-stream;base64,{encoded}">Download offline HTML report</a>',
                                unsafe_allow_html=True,
                            )
            with tabs[3]:
                st.subheader("Model evidence and provenance")
                st.write("Model bundles are fitted before the race; training, tuning and calibration lineage must precede the target. The app never trains models.")
                st.write("Snapshot", snapshot.ref.snapshot_id, "Model", model_id or "unavailable")
                _scorecards(root, local=local)
                _json({"snapshot_quality": snapshot.manifest.quality, "availability": {k: v.model_dump(mode="json") for k, v in snapshot.manifest.availability.items()}, "source_ids": snapshot.manifest.source_ids, "event_metadata": snapshot.manifest.event_metadata})
                payload = st.session_state.get("decision_payload")
                if payload and payload.get("snapshot_id") == state.snapshot_id and payload.get("driver_number") == driver and payload.get("completed_lap") == int(completed):
                    if not local:
                        _authorize_view(payload["source_lineage"])
                    _json({k: payload[k] for k in ("model_id", "model_data_cutoff_utc", "source_lineage", "forecast_series", "quality_flags", "evidence")})
    except WorkbenchError as exc:
        st.error(f"{exc.code}: {exc.detail}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Input or artifact unavailable: {type(exc).__name__}")

"""Real CLI/core synthetic roundtrip, not mocked analytics."""
import json
from pathlib import Path
from typer.testing import CliRunner
from f1_strategy.cli import app


def test_offline_cli_workflow_and_saved_payload(tmp_path):
    runner = CliRunner()
    root = tmp_path / "demo"
    def command(*args):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)
    generated = command("synthetic", "--output-root", str(root))
    index = json.loads(Path(generated["index_path"]).read_text())
    trained = command("train", "--root", str(root), "--spec", index["training_spec_path"])
    checked = command("snapshot-check", "--root", str(root), "--snapshot", index["target_snapshot_id"])
    assert checked["valid"] is True
    prefix = command("replay", "--root", str(root), "--snapshot", index["target_snapshot_id"],
                     "--driver", index["driver_number"], "--lap", str(index["completed_lap"]))
    assert prefix["max_feature_release_ns"] <= prefix["cutoff_ns"]
    assert prefix["drivers"]["44"]["completed_laps"] == index["completed_lap"]
    assert_future_mutation_forecast_invariant(root, index, trained["model_id"])
    result = command("compare", "--root", str(root), "--snapshot", index["target_snapshot_id"],
                     "--model", trained["model_id"], "--driver", index["driver_number"],
                     "--lap", str(index["completed_lap"]), "--scenario", index["scenario_path"],
                     "--draws", "100", "--output-dir", str(tmp_path / "analyses"))
    decision = result["decision"]
    assert decision["status"] == "ok", decision
    assert decision["delta_stay_minus_pit_s"] < 0  # independently expected costly early stop
    assert set(decision["selected_plan_ids"]) == {"pit_now", "stay_out"}
    assert decision["simulation_series"]
    assert decision["source_lineage"]
    assert decision["scenario"]["inventory_provenance"].startswith("assumed")
    rendered = command("report", "--analysis", result["analysis_path"], "--output-dir", str(tmp_path / "rendered"))
    html = Path(rendered["path"]).read_text()
    assert decision["analysis_id"] in html
    assert "Event-time reconstructed replay" in html
    # A complete saved report does not require its original snapshot or model root.
    import shutil
    shutil.rmtree(root)
    command("report", "--analysis", result["analysis_path"], "--output-dir", str(tmp_path / "rendered-again"))
    payload_path = Path(result["analysis_path"])
    payload = json.loads(payload_path.read_text())
    payload["delta_stay_minus_pit_s"] += 1
    payload_path.write_text(json.dumps(payload))
    rejected = runner.invoke(app, ["report", "--analysis", str(payload_path), "--output-dir", str(tmp_path / "tampered")])
    assert rejected.exit_code == 4
    assert "corrupt_analysis" in rejected.output


def assert_future_mutation_forecast_invariant(root, index, model_id):
    from dataclasses import replace
    import numpy as np
    import pyarrow as pa
    from f1_strategy.contracts import PaceAssumptions
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.pace.artifacts import load_model
    from f1_strategy.pace.hierarchical import forecast_stint
    from f1_strategy.replay import build_race_state
    snapshot = open_snapshot(root, index["target_snapshot_id"])
    state = build_race_state(snapshot, driver_number="44", completed_lap=10)
    model = load_model(root, model_id, target_start_utc=state.race_start_utc)
    tables = dict(snapshot.tables)
    for name in ("laps", "weather", "track_status", "pit_events"):
        table = tables[name]
        rows = table.to_pylist()
        for row in rows:
            if name == "laps":
                row["final_deleted"], row["source_position"] = True, 999
            if row["release_ns"] is None or row["release_ns"] <= state.cutoff_ns:
                continue
            if name == "laps":
                row["lap_time_s"] += 100.0
                row["compound"], row["tyre_age_laps"] = "WET", 99
            elif name == "weather":
                row["rainfall"], row["track_temp_c"] = True, 99.0
            elif name == "track_status":
                row["status"], row["source_status"] = "RED", "5"
            elif name == "pit_events" and row["kind"] == "exit":
                row["release_ns"] += 1_000_000_000
        tables[name] = pa.Table.from_pylist(rows, schema=table.schema)
    changed = build_race_state(replace(snapshot, tables=tables), driver_number="44", completed_lap=10)
    def forecast(s):
        return forecast_stint(s, model, driver_number="44", compound="MEDIUM",
                              tyre_age_laps=s.drivers["44"].tyre_age_laps, horizon_laps=3,
                              assumptions=PaceAssumptions())
    before, after = forecast(state), forecast(changed)
    np.testing.assert_array_equal(before.mean_s, after.mean_s)
    np.testing.assert_array_equal(before.parameter_mean_s, after.parameter_mean_s)
    np.testing.assert_array_equal(before.interval95_s, after.interval95_s)

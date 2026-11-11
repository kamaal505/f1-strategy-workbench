# Local workflow

```sh
uv sync --locked --dev
uv run --locked f1-strategy synthetic --output-root data/demo
```

Read `index_path` from stdout, then read that generated JSON. Use its literal `training_spec_path`, `target_snapshot_id`, `scenario_path`, `driver_number` and `completed_lap` fields; do not substitute invented IDs.

```sh
uv run --locked f1-strategy train --root data/demo --spec "$TRAINING_SPEC"
# MODEL_ID is model_id from train stdout.
uv run --locked f1-strategy snapshot-check --root data/demo --snapshot "$SNAPSHOT_ID"
uv run --locked f1-strategy replay --root data/demo --snapshot "$SNAPSHOT_ID" --driver "$DRIVER" --lap "$LAP"
uv run --locked f1-strategy compare --root data/demo --snapshot "$SNAPSHOT_ID" --model "$MODEL_ID" --driver "$DRIVER" --lap "$LAP" --scenario "$SCENARIO" --draws 1000 --seed 2026
# ANALYSIS_PATH is analysis_path from compare stdout.
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports/demo
F1_STRATEGY_MODE=local uv run --locked streamlit run streamlit_app.py --server.headless=true --server.port=8501
```

The saved HTML is self-contained. Rendering only requires the complete checksummed analysis, not credentials, snapshots or a model. Recomputing requires the exact immutable inputs, model, rules, code and dependency lock. Public use is a separate permission decision.

## Real-source ingestion

The pinned TracingInsights public archive is fetched anonymously; no account, F1TV subscription, API key or environment confirmation is required. Check applicable access and retention rights independently. Public use requires separate full-lineage permission.

```sh
uv run --locked f1-strategy ingest --year 2023 --round 7
```

Hosted acquisition remains limited to curated races without telemetry, with parent-side source-access and retention authorization. Workers receive no credentials or authorization-record path.

For an as-of model, resolve actual snapshot IDs from the immutable catalog. Require at least five earlier eligible fitting events and disjoint earlier calibration events. Current-race adaptation is prefix-only. A model trained later than the target race start is invalid, even if coefficient dimensions match.

## Exit behavior

Successful analyses, including honest analytical abstentions, return 0. Invalid controls return 2; missing/acquisition prerequisites 3; corrupt/future artifacts 4; numerical/internal failures 5. Diagnostics are sanitized JSON on stderr. An unavailable statistic is null with its reason, never a fabricated zero.

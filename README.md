# F1 strategy workbench

Run synthetic decision replay, fit as-of pace models, compare pit/stay strategies and export offline HTML through a shared CLI and Streamlit app. The historical research domain is **2022–2024 dry Grand Prix race sessions**; sprint sessions, wet forecasting and red-flag/restart simulation are unsupported. Actual-race acquisition, evaluation, case studies and public launch remain gated by source rights and account access.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

```sh
uv sync --locked --dev
uv run --locked f1-strategy synthetic --output-root data/demo
uv run --locked f1-strategy snapshot-list --root data/demo
uv run --locked f1-strategy snapshot-check --root data/demo --snapshot <snapshot_id>
```

Uses the existing Python 3.12 environment. FastF1 3.9.2 real acquisition requires its documented source account access; a subscription does not grant public-display or redistribution permission. No real observations, learned artifacts or reports may be publicly enabled without applicable full-lineage authorization. Default public operation is labelled synthetic only. Public real-race launch additionally needs an operator-owned GitHub repository and connected Streamlit deployer with admin access.

## Shared CLI and app

`data/demo/index.json` records `target_snapshot_id`, `training_spec_path`, `scenario_path`, `driver_number`, `completed_lap` and `expected_direction`. Populate the shell variables below from that actual index; `MODEL_ID` and `ANALYSIS_PATH` come from command stdout.

```sh
uv run --locked f1-strategy train --root data/demo --spec "$TRAINING_SPEC"
uv run --locked f1-strategy replay --root data/demo --snapshot "$SNAPSHOT_ID" --driver "$DRIVER" --lap "$LAP"
uv run --locked f1-strategy compare --root data/demo --snapshot "$SNAPSHOT_ID" --model "$MODEL_ID" --driver "$DRIVER" --lap "$LAP" --scenario "$SCENARIO" --draws 1000 --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports/demo
F1_STRATEGY_MODE=local uv run --locked streamlit run streamlit_app.py --server.headless=true --server.port=8501
uv run --locked f1-strategy deployment-check
```

Pit now means the **next** entry opportunity in lap N+1. Fuel gain, track drift, future tyre inventory, pit losses and rival policies are explicit assumptions. Pace intervals do not establish causal strategy gains. Saved HTML renders offline from the complete analysis; recomputation requires exact snapshots, trusted models and code/lock identity.

Public mode is the default. It must not expose arbitrary ingestion, training, paths, URLs or uploads. Real-race authorization is checked across the whole observation/model/calibration lineage, including cached results and downloads. Missing permission is an unavailable state, not synthetic analytics under a real-race label.

See [local workflow](docs/local-workflow.md), [data contracts](docs/data-contracts.md), [methodology](docs/methodology.md), [architecture](docs/architecture.md), [executed verification](docs/verification.md) and [case-study prerequisites](docs/case-studies/2023-spain.md).

# F1 strategy workbench

Historical F1 strategy research through a shared local CLI and Streamlit app: reconstruct a driver’s state after a completed lap, forecast conditional pace, compare pit-now/stay-out plans and export self-contained HTML.

Supported domain: **2022–2024 dry Grand Prix races**, including Grands Prix on sprint weekends. Sprint sessions, wet forecasting and red-flag/restart simulation are unsupported. A replay can still show an unsupported prefix without manufacturing a recommendation.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

## Quick start: synthetic demonstration

Use Python 3.12 and the existing `.venv`; `uv.lock` is the dependency lock.

```sh
uv sync --locked --dev
uv run --locked f1-strategy synthetic --output-root data/demo
F1_STRATEGY_MODE=local uv run --locked streamlit run streamlit_app.py --server.headless=true --server.port=8501
```

`synthetic` creates eight labelled fictional snapshots, a fitted hierarchical model, training/scenario specifications and `data/demo/index.json`. It already fits the demo model; a second training run is unnecessary. The four views are **Decision replay**, **Tyres & pace**, **Pit windows & scenarios**, and **Model evidence**. Synthetic results demonstrate execution, not real-race accuracy.

For a runnable CLI replay/comparison/report sequence using the emitted IDs, see [local workflow](docs/local-workflow.md).

## Free historical inputs

Local ingestion downloads commit-pinned public JSON from [TracingInsights](https://tracinginsights.com/analysis/download-raw-data/). **No F1TV subscription, source login, API key or authentication environment variables are required.** FastF1 is not a runtime dependency; the archive contains FastF1-processed timing and Jolpica/OpenF1-derived fields.

[OpenF1](https://openf1.org/) provides free historical data from 2023 onward. The archive was selected to preserve the complete **68-race 2022–2024** domain rather than omit 2022.

```sh
uv run --locked f1-strategy ingest --year 2023 --round 7 --root data
uv run --locked f1-strategy snapshot-list --root data
```

Spain 2023 is round **7**, not preseason calendar slot 8. Ingestion returns the actual `snapshot_id` and path. Snapshots retain source JSON, canonical Parquet tables, provenance and quality findings; subsequent reads/replays are offline. A single race is not enough to fit an as-of strategy model: earlier fitting and calibration inputs are separate prerequisites.

Free access is not unrestricted data licensing. Check the source’s applicable access/retention terms; library or repository code licences do not establish upstream F1 display or redistribution rights.

### Telemetry limitation

Optional `ingest --telemetry` retains processed per-lap slices, not native separate car/position streams. Their clocks are slice-relative: absolute sample event/release times remain unknown and those samples are excluded from timed replay. Missing/partial telemetry is recorded and does not prevent strategy analysis.

## What is verified—and what remains

- **Free ingestion:** all 68 races were acquired into the private local `data/free/` root. Offline prefixes were exercised for Bahrain 2022, Spain 2023, Singapore 2023 and Australia 2024. These local datasets are ignored by Git and are not included in a fresh clone.
- **Local implementation:** synthetic CLI/UI/report behavior was exercised; the free-source cutover additionally passed 26 targeted offline tests and installed-wheel comparison/report execution without FastF1.
- **M3 historical decision:** real inputs and prefixes are available. A real as-of model, evaluated strategy report and matching real-race browser interaction remain to be executed.
- **M4 evidence:** no real-data champion or frozen settings have been established. Development/calibration still need to run; 2024 has not been scored or used for selection.
- **M5/M6 delivery:** the app/report/coordinator implementation exists, but authorized real-race public launch and complete analytical case studies are not verified. The [GitHub repository](https://github.com/kamaal505/f1-strategy-workbench) exists; Streamlit account consent, applicable public-use rights and actual provider/browser proof remain prerequisites.

See [executed verification](docs/verification.md) for exact artifacts and the distinction between pre-migration and current checks.

## Interpretation and public use

Pit now means the **next** entry opportunity, in lap N+1. Fuel gain, track drift, future tyre inventory, pit losses, traffic and rival policies are explicit assumptions. Strategy gains are conditional simulations, not observed causal effects; pace calibration does not validate counterfactual strategy coverage.

Streamlit defaults to **public** mode: labelled synthetic data only unless operator-authorized curated real races and their complete model lineage are admitted. Local mode is server configuration, not a visitor control. Public access never enables arbitrary paths, uploads, source URLs or training. Reports, cached returns and downloads retain scope/expiry checks; no source credentials need to be configured.

Saved HTML renders from its sealed analysis without snapshots, models or network access. Recomputing requires compatible immutable inputs, models and recorded code/lock/rules. The source cutover changed the lock and regenerated packaged demo artifacts; old-lock models are not silently reused.

## Documentation

- [Local CLI, as-of preparation and app catalogues](docs/local-workflow.md)
- [Snapshot schemas, source layout and time semantics](docs/data-contracts.md)
- [Modelling, assumptions and chronological evaluation](docs/methodology.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Public hosting and authorization](docs/deployment.md)
- [Verified behavior and remaining gates](docs/verification.md)
- Case studies: [Spain 2023](docs/case-studies/2023-spain.md), [Singapore 2023](docs/case-studies/2023-singapore.md), [Australia 2024](docs/case-studies/2024-australia.md)

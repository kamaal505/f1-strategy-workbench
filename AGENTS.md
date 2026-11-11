# Project map

This file maps the repository; `CLAUDE.md` is a relative symlink to this file. Keep the map current when files move or responsibilities change. Generated hash-named artifacts are described by pattern rather than enumerated individually.

## Project and entrypoints

Historical F1 race-strategy research for 2022–2024 Grand Prix races: a shared Python core serves a local CLI, Streamlit views and offline HTML reports. Forecasting supports dry races, not sprint sessions, wet forecasting or red-flag/restart simulation.

- [`README.md`](README.md) — Quick start, free-source access, verified capabilities and remaining milestones.
- [`AGENTS.md`](AGENTS.md) — Canonical file/directory/data map for coding agents.
- [`CLAUDE.md`](CLAUDE.md) — Symlink exposing this same map to Claude tooling; do not maintain a second copy.
- [`pyproject.toml`](pyproject.toml) — Package/build metadata, runtime/dev dependencies, CLI entrypoint and pytest configuration.
- [`uv.lock`](uv.lock) — Locked dependency resolution used for installation and artifact compatibility.
- [`.python-version`](.python-version) — Pins the project’s Python minor version to 3.12.
- [`.gitignore`](.gitignore) — Excludes private datasets, generated reports/builds, environments, caches and secrets.
- [`streamlit_app.py`](streamlit_app.py) — Minimal hosted entrypoint calling the installed application’s `main()`.
- [`.streamlit/config.toml`](.streamlit/config.toml) — Headless Streamlit server and minimal-toolbar settings.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — Linux locked-install, offline tests, package build, wheel-import and synthetic CLI/report verification.

## Application/core — `src/f1_strategy/`

- [`__init__.py`](src/f1_strategy/__init__.py) — Package identity/version without acquisition side effects.
- [`contracts.py`](src/f1_strategy/contracts.py) — Validated requests, manifests, scenarios, runtime records, model protocol and typed errors.
- [`identity.py`](src/f1_strategy/identity.py) — Canonical JSON, checksums, lock/code identities, payload sealing and confined paths.
- [`cli.py`](src/f1_strategy/cli.py) — Typer commands for generation, ingestion, inspection, fitting, replay, comparison, reporting and backtests.
- [`app.py`](src/f1_strategy/app.py) — Streamlit’s Decision replay, Tyres & pace, Pit windows & scenarios and Model evidence views.
- [`replay.py`](src/f1_strategy/replay.py) — Cutoff-bounded race state, released deletion messages, tyre/pit history and approximate progress.
- [`rules.py`](src/f1_strategy/rules.py) — Dry-compound obligations, set-life/directive checks and explicit unsupported or conditional legality.
- [`synthetic.py`](src/f1_strategy/synthetic.py) — Deterministic fictional snapshots, training/scenario specifications and fitted demo generation.
- [`payloads.py`](src/f1_strategy/payloads.py) — Replay-to-JSON conversion and validated, atomically saved/reloaded analysis payloads.
- [`reporting.py`](src/f1_strategy/reporting.py) — Shared Plotly charts and self-contained HTML/manifest rendering from sealed analyses.
- [`deployment.py`](src/f1_strategy/deployment.py) — Public permissions/catalogues, packaged-demo materialization, leases, quotas and bounded child coordination.
- [`worker.py`](src/f1_strategy/worker.py) — Private JSON subprocess protocol for registered ingestion/comparison jobs.
- [`report_writer.py`](src/f1_strategy/report_writer.py) — Fixed private report-rendering subprocess operating inside coordinator-owned staging.

## Source inputs and snapshots — `src/f1_strategy/data/`

- [`__init__.py`](src/f1_strategy/data/__init__.py) — Data subpackage marker.
- [`archive.py`](src/f1_strategy/data/archive.py) — Pinned season revisions, bounded anonymous JSON transport, strict field decoding and driver-identity joins.
- [`acquire.py`](src/f1_strategy/data/acquire.py) — Fetches race/archive streams and optional processed telemetry, records provenance/quality and publishes snapshots.
- [`normalize.py`](src/f1_strategy/data/normalize.py) — Projects timing frames into canonical released tables and reports suspect observations without repairing them.
- [`schemas.py`](src/f1_strategy/data/schemas.py) — Central Arrow schemas and canonical table construction.
- [`snapshots.py`](src/f1_strategy/data/snapshots.py) — Content-addressed Parquet publication, manifest/hash/path validation and local snapshot listing.
- [`query.py`](src/f1_strategy/data/query.py) — Request-scoped DuckDB queries and parameterized release-cutoff views over verified tables.

## Pace modelling and evaluation — `src/f1_strategy/pace/`

- [`__init__.py`](src/f1_strategy/pace/__init__.py) — Pace subpackage marker.
- [`features.py`](src/f1_strategy/pace/features.py) — Past-only lap covariates, clean-lap eligibility, warm-up indicators and explicit exclusion reasons.
- [`training.py`](src/f1_strategy/pace/training.py) — Earlier-event admission, disjoint chronological roles and reconstructed issued-prefix histories.
- [`design.py`](src/f1_strategy/pace/design.py) — Deterministic sparse partial-pooling design and converged robust regression fitting.
- [`selection.py`](src/f1_strategy/pace/selection.py) — Pooling-penalty selection using strictly earlier development folds.
- [`bootstrap.py`](src/f1_strategy/pace/bootstrap.py) — Whole-event/driver-stint resampling for shared parameter uncertainty.
- [`adaptation.py`](src/f1_strategy/pace/adaptation.py) — Released-prefix pace anchors and supported current-stint slope adaptation.
- [`out_of_event.py`](src/f1_strategy/pace/out_of_event.py) — Leave-one-event-out residuals with past-only adaptation inside the omitted event.
- [`residuals.py`](src/f1_strategy/pace/residuals.py) — Centered three-lap within-stint residual blocks with disclosed spec/compound pooling.
- [`calibration.py`](src/f1_strategy/pace/calibration.py) — Finite-sample interval widening with explicit sparse-evidence outcomes.
- [`heldout.py`](src/f1_strategy/pace/heldout.py) — Issued parameter paths and held-out lap/cumulative interval calibration.
- [`hierarchical.py`](src/f1_strategy/pace/hierarchical.py) — Model-kind fitting dispatch, hierarchical assumption-grid bundles and supported future-stint forecasts.
- [`model.py`](src/f1_strategy/pace/model.py) — In-memory hierarchical model predictions, parameter replicates and residual access.
- [`baseline.py`](src/f1_strategy/pace/baseline.py) — Weak latest-five-eligible-lap baseline without compound-switch advice.
- [`rolling_model.py`](src/f1_strategy/pace/rolling_model.py) — Persisted identity/provenance wrapper for the prefix-dependent rolling baseline.
- [`challenger.py`](src/f1_strategy/pace/challenger.py) — Chronological HGB residual/quantile training, numeric tree persistence and recursive forecasts.
- [`uncertainty.py`](src/f1_strategy/pace/uncertainty.py) — Maps base block ranks through conditional HGB quantile distributions without double-counting noise.
- [`bundle.py`](src/f1_strategy/pace/bundle.py) — Numeric-only model manifests/NPZ storage, content identities and runtime compatibility checks.
- [`artifacts.py`](src/f1_strategy/pace/artifacts.py) — Recursive model-lineage admission and verified loading before the target race.
- [`corpus.py`](src/f1_strategy/pace/corpus.py) — Verified offline schedule loading, local snapshot resolution and as-of training-spec construction.
- [`validation.py`](src/f1_strategy/pace/validation.py) — Chronological backtests, censoring/metrics, development champion gates and frozen holdout admission.

## Strategy simulation — `src/f1_strategy/strategy/`

- [`__init__.py`](src/f1_strategy/strategy/__init__.py) — Strategy subpackage marker.
- [`plans.py`](src/f1_strategy/strategy/plans.py) — Expands assumed inventories/scenarios and performs bounded deterministic tyre-set plan search.
- [`pit.py`](src/f1_strategy/strategy/pit.py) — Estimates pit loss and absolute occupancy from entirely released completed visits.
- [`occupancy.py`](src/f1_strategy/strategy/occupancy.py) — Conditions remaining absolute pit-lane occupancy on time already spent in the lane.
- [`field.py`](src/f1_strategy/strategy/field.py) — Opponent age policies, crossing-clock simulation, shared neutralisations and approximate field positions.
- [`simulate.py`](src/f1_strategy/strategy/simulate.py) — Paired plan screening/evaluation, fixed-plan sensitivities, recommendation gates and sealed decisions.

## Packaged resources — `src/f1_strategy/assets/`

Resources are accessed through `importlib.resources`; they must work from an installed wheel, not just a source checkout.

- [`lock.sha256`](src/f1_strategy/assets/lock.sha256) — Embedded dependency-lock identity for installed artifact compatibility.
- [`report.html.j2`](src/f1_strategy/assets/report.html.j2) — Autoescaped HTML report template with embedded charts and provenance.
- [`rules/seasons.json`](src/f1_strategy/assets/rules/seasons.json) — Sporting-regulation references and explicitly assumed mandatory-compound defaults.
- [`events/directives.json`](src/f1_strategy/assets/events/directives.json) — Event exceptions including Qatar set-life limits and alternative-allocation caveats.
- [`synthetic/specification.json`](src/f1_strategy/assets/synthetic/specification.json) — Fictional driver, pace-curve, noise, timing and inventory specification.
- [`scenarios/assumed-green.json`](src/f1_strategy/assets/scenarios/assumed-green.json) — Assumed inventory with no new neutralisation.
- [`scenarios/assumed-sc.json`](src/f1_strategy/assets/scenarios/assumed-sc.json) — Hypothetical safety-car scenario after one leader lap.
- [`scenarios/assumed-vsc.json`](src/f1_strategy/assets/scenarios/assumed-vsc.json) — Hypothetical virtual-safety-car scenario after one leader lap.
- [`evaluation/corpus.json`](src/f1_strategy/assets/evaluation/corpus.json) — Race membership, stable year/round mapping, partitions and vendored schedule hashes.
- [`evaluation/schedule_2022.json`](src/f1_strategy/assets/evaluation/schedule_2022.json) — Retained upstream 2022 schedule used for offline race identity/provenance.
- [`evaluation/schedule_2023.json`](src/f1_strategy/assets/evaluation/schedule_2023.json) — Retained upstream 2023 schedule, where Spain is round 7 after cancelled Imola.
- [`evaluation/schedule_2024.json`](src/f1_strategy/assets/evaluation/schedule_2024.json) — Retained upstream 2024 schedule for holdout race membership.
- [`evaluation/protocol.json`](src/f1_strategy/assets/evaluation/protocol.json) — Fixed chronological partitions, forecast horizons and evaluation/holdout policy.
- [`evaluation/development.json`](src/f1_strategy/assets/evaluation/development.json) — Development backtest controls for 2023 rounds 1–16.
- [`evaluation/holdout-2024.json`](src/f1_strategy/assets/evaluation/holdout-2024.json) — Holdout controls requiring a pre-existing valid frozen-settings manifest, not supplied results.
- [`evaluation/asof-2023-07-R.json`](src/f1_strategy/assets/evaluation/asof-2023-07-R.json) — Unresolved Spain as-of descriptor to resolve against actual earlier snapshots before training.
- [`evaluation/asof-2023-15-R.json`](src/f1_strategy/assets/evaluation/asof-2023-15-R.json) — Unresolved Singapore as-of descriptor to resolve against actual earlier snapshots before training.
- [`evaluation/asof-2024-03-R.json`](src/f1_strategy/assets/evaluation/asof-2024-03-R.json) — Unresolved Australia as-of descriptor, not proof of frozen-model holdout evaluation.
- [`demo/index.json`](src/f1_strategy/assets/demo/index.json) — Portable packaged synthetic snapshot/model/scenario IDs and expected demonstration direction.
- [`demo/training.json`](src/f1_strategy/assets/demo/training.json) — Explicit synthetic fitting/calibration snapshot IDs and training controls.
- [`demo/scenario.json`](src/f1_strategy/assets/demo/scenario.json) — Conditional inventory, dry-weather and rules assumptions for the packaged demonstration.
- `demo/snapshots/<snapshot_id>/` — Small labelled synthetic snapshot publications using the same file layout as local snapshots below.
- `demo/models/<model_id>/` — Verified packaged synthetic model manifest and numeric arrays; never real-race weights under synthetic labels.

## Behavioral checks — `tests/`

- [`test_contracts.py`](tests/test_contracts.py) — Request types, supported domain, finite uncertainties and unique/usable tyre-set controls.
- [`test_archive_parsing.py`](tests/test_archive_parsing.py) — Archive null/false decoding, identity joins, future pit-exit exclusion, payload limits and partial telemetry.
- [`test_snapshots.py`](tests/test_snapshots.py) — Immutable identity, source/canonical handling, offline queries, checksums and path/symlink rejection.
- [`test_replay_rules.py`](tests/test_replay_rules.py) — Released-prefix invariance, deletion/pit state, dry/rule/directive boundaries and explicit abstentions.
- [`test_prefix_evidence.py`](tests/test_prefix_evidence.py) — Entrants without a released crossing remain unknown rather than silently retired.
- [`test_pace.py`](tests/test_pace.py) — Conditional pace fitting, support, warm-up, calibration and temporal training boundaries.
- [`test_strategy.py`](tests/test_strategy.py) — Hand-calculable paired timing, plans, inventory, occupancy, traffic, neutralisation and finish rules.
- [`test_validation_ml.py`](tests/test_validation_ml.py) — Chronological ML/evaluation leakage boundaries, metrics and frozen holdout behavior.
- [`test_challenger_uncertainty.py`](tests/test_challenger_uncertainty.py) — HGB conditional path uncertainty, future-stint fallbacks and transitive holdout rejection.
- [`test_artifact_boundaries.py`](tests/test_artifact_boundaries.py) — Rejects incompatible, incomplete or cyclic model dependencies before reading numeric arrays.
- [`test_public_boundaries.py`](tests/test_public_boundaries.py) — Permission scopes/expiry, credential isolation, registry safety and bounded worker/report cleanup.
- [`test_workflow.py`](tests/test_workflow.py) — Real synthetic CLI/core fitting, replay, comparison and offline-report integration.

## Reference documentation — `docs/`

- [`local-workflow.md`](docs/local-workflow.md) — Executable CLI recipes, as-of resolution, catalogue roots and compatibility guidance.
- [`data-contracts.md`](docs/data-contracts.md) — Source-payload layout, canonical schemas, time/release rules and quality policy.
- [`methodology.md`](docs/methodology.md) — Conditional estimands, model assumptions, simulation and chronological evaluation design.
- [`architecture.md`](docs/architecture.md) — Core/module boundaries, artifact identities and public/local trust boundaries.
- [`deployment.md`](docs/deployment.md) — Streamlit setup, authorization records, public roots, capacity and external launch prerequisites.
- [`verification.md`](docs/verification.md) — Exercised checks/artifact evidence, historical versus current results and remaining gaps.
- [`case-studies/2023-spain.md`](docs/case-studies/2023-spain.md) — Spain tyre/layout-transfer context, observed prefix and pending strategy analysis.
- [`case-studies/2023-singapore.md`](docs/case-studies/2023-singapore.md) — Singapore traffic/SC/VSC questions, observed prefix and pending paired scenarios.
- [`case-studies/2024-australia.md`](docs/case-studies/2024-australia.md) — Australia input evidence and strictly reserved frozen-model holdout analysis.

## Data we use

- **Primary historical source:** [TracingInsights public archives](https://tracinginsights.com/analysis/download-raw-data/) in its [2022](https://github.com/TracingInsights/2022), [2023](https://github.com/TracingInsights/2023) and [2024](https://github.com/TracingInsights/2024) repositories; exact pinned commits are maintained in `data/archive.py`.
- **Observed archive inputs:** lap/sector durations, estimated lap boundaries, compound/tyre life, independent pit entry/exit times, driver identities, weather, track/session status and race-control messages; original JSON bytes are retained alongside normalized tables.
- **Processing lineage:** archive fields include FastF1 processing and Jolpica/OpenF1-derived content; FastF1 is not a runtime dependency and no F1TV login/API key is required.
- **Alternative source boundary:** [OpenF1](https://openf1.org/) provides free historical data from 2023 onward, so it is not the sole provider for the full 2022–2024 domain.
- **Schedule data:** vendored [theOehrly/f1schedule](https://github.com/theOehrly/f1schedule) JSON supplies stable year/round membership, not proof that timing data or public-use permission exists.
- **Rules/event context:** FIA regulation/directive and tyre-nomination references inform checks; unknown event-date applicability, physical tyre specification and inventory remain conditional rather than inferred.
- **Synthetic data:** deterministic fictional observations exercise the same schemas/models/simulator, with `SYN-...` sessions and separate synthetic source IDs; they are not real-race validation.
- **Optional telemetry:** processed/interpolated per-lap speed, RPM, gear, throttle, binary brake/DRS and position channels have slice-relative clocks; unknown absolute event/release times exclude them from timed replay.
- **Not observed:** original message arrival times, actual fuel, tyre temperatures/pressures, complete remaining tyre inventory, precise stationary service or future rival decisions; scenario values are assumptions.
- **Chronological split:** 2022 initializes fitting, 2023 rounds 1–16 develop/select, rounds 17–22 calibrate, and 2024 is reserved for frozen-model holdout; downloading inputs is not scoring or model selection.
- **Rights boundary:** free access or a code licence does not grant upstream public data/model/report rights; real public operation requires complete-lineage scope authorization.

## Local/generated directories and file patterns

`data/`, `reports/` and `dist/` are ignored local outputs, not datasets shipped in a fresh clone. Use actual emitted IDs rather than copying old hashes; lock/version changes can invalidate older model artifacts.

- `data/free/` — Verified private acquisition corpus for the 68 real 2022–2024 races.
- `data/free/acquisition-proof.json` — Recorded actual snapshot IDs, retained-source counts and offline-prefix smoke evidence.
- `data/demo/` — Default local synthetic runs/experiments; older artifacts may use earlier locks.
- `data/free-demo/` — Synthetic artifacts regenerated for the free-source dependency lock.
- `<root>/index.json` — Generated demo’s chosen snapshot/model IDs, specification paths, driver/lap and expected direction.
- `<root>/training.json` — Generated demo’s fitting/calibration inputs and model-training controls.
- `<root>/scenario.json` — Generated demo’s explicit conditional simulation assumptions.
- `data/demo/hgb-smoke.json` — Local experimental synthetic HGB training specification, not a selected real-data champion.
- `data/demo/proof-analysis.json` / `proof-hgb-analysis.json` — Local smoke-command output records pointing to saved synthetic analyses.
- `<root>/snapshots/<snapshot_id>/manifest.json` — Snapshot identity, source metadata, availability, quality and member checksums.
- `<root>/snapshots/<snapshot_id>/laps.parquet` — Canonical completed-lap observations with released tyre/timing and retrospective audit fields.
- `<root>/snapshots/<snapshot_id>/pit_events.parquet` — Independently released pit entry/exit events.
- `<root>/snapshots/<snapshot_id>/weather.parquet` — Timestamped weather observations, not weather forecasts.
- `<root>/snapshots/<snapshot_id>/track_status.parquet` — Timestamped normalized green/yellow/SC/VSC/red status.
- `<root>/snapshots/<snapshot_id>/session_status.parquet` — Timestamped session lifecycle/start/abort/finish status.
- `<root>/snapshots/<snapshot_id>/messages.parquet` — Timestamped race-control text, flags and optional driver/lap attribution.
- `<root>/snapshots/<snapshot_id>/source/session_info.parquet` — Original archive JSON for the stream origin, race distance/start and status timelines.
- `<root>/snapshots/<snapshot_id>/source/drivers.parquet` — Original archive JSON for driver codes, available numbers and entry metadata.
- `<root>/snapshots/<snapshot_id>/source/session_laptimes.parquet` — Original parallel lap/tyre/pit arrays, including unconsumed and retrospective fields.
- `<root>/snapshots/<snapshot_id>/source/weather.parquet` — Original standalone timestamped weather arrays.
- `<root>/snapshots/<snapshot_id>/source/rcm.parquet` — Original timestamped race-control message arrays.
- `<root>/snapshots/<snapshot_id>/source/<stream>.parquet` — Labelled synthetic source-table stand-ins when using fictional snapshots.
- `<root>/snapshots/<snapshot_id>/source/telemetry/<code>/<lap>.parquet` — Optional original processed telemetry JSON payload for one driver/lap.
- `<root>/snapshots/<snapshot_id>/car_samples.parquet` / `position_samples.parquet` — Optional normalized processed telemetry channels retaining unknown absolute clocks.
- `<root>/models/<model_id>/manifest.json` — Semantic model identity, runtime versions, training roles, support, calibration and transitive source lineage.
- `<root>/models/<model_id>/arrays.npz` — Numeric coefficients, bootstraps, residuals and/or exported trees loaded without pickle.
- `<root>/evaluation/<evaluation_id>/audit.json` — Backtest input identities, availability, exclusion and chronological provenance audit.
- `<root>/evaluation/<evaluation_id>/scorecard.json` — Scored/censored coverage, errors and champion-selection evidence, including unavailable outcomes.
- `<root>/evaluation/<evaluation_id>/frozen-settings.json` — Conditional output only when development can produce valid frozen champion/model settings.
- `<root>/public_registry.json` — Optional operator-owned public race/snapshot/model/scenario registry, never a visitor upload.
- `<root>/staging/` — Coordinator-owned private temporary child-job/report workspaces.
- `reports/demo/`, `reports/free-demo/`, `reports/hgb-smoke/` — Local synthetic report groups from different verification/experiment runs.
- `<output_dir>/<analysis_id>/analysis.json` — Complete sealed decision payload that can be rendered without loading data/models.
- `<output_dir>/<analysis_id>/report.html` — Self-contained HTML report with embedded Plotly JavaScript/data.
- `<output_dir>/<analysis_id>/manifest.json` — Analysis/payload/report checksums protecting a rendered report publication.
- `dist/*.whl` / `dist/*.tar.gz` — Built installable wheel and source distribution.
- `.venv/` — Existing managed Python environment; reuse it rather than recreating it.
- `.omp/` — Ignored local assistant configuration; preserve it and keep it out of project artifacts.
- `.pytest_cache/`, `__pycache__/`, `.hypothesis/` — Disposable test/bytecode/property-test caches, not source or research evidence.
- `.git/` — Repository objects and history, not application inputs.

## Working boundaries

- Install with `uv sync --locked --dev`; use the shared core rather than duplicating analytics in UI/CLI.
- Run relevant offline checks with `uv run --locked pytest --disable-socket --allow-unix-socket <test_paths>`; build installable resources with `uv build` when packaging changes.
- Keep private source data, authorization evidence and secrets out of Git; public source/model/report access remains default-deny without the appropriate scopes.
- Gate every replay input by release time before aggregation; never resolve ambiguity using final results or later events.
- Do not treat old-lock models as compatible, infer unobserved inventory/weather or claim a real-data champion/public deployment from synthetic proof; consult `docs/verification.md`.

Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

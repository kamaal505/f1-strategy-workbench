# Local workflow

Run from the repository root with Python 3.12. Reuse `.venv` and the locked project; no F1TV account, source API key or authentication setup is needed.

## Synthetic replay, comparison and report

```sh
uv sync --locked --dev
uv run --locked f1-strategy synthetic --output-root data/demo
```

The generator already fits its hierarchical model. Resolve shell variables from its actual index, rather than copying historical artifact IDs:

```sh
eval "$(uv run --locked python - <<'PY'
import json
import shlex
from pathlib import Path

index = json.loads(Path('data/demo/index.json').read_text())
for name, key in (
    ('SNAPSHOT_ID', 'target_snapshot_id'),
    ('MODEL_ID', 'model_id'),
    ('DRIVER', 'driver_number'),
    ('LAP', 'completed_lap'),
    ('SCENARIO', 'scenario_path'),
    ('TRAINING_SPEC', 'training_spec_path'),
):
    print(f'{name}={shlex.quote(str(index[key]))}')
PY
)"
uv run --locked f1-strategy snapshot-check --root data/demo --snapshot "$SNAPSHOT_ID"
uv run --locked f1-strategy replay --root data/demo --snapshot "$SNAPSHOT_ID" --driver "$DRIVER" --lap "$LAP"
uv run --locked f1-strategy compare --root data/demo --snapshot "$SNAPSHOT_ID" --model "$MODEL_ID" --driver "$DRIVER" --lap "$LAP" --scenario "$SCENARIO" --draws 1000 --seed 2026 --output-dir reports/demo > data/demo/comparison-output.json
ANALYSIS_PATH="$(uv run --locked python -c 'import json; print(json.load(open("data/demo/comparison-output.json"))["analysis_path"])')"
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports/demo
```

`comparison-output.json` captures CLI stdout; the sealed analysis and HTML live beneath `reports/demo/<analysis_id>/`. The report needs only the complete checksummed analysis, not source access or a model. Recomputing requires exact compatible snapshots, model, rules, code and dependency lock.

To fit a deliberately changed training specification, use `uv run --locked f1-strategy train --root data/demo --spec "$TRAINING_SPEC"` and take `model_id` from that command’s stdout. Do not assume the previous model ID still applies.

## Free real-source ingestion

The [TracingInsights public archive](https://tracinginsights.com/analysis/download-raw-data/) is commit-pinned and fetched anonymously. Check applicable access/retention rights independently; public use is a separate permission decision.

```sh
uv run --locked f1-strategy ingest --year 2023 --round 7 --root data
uv run --locked f1-strategy snapshot-list --root data
```

Take the real snapshot ID from stdout, then use `snapshot-check` and `replay` with that same root. Spain’s recorded smoke prefix is driver 44 after lap 20; the [case study](case-studies/2023-spain.md) records its observed ID/cutoff, not a completed strategy result.

The verified full-corpus acquisition on the development machine is under `data/free/`, with `data/free/acquisition-proof.json` recording its actual IDs. To use those existing artifacts, pass `--root data/free`. They are private local files, ignored by Git; a fresh clone must acquire its own inputs.

Optional acquisition adds `--telemetry`. It may fetch many per-lap files and records bounds/partial availability. These are processed slices with unknown absolute sample clocks, not timed replay observations. Strategy does not require telemetry.

## Prepare an as-of model

Ingest the target and its earlier history into one root first. One target snapshot cannot supply an earlier-event model. Bundled `evaluation/asof-*.json` files are unresolved descriptors, **not** ready-to-train `TrainingSpec` files.

For Spain, after the prior snapshots are present:

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from f1_strategy.pace.corpus import resolve_asof_training

root = Path('data')  # Use Path('data/free') for the existing verified local corpus.
spec = resolve_asof_training('evaluation/asof-2023-07-R.json', root=root)
output = root / 'asof-2023-07-R.json'
output.write_text(spec.model_dump_json(indent=2) + '\n')
print(output)
PY
uv run --locked f1-strategy train --root data --spec data/asof-2023-07-R.json
```

Keep the resolver and training roots consistent if you change them. The resolver fails with the missing session IDs instead of silently choosing an incomplete history. It reserves the latest 20% of prior non-holdout events, at least two, for calibration and requires at least five distinct fitting events. Fit/tuning/calibration sources must predate the target; current-race adaptation is prefix-only. Model suitability and interval-calibration thresholds still apply after resolution.

Use the emitted real `model_id`, target snapshot ID and an explicitly reviewed scenario for `compare`; then render the returned `analysis_path`. Future tyre inventory, pit losses, dry assumptions and event-rule overrides are not observations. Unsupported evidence may produce an honest abstention rather than advice.

The fixed evaluation split is 2022 initialization, 2023 rounds 1–16 development, rounds 17–22 final calibration and 2024 holdout. Do not score holdout before a valid frozen-settings manifest exists or use it to retune. See [methodology](methodology.md).

## Local app and catalogue roots

```sh
F1_STRATEGY_MODE=local uv run --locked streamlit run streamlit_app.py --server.headless=true --server.port=8501
```

The app discovers the generated `data/demo/index.json`, with a packaged synthetic fallback. In local mode it also offers the `data/` catalogue when `data/snapshots/` exists. Arbitrary roots such as `data/free/` are supported by CLI `--root`, but are **not automatically discovered by the local UI**. Use `--root data` for inputs/models intended for that local catalogue. Ingestion alone does not fit a model.

Without `F1_STRATEGY_MODE=local`, the app defaults to public mode. Hosted acquisition is limited to curated races without telemetry, with parent-side source-access and retention authorization. Workers receive no source credentials or authorization-record path. Public display, model use, reports and artifact distribution require their own complete-lineage scopes; see [deployment](deployment.md).

## Exit behavior and artifact compatibility

Successful analyses, including analytical abstentions, return 0. Invalid controls return 2; missing/acquisition prerequisites 3; corrupt/future artifacts 4; numerical/internal failures 5. Diagnostics are sanitized JSON on stderr. An unavailable statistic is null with its reason, never a fabricated zero.

The free-source cutover removed FastF1 from runtime dependencies and changed `uv.lock`. Packaged synthetic snapshots/models were regenerated. Older private models may fail strict lock/version compatibility; generate or fit compatible artifacts rather than editing their manifests or suppressing checks.

Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

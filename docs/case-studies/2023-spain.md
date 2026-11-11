# 2023 Spanish Grand Prix — tyre pace and pit timing

**Status: real snapshot and offline prefix verified; strategy analysis pending.** Free acquisition needs no F1TV login. A real as-of model, strategy report, browser proof and full analytical conclusions have not been established; public display and derivative permissions remain separate.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

The 2023 Barcelona layout removed the final chicane and the C1 specification changed. Do not identify relative-compound labels with unchanged physical chemistry or pool circuit-layout intercepts without source-backed provenance. Show the mandatory nine fuel/track assumptions; a fitted age slope is conditional pace, not measured wear.

Sources: [Pirelli, Monaco and Spain 2023](https://press.pirelli.com/2023-tyre-compound-choices--monaco-and-spain/), [FastF1 2023 schedule](https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_2023.json). Spain is FastF1 round **7**, not preseason calendar slot 8.

The first-mile selection is driver 44 after lap 20, which passed the input replay check below. Model support is a separate test. If that decision lacks evidence, consider the earliest supported lap 10–30 for driver 44, then 1, then 14; record the actual choice without lowering thresholds.

Private acquisition and offline replay are verified for snapshot `fd793d5d5a2e96e6cbe85fbf12a415bf2c451cf05ce16d15f4f60d16deabe801`, driver 44 after lap 20, cutoff `5354336000000` ns. This is source/prefix evidence only, not a fitted-model recommendation.

The recorded files live in the development machine’s private `data/free/` catalogue, not in the repository. A fresh clone must ingest its own inputs. See [local workflow](../local-workflow.md) for catalogue roots and resolving the bundled as-of descriptor against earlier snapshots.

## Reproduction after prerequisites

```sh
ROOT=data
uv run --locked f1-strategy ingest --year 2023 --round 7 --root "$ROOT"
# Ingest prior history and resolve evaluation/asof-2023-07-R.json first.
# ASOF_TRAINING_SPEC is the generated TrainingSpec, not that descriptor.
uv run --locked f1-strategy train --root "$ROOT" --spec "$ASOF_TRAINING_SPEC"
# Set IDs from ingest/train stdout and review the explicit scenario assumptions.
uv run --locked f1-strategy compare --root "$ROOT" --snapshot "$REAL_SNAPSHOT_ID" --model "$ASOF_MODEL_ID" --driver 44 --lap 20 --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

# 2024 Australian Grand Prix — undercut and remaining life

**Status: blocked by missing authorized source inputs.** No race snapshot/model/analysis ID exists for this case. Source authentication reported `Not authenticated`; source access and retention permission were not supplied. Public display and full-lineage derivative permissions are separately absent. This is a named missing race, not a substituted synthetic case study.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

This race belongs to untouched 2024 holdout. Development chooses the champion, disjoint late-2023 calibration adjusts intervals, then frozen settings are tested once. No selection or retuning may inspect this race. A missing frozen-settings manifest is a prerequisite failure, not a holdout score.

The [2024 C3/C4/C5 nomination](https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-australian-grand.6VUbZ0PCx0V9XHZxIcpC2d) differs from 2023. [Official team comments](https://www.formula1.com/en/latest/article/what-the-teams-said-race-day-in-australia-2024.1Ze7HfHIYuMT5QzDOtms37) contain driver-reported graining; that is hindsight testimony, not a measured tyre state or model inference.

## Reproduction after prerequisites

```sh
# Only after confirming applicable source access and snapshot retention rights:
export F1_STRATEGY_AUTHORIZATION_FILE="$PRIVATE_AUTHORIZATION_FILE"
export F1_SOURCE_ACCESS_CONFIRMED=1 F1_SNAPSHOT_RETENTION_CONFIRMED=1
uv run --locked f1-strategy ingest --year 2024 --round 3
# Resolve prior actual snapshots into an as-of TrainingSpec; no invented IDs.
uv run --locked f1-strategy train --spec "$ASOF_TRAINING_SPEC"
uv run --locked f1-strategy compare --snapshot "$REAL_SNAPSHOT_ID" --model "$ASOF_MODEL_ID" --driver 55 --lap 20 --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors remain **unavailable because acquisition is blocked**. They must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

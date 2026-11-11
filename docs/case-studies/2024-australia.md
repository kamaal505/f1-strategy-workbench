# 2024 Australian Grand Prix — undercut and remaining life

**Status: real snapshot and input-boundary replay verified; holdout analysis pending.** Free acquisition needs no F1TV login. No frozen-champion holdout score, strategy report or full case-study conclusion has been produced; public display and derivative permissions remain separate.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

This race is reserved for 2024 holdout. Development chooses the champion, disjoint late-2023 calibration adjusts intervals, then frozen settings are tested once. This race’s outcomes must not guide selection or retuning. A missing frozen-settings manifest is a prerequisite failure, not a holdout score.

The [2024 C3/C4/C5 nomination](https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2024-australian-grand.6VUbZ0PCx0V9XHZxIcpC2d) differs from 2023. [Official team comments](https://www.formula1.com/en/latest/article/what-the-teams-said-race-day-in-australia-2024.1Ze7HfHIYuMT5QzDOtms37) contain driver-reported graining; that is hindsight testimony, not a measured tyre state or model inference.

Private acquisition and an input-boundary smoke replay are verified for snapshot `19e6a84249db41d2de256d6b7eb4b42eacfcb967d60ae1d88b681bd9bf97b321`, driver 44 after lap 10, cutoff `4372328000000` ns. No holdout scoring, model selection or retuning was performed.

The recorded files live in the development machine’s private `data/free/` catalogue, not in the repository. The input smoke check does not authorize using this race to choose a model, settings or a favourable decision prefix. See [methodology](../methodology.md) for the fixed development/calibration/holdout split and [local workflow](../local-workflow.md) for root selection.

## Reproduction after prerequisites

```sh
ROOT=data
uv run --locked f1-strategy ingest --year 2024 --round 3 --root "$ROOT"
# Only after development selection is frozen and disjoint calibration is ready:
# FROZEN_TRAINING_SPEC must implement that frozen choice, not a retuned default.
uv run --locked f1-strategy train --root "$ROOT" --spec "$FROZEN_TRAINING_SPEC"
# Set IDs from stdout and use the frozen evaluation protocol's DRIVER/LAP.
uv run --locked f1-strategy compare --root "$ROOT" --snapshot "$REAL_SNAPSHOT_ID" --model "$FROZEN_MODEL_ID" --driver "$DRIVER" --lap "$LAP" --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

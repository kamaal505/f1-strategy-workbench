# 2023 Singapore Grand Prix — neutralisation and traffic

**Status: blocked by missing authorized source inputs.** No race snapshot/model/analysis ID exists for this case. Source authentication reported `Not authenticated`; source access and retention permission were not supplied. Public display and full-lineage derivative permissions are separately absent. This is a named missing race, not a substituted synthetic case study.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

Select one released prefix before a neutralisation and one while it is active. The future SC/VSC timeline is not a replay input. An active status requires an explicit hypothetical remaining duration. Compare baseline, SC-after-one-leader-lap and VSC-after-one-leader-lap independently; no probability-weighted forecast is implied.

The [official race report](https://www.formula1.com/en/latest/article/sainz-holds-off-norris-and-fast-charging-mercedes-pair-to-take-sensational.16sNsRUz2MAFyXxSE3RdwX) describes leaders stopping under SC and Mercedes changing tyres under a later VSC. This is **hindsight context only**. No numerical recommendation, traffic gain or observed forecast error is asserted here.

[Pirelli nomination context](https://press.pirelli.com/unchanged-nominations-for-singapore-japan-and-qatar--in-name-at-least/) must be reconciled with event-date rules and physical-spec evidence.

## Reproduction after prerequisites

```sh
# Only after confirming applicable source access and snapshot retention rights:
export F1_STRATEGY_AUTHORIZATION_FILE="$PRIVATE_AUTHORIZATION_FILE"
export F1_SOURCE_ACCESS_CONFIRMED=1 F1_SNAPSHOT_RETENTION_CONFIRMED=1
uv run --locked f1-strategy ingest --year 2023 --round 15
# Resolve prior actual snapshots into an as-of TrainingSpec; no invented IDs.
uv run --locked f1-strategy train --spec "$ASOF_TRAINING_SPEC"
uv run --locked f1-strategy compare --snapshot "$REAL_SNAPSHOT_ID" --model "$ASOF_MODEL_ID" --driver 55 --lap 20 --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors remain **unavailable because acquisition is blocked**. They must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

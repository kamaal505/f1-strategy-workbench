# 2023 Singapore Grand Prix — neutralisation and traffic

**Status: no real model or analysis established.** Anonymous archive acquisition does not establish source retention or public display and full-lineage derivative permissions. Do not substitute a synthetic case study or invent results.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

Select one released prefix before a neutralisation and one while it is active. The future SC/VSC timeline is not a replay input. An active status requires an explicit hypothetical remaining duration. Compare baseline, SC-after-one-leader-lap and VSC-after-one-leader-lap independently; no probability-weighted forecast is implied.

The [official race report](https://www.formula1.com/en/latest/article/sainz-holds-off-norris-and-fast-charging-mercedes-pair-to-take-sensational.16sNsRUz2MAFyXxSE3RdwX) describes leaders stopping under SC and Mercedes changing tyres under a later VSC. This is **hindsight context only**. No numerical recommendation, traffic gain or observed forecast error is asserted here.

[Pirelli nomination context](https://press.pirelli.com/unchanged-nominations-for-singapore-japan-and-qatar--in-name-at-least/) must be reconciled with event-date rules and physical-spec evidence.

Private acquisition and offline replay are verified for snapshot `a205207b8640d523a22b797a6c55f0533688a2284d0481825469950c0acde214`, driver 44 after lap 10, cutoff `4777594000000` ns. This verifies released-input boundaries, not yet the two neutralisation comparisons above.

## Reproduction after prerequisites

```sh
# Only after confirming applicable source access and snapshot retention rights:
uv run --locked f1-strategy ingest --year 2023 --round 15
# Resolve prior actual snapshots into an as-of TrainingSpec; no invented IDs.
uv run --locked f1-strategy train --spec "$ASOF_TRAINING_SPEC"
uv run --locked f1-strategy compare --snapshot "$REAL_SNAPSHOT_ID" --model "$ASOF_MODEL_ID" --driver 55 --lap 20 --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

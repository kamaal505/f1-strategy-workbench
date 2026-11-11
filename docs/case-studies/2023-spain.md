# 2023 Spanish Grand Prix — tyre pace and pit timing

**Status: no real model or analysis established.** Anonymous archive acquisition does not establish source retention or public display and full-lineage derivative permissions. Do not substitute a synthetic case study or invent results.

> Event-time reconstructed replay; historical corrections and original arrival times are not fully observable.

The 2023 Barcelona layout removed the final chicane and the C1 specification changed. Do not identify relative-compound labels with unchanged physical chemistry or pool circuit-layout intercepts without source-backed provenance. Show the mandatory nine fuel/track assumptions; a fitted age slope is conditional pace, not measured wear.

Sources: [Pirelli, Monaco and Spain 2023](https://press.pirelli.com/2023-tyre-compound-choices--monaco-and-spain/), [FastF1 2023 schedule](https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_2023.json). Spain is FastF1 round **7**, not preseason calendar slot 8.

After acquisition, first attempt driver 44 lap 20. If unsupported, choose the earliest supported lap 10–30 for driver 44, then 1, then 14. Record that actual prefix; never lower the evidence thresholds.

Private acquisition and offline replay are verified for snapshot `fd793d5d5a2e96e6cbe85fbf12a415bf2c451cf05ce16d15f4f60d16deabe801`, driver 44 after lap 20, cutoff `5354336000000` ns. This is source/prefix evidence only, not a fitted-model recommendation.

## Reproduction after prerequisites

```sh
# Only after confirming applicable source access and snapshot retention rights:
uv run --locked f1-strategy ingest --year 2023 --round 7
# Resolve prior actual snapshots into an as-of TrainingSpec; no invented IDs.
uv run --locked f1-strategy train --spec "$ASOF_TRAINING_SPEC"
uv run --locked f1-strategy compare --snapshot "$REAL_SNAPSHOT_ID" --model "$ASOF_MODEL_ID" --driver 44 --lap 20 --scenario "$REAL_SCENARIO" --seed 2026
uv run --locked f1-strategy report --analysis "$ANALYSIS_PATH" --output-dir reports
```

Snapshot, model, scenario and analysis identities, exact cutoff, installed-code/lock digests, uncertainty, sensitivity, observed later outcome and forecast errors must be populated from executed results; this document makes no numerical strategy-gain claim. Public charts/reports may be distributed only within their applicable granted scope.

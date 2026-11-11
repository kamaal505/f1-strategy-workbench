# Methodology

## Units and estimands

Event time is signed int64 nanoseconds from the stream origin, not original receipt time. Lap durations and uncertainty are seconds. Tyre age includes prior use. Fuel gain is an assumed seconds-per-race-lap effect, not measured fuel kilograms. Fuel, track drift and tyre-age effects are not separately identifiable from a single uninterrupted stint.

Replay is event-time reconstructed. Source historical corrections may remain retrospective. The focal cutoff is its completed lap crossing; pit now means the next entry opportunity in lap N+1. No final positions, final deletion flags or later pit/weather/status observations may become decision features.

Forecasts are conditional clean lap times. Strategy timing compares equal scheduled distance, not different-distance flag timestamps. Simulated pit losses, inventory, traffic, neutralisations and rival policies are assumptions, not observed counterfactual outcomes. Interval calibration of pace does not establish joint counterfactual strategy coverage.

## Replay and conditional legality

Every stream is gated before aggregation. Mounted tyres become observable at source-lap completion; pit entry/exit release independently. Final deletion/position fields are removed. Deletion and reinstatement messages require unambiguous driver/lap attribution. Approximate progress becomes stale after 1.5 recent typical laps without a crossing; stale absence is not retirement. Sectors are withheld without their own completion timestamps.

Two distinct dry specifications and at least one designated mandatory specification are required; using both designated specifications is not required. Default MEDIUM/HARD designations and virtual inventory remain explicitly assumed. Season-level rule citations do not establish exact event-date applicability.

The final [2023 Qatar directive](https://www.fia.com/news/fia-statement-treatment-tyres-2023-fia-formula-1-qatar-grand-prix-update) sets 18 total laps per set including prior use. Individual remaining-life/counting criteria were communicated to teams; without documented applicability and set-age evidence, strategy abstains rather than inventing a three-stop rule.

## Conditional pace model

The main estimator is a partially pooled **robust regression**, not a Bayesian posterior or measured degradation. It fits `lap_time_s + (g-d)*(race_lap-1)` and reverses that assumed effect at prediction. MEDIUM-reference contrasts, per-compound age/tail hinges, full-post-outlap warm-up, latest known track-temperature deviation/missingness and driver/event/layout/spec-era effects form a deterministic sparse design. Unseen groups contribute zero deviation; that is pooled transfer, not identical tyre chemistry.

Huber loss uses epsilon 1.35, alpha 0.0001, maximum 1,000 iterations and tolerance 0.000001. Per-column penalty scaling implements partial pooling. Median target translation and bounded sparse robust-ridge initial values condition the optimizer; the final sklearn Huber solve retains its objective and must converge. Nonconvergence is an error, never a usable fit.

The bundle contains nine `(g,d)` settings: `g={0.04,0.06,0.08}`, `d={-0.02,0,0.02}` seconds/lap. Equal `g-d` settings share fits and resamples. Each distinct fit uses 200 whole-event/driver-stint parameter bootstrap replicates. Parameter draws are global across cars/laps; residual paths use centered three-lap within-stint blocks from event-held-out fits with prefix-only adaptation. Models use numeric-only NPZ and semantic metadata identities; no arbitrary pickle loading is needed for the hierarchical model.

The latest five eligible released laps anchor a driver's pace. Current-stint slope adaptation requires eight laps spanning six ages; otherwise shape is pooled. Advice requires three prefix laps, candidate-compound evidence from 30 laps/three stints/two earlier events, spec-aware transfer support, residual support and no more than two laps beyond observed age support. Uncertainty is not eliminated when tyre slopes are negative.

Lap-h and sum-of-next-h intervals are distinct targets for horizons 1,3,5,10. Earlier event-held-out 80%/95% intervals are widened using nonconformity scores only when 30 scores span two events. Held-out predictions use the same prefix anchors and hold-latest temperature as serving forecasts. Future pit segments and other horizons remain raw/uncalibrated. Correlated prefixes and era drift mean empirical marginal coverage, not a joint or counterfactual guarantee.

The rolling baseline uses up to five eligible released laps, requires three, and holds tyre effect constant. It cannot provide cross-compound strategic advice. Management, damage, actual fuel load and tyre temperatures are unobserved.

## Conditional strategy simulation

Inventory is consumed, not replenished: the default one unused HARD/MEDIUM/SOFT set per driver is explicitly assumed. Pit-now first enters on lap N+1; the new set applies from its outlap. Net transit, stationary service and assumed outlap warm-up are separate quantities. Released peer-stop evidence estimates their total loss; the split remains assumed. For HGB, this clean-lap pit-loss estimator explicitly uses its hierarchical base, not missing ML-history features.

A width-64, depth-at-most-three beam retains legal completions; the three lowest isolated-time plans per branch receive 256 screening draws. Published draws use an independent stream, paired across the selected fixed plans. Search exhaustion is not proof of infeasibility. Both branches project the same scheduled distance, including lapped cars. Flag-time classification is separate: a two-hour expiry at a leader crossing still requires the following leader lap, capped by scheduled distance.

All cars share one simulated leader-crossing neutralisation clock. In-progress pace components rescale at that boundary; stationary service is not discounted. VSC preserves order; SC applies approximate queue compression without automatic unlapping. Hypothetical closures are evaluated on this clock, not a lapped car's own lap count. Unsupported/stale opponents or absolute pit occupancy suppress field probabilities rather than invent retirement. Exact sector deltas, pit lights, red-flag restarts and precise pit geometry remain unsupported.

Positive stay-minus-pit delta favours pit now. Advice requires both expected-time direction and probability threshold (pit: positive mean and probability at least 0.8; stay: negative mean and probability at most 0.2), unchanged direction across the mandatory fuel/track, pit-loss and traffic sensitivities. Sensitivities retain the screened plan pair rather than reoptimizing. Position probabilities are conditional on assumed rival policies and no unmodelled retirement; these are not causal win probabilities.

## Chronological ML evidence

Training roles admit one immutable acquisition per distinct session; corrected copies cannot inflate event counts or enter each other's held-out fits. Dry fitting excludes observed rain on slicks and unknown weather. Issued residual/calibration anchors reconstruct released deletion/reinstatement state; final corrections select historical evaluation targets only. Calibration and serving choose the same spec-era/compound/all-dry residual fallback. Cross-compound and future-stop forecasts do not inherit continuation calibration.

The HGB challenger learns event-chronological issued-forecast residual corrections and lag histories, not whole-event fitted residuals. Future recursion uses generated predictions; a planned stop resets its stint origin/history and uses hierarchical warm-up fallback. Matching hierarchical parameter replicates remain shared. Conditional quantiles supply piecewise-linear inverse-CDF marginals with linear tails, centered around the separately predicted mean correction; base three-lap blocks supply serial rank dependence, not a second noise source. Display, calibration and strategy sampling use that same distributional assumption. ML parameter uncertainty is explicitly **not bootstrapped**.

2022 initializes fitting; 2023 rounds 1–16 select on expanding development folds; rounds 17–22 are disjoint final calibration; all 2024 is untouched holdout. Every transitive frozen-model source is checked, so an earlier 2024 race cannot enter later holdout scoring. Same-stint clean targets determine evaluation eligibility only; stint-ending censoring is informative. Errors, pinball loss, widths and empirical coverage are race-macro summaries with event-block paired uncertainty, not lap-wise significance.

Hierarchical remains the configured default. HGB may replace it only after development evidence meets the specified 5% one-lap MAE, coverage-error and compound-group gates. **No actual-data winner has been selected here:** the development run recorded 0 scored events out of 16 missing races. No valid frozen-settings manifest exists, so 2024 holdout has not been inspected. Synthetic fit/load/forecast proof does not establish real-race accuracy or a challenger victory.

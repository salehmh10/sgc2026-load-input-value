# Scientific methods and implementation contract

This document describes the current manuscript, *Incremental Value of Weather and Calendar Inputs in Hour-Ahead Load Forecasting*, and the locked I14/I15 methods from which the portable implementation was extracted. The study is a retrospective comparison of conditional forecasting value. It does not estimate causal contributions to electricity consumption.

## Forecast task and chronological splits

For target hour `t`, a forecast uses load history `[t−168 h, t)` and predicts one hour ahead. Models retain fixed weights during rolling inference; later origins can incorporate earlier observations as they become available. This is not retraining at every origin.

All bounds below include the start and exclude the end, at recorded midnight. Timestamps are naive, retained as recorded; no timezone conversion is justified by the available metadata.

| Window | Train | Validation | Evaluation |
|---|---|---|---|
| Warm (`warm_late_summer`) | 2024-08-04 to 2025-08-04 | 2025-08-04 to 2025-09-01 | 2025-09-01 to 2025-09-29 |
| Cold (`cold`) | 2024-11-03 to 2025-11-03 | 2025-11-03 to 2025-12-01 | 2025-12-01 to 2025-12-29 |

Each split has 365 training days, 28 validation days and 28 evaluation days. Its grid starts 168 hours before Train to preserve the first full history. These input-only context rows never fit scalers and are never targets. Historical observed target counts are 7,088 / 611 / 626 for warm and 7,196 / 662 / 669 for cold. Each evaluation grid has 672 origins; only 10 warm and 25 cold days have all 24 observed targets. The warm training **sample count** of 7,088 and the common **feature count** of 7,088 are distinct quantities.

Warm Evaluation is inside cold Train. The windows are therefore reported separately and are not independent test replications. Both windows had prior design exposure; see [limitations](limitations.md).

## Preprocessing and observation support

Load is measured in kW. Observed load outliers are retained without clipping, winsorization or deletion. Only originally observed target loads enter training loss, validation selection or scoring. A prediction must nevertheless be finite at every scheduled origin, including origins with missing targets.

Historical missing load is reconstructed using only earlier **original observations**. The ordered hierarchy is:

1. Median of available observed weekly lags 168, 336, 504 and 672 hours.
2. Median of past observations with the same hour and working-day type.
3. Median of past observations at the same hour of week.
4. Median of all past observations.
5. Physical zero sentinel when no prior observation exists.

Filled values never become donors. In the historical cold split, the first 31 pre-training hours use the zero sentinel, with unavailable-data indicators. Quality channels preserve observation status, reconstruction status, gap age, recent observation fraction and long-gap status. Gap age is transformed by `log1p` before standardization; recent observation fraction is the trailing 168-hour observation mean. The long-gap flag marks at least 24 consecutive missing hours.

Weather gaps use fixed per-variable medians of the split's observed Train weather. Load scaling is fitted on observed Train loads; weather scaling uses the median-filled Train grid. The two continuous quality quantities use the Train grid. Binary and cyclic calendar features are not standardized. These parameters are frozen for Validation and Evaluation. Fixed Train weather medians can summarize observations later than an individual **training** origin; they remain legitimate fitted preprocessing parameters for the subsequent fixed-model forecast experiment. They are not target-dependent interpolation or operational expanding-origin preprocessing.

Calendar channels are sine/cosine pairs for hour, weekday and hour of week; twelve Jalali-month indicators; and working-day, holiday, pre-holiday and post-holiday indicators. Holidays follow Iranian national holidays or Friday. Adjacent-day holiday flags use a known calendar, not future load or weather. Emergency closures are not independently documented. The historical `holidays` implementation version was 0.103.

## Tensor schema and flattening

All model families receive a historical tensor `(168, 42)` and target covariates `(1, 32)`. The exact channel order is:

| Historical positions | Channels |
|---|---|
| 0 | Scaled load history |
| 1–9 | Past path: temperature, humidity, temperature missing, humidity missing, load observed, load imputed, scaled log gap age, scaled recent observed fraction, long-gap flag |
| 10–35 | Historical future path: temperature, humidity, temperature missing, humidity missing, then 22 calendar channels |
| 36–41 | Six scaled thermal features in the order below |

The 22 calendar channels are `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `hour_of_week_sin`, `hour_of_week_cos`, `shamsi_month_01` through `shamsi_month_12`, `working_day`, `holiday`, `pre_holiday`, `post_holiday`. Target covariates contain the 26 future-path base channels followed by six thermal channels. There is no target-hour load predictor.

Ridge and XGBoost use float32 input values flattened in C order: history rows from `t−168` through `t−1`, within-row channels in the order above, followed by the target row. Thus `168 × 42 + 32 = 7,088` columns. Ridge promotes those exact float32 values to float64 for fitting. Duplicated historical weather and missingness channels remain duplicated, as in TiDE. No lag selection or family-specific predictor redesign is introduced.

## Thermal features and selection

For each forecast's 169-hour window `[t−168, t]`, heating excess is `max(heat_threshold − temperature, 0)` and cooling excess is `max(temperature − cool_threshold, 0)`. The feature order is:

1. Heating excess.
2. Cooling excess.
3. Heating degree-hours over the preceding six hours.
4. Cooling degree-hours over the preceding six hours.
5. Heating degree-hours over the preceding 24 hours.
6. Cooling degree-hours over the preceding 24 hours.

Every accumulated feature excludes its current hour and is truncated at the beginning of **its own forecast window**, with an empty sum equal to zero. Consequently early historical rows do not use temperatures outside the 168-hour history. Target accumulations use all six or 24 preceding hours. The two intensity features have units °C, and the accumulated features have units °C·h.

An auxiliary Ridge model with penalty 10 selects thresholds. This is distinct from the full Ridge-ARX comparator. It uses physical reconstructed load lags 1, 24 and 168, all 22 target calendar channels, and heating/cooling excess, with a freshly fitted StandardScaler in each prefix. Four expanding Train prefixes end after days 181, 227, 273 and 319; each has a subsequent 46-day inner validation interval. Weather medians use only the current prefix. Candidate heating thresholds are 10, 14, 18 and 22°C; cooling thresholds are 18, 22, 26 and 30°C, restricted to heating ≤ cooling. Every candidate must have at least 100 jointly observed load–temperature targets **strictly below and strictly above each threshold in every training prefix**. All observed validation targets remain scored. Selection minimizes the equally weighted mean of the four inner mean daily WAPE scores, breaking exact ties by lower heating then cooling threshold. No eligible pair is a hard failure, not an invitation to tune on Evaluation.

Historical selected pairs are `(10, 18)`°C warm and `(10, 22)`°C cold. Main thermal standardization uses all 169-row occurrences from **observed Train origins**, retaining multiplicity from overlapping windows but excluding occurrences whose row timestamps precede Train. The median over these same occurrences supplies each thermal masking constant. Thresholds and fitted preprocessing are shared across all three model families within a split.

## Seven retrained conditions

Channel removal means replacement with a fixed Train marginal median during both fitting and inference. Dimensions and channel order remain unchanged. Base-channel medians use the Train grid; thermal medians use the occurrence-weighted fit described above. Every duplicated pathway is masked. Load history and its quality flags are always retained.

| Condition | Change relative to FULL |
|---|---|
| `FULL` | Retain all channels; retained target weather is realized weather. |
| `NO_DERIVED` | Replace the six thermal channels. |
| `NO_TEMP` | Replace temperature, its missingness flag and all six thermal descendants. |
| `NO_HUMIDITY` | Replace humidity and its missingness flag. |
| `NO_CALENDAR` | Replace all 22 explicit calendar channels. |
| `NO_WEATHER` | Replace temperature, humidity, their missingness flags and the thermal features. |
| `FULL_PROXY` | Retrain using separately constructed past-only target weather estimates. |

For `FULL_PROXY`, temperature and humidity independently use their latest original observation in `[t−168, t)`, falling back to the relevant Train median. Target missingness flags indicate donor availability (0) or fallback (1); actual target-hour missingness is not an input. Historical weather remains unchanged. Target thermal features are recomputed after temperature replacement, using FULL thresholds and scalers. This is a persistence proxy, not an archived operational weather forecast. `NO_CALENDAR` removes explicit channels only: calendar-informed load reconstruction and encoded calendar information in past loads remain.

## Fitting the three families

**TiDE.** The native 168-to-1 configuration uses two encoder blocks, one decoder block, hidden width 128, decoder output width 16, past/future temporal projection widths 4, past/future temporal hidden widths 16, temporal decoder hidden width 32, dropout 0.1 and no reversible instance normalization. The local correction replaces only `temporal_decoder.layer_norm` on its scalar output with an identity. Five other LayerNorm modules and the lookback skip connection remain. The resulting parameter count is exactly 477,491; an unmodified library TiDE is not an equivalent replacement.

Seeds 42, 314 and 2718 produce 42 final fits across two windows and seven conditions. Initial states match across conditions within each split–seed pair. AdamW uses learning rate 0.0003, weight decay 0.0001 and L1 loss on scaled load. Training uses chronological batches of 64, gradient clipping at 1, deterministic CPU execution and four threads. Validation early stopping has patience 12 and minimum improvement 0.0001. The limits are 200 epochs and 1,800 seconds per fit. Restore the best checkpoint without Train–Validation refitting. Neither limit was reached in the historical TiDE fits.

**Ridge-ARX.** Fit an intercept and all flattened features to observed kW targets, minimizing residual sum of squares plus `alpha × sum(coefficient²)`; the intercept is not penalized. Historical fitting uses centered float64 primal normal equations and Cholesky, checking relative stationarity residual ≤ 1e−7. Candidate penalties are 0.01, 0.1, 1, 10, 100, 1000 and 10000. Select on FULL observed-target Validation MAE, breaking ties by candidate order, then reuse the selected penalty across conditions. Historical selections are 100 warm and 10 cold. The predictor is deterministic: 14 final fits, without artificial seed replication.

**XGBoost.** Squared-error regression uses kW targets, CPU histogram trees, maximum 64 bins, no L1 penalty and gamma 0. Twelve prespecified candidates vary depth, learning rate, minimum child weight, row sampling, column sampling and L2 penalty. The locked candidate list is the authoritative configuration; its historical sampling used Python `random.Random(20260926).sample` over the documented Cartesian grid. FULL selection uses model seed 42 and observed-target Validation MAE. Final fits reuse selected hyperparameters across conditions but stop separately for each condition and seed, with patience 50 and a 2,000-round ceiling. Predictions retain `best_iteration + 1` trees. Three seeds produce 42 final fits. Both windows select learning rate 0.03; warm/cold depth is 4/2, child weight 5/1, row sampling 0.8/1.0, column sampling 1.0/0.6 and L2 penalty 10/1. Seven final cold fits reached the ceiling; convergence beyond it was not assessed.

All final fitting uses Train targets only. Validation selects checkpoints or hyperparameters, and Evaluation scoring follows finalized models. The historical I15 comparison reused I14 TiDE predictions unchanged. New portable runs create their own provenance and do not claim to be those historical runs. L1 neural loss, squared-error tree fitting and penalized linear squared error differ, as do capacity and search budgets.

## Evaluation, ensembles and intervals

For each day, daily WAPE is `100 × sum(abs(prediction − actual)) / sum(abs(actual))` over observed target hours. **Mean daily WAPE** averages daily ratios equally. **Pooled WAPE** first aggregates numerator and denominator across the whole window. All 28 days, including incomplete days, enter the historical primary mean. A zero denominator or day with no observations is explicitly undefined. MAE, RMSE and signed bias (`prediction − actual`) use observed targets in kW; daily-WAPE P90 uses the 0.9 quantile.

For TiDE and XGBoost, average the three aligned predictions in kW **before** computing ensemble metrics. Averaging seed metrics is not equivalent. Report single-seed values separately; their sample standard deviation uses `ddof=1`. Ridge remains a single deterministic predictor.

On days with all 24 observed targets, peak-magnitude absolute error is `abs(max(prediction) − max(actual))`. Peak timing is the absolute hour distance between the earliest predicted and observed maxima. The distance is not circular across midnight. Average these daily errors over eligible days (10 warm, 25 cold). Peak contrasts are descriptive, without confidence intervals.

Every removal/proxy contrast is `other − FULL`, in percentage points for WAPE: **positive favors FULL**. `NO_WEATHER` and `NO_TEMP` are primary; other conditions and direct family rankings are supplementary. Pair identical dates and observation support. For exact historical confidence-interval compatibility, initialize NumPy `default_rng(20260926)` separately for each contrast, consume 10,000 IID day resamples first, then draw 10,000 noncircular moving-block samples of length three from the same stream. Select `ceil(n_days / 3)` blocks from starts `0` through `n_days−3`, concatenate, truncate to `n_days`, and average. The 2.5th and 97.5th percentiles are the reported block intervals. The IID interval is supplementary. Fewer than three days or any undefined paired daily difference yields no interval. Intervals quantify conditional day variation within the window, not retraining uncertainty, and are not multiplicity-adjusted.

Simple references predict lag-1, lag-24 or lag-168 reconstructed historical load, using the same observed targets. The publication's stored reference scores preserve their original two-decimal precision. The TiDE leave-one-seed-out check averages the remaining two predictions before scoring. The separate TiDE September sensitivity excludes 23–27 September inclusive from scoring without refitting, leaving 23 days and 507 observed targets. It does not replace the primary analysis and was not extended to Ridge or XGBoost in the current paper.

## Evidence boundary

The paper and the I14/I15 locked protocols, scientific source files, feature schemas, training manifests and publication summary CSVs were compared during extraction. The seven I14 scientific source hashes and four I15 training-source hashes match their historical locks. No numerical contradiction was found in the principal reported contrasts, their signs, block limits or peak differences. Historical execution modules were inspected as text, without importing their side-effecting bootstrap code.

An independent comparison extracted only selected pure functions from the historical source syntax tree, without importing original modules. On synthetic arrays, load reconstruction, fitted preprocessing and transformed frames matched exactly; all seven conditions matched at seven selected origins with zero numerical difference. Separate checks verified chronological flatten indices, truncated degree-hour sums, and invariance of a proxy forecast's inputs to changed target/future load and target weather after preprocessing was fixed. The 84 representative metric rows, 72 contrast rows and 24 primary seed-point rows match historical summary exports exactly. These focused checks support extraction fidelity; they are not full retraining or verification of private observations.

Earlier I11–I13 studies explain calendar/reconstruction ancestry and the TiDE correction. Their runtime code and caches are not required: I14 already contains the needed implementations. Historical jump-event and attribution modules do not support a figure or claim in the current manuscript and are outside this companion's current-paper reproduction scope.

Artifact reproduction reads authorized summary CSVs and redraws figures/tables. It does **not** retrain models, recover private hourly predictions or independently recompute bootstrap intervals from them. New private-data runs and focused implementation tests provide a separate level of verification; their scope and results are documented in [reproduction](reproduction.md).

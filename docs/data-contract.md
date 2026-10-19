# Private input contract

The full experiment requires an authorized local CSV. The repository does not contain the real hourly load, weather series, reconstructed frames, predictions, or model weights. Values and timestamps must retain their original alignment; the runner does not shift timezone or reconstruct a different sampling convention.

| Column | Meaning and validation |
|---|---|
| `TIMESTAMP` | Unique, ascending, timezone-naive local timestamp on an exact hour. The original convention is retained without timezone conversion. |
| `LOAD` | Original observed load in kW; missing values may be blank/NaN. Observed outliers are retained. Infinite values are rejected. |
| `TEMP_HOUR` | Hourly temperature in degrees Celsius; missing values permitted, infinity rejected. |
| `HUMIDITY` | Relative humidity in percent; missing values permitted, infinity rejected. |
| `HOLIDAY` | Original daily holiday flag, constant across present rows in a date. Used to audit disagreement with the national calendar, not as the model holiday input. |
| `SHAMSI` | Jalali date as integer `YYYYMMDD`, checked against the local arithmetic conversion for every present value. |

Missing rows are reindexed onto a complete hourly grid. At minimum, provide the 168 hours preceding each Train start through its Evaluation end. The original protocol forbids modeling or donor values on or after 2026-01-08. The reader first reads timestamps only, then reads the six selected columns only within the requested fold interval. It does not use later rows as donors or fit observations.

| Fold | Train [start, end) | Validation [start, end) | Evaluation [start, end) | Observed targets Train / Validation / Evaluation |
|---|---|---|---|---|
| Warm late summer | 2024-08-04 — 2025-08-04 | 2025-08-04 — 2025-09-01 | 2025-09-01 — 2025-09-29 | 7088 / 611 / 626 |
| Cold | 2024-11-03 — 2025-11-03 | 2025-11-03 — 2025-12-01 | 2025-12-01 — 2025-12-29 | 7196 / 662 / 669 |

The supplied paper configuration rejects different observed-target counts. It also rejects unsupported edits to scientific settings, so a manifest cannot claim altered thresholds or a bootstrap seed that the implementation did not use. Paths for input and output remain freely configurable.

## Reconstructed context and fitting scope

Missing load is filled using observed donors strictly earlier than the context hour: the median of available weekly lags 168/336/504/672, then earlier observations with the same hour and working-day type, then the same hour of week, then all earlier observed load. If none exists, physical zero is a sentinel with no donor. Filled values never become donors. The context includes quality channels: observed/imputed indicators, log gap age, recent observed fraction, and a 24-hour gap flag. Missing targets are excluded from training loss and scoring; predictions are nevertheless produced at every Validation and Evaluation hour.

Load scaling is fit on observed Train targets. Weather missing values use fixed per-variable Train medians; weather scaling uses the filled Train grid. Quality scaling is also Train-only. No pre-Train context, Validation, or Evaluation row fits these scalers. Holiday inputs are the Iran national schedule or Friday, with known pre-/post-holiday flags; emergency closures are not inferred.

Thermal thresholds are selected using the original four expanding Train prefixes and auxiliary Ridge regressions. Each candidate needs at least 100 jointly observed load/temperature targets on each side of both thresholds in every training prefix. Selection minimizes equally weighted inner mean daily WAPE; ties favor smaller heating then cooling thresholds. Insufficient support fails without a fallback. The six derived features use heating/cooling excess and preceding 6/24-hour sums excluding the current hour. Every 169-row origin window resets the sums. Their scaler and mask medians count overlapping occurrences at Train row timestamps, including target-hour occurrences, and exclude all pre-Train occurrences.

## Exact model input

Each origin uses `[t−168h, t)` load history, 9 past covariates, and 32 historical future covariates; target-hour covariates have 32 channels. History order is load, past covariates, then future covariates including the six thermal features. Ridge and XGBoost flatten chronological history in C order and append target covariates: `168 × 42 + 1 × 32 = 7088` float32 values. No current or future load enters this tensor. Duplicate weather and missingness channels are deliberately retained.

The seven conditions are `FULL`, `NO_DERIVED`, `NO_TEMP`, `NO_HUMIDITY`, `NO_CALENDAR`, `NO_WEATHER`, and `FULL_PROXY`. Ablations replace every matching channel path and associated missingness flag by its fixed Train median without deleting dimensions. `NO_TEMP` also masks thermal features; `NO_WEATHER` masks temperature, humidity, flags, and thermal features. `NO_CALENDAR` removes explicit channels only: context reconstruction remains calendar-informed.

`FULL` includes realized target-hour weather and is retrospective. `FULL_PROXY` independently takes the last real observation of each weather variable in `[t−168h,t)`, or its Train median when none exists. Its target flags indicate donor/fallback availability and never inspect target missingness. Thermal features are rebuilt after replacement. This is a persistence proxy, not an archived weather forecast.

## Execution and outputs

```bash
liv experiment run --data /path/to/private/input.csv --output /path/to/new-private-run --config configs/paper.json
```

The output must be new or empty. The run records input and code hashes, configuration, package versions, preprocessing lineage, selected parameters, all models and reload checks, aligned predictions, and summaries. These files can contain private hourly observations and absolute local paths in library logs. Keep the entire output private; it is not automatically approved for publication. All paper-sized fits are CPU-only and serial. Every saved final model is reloaded in a fresh process and checked against Validation and Evaluation predictions before the frozen model manifest permits Evaluation scoring.

```bash
liv experiment smoke --output runs/synthetic-smoke --config configs/paper.json
```

This creates an explicitly synthetic CSV and a separate `execution/` folder. It uses a 48-hour Train interval, 24-hour Validation and Evaluation intervals, prescribed smoke-only thermal thresholds, one seed, FULL-only model fitting, one TiDE epoch, and two XGBoost trees. All seven input conditions are built and shape-checked. Ridge retains the exact 7088-column centered primal solver. Two small baseline selection fits and three final fits exercise saving, fresh-process loading, inference, and scoring. Smoke results provide no evidence for the paper's numerical findings or training convergence.

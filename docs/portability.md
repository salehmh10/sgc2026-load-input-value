# Extraction and portability changes

The portable package contains the necessary I14 preprocessing, thermal feature, masking and local TiDE implementation plus I15 baseline fitting and evaluation semantics. It never imports or launches a historical iteration directory. Earlier iterations are not runtime dependencies. Public artifact reproduction uses the approved summary CSVs through a separate script and does not import the training stack.

| Historical responsibility | Portable location | Engineering change |
|---|---|---|
| I14 `study_data.py` | `src/load_input_value/data.py` | Explicit input and output paths; fresh fold reconstruction replaces implicit raw path and historical cache lookup. Pure calendar, causal donors, scaling and transformation bodies retained. |
| I14 `temperature_features.py` | `src/load_input_value/features.py` | Explicit feature output directory replaces historical review files and global root. CV design, eligibility, objective, tie rules and weighted occurrence scaling retained. |
| I14 `model_inputs.py`; I15 `i15_data.py` | `src/load_input_value/inputs.py` | Parameters supplied by the reconstructed fold; original native input construction and masks retained, with the I15 C-order flatten contract. |
| I14 `tide_fixed.py` | `src/load_input_value/tide.py` | Local subclass retained, including scalar temporal-decoder LayerNorm→Identity. Five other LayerNorm layers and 477491 parameters are asserted. |
| I14 `study_training.py` | `src/load_input_value/training.py` | Matrix-backed datasets emit the same native Darts tuple. Output/cache locations are explicit; callback is a serializable top-level class; strict best checkpoint reload retained. |
| I15 `i15_training.py` | `src/load_input_value/baselines.py`, `experiments.py` | Centered float64 primal Cholesky equations and stationarity check retained. FULL-only tuning is shared across conditions. |
| I15 `i15_evaluation.py` | `src/load_input_value/evaluation.py` | Pure metric, ensemble, contrast and bootstrap functions retained. Historical file loading and publication-state gates replaced by new-execution manifests. |
| Historical protocol locks | `configs/paper.json`, `protocol.py` | Scientific settings transcribed and hashed; private environment paths and obsolete workflow-owner fields removed. Unsupported configuration edits fail explicitly. |

Historical hashes and detailed absolute source-to-target mapping were recorded privately during extraction. No historical manifest was rewritten to imply these are the original executions. New results always have their own code/configuration fingerprints, separate output directories, and `historical_execution: false`.

The full runner fits 42 TiDE models, 14 deterministic Ridge models and 42 XGBoost models, with auxiliary threshold-CV fits and FULL tuning fits in addition. TiDE retains L1 loss, AdamW, 200-epoch ceiling, 1800-second budget per fit, patience 12, minimum improvement 0.0001, batch size 64, no shuffle, and fixed paired initialization across conditions for each fold/seed. Ridge uses squared residuals plus L2; XGBoost uses squared-error trees with observed Validation MAE early stopping. The three objectives are not matched.

Fresh-process replay covers every Validation and Evaluation input. TiDE inverse scaling preserves the original float32 in-place operations using NumPy float64 parameters, including when parameters travel through JSON. TiDE tolerance is 0.001 kW; the baselines use absolute tolerance 0.000001 kW and relative tolerance 1e-12. These checks concern newly trained models; no original private model was re-trained or reloaded during packaging.

Only after all requested final models and replay evidence are hashed does the full run compute Evaluation metrics. Seeds 42, 314 and 2718 are averaged as predictions in kW before ensemble metrics. Inference rolls the observed/reconstructed history forward with fixed weights; models are not refit at each origin. Scoring retains the original equal-day WAPE, pooled WAPE, complete-day peak rules, and exact iid-then-block bootstrap RNG stream.

The full run also generates causal lag-1/24/168 references, seed variability and contrasts, Validation/Evaluation shifts, regime summaries, TiDE two-seed sensitivity, and the TiDE-only post-hoc omission of September 23–27. These sensitivities do not refit models, do not apply to the other model families, and do not establish independent replication. Exploratory attribution and temperature-jump analyses absent from the current paper are not part of this runner.

## Scope of validation

Tests cover causal donors, Train-only scalers and derived-feature occurrence weighting, absence of target/future load in inputs, all duplicate mask paths, target-weather-independent persistence, flattening order, Ridge stationarity, prediction averaging, missing-target scoring, daily versus pooled WAPE, earliest peak ties, and exact block bootstrap RNG order. The small synthetic command verifies the three model families without full retraining. The repository's verification report records which environments and commands were actually exercised; a successful synthetic check does not establish equality of re-trained paper results.

Full private-data training is intentionally not run for packaging. Calendar/library versions and floating-point libraries can affect future runs, and the historical environment was a larger personal environment. The tested installation is an isolated environment with explicit dependencies; platform support should be read from the verification report, not inferred from path portability. The full run is deliberately conservative about output reuse: failures preserve partial evidence and require a new output directory, instead of silently resuming incomplete models.

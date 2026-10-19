# Interpretation and reproducibility limits

The evidence concerns the incremental predictive value of explicit inputs conditional on a 168-hour load history. It does not identify causal weather or calendar effects on consumption, and the input-removal differences cannot be summed into a demand decomposition. No condition removes load history. Even `NO_CALENDAR` retains calendar-informed historical-load reconstruction.

## Temporal and data limitations

The two 28-day evaluation windows were previously inspected, and the design was developed retrospectively. Earlier experiments screened architectures on 8 October–10 November 2025 and selected TiDE settings on 11–26 November. The thermal-feature family was later selected on the same intervals with the corrected model. Warm evaluation dates had appeared in prior training. Cold 1–21 December had served as ensemble validation, 22 December as context, and 23–28 December as test data. These are not pristine holdouts. The cross-family comparison followed inspection of TiDE results; its search and analysis rules were locked before scoring the new family predictions.

Warm Evaluation also occurs within cold Train. Shared histories, different missingness and fitted thresholds mean that differences between windows cannot be attributed solely to season. The warm window is relatively warm late summer, not a defined heat wave or representation of a complete season.

The location is an anonymous city in Iran. Available metadata do not establish network aggregation level, the particular OpenWeatherMap product, or a verified timezone. Load and weather observation masks coincide, and warm-weather training coverage is incomplete: June and July 2025 contain 404/720 and 258/744 hourly observations. The missing hours' temperatures and their effects on the learned relation are unknown. Large load-level changes and the September decline have no established physical or operational explanation in the records. Observed extreme values remain in the primary analysis.

## Model and objective limitations

The models share predictors but differ in capacity, optimization objective and tuning budget. The fixed 7,088-column representation preserves duplicated channels and was not optimized separately for each family. Hyperparameters selected on FULL control tuning choices across removals; they need not be optimal after correlated predictors are masked. Direct rankings apply to these fitted configurations and do not prove general architectural superiority.

The local TiDE correction removes only a scalar-output LayerNorm that otherwise destroys that branch's input dependence. Retaining the remaining normalization and skip paths is part of the method, not a cosmetic implementation detail. A different TiDE implementation needs demonstrated equivalence before its results can be described as reproduction.

Both heating thresholds and the warm cooling threshold lie at the lower grid boundary. Cold cooling candidates above 22°C fail support requirements. Small runner-up differences and a constrained grid limit threshold identification. These predictive transformation parameters are not physical balance temperatures. Thermal activation also differs between the two windows, so six-channel removal is not a comparison of equivalent physical heating and cooling effects.

Seven final cold XGBoost fits reached the 2,000-round ceiling. The best validation iteration was retained, but a larger ceiling was not studied. Three stochastic seeds provide limited evidence about training variability. TiDE and XGBoost prediction ensembles are not single-fit results; lower ensemble error does not imply that every individual fit has the same ranking.

## Weather availability and uncertainty

Except for `FULL_PROXY`, retained target-hour weather is realized weather, not a verified forecast available at the origin. The proxy independently persists the latest earlier observed temperature and humidity. It retains historical weather and is not the same comparison as removing all weather. Neither archived weather forecasts nor a complete set of ablations under restricted weather availability were evaluated. Proxy improvements do not establish general superiority of persistence over weather forecasts or observations.

Mean daily WAPE gives equal weight to days, including incomplete ones; smaller daily load denominators can increase WAPE without proportional increases in absolute error. Peak metrics use only 10 warm and 25 cold complete days, and no peak uncertainty intervals are supplied. Their observation support differs from the primary score's 28 days.

The paired three-day block intervals condition on fixed models or prediction ensembles and the observed window. They omit retraining uncertainty and multiplicity adjustment. Agreement between model families does not create independent replications. Warm TiDE seed disagreement and differing peak responses limit any broad recommendation.

The September exclusion and leave-one-seed-out checks are descriptive TiDE sensitivities. The September exclusion removes 23–27 September only from scoring without refitting; every day remains in the primary evaluation. The current paper does not extend that exclusion to Ridge or XGBoost.

## Reproduction levels and source reconciliation

| Item | What is preserved and what remains limited |
|---|---|
| Paper figures and tables | Authorized historical summary CSVs support numerical and visual regeneration. Font/layout or PDF metadata may change image hashes without changing numerical values. |
| Bootstrap limits | Stored limits can be checked for correct plotting and signs. Summary CSVs cannot independently regenerate bootstrap samples; private hourly predictions or daily paired differences are required. |
| Historical simple references | Stored values retain original manuscript precision (two decimals). They were inherited from earlier validated manuscript scores, not newly computed by the I15 exporter. A new private-data run calculates references afresh. |
| Historical TiDE provenance | I15 reused I14 predictions unchanged. A new portable run has new code/environment/run metadata and must not masquerade as the historical I14 run. |
| September and leave-one-seed-out evidence | Supporting historical aggregates exist, but standalone generation functions were not found in the I14/I15 execution modules inspected. New explicit scoring implementations must be distinguished from historical output. Private review files are excluded. |
| Earlier figures and scripts | The source paper directory also contains earlier TiDE-only figures and exports. The current manuscript uses `study_windows`, `i15_primary_value`, `i15_mean_peak_tradeoff`, and the two I15 tables. Earlier assets are not substitutes for the current figures. |
| Original execution environment | Historical manifests describe a Windows Python 3.12.3 environment, including I15 use of system-site packages. Those records are evidence, not an independently installable lockfile. The companion documents its clean tested environment separately. |
| Full rerun | Private load/weather input is not distributed. Full training is deliberately not repeated while assembling this repository. Synthetic smoke tests exercise software only and cannot establish reproduction of paper scores. |

No conflict was found between the current manuscript's central methods and the locked I14/I15 execution code. The checked scientific-source hashes match their historical locks. This does not certify unseen private data, every historical intermediate artifact or bitwise equivalence under another library version. The source checks, the new implementation tests and any full rerun are distinct evidence.

Recorded historical versions include NumPy 2.2.6, pandas 2.2.2, SciPy 1.13.0, scikit-learn 1.7.2, matplotlib 3.8.4, holidays 0.103, Darts 0.41.0, PyTorch 2.13.0, PyTorch Lightning 2.5.2, PyArrow 25.0.0 and XGBoost 3.4.0. These are reported as found in manifests, not asserted compatible with every platform or installable index. Only platforms and commands actually exercised by the companion's validation may be called tested.

Raw hourly series, aligned predictions containing actual loads, processed frames, model weights, checkpoints and caches are deliberately excluded. Reproducing the full experiment requires the authorized private data and an explicit execution command. Software licensing, paper copyright and data access are separate; a software license never grants access to withheld data or automatically licenses the manuscript.

# Fixed publication reference

These are reviewed aggregate results associated with the current manuscript. They contain partition-level metrics, paired comparison summaries, three-seed contrast points, six rounded persistence scores, and chronological-window summaries. They contain no hourly actual-load observations, predictions, private model weights, or location coordinates.

`SHA256SUMS.json` fixes the admitted CSVs and historical table references. The artifact command verifies these checksums and compares all rendered table cells, plotted contrasts, peak changes, and confidence bounds with the references. The plotting command reads these files directly; it does not discover files in the original project or rerun training.

`i15_accuracy.csv` has 84 rows (two partitions, two windows, three families, seven conditions). Ridge uses its deterministic fit; TiDE and XGBoost use the mean of three predictions before scoring. `i15_contrasts.csv` has 72 condition-minus-FULL summaries. `i15_primary_seed_points.csv` has 24 individual stochastic-seed contrasts for the two primary evaluation comparisons. `i15_simple_references.csv` preserves six manuscript values at their original two-decimal precision; these were inherited, not recomputed by the I15 exporter. `study_windows.csv` holds six chronological-partition summaries, including observed-weather aggregate statistics used by Figure 1.

Confidence intervals are historical paired day-bootstrap results. This package verifies their stored values and draws them; summary CSVs cannot independently reconstruct the bootstrap sample distribution. The original hourly predictions required for that operation are private. New results belong in a separate user-selected output directory, never over these references.

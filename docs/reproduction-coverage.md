# Reproduction coverage

The repository distributes the software needed to reproduce the current study from authorized input data through training, evaluation, descriptive diagnostics and research figures/tables. It also supports figure/table regeneration from the included historical aggregates. The manuscript text and typesetting bundle are not required for either scientific workflow.

## Executable stages

| Stage | Implementation | Configuration or output |
|---|---|---|
| Input schema, hourly grids and observed-target masks | `src/load_input_value/data.py` | Explicit CSV path; [data contract](data-contract.md) |
| Causal load reconstruction, calendar encoding and Train-only standardization | `src/load_input_value/data.py` | Fold preprocessing metadata, audit, scalers and registries |
| Thermal threshold selection and six derived features | `src/load_input_value/features.py` | Cross-validation candidates, support and selected parameter files |
| Native 168-hour inputs, 7,088-column flattening and all seven conditions | `src/load_input_value/inputs.py` | Condition-specific tensors; variable-specific persistence proxy |
| Corrected TiDE model and training | `src/load_input_value/tide.py`, `training.py` | Local scalar-decoder normalization correction, checkpoints and fresh-process replay |
| Ridge-ARX and XGBoost | `src/load_input_value/baselines.py`, `experiments.py` | FULL tuning, frozen settings across conditions, final fits and replay |
| Full two-fold experiment | `src/load_input_value/experiments.py`, `protocol.py` | `configs/paper.json`; 42 TiDE, 14 Ridge and 42 XGBoost final fits plus selection fits |
| Prediction ensembles, error metrics and paired uncertainty | `src/load_input_value/evaluation.py` | Equal-day and pooled WAPE, complete-day peaks, iid and three-day block bootstrap |
| Baselines and descriptive sensitivities | `experiments.py::supplemental_tables` | Lag-1/24/168, seed variability, regime scores, TiDE leave-one-seed-out and September exclusion |
| Descriptive input audit | `src/load_input_value/diagnostics.py` | Coverage, observation masks, daily/load/weather changes, year-over-year comparisons, proxy donor age/fallback and fitted thermal activation |
| New-run publication export | `src/load_input_value/artifacts.py` | Five aggregate CSVs, checksums and source-execution provenance |
| Three figures and three tables | `scripts/reproduce_artifacts.py` | PDF/PNG figures, two numerical LaTeX tables and one fixed protocol table |
| Validation and release audit | `tests/`, `scripts/audit_release.py` | Synthetic scientific controls, numerical artifact checks and publication privacy checks |

Commands for the complete sequence are in [reproduction.md](reproduction.md). Every new-run output stays separate from the immutable historical reference files. The export command requires a completed full experiment; the small synthetic model smoke is only an execution check. Descriptive audits can run without training; thermal activation additionally requires the matching completed run's saved thresholds.

## Why the distribution is smaller than the working project

The implementation consolidates the current I14/I15 scientific definitions into a Python package. Original notebooks, duplicated runners, workflow-owner gates, private review/provenance snapshots and generated narrative reports are not runtime dependencies. The manuscript's exploratory predecessors, integrated-gradient/permutation attribution and jump-event analyses do not support the current study's figures or claims and are outside this reproduction scope. Their omission does not remove the current model-training pipeline.

Scientific libraries such as Darts, PyTorch, scikit-learn and XGBoost are installed at the versions declared in `pyproject.toml`; their source archives are not vendored. Earlier iterations supply methodological provenance, not hidden imports or cache dependencies. [Source mappings](source-map.json), [methodology](methodology.md) and [portability notes](portability.md) identify the retained definitions.

The completeness review identified two integration gaps in the initial distribution: executable descriptive-data summaries, and the bridge from a new full experiment to plotting CSVs. The `audit-data` and `export` commands supply those stages. The fitted-model protocol and historical numerical results were not changed by these additions.

## Materials supplied separately

Authorized private observations must be supplied by the researcher. Model weights, reconstructed hourly frames, predictions containing actual load and all private execution outputs are generated locally. They are excluded from Git, together with credentials, private reviews and personal paths. The manuscript itself is excluded from the current tree at the owner's request; research figures/tables remain reproducible without it. Earlier Git revisions retain the previously distributed manuscript bundle.

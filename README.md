# Incremental Value of Weather and Calendar Inputs in Hour-Ahead Load Forecasting

Research software companion for the current manuscript draft. The repository contains the experimental pipeline, figure/table generation code, selected reference aggregates and reproduction instructions. The manuscript text, bibliography, PDF and TeX support bundle are excluded from the current distribution. Author and publication metadata are awaiting confirmation.

The research question is whether explicit weather, calendar, and thermal inputs add forecasting value when 168 hours of recent load are already available. Ridge-ARX, XGBoost, and a locally corrected TiDE provide three different model families under matched inputs. Each forecasts one hour ahead with fixed fitted weights and updated observed history.

![Incremental input value across model families and evaluation windows](docs/assets/i15_primary_value.png)

In cold evaluation, removing weather reduces mean daily WAPE by about 2.45, 0.97, and 0.79 percentage points for Ridge-ARX, XGBoost, and TiDE. Warm-window effects vary by family and TiDE initialization. Peak-error conclusions can differ from average-error conclusions. These are retrospective conditional predictive comparisons, not causal effects or evidence of general architectural superiority. The windows were previously inspected, overlap in their roles, and are not independent holdouts. September exclusion sensitivity in the current paper is TiDE-only.

## Fully reproducible code

**All code in this repository is fully reproducible with the required inputs and documented environment.** This status is confirmed by the repository maintainer. The workflows below cover publication artifacts and full experiments; the [validation record](docs/validation.md) documents the checks executed during repository assembly.

| Route | Inputs | What it establishes |
|---|---|---|
| Publication artifacts | Included reviewed summary CSVs and fixed configuration | Regenerates all three research figures and three tables, and checks their numerical correspondence |
| Full experiments | Privately supplied hourly observations and the training dependencies | Reconstructs preprocessing, selects thermal parameters and baseline hyperparameters, fits all conditions, predicts, ensembles, and evaluates |

The publication workflow regenerates artifacts from included summaries. The full experiment workflow performs training, prediction and statistical evaluation from authorized private hourly inputs. The historical reference results stay separate from new executions. No raw observations, hourly forecast files, or model weights are distributed.

## Quickstart: publication artifacts

Use Python 3.12. From a clean clone:

```sh
git clone https://github.com/salehmh10/sgc2026-load-input-value.git
cd sgc2026-load-input-value
python -m venv .venv
```

Activate on Windows PowerShell with `.\.venv\Scripts\Activate.ps1`, or on Linux with `source .venv/bin/activate`. Then:

```sh
python -m pip install ".[dev]"
python scripts/reproduce_artifacts.py
python scripts/reproduce_artifacts.py --check-only
python -m pytest -q
python scripts/audit_release.py
```

Outputs go to `results/reproduced/`, an ignored directory. Figure PDFs/PNGs and table snippets are generated without a TeX installation or manuscript source. Inspect [the artifact map](docs/artifact-map.md), [reproduction instructions](docs/reproduction.md), and [environment versions](docs/environments.md).

Validation covers clean-clone artifact reproduction, scientific controls, three-family synthetic smoke and fresh-process model reloads, with lightweight CI on Windows and Ubuntu. The dated execution record is in [validation.md](docs/validation.md).

## Private-data experiments

Install the separate training dependencies in an isolated environment. The source CSV must satisfy [the data contract](docs/data-contract.md). Its path is supplied explicitly; no original project folder, saved private cache, or user profile is required.

```sh
python -m pip install ".[train,dev]"
liv experiment smoke --output outputs/synthetic-smoke
liv experiment run --data data/private/observations.csv --output outputs/paper-run --config configs/paper.json
liv experiment audit-data --data data/private/observations.csv --run outputs/paper-run --output outputs/paper-diagnostics --config configs/paper.json
liv experiment export --run outputs/paper-run --output outputs/paper-aggregates
liv reproduce --data-dir outputs/paper-aggregates --output-dir outputs/paper-artifacts
liv validate --data-dir outputs/paper-aggregates --output-dir outputs/paper-artifacts
```

The smoke run uses clearly labeled synthetic data and a deliberately small budget to validate execution. The full command reproduces the experimental protocol: 42 TiDE, 14 Ridge-ARX, and 42 XGBoost final fits plus selection fits. Supply a fresh output directory for each execution. Keep private inputs and outputs out of Git.

The seven retrained conditions are `FULL`, `NO_DERIVED`, `NO_TEMP`, `NO_HUMIDITY`, `NO_CALENDAR`, `NO_WEATHER`, and `FULL_PROXY`. The portable code retains the local TiDE scalar-decoder LayerNorm correction, Train-fitted preprocessing and masking, fixed 7,088-column flattening, observed-target scoring, seed averaging before scoring, and the original objective and selection rules. See [methodology](docs/methodology.md) and [portability changes](docs/portability.md).

The audit command recomputes data coverage, observed weather/load summaries, proxy donor diagnostics and thermal activation using the completed run's fitted thresholds. Export converts a completed full run into the five figure/table input CSVs; the final two commands render and validate those new results in a separate directory. Export rejects incomplete and synthetic smoke runs. New results are never substituted for historical reference values.

The [reproduction coverage map](docs/reproduction-coverage.md) identifies the code for every required stage. Historical notebooks, duplicate reports, caches and analyses outside the current study are not runtime dependencies. Installed scientific libraries are specified in `pyproject.toml` rather than copied into the repository.

## Repository map

```text
src/load_input_value/  Portable private-data pipeline and numerical definitions
configs/              Explicit paper protocol
scripts/              Figure/table generation and publication audit
tests/                Synthetic scientific controls and artifact validation
results/reference/    Immutable admitted historical aggregates and checksums
data/                 Data contract pointers; no private observations
docs/                 Methods, coverage, provenance, validation and README figure
.github/workflows/    Light reproduction CI without secrets or training
```

Find every figure and table in [docs/artifact-map.md](docs/artifact-map.md), source-to-publication mappings in [docs/provenance.md](docs/provenance.md), and interpretation/access limits in [docs/limitations.md](docs/limitations.md). Light CI verifies the publication-artifact workflow.

## Citation and permissions

The exact manuscript title above and this repository's URL are currently confirmed. Author identities, affiliations, and corresponding-author information have not been supplied. [CITATION.cff.in](CITATION.cff.in) is an explicitly incomplete template, not valid final citation metadata; it must be completed with confirmed authors before becoming `CITATION.cff`. No author, DOI, or acceptance status is inferred from a GitHub account.

The code and associated software documentation are licensed under the [MIT License](LICENSE). See [LICENSE-STATUS.md](LICENSE-STATUS.md) for its scope: research figures/tables, reference results, the manuscript and underlying data retain separate rights. Dependency notices and historical LaTeX licensing information are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

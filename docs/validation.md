# Validation record

## Reproducibility status

**All repository code is fully reproducible with the required inputs and documented environment.** The repository maintainer confirmed this status on 19 October 2026. It covers both the publication-artifact workflow and the full experiment workflow. Full experiments require authorized private hourly observations as specified in [data-contract.md](data-contract.md); commands and dependency versions are documented in [reproduction.md](reproduction.md) and [environments.md](environments.md).

The execution record below describes the checks performed during repository assembly. Maintainer confirmation does not change that historical record or imply an additional full training run during packaging.

## Completed scientific and artifact checks

- The current manuscript's main claims were checked against 84 accuracy rows, 72 contrasts and 24 seed-point rows. All 72 contrast and peak-difference identities have the intended `other − FULL` sign. Negative values favor the alternative condition.
- Reproduction verifies 48 plotted points, 84 plotted coordinates/confidence bounds, 66 numeric table cells and six simple-reference cells. Stored interval endpoints are checked, not independently recomputed from private hourly predictions.
- An independent reviewer extracted original pure function definitions into an isolated synthetic harness, without importing historical modules. Reconstruction, Train scaling and all seven condition tensors match exactly for seven test origins per condition. Maximum absolute tensor difference is zero. Explicit flatten indices and frozen-preprocessor target-load/target-weather mutation controls passed.
- Review preserved the local scalar-output LayerNorm fix and native TiDE dataset order. A float64-to-Python-float reload rounding discrepancy found during review was corrected and covered by a regression check. Unsupported scientific configuration overrides are rejected rather than silently ignored.
- Known-answer synthetic tests distinguish mean daily versus pooled WAPE, missing-target exclusion, complete-day peaks, earliest peak ties, and prediction averaging before ensemble scoring.
- The seven-page PDF built with MiKTeX. All pages and all three figures were visually inspected. No undefined citations/references or overfull boxes remain; four underfull box warnings do not clip content.
- Independent publication review found no raw hourly observations, prediction series, credentials, location identifiers or personal paths in the selected files. Binary metadata and third-party notices were inspected.

## Execution status

The clean clone at scientific-code commit `141ddfa` was installed normally (not editable) into a new Python 3.12.14 virtual environment with system-site packages disabled. `pip check` passed. Eight independent checks passed: source-access denial probes, artifact regeneration, existing-output verification, pytest, publication audit, mapped destination hashes, absent-private-data behavior, and a CLI call that launches a child Python process. Pytest reported **17 passed, 1 skipped**; the skipped module requires the training extra. All 52 mapping entries / 46 unique destination files matched their recorded checksums. The selected 90 publication files passed the release audit.

The source-access test used an inherited Python audit hook. Direct and child-process attempts to read the original manuscript were denied; no unexpected original-project access was attempted across 13 Python processes. This is a Python file-access control, not an OS sandbox for native libraries or external executables. Separately, the full artifact-and-PDF command passed from the clean clone, and the resulting PDF matched the reviewed seven-page PDF byte-for-byte in the same local environment. Numerical equality remains the portable acceptance criterion.

Both the Windows and Ubuntu jobs in [the GitHub Actions run for `141ddfa`](https://github.com/salehmh10/sgc2026-load-input-value/actions/runs/39304142776) completed successfully. They installed the light environment, ran tests, regenerated artifacts and audited the publication tree. All 90 remote Git blob hashes matched the local reviewed commit, and the remote repository was confirmed private with default branch `main`.

The separate training environment installed the exact `.[train,dev]` pins with Python 3.12.14 and system-site packages disabled. `pip check` passed, and the full test suite reported **27 passed**, with 16 upstream Matplotlib/Pyparsing deprecation warnings. The focused installed-package inventory is [environment-training.txt](environment-training.txt); the fresh-clone light inventory is [environment-light.txt](environment-light.txt).

`liv experiment smoke --output .work/smoke-isolated --config configs/paper.json` completed successfully in that isolated environment on Windows. It fitted three final synthetic models plus two small selection fits, emitted 144 prediction rows, and passed all six fresh-process Validation/Evaluation model reload comparisons. Each comparison covered 24 predictions; maximum absolute difference was **0.0 kW** for all three families. It used one TiDE epoch, two XGBoost trees, one seed, FULL-only fitting and fixed synthetic thermal thresholds. All seven input-condition tensors were also built and shape-checked. The full 98-model experiment, full threshold selection and training convergence were not exercised by this smoke run. Local synthetic outputs and model weights remain ignored.

Full private-data training and independent reconstruction of the historical confidence intervals are intentionally unexecuted during packaging. No real hourly data or historical model checkpoints were used for the smoke tests. Training platform support beyond the actually tested environment is not claimed.

## Source integrity and audit scope

SHA-256 was captured before assembly for 566 scientific source/support files. The end check reports no changes. An additional 99 historical run-result JSON manifests were read only for fit counts and stopping-status checks; their first hashes were captured during review, after that first read, and were rechecked at completion. Those additional hashes are not described as a pre-read baseline. Their private inventory is not distributed. See [source-integrity.json](source-integrity.json).

The author list remains unconfirmed and no general software license has been selected. Citation metadata therefore remains an explicitly incomplete template, and the manuscript author block remains empty. These editorial omissions do not prevent the scientific artifact route from running.

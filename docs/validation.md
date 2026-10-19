# Validation record

This document distinguishes executed checks from planned or deliberately unexecuted work. It is updated at release verification; the workflow alone is not evidence of a passing CI run.

## Completed scientific and artifact checks

- The current manuscript's main claims were checked against 84 accuracy rows, 72 contrasts and 24 seed-point rows. All 72 contrast and peak-difference identities have the intended `other − FULL` sign. Negative values favor the alternative condition.
- Reproduction verifies 48 plotted points, 84 plotted coordinates/confidence bounds, 66 numeric table cells and six simple-reference cells. Stored interval endpoints are checked, not independently recomputed from private hourly predictions.
- An independent reviewer extracted original pure function definitions into an isolated synthetic harness, without importing historical modules. Reconstruction, Train scaling and all seven condition tensors match exactly for seven test origins per condition. Maximum absolute tensor difference is zero. Explicit flatten indices and frozen-preprocessor target-load/target-weather mutation controls passed.
- Review preserved the local scalar-output LayerNorm fix and native TiDE dataset order. A float64-to-Python-float reload rounding discrepancy found during review was corrected and covered by a regression check. Unsupported scientific configuration overrides are rejected rather than silently ignored.
- Known-answer synthetic tests distinguish mean daily versus pooled WAPE, missing-target exclusion, complete-day peaks, earliest peak ties, and prediction averaging before ensemble scoring.
- The seven-page PDF built with MiKTeX. All pages and all three figures were visually inspected. No undefined citations/references or overfull boxes remain; four underfull box warnings do not clip content.
- Independent publication review found no raw hourly observations, prediction series, credentials, location identifiers or personal paths in the selected files. Binary metadata and third-party notices were inspected.

## Execution status

The isolated lightweight environment passes the artifact and numerical controls; tests requiring the training extra are explicitly skipped there. A preliminary three-family synthetic smoke completed with fresh-process model reload checks. Clean-clone verification and a separate fully isolated training smoke are recorded below when complete. These preliminary checks must not be read as full retraining or historical prediction equality.

Full private-data training and independent reconstruction of the historical confidence intervals are intentionally unexecuted during packaging. No real hourly data or historical model checkpoints were used for the smoke tests. Training platform support beyond the actually tested environment is not claimed.

## Source integrity and audit scope

SHA-256 was captured before assembly for 566 scientific source/support files. The end check reports no changes. An additional 99 historical run-result JSON manifests were read only for fit counts and stopping-status checks; their first hashes were captured during review, after that first read, and were rechecked at completion. Those additional hashes are not described as a pre-read baseline. Their private inventory is not distributed. See [source-integrity.json](source-integrity.json).

The author list remains unconfirmed and no general software license has been selected. Citation metadata therefore remains an explicitly incomplete template, and the manuscript author block remains empty. These editorial omissions do not prevent the scientific artifact route from running.

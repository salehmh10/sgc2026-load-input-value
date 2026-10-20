# Reproduction workflows and commands

All repository code is fully reproducible with the required inputs and documented environment, as confirmed by the repository maintainer. Publication artifacts use the included reviewed summaries; full experiments use authorized private observations under the [data contract](data-contract.md). The [validation record](validation.md) separately records the checks executed during repository assembly.

The publication route reads only reviewed files in this clone. It needs Python 3.12, NumPy and Matplotlib; the development extra adds the test/audit tools. It does not access an original project directory, load hourly observations, use a GPU, or train a model.

```sh
python -m venv .venv
# Activate .venv for your platform, as shown in README.md.
python -m pip install ".[dev]"
python scripts/reproduce_artifacts.py
python scripts/reproduce_artifacts.py --check-only
python -m pytest -q
python scripts/audit_release.py
```

These commands generate research figures and table snippets without TeX or the manuscript source. Figure data, chart coordinates, signs, stored confidence endpoints, and rounded table cells are validated independently of PDF/PNG file hashes. Differences in fonts, renderer versions or metadata may change image bytes without changing numerical results. Reference CSV checksums preserve the originally admitted bytes, including their line endings.

The stored bootstrap endpoints originate in the historical executions. Regenerating a chart of these endpoints does not recreate the bootstrap from private hourly forecasts. Full experiments recompute prediction-based metrics and bootstrap intervals. The condition-definition table is a fixed protocol artifact; see [artifact-map.md](artifact-map.md).

For an authorized full experiment install `.[train,dev]`, follow [data-contract.md](data-contract.md), and invoke:

```sh
liv experiment smoke --output outputs/synthetic-smoke
liv experiment run --data data/private/observations.csv --output outputs/paper-run --config configs/paper.json
liv experiment audit-data --data data/private/observations.csv --run outputs/paper-run --output outputs/paper-diagnostics --config configs/paper.json
liv experiment export --run outputs/paper-run --output outputs/paper-aggregates
liv reproduce --data-dir outputs/paper-aggregates --output-dir outputs/paper-artifacts
liv validate --data-dir outputs/paper-aggregates --output-dir outputs/paper-artifacts
```

The smoke command uses synthetic data and shortened training only. The run command performs fresh Train-only fitting and selection followed by observed-target evaluation under the paper protocol. The audit command recomputes descriptive data statistics and uses the completed run's selected thresholds for thermal activation. It can also run before training by omitting `--run`; that mode explicitly leaves fitted thermal diagnostics unavailable. It does not train models or select new thresholds.

Export verifies a completed full experiment and creates the five aggregate CSVs used by the figure/table renderer, together with checksums and execution provenance. It rejects smoke and incomplete runs. Rendering with `--data-dir` uses those new results and validates against that bundle; it never labels them as historical paper scores. New manifests identify a new execution and are never substituted for the historical manifests. Do not compare a synthetic score to a published paper score as evidence of reproduction.

All commands are listed by `liv --help` and `liv experiment --help`. The [coverage map](reproduction-coverage.md) connects each research stage to its implementation. The distributed software does not require the original project directory or the manuscript text.

Keep `results/reference/` immutable. Outputs from artifact regeneration live in `results/reproduced/`; private experiment outputs belong under ignored `outputs/` or an explicit external output directory. The root `.gitignore` is a safeguard, not a substitute for inspecting files before staging.

The manuscript source, bibliography, PDF builder and TeX support files were removed from the current tree at the owner's request. Earlier Git revisions still contain that historical material; this was a normal deletion commit, not a rewrite of public history.

For the executed-check record see [validation.md](validation.md). Dependency versions and historical-versus-tested environment distinctions are recorded in [environments.md](environments.md).

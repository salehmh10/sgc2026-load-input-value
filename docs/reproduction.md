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
python scripts/build_paper.py
```

The last command additionally needs `pdflatex` and `bibtex`. Figure data, chart coordinates, signs, stored confidence endpoints, and rounded table cells are validated independently of PDF/PNG file hashes. Differences in fonts, renderer versions or metadata may change image bytes without changing numerical results. Reference CSV checksums preserve the originally admitted bytes, including their line endings.

The stored bootstrap endpoints originate in the historical executions. Regenerating a chart of these endpoints does not recreate the bootstrap from private hourly forecasts. Likewise, compiling the manuscript verifies document construction, not all scientific claims. The paper includes fixed narrative summaries and a fixed condition-definition table; their status is listed in [artifact-map.md](artifact-map.md).

For an authorized full experiment install `.[train,dev]`, follow [data-contract.md](data-contract.md), and invoke:

```sh
liv experiment smoke --output outputs/synthetic-smoke
liv experiment run --data data/private/observations.csv --output outputs/paper-run --config configs/paper.json
```

The first command uses synthetic data and shortened training only. The second performs fresh Train-only fitting and selection followed by observed-target evaluation under the paper protocol. New manifests identify a new execution and are never substituted for the historical manifests. Do not compare a synthetic score to a published paper score as evidence of reproduction.

Keep `results/reference/` immutable. Outputs from artifact regeneration live in `results/reproduced/`; private experiment outputs belong under ignored `outputs/` or an explicit external output directory. The root `.gitignore` is a safeguard, not a substitute for inspecting files before staging.

For the executed-check record see [validation.md](validation.md). Dependency versions and historical-versus-tested environment distinctions are recorded in [environments.md](environments.md).

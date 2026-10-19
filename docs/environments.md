# Historical and supported environments

The historical manifests describe Python 3.12.3 on Windows 11. I15 used an environment with `include-system-site-packages = true`. Its broad package freeze includes unrelated software and is not an installation specification for this repository. No global Python or Git settings were changed during assembly.

| Package | Historical manifest | Repository direct pin |
|---|---:|---:|
| NumPy | 2.2.6 | 2.2.6 |
| pandas | 2.2.2 | 2.2.3 |
| SciPy | 1.13.0 | 1.15.3 |
| scikit-learn | 1.7.2 | 1.7.2 |
| Matplotlib | 3.8.4 | 3.10.3 |
| holidays | 0.103 | 0.103 |
| Darts | 0.41.0 | 0.41.0 |
| PyTorch | 2.13.0 | 2.13.0 |
| PyTorch Lightning | 2.5.2 | 2.5.2 |
| XGBoost | 3.4.0 | 3.4.0 |
| PyArrow | 25.0.0 | 25.0.0 |

The lightweight base installs only NumPy and Matplotlib plus their dependencies. The `dev` extra supplies pytest, pandas and metadata-audit tools. The `train` extra adds the private-data scientific and model dependencies. Dependencies are installed into fresh virtual environments without system-site packages. Python 3.12 is the supported interpreter series; other versions have not been validated.

Darts is pinned because the local TiDE adapter relies on its private module and dataset interfaces. The adapter removes only the scalar temporal-decoder output LayerNorm, keeps the other five normalization modules, and asserts 477,491 parameters. Replacing it with a stock TiDE model changes the method.

The compatible pandas, SciPy and plotting pins differ from the historical environment. The repository does not claim bitwise equality of a new full training run. Synthetic numerical parity and execution checks are reported separately from paper artifact equality. The installed dependency inventories in `docs/environment-*.txt` identify the clean environments actually exercised; these focused inventories are evidence, not a copy of the original personal environment.

The local build used Windows and MiKTeX. GitHub Actions tests the lightweight artifact route on Windows and Linux; actual completed runs, rather than the workflow's mere presence, determine the compatibility claim. Training smoke coverage is limited to the platform recorded in [validation.md](validation.md).

# Private input contract

No source observations are distributed here. Read [docs/data-contract.md](../docs/data-contract.md) before running the private-data pipeline. Place an authorized input under ignored `data/private/`, or pass an external path with `--data`.

`liv experiment smoke --output outputs/synthetic-smoke` creates its own labeled synthetic example. It does not imitate or establish the paper's numerical findings. Predictions, model weights, hourly prepared frames, and execution caches remain in ignored local output directories.

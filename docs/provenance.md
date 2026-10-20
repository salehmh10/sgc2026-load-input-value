# Source and execution provenance

The scientific code and historical summaries were selected from the current research project and the I14/I15 experiments. Earlier iterations supplied methodological context only; the distributed pipeline does not execute them or read their private artifacts. The current distribution excludes the manuscript text, bibliography and TeX support bundle.

[source-map.json](source-map.json) records each imported or adapted scientific source, its SHA-256, the publication destination, and the destination checksum. Source names are relative to the original research project, without personal absolute paths. The detailed local inventory remains private. [source-integrity.json](source-integrity.json) reports the before/after SHA-256 check of the source files used and supporting scientific files inventoried at assembly.

| Source group | Publication destination | Treatment |
|---|---|---|
| I14 reconstruction, thermal features, model inputs and fixed TiDE | `src/load_input_value/` | Extracted numerical definitions; configurable private input and output, removed bootstrap/cache side effects |
| I15 baselines and evaluation | `src/load_input_value/` | Exact input order, objective, selection, metrics and bootstrap definitions; fresh execution manifests |
| I14/I15 protocols | `configs/paper.json` | Scientific settings selected into an anonymous configuration; historical path/host details omitted |
| Current paper's reviewed summary CSVs | `results/reference/` | Admitted aggregate values preserved; original reference checksums retained |
| Condition-definition table | `results/reference/expected-tables/conditions.tex` | Fixed protocol table retained independently of manuscript source |
| Reviewed primary-value figure | `docs/assets/i15_primary_value.png` | README illustration retained; all research figures can be regenerated |
| Current plot script | `scripts/reproduce_artifacts.py` | Portable inputs, separate generated outputs and explicit numerical checks |

The current source map lists surviving destinations only. Original source hashes identify the versions used when each item was first extracted; publication hashes identify the current distributed files. The manuscript and its support files remain in earlier Git revisions; removing them from the current tree does not erase public history. The original source-integrity counts and historical PDF checks remain dated assembly evidence, not a claim that the manuscript bundle is still distributed.

`results/reference/SHA256SUMS.json` is the byte-integrity authority for fixed reference assets. A mismatch is an error; do not regenerate the manifest to conceal changed reference data. The manuscript's rounded tables and plot points are additionally validated numerically, independently of renderer-dependent image bytes.

New experiment outputs carry their own execution configuration, package versions, input digest, model/checkpoint evidence, fitted preprocessing, selections, prediction files and metrics. They are stored locally under the output path. Historical manifests have not been rewritten to identify the new code as the old execution. Reference performance belongs to the historical run, while the portable code has separate validation evidence.

Raw hourly inputs, hourly prepared frames, observation-bearing predictions, learned weights, caches, private review reports and conversation history are excluded. To replace these privately, use the full experiment command and the [data contract](data-contract.md). Public chart reproduction uses reviewed summary CSVs instead of these files. Private audit reports were used only for local reconciliation where necessary and were not copied into the publication surface.

The repository maintainer confirms that all code is fully reproducible with the required inputs and documented environment. Exact preservation of core functions and synthetic parity provide implementation evidence. The assembly execution record is maintained in [validation.md](validation.md), with dependency details in [environments.md](environments.md).

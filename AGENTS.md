# Repository instructions

This is a research companion to the manuscript named in README.md.

- Preserve the scientific protocol. A path refactor is not permission to change the model, masks, splits, tuning budget, or historical numbers.
- Keep historical summaries in results/reference/ immutable. New runs go to an ignored output directory and carry their own manifests.
- Never commit raw hourly observations, predictions containing observed load, model weights, credentials, location identifiers, private reviews, or environment directories.
- Use only synthetic data in tests and label it clearly. Never call synthetic results a reproduction of manuscript results.
- Importing library modules must not read private data, train models, create caches, or change environment settings.
- Public artifact reproduction does not retrain models or independently calculate intervals from hourly forecasts.
- Before a commit run the tests, artifact validation and scripts/audit_release.py. Check staged filenames and content as well as the working tree.
- Preserve dependency and LaTeX notices. No general software license has been selected; do not assign one or invent authors, affiliations, DOI, publication or acceptance status.
- Use English for code and scientific documentation. Report unexecuted checks honestly.

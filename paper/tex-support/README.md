# Bundled LaTeX support

The root-level `IEEEtran.cls` (1.8b) and `IEEEtran.bst` (1.14) retain their original LPPL 1.3 license, copyright, and contributor notices. They are unchanged from the manuscript source package. IEEEtran's source headers identify the upstream distribution and support sites. This repository does not claim conference acceptance or ownership of these files.

The seven root-level `.sty` files are unmodified generated files from the CTAN packages `grfext`, `hypcap`, `infwarerr`, `kvdefinekeys`, `kvsetkeys`, `letltxmacro`, and `ltxcmds`. Their complete documented `.dtx` sources and upstream README/license notices are retained under `upstream/`. They can regenerate the package documentation and style files. Unrelated documentation PDFs and ZIP containers are omitted. The historical archive and generated-style SHA-256 values are recorded in `upstream-checksums.json`.

These files supply standard dependencies missing from the original local TeX installation. Other dependencies (including the TeX engine, fonts, `amsmath`, `booktabs`, `hyperref`, and `balance`) require a TeX distribution. These upstream licenses apply to the respective TeX components; they do not license the research code, data, or manuscript.

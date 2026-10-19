"""Build the manuscript in a separate output directory using a local TeX installation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced/paper")
    parser.add_argument("--skip-artifacts", action="store_true", help="Use already generated paper figures and tables.")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    source = ROOT / "paper"
    if output == source or source in output.parents:
        parser.error("Build outputs must be separate from paper sources.")
    commands = {name: shutil.which(name) for name in ["pdflatex", "bibtex"]}
    missing = [name for name, path in commands.items() if path is None]
    if missing:
        parser.exit(2, "Missing optional TeX tools: " + ", ".join(missing) + ". Install a complete TeX Live or MiKTeX distribution.\n")
    if not args.skip_artifacts:
        subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_artifacts.py")], check=True)
    output.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = output / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if not args.skip_artifacts:
        # Keep the reviewed publication snapshots untouched. Only the isolated
        # build receives newly generated figures and tables.
        for folder in ["figures", "tables"]:
            for path in (ROOT / "results/reproduced" / folder).iterdir():
                if path.is_file():
                    shutil.copyfile(path, output / folder / path.name)
    environment = os.environ.copy()
    # Fixed epoch keeps supported engines from inserting the current build time.
    environment["SOURCE_DATE_EPOCH"] = "0"
    environment["FORCE_SOURCE_DATE"] = "1"
    latex = [commands["pdflatex"], "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"]
    passes = [latex, [commands["bibtex"], "main"], latex, latex]
    for number, command in enumerate(passes, 1):
        result = subprocess.run(command, cwd=output, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        (output / f"build-pass-{number}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            print(result.stdout[-6000:], file=sys.stderr)
            raise SystemExit(f"TeX build failed on pass {number}; inspect {output / f'build-pass-{number}.log'}")
    log = (output / "main.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:Citation|Reference) .+ undefined|There were undefined references|Label\(s\) may have changed", log):
        raise SystemExit("Unresolved manuscript citations/references; inspect the final TeX log.")
    boxes = re.findall(r"(?:Over|Under)full \\[hv]box[^\n]*", log)
    print(f"Built {output / 'main.pdf'}")
    print(f"Final pass: no unresolved citations or references; {len(boxes)} TeX box warning(s). Inspect the PDF before release.")


if __name__ == "__main__":
    main()

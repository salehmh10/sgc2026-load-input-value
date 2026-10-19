"""Check the reviewed publication surface, optionally the exact staged blobs.

This check is a guardrail and does not replace human review of numeric granularity.
Extra confidential terms may be supplied from an ignored local JSON list.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ("src", "configs", "scripts", "tests", "docs", "results/reference", "paper", "data", ".github")
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".work", ".git"}
DENIED_SUFFIXES = {".ckpt", ".pt", ".pth", ".ubj", ".pkl", ".joblib", ".parquet", ".arrow", ".feather", ".npz", ".npy", ".ipynb", ".env", ".key", ".pem"}
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|SHARIF)[\\/]|/(?:home|Users)/[A-Za-z0-9_.-]+/)", re.I)
SECRET = re.compile(r"(?:gh[pousr]_" + r"[A-Za-z0-9_]{20,}|github_pat_" + r"[A-Za-z0-9_]{20,}|-----BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY-----)")
PRIVATE_COLUMNS = {"timestamp", "datetime", "y_true", "y_pred", "actual", "prediction", "load", "load_raw", "temp_hour", "humidity"}


def files(staged: bool):
    if staged:
        raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=ROOT)
        for name in raw.decode().split("\0"):
            if name:
                yield Path(name), subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        return
    candidates = [p for p in ROOT.iterdir() if p.is_file()]
    for folder in PUBLIC_DIRS:
        path = ROOT / folder
        if path.exists():
            candidates.extend(p for p in path.rglob("*") if p.is_file())
    for path in sorted(set(candidates)):
        relative = path.relative_to(ROOT)
        if any(p in IGNORED_PARTS or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.suffix in {".pyc", ".aux", ".log", ".out", ".blg", ".bbl", ".fls"} or relative.as_posix() == "paper/main.pdf":
            continue
        yield relative, path.read_bytes()


def inspect(path: Path, content: bytes, confidential_terms: list[str]):
    issues = []
    name = path.as_posix()
    if any(p in {".work", ".venv", "private", "predictions", "models", "cache"} for p in path.parts):
        issues.append("private/output directory")
    if path.suffix.lower() in DENIED_SUFFIXES or path.name.startswith(".env"):
        issues.append("unapproved binary, notebook, model or secret-file type")
    if len(content) > 5_000_000:
        issues.append("file exceeds reviewed publication size limit")
    searchable = name + "\n" + content.decode("utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        searchable += str(reader.metadata) + "\n" + "\n".join(p.extract_text() or "" for p in reader.pages)
        if reader.metadata and reader.metadata.get("/Author"):
            issues.append("PDF contains an author metadata field requiring review")
    if path.suffix.lower() == ".png":
        from PIL import Image
        with Image.open(io.BytesIO(content)) as image:
            searchable += str(image.info)
            if image.getexif():
                issues.append("image contains unreviewed EXIF metadata")
    if PERSONAL_PATH.search(searchable):
        issues.append("personal absolute path")
    if SECRET.search(searchable):
        issues.append("credential-shaped content")
    if any(term.casefold() in searchable.casefold() for term in confidential_terms if term):
        issues.append("confidential location/identity term")
    if path.suffix.lower() == ".csv":
        rows = csv.reader(io.StringIO(content.decode("utf-8-sig")))
        header = next(rows, [])
        if PRIVATE_COLUMNS.intersection(c.strip().lower() for c in header):
            issues.append("hourly observations/predictions require separate approval")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="inspect exact staged content")
    parser.add_argument("--private-terms", type=Path, help="ignored local JSON list, never commit it")
    args = parser.parse_args()
    terms = json.loads(args.private_terms.read_text(encoding="utf-8-sig")) if args.private_terms else []
    failures, count = [], 0
    for path, content in files(args.staged):
        count += 1
        for issue in inspect(path, content, terms):
            failures.append({"file": path.as_posix(), "reason": issue})
    print(json.dumps({"files_checked": count, "issues": failures, "scope": "staged" if args.staged else "publication files"}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

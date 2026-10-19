"""Public summary-only checks; no private inputs or model dependencies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts():
    spec = importlib.util.spec_from_file_location("artifact_renderer", ROOT / "scripts/reproduce_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_reproduction_and_checks(artifacts, tmp_path):
    artifacts.main(["--output-dir", str(tmp_path)])
    report = json.loads((tmp_path / "validation/artifact-validation.json").read_text())
    assert report["status"] == "PASS"
    assert report["numeric_fidelity"]["coordinates_and_ci_bounds"] == 84
    assert report["numeric_fidelity"]["rounded_table_cells"] == 66
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 3
    artifacts.main(["--output-dir", str(tmp_path), "--check-only"])


def test_tampered_reference_rejected(artifacts, tmp_path):
    shutil.copytree(artifacts.DATA, tmp_path / "reference")
    artifacts.DATA = tmp_path / "reference"
    path = artifacts.DATA / "i15_contrasts.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.verify_reference()


def test_flipped_plot_contrast_rejected(artifacts, tmp_path):
    artifacts.OUT = tmp_path / "figures"
    artifacts.TABLES = tmp_path / "tables"
    artifacts.VALIDATION = tmp_path / "validation"
    artifacts.TABLES.mkdir()
    artifacts.VALIDATION.mkdir()
    for path in (artifacts.DATA / "expected-tables").glob("*.tex"):
        shutil.copyfile(path, artifacts.TABLES / path.name)
    # Generate records using the plotting functions, then corrupt a scientific value.
    artifacts.OUT.mkdir()
    artifacts.primary_value(artifacts.read("i15_contrasts.csv"), artifacts.read("i15_primary_seed_points.csv"))
    artifacts.mean_peak(artifacts.read("i15_contrasts.csv"))
    artifacts.DRAWN[0]["x"] *= -1
    (artifacts.VALIDATION / "plotted_points.json").write_text(json.dumps(artifacts.DRAWN))
    with pytest.raises(ValueError, match="contrast mismatch"):
        artifacts.verify_rendered()

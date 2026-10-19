"""The public CLI must work without private inputs or the model stack."""
import os
from pathlib import Path
import subprocess
import sys


def test_public_import_and_help_have_no_workspace_side_effects(tmp_path):
    script = "import sys; import load_input_value; import load_input_value.cli; assert 'torch' not in sys.modules; assert 'darts' not in sys.modules"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True)
    result = subprocess.run([sys.executable, "-m", "load_input_value.cli", "--help"], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert "experiment" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_missing_data_is_an_actionable_error_before_training_import(tmp_path):
    output = tmp_path / "must-not-be-created"
    result = subprocess.run([sys.executable, "-m", "load_input_value.cli", "experiment", "run",
                             "--data", str(tmp_path / "absent.csv"), "--output", str(output)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Private input CSV not found" in result.stderr
    assert "--data" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()

"""Synthetic fixtures verify the full results-to-figures bridge, not paper scores."""
import csv
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_input_value.artifacts import export_publication_data
from load_input_value.evaluation import add_or_check_ensembles, metrics, tables_from_predictions
from load_input_value.protocol import ARMS

ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _renderer():
    spec = importlib.util.spec_from_file_location("new_execution_renderer", ROOT / "scripts/reproduce_artifacts.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    return renderer


@pytest.fixture
def synthetic_run(tmp_path):
    # This mimics the *file contract* of a full completed execution, using only
    # analytic synthetic predictions. It does not claim any model was trained.
    run = tmp_path / "SYNTHETIC_CONTRACT_FIXTURE"
    run.mkdir()
    (run / "metrics").mkdir()
    config = {"conditions": ARMS, "seeds": [42, 314, 2718], "folds": {
        "warm_late_summer": {"train": ["2025-01-01", "2025-01-04"],
                             "validation": ["2025-01-04", "2025-01-07"],
                             "evaluation": ["2025-01-07", "2025-01-10"]},
        "cold": {"train": ["2025-01-01", "2025-01-10"],
                 "validation": ["2025-01-10", "2025-01-13"],
                 "evaluation": ["2025-01-13", "2025-01-16"]}}}
    _json(run / "execution.json", {"synthetic_smoke_only": False, "configuration": config})
    _json(run / "completed.json", {"completed": True, "full_retraining": True, "synthetic_smoke_only": False})
    predictions, lags = [], []
    for fold, bounds in config["folds"].items():
        index = pd.date_range(bounds["train"][0], bounds["evaluation"][1], freq="h", inclusive="left")
        meta = pd.DataFrame({"LOAD": 100 + np.sin(np.arange(len(index)) / 3),
                             "temperature_raw": 10 + np.sin(np.arange(len(index)) / 24),
                             "temperature_observed": True}, index=index)
        for start, _ in bounds.values():
            meta.loc[pd.Timestamp(start), "LOAD"] = np.nan
            # An extreme imputed value must not enter observed-weather summaries.
            meta.loc[pd.Timestamp(start), "temperature_raw"] = 99999.
            meta.loc[pd.Timestamp(start), "temperature_observed"] = False
        directory = run / "preprocessing" / fold
        directory.mkdir(parents=True)
        meta.to_parquet(directory / "metadata.parquet")
        _json(directory / "audit.json", {"identity": {"bounds": bounds}})
        for part in ("validation", "evaluation"):
            start, end = bounds[part]
            selected = meta.loc[(meta.index >= start) & (meta.index < end)]
            base = pd.DataFrame({"timestamp": selected.index, "actual": selected.LOAD.to_numpy(),
                                 "observed": selected.LOAD.notna().to_numpy(),
                                 "prediction": selected.LOAD.fillna(100).to_numpy() + 1.23456789})
            for lag in (1, 24, 168):
                lags.append({"fold": fold, "partition": part.capitalize(), "lag_hours": lag, **metrics(base)})
            for family in ("Ridge-ARX", "XGBoost", "TiDE"):
                for arm_number, arm in enumerate(ARMS):
                    for seed, error in ([("deterministic", 2)] if family == "Ridge-ARX"
                                        else [("42", 2), ("314", -2), ("2718", 3)]):
                        table = base.copy()
                        day = np.arange(len(table)) // 24 + 1
                        table.prediction = selected.LOAD.fillna(100).to_numpy() + error + arm_number * day / 8
                        table["family"], table["fold"], table["arm"] = family, fold, arm
                        table["seed"], table["partition"] = seed, part.capitalize()
                        predictions.append(table)
    all_predictions = add_or_check_ensembles(pd.concat(predictions, ignore_index=True))
    computed = tables_from_predictions(all_predictions)
    for name in ("overall_metrics", "paired_comparisons"):
        computed[name].to_csv(run / "metrics" / f"{name}.csv", index=False)
    pd.DataFrame(lags).to_csv(run / "metrics/lag_references.csv", index=False)
    return run


def test_export_preserves_seed_selection_sign_intervals_and_raw_weather(synthetic_run, tmp_path):
    output = tmp_path / "aggregates"
    report = export_publication_data(synthetic_run, output)
    assert report["origin"] == "new_full_experiment" and report["historical_reference"] is False
    assert {name: info["rows"] for name, info in report["exports"].items()} == {
        "i15_accuracy.csv": 84, "i15_contrasts.csv": 72, "i15_primary_seed_points.csv": 24,
        "i15_simple_references.csv": 3, "study_windows.csv": 6}
    original = _rows(synthetic_run / "metrics/paired_comparisons.csv")
    key = lambda row: tuple(row[k] for k in ("fold", "family", "partition", "arm", "seed"))
    by_key = {key(row): row for row in original}
    for name in ("i15_contrasts.csv", "i15_primary_seed_points.csv"):
        for row in _rows(output / name):
            # Exact CSV strings retain all interval endpoints and delta precision.
            assert row == by_key[key(row)]
            assert float(row["delta_percentage_points"]) == pytest.approx(float(row["other_WAPE"]) - float(row["FULL_WAPE"]))
    seeds = _rows(output / "i15_primary_seed_points.csv")
    assert {row["family"] for row in seeds} == {"TiDE", "XGBoost"}
    assert {row["seed"] for row in seeds} == {"42", "314", "2718"}
    assert {row["arm"] for row in seeds} == {"NO_TEMP", "NO_WEATHER"}
    accuracy = _rows(output / "i15_accuracy.csv")
    assert all(row["seed"] == ("deterministic" if row["family"] == "Ridge-ARX" else "mean_42_314_2718")
               for row in accuracy)
    full_tide = next(row for row in accuracy if row["family"] == "TiDE" and row["arm"] == "FULL")
    raw_scores = _rows(synthetic_run / "metrics/overall_metrics.csv")
    seed_scores = [float(row["MeanDailyWAPE"]) for row in raw_scores
                   if (row["fold"], row["partition"], row["family"], row["arm"])
                   == (full_tide["fold"], full_tide["partition"], "TiDE", "FULL")
                   and row["seed"] in {"42", "314", "2718"}]
    assert float(full_tide["MeanDailyWAPE"]) != pytest.approx(np.mean(seed_scores))
    windows = _rows(output / "study_windows.csv")
    assert all(float(row["temperature_max_C"]) < 12 for row in windows)
    assert all(int(row["observed_temperature_count"]) == int(row["grid_hours"]) - 1 for row in windows)
    assert not any(p.suffix == ".parquet" for p in output.iterdir())
    with pytest.raises(FileExistsError, match="new or empty"):
        export_publication_data(synthetic_run, output)


@pytest.mark.parametrize("corruption", ["missing_seed", "flipped_sign", "invalid_interval", "smoke", "window_support"])
def test_invalid_execution_is_rejected_without_output(synthetic_run, tmp_path, corruption):
    if corruption == "smoke":
        path = synthetic_run / "completed.json"
        marker = json.loads(path.read_text())
        marker["synthetic_smoke_only"] = True
        _json(path, marker)
    elif corruption == "window_support":
        path = synthetic_run / "preprocessing/cold/metadata.parquet"
        metadata = pd.read_parquet(path)
        metadata = metadata.iloc[1:]
        metadata.to_parquet(path)
    else:
        path = synthetic_run / "metrics/paired_comparisons.csv"
        rows = _rows(path)
        if corruption == "missing_seed":
            rows.pop()
        elif corruption == "flipped_sign":
            rows[0]["delta_percentage_points"] = str(-float(rows[0]["delta_percentage_points"]))
        else:
            rows[0]["block3_CI_low"], rows[0]["block3_CI_high"] = "2", "-2"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError):
        export_publication_data(synthetic_run, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_unexpected_private_columns_are_never_exported(synthetic_run, tmp_path):
    for name in ("overall_metrics", "paired_comparisons"):
        path = synthetic_run / "metrics" / f"{name}.csv"
        rows = _rows(path)
        for row in rows:
            row.update(private_location="PRIVATE_SENTINEL_LOCATION",
                       raw_file_path="PRIVATE_SENTINEL_PATH",
                       hourly_observations="PRIVATE_SENTINEL_HOURLY_VALUES")
        _csv(path, rows)
    output = tmp_path / "safe_aggregates"
    export_publication_data(synthetic_run, output)
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "PRIVATE_SENTINEL" not in text
        assert "private_location" not in text
        assert "raw_file_path" not in text
        assert "hourly_observations" not in text


def test_new_execution_exports_render_without_historical_score_substitution(synthetic_run, tmp_path):
    aggregates, rendered = tmp_path / "aggregates", tmp_path / "rendered"
    export_publication_data(synthetic_run, aggregates)
    renderer = _renderer()
    renderer.main(["--data-dir", str(aggregates), "--output-dir", str(rendered)])
    report = json.loads((rendered / "validation/artifact-validation.json").read_text())
    assert report["origin"] == "new_full_experiment"
    assert report["numeric_fidelity"]["historical_tables"].startswith("not compared")
    assert report["numeric_fidelity"]["rounded_table_cells"] == 66
    assert "71 observed targets" in (rendered / "tables/i15_accuracy.tex").read_text()
    assert len(list((rendered / "figures").glob("*.png"))) == 3
    renderer.main(["--data-dir", str(aggregates), "--output-dir", str(rendered), "--check-only"])
    with pytest.raises(FileExistsError, match="new or empty"):
        renderer.main(["--data-dir", str(aggregates), "--output-dir", str(rendered)])
    with pytest.raises(SystemExit):
        renderer.main(["--data-dir", str(aggregates)])
    for target in (renderer.REFERENCE, renderer.REFERENCE / "nested", renderer.REFERENCE.parent):
        with pytest.raises(ValueError, match="must not overlap"):
            renderer.main(["--output-dir", str(target)])
    for target in (aggregates, aggregates / "nested", aggregates.parent):
        with pytest.raises(ValueError, match="must not overlap"):
            renderer.main(["--data-dir", str(aggregates), "--output-dir", str(target)])


def test_decimal_halfway_values_render_and_validate_identically(synthetic_run, tmp_path):
    aggregates, rendered = tmp_path / "halfway", tmp_path / "halfway_rendered"
    export_publication_data(synthetic_run, aggregates)
    path = aggregates / "i15_accuracy.csv"
    rows = _rows(path)
    for row in rows:
        row["MeanDailyWAPE"] = {"FULL": "2.675", "NO_TEMP": "2.670"}.get(row["arm"], "2.680")
    _csv(path, rows)
    path = aggregates / "i15_contrasts.csv"
    rows = _rows(path)
    for row in rows:
        negative = row["arm"] == "NO_TEMP"
        row.update(FULL_WAPE="2.675", other_WAPE="2.670" if negative else "2.680",
                   delta_percentage_points="-0.005" if negative else "0.005",
                   iid_CI_low="-0.02", iid_CI_high="0.02",
                   block3_CI_low="-0.02", block3_CI_high="0.02",
                   days_FULL_better="0" if negative else row["days"],
                   days_FULL_worse=row["days"] if negative else "0", days_tied="0")
    _csv(path, rows)
    path = aggregates / "i15_simple_references.csv"
    rows = _rows(path)
    for row in rows:
        row.update(warm_mean_daily_WAPE_percent="2.675", cold_mean_daily_WAPE_percent="0.005")
    _csv(path, rows)
    hashes = json.loads((aggregates / "SHA256SUMS.json").read_text())
    manifest = json.loads((aggregates / "publication-export.json").read_text())
    for name in hashes:
        hashes[name] = hashlib.sha256((aggregates / name).read_bytes()).hexdigest()
        manifest["exports"][name]["sha256"] = hashes[name]
    _json(aggregates / "SHA256SUMS.json", hashes)
    _json(aggregates / "publication-export.json", manifest)
    renderer = _renderer()
    renderer.main(["--data-dir", str(aggregates), "--output-dir", str(rendered)])
    accuracy = (rendered / "tables/i15_accuracy.tex").read_text()
    contrast = (rendered / "tables/i15_validation.tex").read_text()
    assert r"\full & 2.68 & 2.68 & 2.68" in accuracy
    assert r"Previous hour & \multicolumn{3}{c}{2.68} && \multicolumn{3}{c}{0.00}" in accuracy
    assert "$+0.00$" in contrast and "$-0.00$" in contrast
    renderer.main(["--data-dir", str(aggregates), "--output-dir", str(rendered), "--check-only"])

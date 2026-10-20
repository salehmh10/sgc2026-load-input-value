"""Known-answer descriptive checks use invented observations only."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from load_input_value.data import gregorian_to_jalali, load_fold
from load_input_value.diagnostics import (
    _coverage_tables, _mask_summary, _proxy_summary, _thermal_summary, _year_over_year, audit_data,
)


def synthetic_input(tmp_path):
    index = pd.date_range("2025-01-01", periods=168 + 48 + 24 + 24, freq="h")
    frame = pd.DataFrame({
        "TIMESTAMP": index, "LOAD": 100.0, "TEMP_HOUR": 20.0, "HUMIDITY": 50.0,
        "HOLIDAY": (index.dayofweek == 4).astype(int),
        "SHAMSI": [y * 10000 + m * 100 + d for y, m, d in [
            gregorian_to_jalali(t.year, t.month, t.day) for t in index]],
    })
    # Entire initial context is absent for temperature, but humidity is observed.
    frame.loc[:167, "TEMP_HOUR"] = np.nan
    frame.loc[216, "LOAD"] = np.nan
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    config = {
        "schema_version": 1, "strict_observed_counts": False,
        "folds": {"warm_late_summer": {
            "train": ["2025-01-08", "2025-01-10"],
            "validation": ["2025-01-10", "2025-01-11"],
            "evaluation": ["2025-01-11", "2025-01-12"],
        }},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return path, config_path, config


def test_raw_coverage_masks_and_daily_change():
    index = pd.date_range("2025-09-22", periods=48, freq="h")
    frame = pd.DataFrame({
        "LOAD": np.r_[np.full(24, 200.), np.full(24, 100.)],
        "temperature_raw": np.r_[np.full(24, 20.), np.full(24, 18.)],
        "humidity_raw": 50.,
    }, index=index)
    frame.loc[index[0], ["LOAD", "temperature_raw"]] = np.nan
    tables = _coverage_tables(frame)
    month = tables["monthly_coverage"].iloc[0]
    day = tables["daily_observed_summary"].iloc[1]
    assert month.observed_load_hours == 47
    assert month.complete_load_days == 1
    assert not month.full_calendar_month_in_scope
    assert day.observed_load_mean_change_kw == -100
    assert day.observed_load_mean_change_percent == -50
    assert day.observed_temperature_mean_change == -2
    mask = _mask_summary(frame)
    assert mask["load_temperature_mask_mismatches"] == 0
    assert mask["load_humidity_mask_mismatches"] == 1
    assert mask["all_three_masks_equal"] is False


def test_year_over_year_compares_same_calendar_window():
    old = pd.date_range("2024-09-01", periods=48, freq="h")
    new = pd.date_range("2025-09-01", periods=48, freq="h")
    frame = pd.DataFrame({"LOAD": np.r_[np.full(48, 200.), np.full(48, 300.)]}, index=old.append(new))
    table = _year_over_year(frame, {"warm": {"evaluation": ["2025-09-01", "2025-09-03"]}})
    row = table.iloc[0]
    assert row.observed_mean_load_change_percent == 50
    assert row.previous_window_fully_in_scope
    assert row.current_window_fully_in_scope
    assert row.previous_observed_load_hours == 48


def test_proxy_fallback_is_variable_specific_and_ages_exclude_fallback(tmp_path):
    path, _, config = synthetic_input(tmp_path)
    data = load_fold("warm_late_summer", path, strict_counts=False, bounds=config["folds"]["warm_late_summer"])
    result = _proxy_summary(data).set_index(["partition", "variable"])
    temp = result.loc[("train", "temperature")]
    humidity = result.loc[("train", "humidity")]
    assert temp.fallback_count == 1
    assert temp.donor_count == 47
    assert temp.donor_age_min_hours == temp.donor_age_max_hours == 1
    assert humidity.fallback_count == 0
    assert humidity.donor_count == 48
    assert humidity.donor_age_median_hours == 1


def test_physical_thermal_counts_use_saved_thresholds_before_scaling(tmp_path):
    path, _, config = synthetic_input(tmp_path)
    data = load_fold("warm_late_summer", path, strict_counts=False, bounds=config["folds"]["warm_late_summer"])
    result = _thermal_summary(data, {"heat_threshold": 10., "cool_threshold": 18.})
    full = result.loc[(result.partition == "evaluation") & (result.condition == "FULL")].set_index("feature")
    assert full.loc["heating_excess", "origins_with_positive_target"] == 0
    assert full.loc["cooling_excess", "origins_with_positive_target"] == 24
    assert full.loc["cooling_degree_hours24", "origins_with_positive_history"] == 24


def test_audit_manifest_no_fitting_no_overwrite_and_matching_run(tmp_path):
    path, config_path, config = synthetic_input(tmp_path)
    output = tmp_path / "audit"
    result = audit_data(path, output, config_path)
    assert result["completed"] and result["model_fits"] == result["threshold_selection_fits"] == 0
    assert result["outputs"]["proxy_availability_summary.csv"]["rows"] == 6
    assert "thermal_activation_summary.csv" not in result["outputs"]
    assert not list(output.glob("*.parquet"))
    with pytest.raises(FileExistsError):
        audit_data(path, output, config_path)
    with pytest.raises(NotADirectoryError, match="supplied path is a file"):
        audit_data(path, path, config_path)
    run = tmp_path / "completed-run"
    feature = run / "features" / "warm_late_summer"
    feature.mkdir(parents=True)
    (run / "completed.json").write_text(
        '{"completed": true, "full_retraining": true, "synthetic_smoke_only": false}', encoding="utf-8")
    params = {
        "heat_threshold": 10., "cool_threshold": 18.,
        "identity": {"data_fingerprint": result["fold_data_fingerprints"]["warm_late_summer"]},
    }
    (feature / "parameters.json").write_text(json.dumps(params), encoding="utf-8")
    checked = audit_data(path, tmp_path / "audit-with-thermal", config_path, run)
    assert checked["outputs"]["thermal_activation_summary.csv"]["rows"] == 36
    params["identity"]["data_fingerprint"] = "wrong input"
    (feature / "parameters.json").write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        audit_data(path, tmp_path / "audit-mismatch", config_path, run)


def test_full_run_thermal_audit_rejects_smoke_and_synthetic_thresholds(tmp_path):
    path, config_path, config = synthetic_input(tmp_path)
    run = tmp_path / "run-markers"
    feature = run / "features" / "warm_late_summer"
    feature.mkdir(parents=True)
    marker = run / "completed.json"
    marker.write_text(
        '{"completed": true, "full_retraining": false, "synthetic_smoke_only": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic smoke"):
        audit_data(path, tmp_path / "reject-smoke", config_path, run)
    marker.write_text('{"completed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unverified run markers"):
        audit_data(path, tmp_path / "reject-incomplete-markers", config_path, run)
    marker.write_text(
        '{"completed": true, "full_retraining": true, "synthetic_smoke_only": false}', encoding="utf-8")
    data = load_fold("warm_late_summer", path, strict_counts=False, bounds=config["folds"]["warm_late_summer"])
    params = {
        "heat_threshold": 10., "cool_threshold": 18., "synthetic_override": True,
        "identity": {"data_fingerprint": data["fingerprint"]},
    }
    (feature / "parameters.json").write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic threshold overrides"):
        audit_data(path, tmp_path / "reject-override", config_path, run)

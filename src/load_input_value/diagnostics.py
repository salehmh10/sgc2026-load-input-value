"""Private-input descriptive audits; no model fitting or physical attribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_fold, validate_input
from .features import F2, raw_features, window_temperatures
from .inputs import proxy_lineage
from .io import sha256, utc, write_json


def _observation_summary(frame):
    """Use raw observed values, never reconstructed load or filled weather."""
    observed = frame.LOAD.notna()
    daily = observed.groupby(frame.index.normalize()).sum()
    result = {
        "grid_hours": len(frame),
        "observed_load_hours": int(observed.sum()),
        "missing_load_hours": int((~observed).sum()),
        "complete_load_days": int(daily.eq(24).sum()),
        "days": len(daily),
        "observed_load_mean_kw": frame.LOAD.mean(),
    }
    for variable in ("temperature", "humidity"):
        values = frame[variable + "_raw"]
        recorded = values.dropna()
        result.update({
            f"observed_{variable}_hours": len(recorded),
            f"observed_{variable}_mean": recorded.mean() if len(recorded) else np.nan,
            f"observed_{variable}_median": recorded.median() if len(recorded) else np.nan,
            f"observed_{variable}_min": recorded.min() if len(recorded) else np.nan,
            f"observed_{variable}_max": recorded.max() if len(recorded) else np.nan,
        })
    return result


def _scope(frame):
    return {
        "scope_start_inclusive": str(frame.index.min()) if len(frame) else None,
        "scope_end_exclusive": str(frame.index.max() + pd.Timedelta(hours=1)) if len(frame) else None,
    }


def _mask_summary(frame):
    load = frame.LOAD.notna()
    temperature = frame.temperature_raw.notna()
    humidity = frame.humidity_raw.notna()
    return {
        "grid_hours": len(frame),
        "load_temperature_mask_mismatches": int((load != temperature).sum()),
        "load_humidity_mask_mismatches": int((load != humidity).sum()),
        "temperature_humidity_mask_mismatches": int((temperature != humidity).sum()),
        "all_three_masks_equal": bool(load.equals(temperature) and load.equals(humidity)),
        "jointly_observed_hours": int((load & temperature & humidity).sum()),
    }


def _coverage_tables(frame):
    monthly = []
    for period, group in frame.groupby(frame.index.to_period("M")):
        monthly.append({
            "year": period.year, "month": period.month, "year_month": str(period),
            **_scope(group), **_observation_summary(group),
            "calendar_month_hours": period.days_in_month * 24,
            "full_calendar_month_in_scope": len(group) == period.days_in_month * 24,
            **_mask_summary(group),
        })
    daily = []
    for date, group in frame.groupby(frame.index.normalize()):
        daily.append({"date": date, **_observation_summary(group)})
    daily = pd.DataFrame(daily)
    adjacent = daily.date.diff().eq(pd.Timedelta(days=1))
    daily["observed_load_mean_change_kw"] = daily.observed_load_mean_kw.diff().where(adjacent)
    previous = daily.observed_load_mean_kw.shift(1)
    daily["observed_load_mean_change_percent"] = (
        100 * daily.observed_load_mean_change_kw / previous
    ).where(previous.ne(0))
    daily["observed_temperature_mean_change"] = daily.observed_temperature_mean.diff().where(adjacent)
    daily["date"] = daily.date.dt.strftime("%Y-%m-%d")
    yearly = pd.DataFrame([
        {"year": int(year), **_scope(group), **_observation_summary(group)}
        for year, group in frame.groupby(frame.index.year)
    ])
    return {
        "monthly_coverage": pd.DataFrame(monthly),
        "annual_coverage": yearly,
        "daily_observed_summary": daily,
    }


def _year_over_year(frame, folds):
    """Same calendar evaluation interval in consecutive years; no causal claim."""
    rows = []
    for fold, bounds in folds.items():
        start, end = map(pd.Timestamp, bounds["evaluation"])
        previous_start, previous_end = start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
        current = frame.loc[(frame.index >= start) & (frame.index < end)]
        previous = frame.loc[(frame.index >= previous_start) & (frame.index < previous_end)]
        current_mean, previous_mean = current.LOAD.mean(), previous.LOAD.mean()
        change = 100 * (current_mean / previous_mean - 1) if pd.notna(previous_mean) and previous_mean != 0 else np.nan
        rows.append({
            "fold": fold, "current_start_inclusive": str(start), "current_end_exclusive": str(end),
            "previous_start_inclusive": str(previous_start), "previous_end_exclusive": str(previous_end),
            "current_grid_hours": len(current), "previous_grid_hours": len(previous),
            "current_window_fully_in_scope": len(current) == int((end - start) / pd.Timedelta(hours=1)),
            "previous_window_fully_in_scope": len(previous) == int((previous_end - previous_start) / pd.Timedelta(hours=1)),
            "current_observed_load_hours": int(current.LOAD.notna().sum()),
            "previous_observed_load_hours": int(previous.LOAD.notna().sum()),
            "current_observed_load_mean_kw": current_mean,
            "previous_observed_load_mean_kw": previous_mean,
            "observed_mean_load_change_percent": change,
        })
    return pd.DataFrame(rows)


def _proxy_summary(data):
    rows = []
    for partition, part in data["partitions"].items():
        lineage = proxy_lineage(data, part["positions"])
        for variable in ("temperature", "humidity"):
            fallback = lineage[variable + "_fallback"].to_numpy(bool)
            ages = lineage.loc[~fallback, variable + "_age_hours"]
            observed = part["observed"]
            rows.append({
                "fold": data["fold"], "partition": partition, "variable": variable,
                "grid_origins": len(lineage),
                "donor_count": int((~fallback).sum()), "fallback_count": int(fallback.sum()),
                "fallback_fraction": float(fallback.mean()),
                "donor_age_min_hours": ages.min() if len(ages) else np.nan,
                "donor_age_median_hours": ages.median() if len(ages) else np.nan,
                "donor_age_p90_hours": ages.quantile(.9) if len(ages) else np.nan,
                "donor_age_max_hours": ages.max() if len(ages) else np.nan,
                "observed_load_targets": int(observed.sum()),
                "donors_at_observed_load_targets": int((~fallback & observed).sum()),
                "fallbacks_at_observed_load_targets": int((fallback & observed).sum()),
                "original_target_variable_observed_count": int(part["metadata"][variable + "_raw"].notna().sum()),
                "age_summary_denominator": "real-donor origins only; fallback ages undefined",
            })
    return pd.DataFrame(rows)


def _thermal_summary(data, params):
    """Count positive physical features before scaling/masking, using saved thresholds."""
    rows = []
    for partition, part in data["partitions"].items():
        positions = part["positions"]
        for condition in ("FULL", "FULL_PROXY"):
            history_active = np.zeros((len(positions), len(F2)), dtype=bool)
            target_active = np.zeros_like(history_active)
            for start in range(0, len(positions), 256):
                selected = positions[start:start + 256]
                temperatures = window_temperatures(data, selected)
                if condition == "FULL_PROXY":
                    temperatures[:, -1] = proxy_lineage(data, selected).temperature.to_numpy(float)
                physical = raw_features(temperatures, params)
                history_active[start:start + len(selected)] = (physical[:, :168] > 0).any(axis=1)
                target_active[start:start + len(selected)] = physical[:, 168] > 0
            for number, feature in enumerate(F2):
                historical = history_active[:, number]
                target = target_active[:, number]
                times = part["timestamps"][historical]
                rows.append({
                    "fold": data["fold"], "partition": partition, "condition": condition, "feature": feature,
                    "heat_threshold": params["heat_threshold"], "cool_threshold": params["cool_threshold"],
                    "grid_origins": len(positions), "observed_load_origins": int(part["observed"].sum()),
                    "origins_with_positive_history": int(historical.sum()),
                    "origins_with_positive_target": int(target.sum()),
                    "observed_load_origins_with_positive_history": int((historical & part["observed"]).sum()),
                    "observed_load_origins_with_positive_target": int((target & part["observed"]).sum()),
                    "first_origin_with_positive_history": str(times.min()) if len(times) else None,
                    "last_origin_with_positive_history": str(times.max()) if len(times) else None,
                    "feature_stage": "physical values before standardization and input-group masks",
                })
    return pd.DataFrame(rows)


def audit_data(data_path, output_directory, config_path, run_directory=None):
    """Write private descriptive tables to a fresh directory; optionally reuse fitted thresholds."""
    data_path = validate_input(data_path)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not config.get("folds"):
        raise ValueError("Expected schema_version 1 with explicit fold bounds.")
    output = Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError("Audit output must be a new or empty directory; the supplied path is a file.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Audit output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        run_path = Path(run_directory).resolve() if run_directory is not None else None
        if run_path is not None:
            completed = json.loads((run_path / "completed.json").read_text(encoding="utf-8"))
            if completed.get("completed") is not True:
                raise ValueError("Thermal diagnostics require a completed experiment run.")
            if completed.get("synthetic_smoke_only") is not False or completed.get("full_retraining") is not True:
                raise ValueError("Thermal diagnostics require a completed full experiment run; synthetic smoke or unverified run markers are not accepted.")
        frames, partitions, masks, proxies, thermal, identities = [], [], [], [], [], {}
        thermal_sources = {}
        for fold, bounds in config["folds"].items():
            data = load_fold(fold, data_path, strict_counts=config.get("strict_observed_counts", True), bounds=bounds)
            identities[fold] = data["fingerprint"]
            frames.append(data["metadata"][["LOAD", "temperature_raw", "humidity_raw"]])
            for partition, part in data["partitions"].items():
                partitions.append({
                    "fold": fold, "partition": partition, **_scope(part["metadata"]),
                    **_observation_summary(part["metadata"]),
                    "context_observed_fraction_min": part["registry"].context_observed_fraction.min(),
                    "context_observed_fraction_mean": part["registry"].context_observed_fraction.mean(),
                })
                masks.append({"fold": fold, "partition": partition, **_mask_summary(part["metadata"])})
            proxies.append(_proxy_summary(data))
            if run_path is not None:
                parameter_file = run_path / "features" / fold / "parameters.json"
                params = json.loads(parameter_file.read_text(encoding="utf-8"))
                if params.get("identity", {}).get("data_fingerprint") != data["fingerprint"]:
                    raise ValueError(f"Saved thermal parameters do not match the supplied data and fold: {fold}")
                if params.get("synthetic_override"):
                    raise ValueError(f"Synthetic threshold overrides cannot be used for full-run thermal diagnostics: {fold}")
                thermal_sources[fold] = sha256(parameter_file)
                thermal.append(_thermal_summary(data, params))
        combined = pd.concat(frames).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        tables = {
            **_coverage_tables(combined),
            "partition_observation_summary": pd.DataFrame(partitions),
            "observation_mask_agreement": pd.DataFrame(masks),
            "evaluation_year_over_year": _year_over_year(combined, config["folds"]),
            "proxy_availability_summary": pd.concat(proxies, ignore_index=True),
        }
        if thermal:
            tables["thermal_activation_summary"] = pd.concat(thermal, ignore_index=True)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        manifest = {
            "completed": True, "created_utc": utc(), "model_fits": 0, "threshold_selection_fits": 0,
            "private_output": True, "input_sha256": sha256(data_path), "configuration_sha256": sha256(config_path),
            "diagnostics_code_sha256": sha256(__file__), "fold_data_fingerprints": identities,
            "scope": "union of configured fold grids including 168-hour context; overlapping hours counted once",
            "coverage": _scope(combined),
            "observations": "raw observed load/weather only; no imputed values in descriptive means or coverage",
            "year_over_year": "same calendar evaluation interval; descriptive change without physical attribution",
            "thermal_status": "saved thresholds from matching completed run" if thermal else "not requested; supply --run for saved selected thresholds",
            "thermal_parameter_sha256": thermal_sources,
            "source_run_completed_sha256": sha256(run_path / "completed.json") if run_path is not None else None,
            "outputs": {
                f"{name}.csv": {"rows": len(table), "sha256": sha256(output / f"{name}.csv")}
                for name, table in tables.items()
            },
        }
        write_json(output / "completed.json", manifest)
        return manifest
    except BaseException as error:
        write_json(output / "failure.json", {
            "exception_type": type(error).__name__, "message": str(error), "partial_outputs_preserved": True,
        })
        raise

"""Export publication aggregates from a completed new experiment execution.

Selection follows the original publication exporters: deterministic Ridge and
means of predictions over the three seeds for TiDE/XGBoost; paired individual
seed points are retained for the two primary evaluation contrasts. No historical
summary, private hourly prediction, or model checkpoint is copied into exports.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("Ridge-ARX", "XGBoost", "TiDE")
FOLDS = ("warm_late_summer", "cold")
PARTITIONS = ("Validation", "Evaluation")
ARMS = ("FULL", "NO_DERIVED", "NO_TEMP", "NO_HUMIDITY",
        "NO_CALENDAR", "NO_WEATHER", "FULL_PROXY")
SEEDS = ("42", "314", "2718")
ENSEMBLE = "mean_42_314_2718"
KEY = ("fold", "partition", "family", "arm", "seed")
ACCURACY_FIELDS = (
    "family", "fold", "partition", "arm", "seed", "MeanDailyWAPE", "P90DailyWAPE",
    "PooledWAPE", "MAE", "RMSE", "SignedBias", "PeakMagnitudeMAE", "PeakTimingMAE",
    "ObservedHours", "MissingHours", "GridHours", "Days", "ScoredDays",
    "CompleteDays", "MeanContextObservedFraction", "NAReason",
)
CONTRAST_FIELDS = (
    "family", "fold", "partition", "seed", "arm", "comparison", "comparison_class",
    "delta_percentage_points", "FULL_WAPE", "other_WAPE", "days_FULL_better",
    "days_FULL_worse", "days_tied", "days", "iid_CI_low", "iid_CI_high",
    "block3_CI_low", "block3_CI_high", "CIReason", "delta_PeakMagnitudeMAE_kW",
    "delta_PeakTimingMAE_hours", "CompleteDays",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(row):
    return tuple(row[field] for field in KEY)


def _order(row):
    return (FOLDS.index(row["fold"]), PARTITIONS.index(row["partition"]),
            FAMILIES.index(row["family"]), ARMS.index(row["arm"]), row["seed"])


def _public_fields(rows, fields):
    """Project only the declared aggregate schema, never arbitrary input fields."""
    if any(set(fields) - set(row) for row in rows):
        raise ValueError("Metric table is missing required publication aggregate fields.")
    return [{name: row[name] for name in fields} for row in rows]


def _index(rows, arms, label):
    expected = {
        (fold, partition, family, arm, seed)
        for fold, partition, family, arm in itertools.product(FOLDS, PARTITIONS, FAMILIES, arms)
        for seed in (("deterministic",) if family == "Ridge-ARX" else (*SEEDS, ENSEMBLE))
    }
    index = {_key(row): row for row in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError(f"{label} does not contain the complete, unique fold/family/condition/seed protocol.")
    return index


def _close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Inconsistent {label}.")


def _check_paired(paired, overall):
    for row in paired:
        key = list(_key(row))
        other = overall[tuple(key)]
        key[3] = "FULL"
        full = overall[tuple(key)]
        if row["comparison"] != row["arm"] + "-FULL":
            raise ValueError("Contrasts must use condition-minus-FULL direction.")
        _close(row["FULL_WAPE"], full["MeanDailyWAPE"], "FULL score")
        _close(row["other_WAPE"], other["MeanDailyWAPE"], "condition score")
        _close(row["delta_percentage_points"],
               float(other["MeanDailyWAPE"]) - float(full["MeanDailyWAPE"]), "paired WAPE")
        for delta, metric in (("delta_PeakMagnitudeMAE_kW", "PeakMagnitudeMAE"),
                              ("delta_PeakTimingMAE_hours", "PeakTimingMAE")):
            _close(row[delta], float(other[metric]) - float(full[metric]), delta)
        if sum(int(row[name]) for name in ("days_FULL_better", "days_FULL_worse", "days_tied")) != int(row["days"]):
            raise ValueError("Paired daily counts do not sum to their support.")
        for prefix in ("iid", "block3"):
            low, high = float(row[prefix + "_CI_low"]), float(row[prefix + "_CI_high"])
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError("Publication export requires finite, ordered confidence intervals.")


def _window_rows(run, config, overall):
    rows = []
    for fold in FOLDS:
        metadata = pd.read_parquet(run / "preprocessing" / fold / "metadata.parquet")
        if not isinstance(metadata.index, pd.DatetimeIndex) or metadata.index.tz is not None:
            raise ValueError("Preprocessing metadata requires a naive DatetimeIndex.")
        if not metadata.index.is_unique or not metadata.index.is_monotonic_increasing:
            raise ValueError("Preprocessing timestamps must be unique and sorted.")
        audit = json.loads((run / "preprocessing" / fold / "audit.json").read_text(encoding="utf-8"))
        if audit["identity"]["bounds"] != config["folds"][fold]:
            raise ValueError("Preprocessing bounds differ from the execution configuration.")
        for partition in ("train", "validation", "evaluation"):
            start, end = config["folds"][fold][partition]
            selected = metadata.loc[(metadata.index >= start) & (metadata.index < end)]
            expected = pd.date_range(start, end, freq="h", inclusive="left")
            if not selected.index.equals(expected):
                raise ValueError("Study window metadata does not cover the complete hourly grid.")
            observed = np.isfinite(selected.LOAD.to_numpy(float))
            counts = pd.Series(observed, index=selected.index).resample("D").agg(["size", "sum"])
            if not counts["size"].eq(24).all():
                raise ValueError("Publication windows must contain whole calendar days.")
            if selected.temperature_observed.dtype != bool:
                raise ValueError("Temperature observation mask must be boolean.")
            temperatures = selected.loc[selected.temperature_observed, "temperature_raw"].dropna()
            if not len(temperatures) or not np.isfinite(temperatures).all():
                raise ValueError("Study windows require finite observed temperature summaries.")
            row = dict(fold=fold, partition=partition, start=start, end_exclusive=end,
                       grid_hours=len(selected), observed_targets=int(observed.sum()),
                       complete_days=int(counts["sum"].eq(24).sum()), days=len(counts),
                       observed_temperature_count=len(temperatures),
                       temperature_median_C=float(temperatures.median()),
                       temperature_min_C=float(temperatures.min()),
                       temperature_max_C=float(temperatures.max()))
            if partition != "train":
                for family in FAMILIES:
                    representative = "deterministic" if family == "Ridge-ARX" else ENSEMBLE
                    for arm in ARMS:
                        score = overall[(fold, partition.capitalize(), family, arm, representative)]
                        for field, metric in (("grid_hours", "GridHours"), ("observed_targets", "ObservedHours"),
                                              ("complete_days", "CompleteDays"), ("days", "Days")):
                            _close(row[field], score[metric], "study-window " + field)
            rows.append(row)
    return rows


def _lag_rows(rows, overall):
    index = {(row["fold"], row["partition"], int(row["lag_hours"])): row for row in rows}
    expected = set(itertools.product(FOLDS, PARTITIONS, (1, 24, 168)))
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("Expected all three persistence references in both partitions and folds.")
    for (fold, partition, _), row in index.items():
        ref = overall[(fold, partition, "Ridge-ARX", "FULL", "deterministic")]
        for name in ("ObservedHours", "GridHours", "Days", "CompleteDays"):
            _close(row[name], ref[name], "persistence " + name)
        if not math.isfinite(float(row["MeanDailyWAPE"])):
            raise ValueError("Persistence scores must be finite.")
    return [
        dict(reference=label,
             warm_mean_daily_WAPE_percent=index[("warm_late_summer", "Evaluation", lag)]["MeanDailyWAPE"],
             cold_mean_daily_WAPE_percent=index[("cold", "Evaluation", lag)]["MeanDailyWAPE"],
             precision="full precision from this execution; round only for presentation")
        for lag, label in ((1, "Previous hour"), (24, "Previous day"), (168, "Previous week"))
    ]


def export_publication_data(run_directory, output_directory):
    """Create five portable aggregate CSVs and their provenance; never overwrite.

    The input is an ``experiment run`` output directory with its completed marker,
    full metric tables and preprocessing metadata. Confidence bounds are copied
    from that run's paired bootstrap results without rounding or recomputation.
    """
    run, output = Path(run_directory).resolve(), Path(output_directory).resolve()
    if output == run or output in run.parents:
        raise ValueError("The export directory cannot contain or replace the experiment directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Publication export requires a new or empty output directory.")
    completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    if (completed.get("completed") is not True or completed.get("full_retraining") is not True
            or completed.get("synthetic_smoke_only") is not False
            or execution.get("synthetic_smoke_only") is not False):
        raise ValueError("Publication export requires a completed full run, not the reduced synthetic smoke.")
    config = execution["configuration"]
    if (set(config["folds"]) != set(FOLDS) or tuple(config["conditions"]) != ARMS
            or tuple(map(str, config["seeds"])) != SEEDS):
        raise ValueError("Execution configuration does not match the publication folds, conditions and seeds.")
    paths = [run / name for name in ("completed.json", "execution.json",
                                    "metrics/overall_metrics.csv", "metrics/paired_comparisons.csv",
                                    "metrics/lag_references.csv")]
    paths += [run / "preprocessing" / fold / name for fold in FOLDS
              for name in ("audit.json", "metadata.parquet")]
    before = {path.relative_to(run).as_posix(): _sha256(path) for path in paths}
    metrics = _read(run / "metrics/overall_metrics.csv")
    paired = _read(run / "metrics/paired_comparisons.csv")
    overall = _index(metrics, ARMS, "Overall metrics")
    _index(paired, ARMS[1:], "Paired comparisons")
    _check_paired(paired, overall)
    representative = lambda row: row["seed"] == ("deterministic" if row["family"] == "Ridge-ARX" else ENSEMBLE)
    exports = {
        "i15_accuracy.csv": _public_fields(sorted(filter(representative, metrics), key=_order), ACCURACY_FIELDS),
        "i15_contrasts.csv": _public_fields(sorted(filter(representative, paired), key=_order), CONTRAST_FIELDS),
        "i15_primary_seed_points.csv": _public_fields(sorted(
            (row for row in paired if row["partition"] == "Evaluation"
             and row["arm"] in ("NO_WEATHER", "NO_TEMP") and row["seed"] in SEEDS), key=_order), CONTRAST_FIELDS),
        "i15_simple_references.csv": _lag_rows(_read(run / "metrics/lag_references.csv"), overall),
        "study_windows.csv": _window_rows(run, config, overall),
    }
    if any(_sha256(run / name) != digest for name, digest in before.items()):
        raise RuntimeError("Experiment inputs changed while publication aggregates were being exported.")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in exports.items():
        _write(output / name, rows)
    hashes = {name: _sha256(output / name) for name in exports}
    manifest = {
        "schema_version": 1, "origin": "new_full_experiment", "historical_reference": False,
        "synthetic_smoke_only": False, "inputs_sha256": before,
        "exports": {name: {"rows": len(rows), "sha256": hashes[name]} for name, rows in exports.items()},
        "selection": "Deterministic Ridge; three-seed mean predictions for TiDE/XGBoost; matched individual primary seed contrasts.",
        "confidence_intervals": "Copied at source precision from this execution's paired_comparisons.csv; bootstrap was performed during evaluation.",
        "weather": "Observed raw temperatures only; missing values never imputed for window summaries.",
        "persistence": "Scores computed in this execution from the same causal context and observed targets.",
        "privacy": "Explicit ordered aggregate schemas only; unexpected input columns are excluded. No hourly observations, predictions, locations or model weights.",
    }
    (output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (output / "publication-export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

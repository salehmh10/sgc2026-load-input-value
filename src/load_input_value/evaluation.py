"""I14/I15 numerical metric and ensemble semantics, without historical I/O."""
import numpy as np
import pandas as pd
from .protocol import ARMS
ENSEMBLE = "mean_42_314_2718"
SEEDS = ["42", "314", "2718"]
KEY = ["family", "fold", "partition", "arm", "seed"]

def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def daily_metrics(table):
    """Literal metric semantics adapted from I14 study_evaluation.py (hashed below)."""
    table = table.sort_values("timestamp").copy()
    table["date"] = table.timestamp.dt.normalize()
    rows = []
    for date, grid in table.groupby("date", sort=True):
        g = grid.loc[grid.observed]
        y, p = g.actual.to_numpy(float), g.prediction.to_numpy(float)
        denom = np.abs(y).sum()
        full = len(grid) == 24 and len(g) == 24
        rows.append({"date": str(date.date()), "observed_hours": len(g), "grid_hours": len(grid),
            "DailyWAPE": ratio(100 * np.abs(p - y).sum(), denom), "denominator": denom,
            "reason": None if denom > 0 else "no observed targets" if not len(g) else "zero denominator",
            "PeakMagnitudeAE": abs(float(p.max() - y.max())) if full else np.nan,
            "PeakTimingAE": abs(float((g.timestamp.iloc[int(p.argmax())] - g.timestamp.iloc[int(y.argmax())]).total_seconds() / 3600)) if full else np.nan,
            "peak_eligible": full})
    return pd.DataFrame(rows)


def metrics(table):
    assert table.timestamp.is_unique
    assert np.isfinite(table.prediction.to_numpy(float)).all(), "Nonfinite predictions cannot be hidden"
    assert np.array_equal(table.observed.to_numpy(bool), np.isfinite(table.actual.to_numpy(float)))
    q = table.loc[table.observed]
    days = daily_metrics(table)
    error = q.prediction.to_numpy(float) - q.actual.to_numpy(float)
    denom = q.actual.abs().sum()
    return {"MeanDailyWAPE": days.DailyWAPE.mean(), "P90DailyWAPE": days.DailyWAPE.quantile(.9),
        "PooledWAPE": ratio(100 * np.abs(error).sum(), denom), "MAE": np.mean(np.abs(error)) if len(q) else np.nan,
        "RMSE": np.sqrt(np.mean(error ** 2)) if len(q) else np.nan, "SignedBias": np.mean(error) if len(q) else np.nan,
        "PeakMagnitudeMAE": days.PeakMagnitudeAE.mean(), "PeakTimingMAE": days.PeakTimingAE.mean(),
        "ObservedHours": len(q), "MissingHours": len(table) - len(q), "GridHours": len(table), "Days": len(days),
        "ScoredDays": int(days.DailyWAPE.notna().sum()), "CompleteDays": int(days.peak_eligible.sum()),
        "MeanContextObservedFraction": q.context_observed_fraction.mean() if "context_observed_fraction" in q else np.nan,
        "NAReason": None if len(q) and denom > 0 else "no observed targets or zero denominator"}


def confidence(deltas):
    """Exact I14 stream: iid draw precedes noncircular length-three block draw."""
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    if n < 3 or not np.isfinite(deltas).all():
        return {"iid_CI_low": np.nan, "iid_CI_high": np.nan, "block3_CI_low": np.nan,
                "block3_CI_high": np.nan, "CIReason": "fewer3days or undefined daily denominator"}
    rng = np.random.default_rng(20260926)
    iid = deltas[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    blocks = np.stack([deltas[i:i + 3] for i in range(n - 2)])
    index = rng.integers(0, n - 2, size=(10000, int(np.ceil(n / 3))))
    mb = blocks[index].reshape(10000, -1)[:, :n].mean(axis=1)
    return {"iid_CI_low": np.quantile(iid, .025), "iid_CI_high": np.quantile(iid, .975),
            "block3_CI_low": np.quantile(mb, .025), "block3_CI_high": np.quantile(mb, .975), "CIReason": None}


def normalize(frame):
    frame = frame.copy()
    if "condition" in frame and "arm" not in frame:
        frame = frame.rename(columns={"condition": "arm"})
    frame["seed"] = frame.seed.astype(str)
    frame["partition"] = frame.partition.astype(str).str.capitalize()
    frame["family"] = frame.family.replace({"Ridge": "Ridge-ARX", "ridge": "Ridge-ARX", "xgboost": "XGBoost", "tide": "TiDE"})
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    assert frame.observed.dtype == bool, "Observed masks must be boolean"
    return frame


def add_or_check_ensembles(frame):
    frame = normalize(frame)
    add = []
    for (family, fold, part, arm), g in frame.groupby(KEY[:-1], sort=False):
        if family == "Ridge-ARX":
            assert set(g.seed) == {"deterministic"}, "No artificial Ridge seeds"
            continue
        assert set(g.seed) in (set(SEEDS), set(SEEDS + [ENSEMBLE]))
        ordered = [g.loc[g.seed == s].sort_values("timestamp").reset_index(drop=True) for s in SEEDS]
        reference = ordered[0]
        for other in ordered[1:]:
            assert np.array_equal(reference.timestamp, other.timestamp)
            assert np.array_equal(reference.actual, other.actual, equal_nan=True)
            assert np.array_equal(reference.observed, other.observed)
        mean = np.mean([q.prediction.to_numpy(float) for q in ordered], axis=0)
        if ENSEMBLE in set(g.seed):
            existing = g.loc[g.seed == ENSEMBLE].sort_values("timestamp")
            np.testing.assert_allclose(existing.prediction, mean, atol=1e-8, rtol=1e-13)
        else:
            table = reference.copy()
            table["seed"], table["prediction"] = ENSEMBLE, mean
            add.append(table)
    return pd.concat([frame, *add], ignore_index=True) if add else frame


def tables_from_predictions(frame):
    overall, daily = [], []
    for keys, table in frame.groupby(KEY, sort=False):
        key = dict(zip(KEY, keys))
        overall.append({**key, **metrics(table)})
        daily.append(daily_metrics(table).assign(**key))
    overall, daily = pd.DataFrame(overall), pd.concat(daily, ignore_index=True)
    paired, differences = [], []
    for keys, g in daily.groupby(["family", "fold", "partition", "seed"], sort=False):
        key = dict(zip(["family", "fold", "partition", "seed"], keys))
        ref = g.loc[g.arm == "FULL"].set_index("date").sort_index()
        for arm in ARMS[1:]:
            other = g.loc[g.arm == arm].set_index("date").sort_index()
            assert list(ref.index) == list(other.index)
            assert np.array_equal(ref.observed_hours, other.observed_hours)
            delta = other.DailyWAPE.to_numpy() - ref.DailyWAPE.to_numpy()
            k = {**key, "arm": arm, "comparison": arm + "-FULL",
                 "comparison_class": "primary" if arm in ("NO_WEATHER", "NO_TEMP") else "supplementary"}
            paired.append({**k, "delta_percentage_points": np.mean(delta), "FULL_WAPE": ref.DailyWAPE.mean(),
                "other_WAPE": other.DailyWAPE.mean(), "days_FULL_better": int((delta > 0).sum()),
                "days_FULL_worse": int((delta < 0).sum()), "days_tied": int((delta == 0).sum()),
                "days": len(delta), **confidence(delta),
                "delta_PeakMagnitudeMAE_kW": other.PeakMagnitudeAE.mean() - ref.PeakMagnitudeAE.mean(),
                "delta_PeakTimingMAE_hours": other.PeakTimingAE.mean() - ref.PeakTimingAE.mean(),
                "CompleteDays": int(ref.peak_eligible.sum())})
            differences.extend({**k, "date": date, "delta_percentage_points": value} for date, value in zip(ref.index, delta))
    paired = pd.DataFrame(paired)
    tables = {"overall_metrics": overall, "daily_metrics": daily, "paired_comparisons": paired,
              "paired_daily_differences": pd.DataFrame(differences)}
    tables["seed_metrics"] = overall.loc[overall.seed != ENSEMBLE]
    tables["ensemble_metrics"] = overall.loc[overall.seed == ENSEMBLE]
    tables["primary_comparisons"] = paired.loc[paired.comparison_class == "primary"]
    tables["weather_availability"] = paired.loc[paired.arm == "FULL_PROXY"]
    tables["peak_metrics"] = overall[KEY + ["PeakMagnitudeMAE", "PeakTimingMAE", "CompleteDays"]]
    tables["full_family_comparison"] = overall.loc[overall.arm == "FULL"]
    stability = []
    for key, group in paired.loc[paired.seed.isin(SEEDS)].groupby(["family", "fold", "partition", "arm"], sort=False):
        values = group.delta_percentage_points.to_numpy(float)
        stability.append({**dict(zip(["family", "fold", "partition", "arm"], key)),
            "seed_count": len(group), "delta_mean": np.mean(values), "delta_sd": np.std(values, ddof=1),
            "delta_min": np.min(values), "delta_max": np.max(values),
            "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum()),
            "all_seeds_same_sign": bool(np.all(values > 0) or np.all(values < 0) or np.all(values == 0))})
    tables["seed_stability"] = pd.DataFrame(stability)
    variability = overall.loc[overall.seed.isin(SEEDS)].groupby(["family", "fold", "partition", "arm"], sort=False)
    tables["seed_metric_variability"] = variability.agg(
        seed_count=("seed", "size"), MeanDailyWAPE_mean=("MeanDailyWAPE", "mean"),
        MeanDailyWAPE_sd=("MeanDailyWAPE", "std"), MeanDailyWAPE_min=("MeanDailyWAPE", "min"),
        MeanDailyWAPE_max=("MeanDailyWAPE", "max"), MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std")).reset_index()
    tables["validation_evaluation_shift"] = paired.pivot(index=["family", "fold", "seed", "arm"], columns="partition", values="delta_percentage_points").reset_index()
    tables["validation_evaluation_shift"]["Evaluation_minus_Validation_pp"] = tables["validation_evaluation_shift"].Evaluation - tables["validation_evaluation_shift"].Validation
    tables["validation_evaluation_shift"]["sign_changed"] = tables["validation_evaluation_shift"].Evaluation * tables["validation_evaluation_shift"].Validation < 0
    return tables


def numeric_compare(recomputed, saved, key):
    a, b = recomputed.set_index(key).sort_index(), saved.set_index(key).sort_index()
    assert a.index.equals(b.index), "Different metric groups"
    rows = []
    for col in b.select_dtypes(include=np.number).columns:
        assert col in a
        av, bv = a[col].to_numpy(float), b[col].to_numpy(float)
        np.testing.assert_allclose(av, bv, atol=1e-9, rtol=1e-12, equal_nan=True, err_msg=col)
        finite = np.isfinite(av) & np.isfinite(bv)
        rows.append({"metric": col, "max_absolute_difference": float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0})
    return rows


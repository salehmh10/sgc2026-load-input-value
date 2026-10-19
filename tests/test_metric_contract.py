"""Independent known-answer controls. All values below are synthetic."""
import numpy as np
import pandas as pd
import pytest

from load_input_value.evaluation import add_or_check_ensembles, confidence, daily_metrics, metrics


def synthetic_table(actual, prediction, start="2025-01-01"):
    actual = np.asarray(actual, dtype=float)
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=len(actual), freq="h"),
                         "actual": actual, "prediction": prediction,
                         "observed": np.isfinite(actual)})


def test_mean_daily_and_pooled_wape_are_distinct():
    frame = synthetic_table([10.] * 24 + [100.] * 24, [12.] * 24 + [110.] * 24)
    scores = metrics(frame)
    assert scores["MeanDailyWAPE"] == pytest.approx(15.)
    assert scores["PooledWAPE"] == pytest.approx(100 * 288 / 2640)
    assert scores["MAE"] == pytest.approx(6.)
    assert scores["SignedBias"] == pytest.approx(6.)


def test_missing_targets_are_excluded_and_peaks_need_complete_days():
    y = np.full(48, 10.)
    y[3] = np.nan
    pred = np.full(48, 11.)
    pred[3] = 999999.
    result = metrics(synthetic_table(y, pred))
    assert result["MeanDailyWAPE"] == pytest.approx(10.)
    assert result["ObservedHours"] == 47
    assert result["CompleteDays"] == 1
    assert result["PeakMagnitudeMAE"] == pytest.approx(1.)


def test_peak_timing_uses_first_argmax_and_absolute_clock_difference():
    y, p = np.ones(24), np.ones(24)
    y[[0, 1]], p[[23]] = 100., 90.
    row = daily_metrics(synthetic_table(y, p)).iloc[0]
    assert row.PeakMagnitudeAE == 10.
    assert row.PeakTimingAE == 23.  # Not wrapped around the 24-hour clock.


def test_predictions_are_averaged_before_scoring():
    frames = []
    for seed, value in [("42", 0.), ("314", 20.), ("2718", 10.)]:
        frame = synthetic_table([10.] * 24, [value] * 24)
        frames.append(frame.assign(family="TiDE", fold="synthetic", partition="Evaluation", arm="FULL", seed=seed))
    frame = add_or_check_ensembles(pd.concat(frames, ignore_index=True))
    ensemble = frame.loc[frame.seed == "mean_42_314_2718"]
    assert metrics(ensemble)["MeanDailyWAPE"] == 0.
    assert np.mean([metrics(f)["MeanDailyWAPE"] for f in frames]) == pytest.approx(200 / 3)


def test_seed_ensemble_requires_matching_observations():
    rows = []
    for seed in ("42", "314", "2718"):
        rows.append(synthetic_table([10.] * 24, [10.] * 24).assign(
            family="XGBoost", fold="synthetic", partition="Evaluation", arm="FULL", seed=seed))
    rows[-1].loc[0, "actual"] = 11.
    with pytest.raises(AssertionError):
        add_or_check_ensembles(pd.concat(rows, ignore_index=True))


def test_block_interval_preserves_negative_contrast():
    ci = confidence([-2.5] * 28)
    assert ci["block3_CI_low"] == -2.5
    assert ci["block3_CI_high"] == -2.5
    assert ci["iid_CI_low"] == -2.5
    assert np.isnan(confidence([1., 2.])["block3_CI_low"])


def test_nonfinite_predictions_fail_even_for_missing_targets():
    frame = synthetic_table([np.nan, 10.], [np.nan, 10.])
    with pytest.raises(AssertionError):
        metrics(frame)

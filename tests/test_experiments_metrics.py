import numpy as np
import pandas as pd
import pytest

from load_input_value.evaluation import metrics, daily_metrics, confidence, add_or_check_ensembles


def table(actual, predicted, start='2025-01-01'):
    return pd.DataFrame({'timestamp': pd.date_range(start, periods=len(actual), freq='h'),
                         'actual': actual, 'observed': np.isfinite(actual), 'prediction': predicted})


def test_daily_equal_weight_and_pooled_weight_differ():
    y = np.r_[np.ones(24), np.full(24, 10.)]
    p = y + 1
    result = metrics(table(y, p))
    assert result['MeanDailyWAPE'] == pytest.approx(55.)
    assert result['PooledWAPE'] == pytest.approx(100 * 48 / 264)
    assert result['SignedBias'] == 1.


def test_missing_targets_excluded_but_predictions_must_be_finite():
    y, p = np.ones(24), np.ones(24)
    y[5], p[5] = np.nan, 100000.
    result = metrics(table(y, p))
    assert result['MAE'] == 0 and result['ObservedHours'] == 23
    assert result['CompleteDays'] == 0 and np.isnan(result['PeakMagnitudeMAE'])
    p[5] = np.nan
    with pytest.raises(AssertionError):
        metrics(table(y, p))


def test_peak_earliest_tie_non_circular():
    y, p = np.ones(24), np.ones(24)
    y[[0, 5]], p[[22, 23]] = 10., 12.
    row = daily_metrics(table(y, p)).iloc[0]
    assert row.PeakTimingAE == 22 and row.PeakMagnitudeAE == 2


def test_bootstrap_preserves_historical_rng_order():
    delta = np.array([.1, -.3, .8, .4, -.5, .9, .2])
    rng = np.random.default_rng(20260926)
    iid = delta[rng.integers(0, 7, size=(10000, 7))].mean(axis=1)
    blocks = np.array([delta[i:i+3] for i in range(5)])
    draws = blocks[rng.integers(0, 5, size=(10000, 3))].reshape(10000, 9)[:, :7].mean(axis=1)
    result = confidence(delta)
    assert result['iid_CI_low'] == np.quantile(iid, .025)
    assert result['block3_CI_low'] == np.quantile(draws, .025)
    assert result['block3_CI_high'] == np.quantile(draws, .975)


def test_ensemble_predictions_before_scoring_and_alignment():
    rows = []
    for seed, pred in [('42', 0.), ('314', 2.), ('2718', 1.)]:
        rows.append(table(np.ones(24), np.full(24, pred)).assign(
            family='TiDE', fold='cold', partition='Evaluation', arm='FULL', seed=seed))
    individual = pd.concat(rows, ignore_index=True)
    result = add_or_check_ensembles(individual)
    ensemble = result[result.seed == 'mean_42_314_2718']
    assert metrics(ensemble)['MAE'] == 0.
    assert np.mean([metrics(r)['MAE'] for r in rows]) > 0.
    individual.loc[24, 'timestamp'] += pd.Timedelta(hours=2)
    with pytest.raises(AssertionError):
        add_or_check_ensembles(individual)

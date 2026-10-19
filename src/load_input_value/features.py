"""Window-bounded F2 and fresh fold-specific auxiliary Ridge preprocessing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .io import write_json
from .data import sha256 as sha

F2 = ['heating_excess', 'cooling_excess', 'heating_degree_hours6',
      'cooling_degree_hours6', 'heating_degree_hours24', 'cooling_degree_hours24']
HEAT_GRID = (10., 14., 18., 22.)
COOL_GRID = (18., 22., 26., 30.)
BOUNDS = {
    'warm_late_summer': ('2024-08-04', '2025-08-04'),
    'cold': ('2024-11-03', '2025-11-03'),
}


def fold_name(data):
    return data['fold']


def raw_features(temperatures, params):
    """Prefix-only six features; arbitrary window size supports manual checks."""
    temp = np.asarray(temperatures, np.float64)
    assert temp.ndim == 2 and temp.shape[1] > 0 and np.isfinite(temp).all()
    heating = np.maximum(float(params['heat_threshold']) - temp, 0.)
    cooling = np.maximum(temp - float(params['cool_threshold']), 0.)
    output = [heating, cooling]
    j = np.arange(temp.shape[1])
    for width in (6, 24):
        for value in (heating, cooling):
            sums = np.concatenate([np.zeros((len(temp), 1)), np.cumsum(value, axis=1)], axis=1)
            output.append(sums[:, j] - sums[:, np.maximum(0, j - width)])
    return np.stack(output, axis=-1)


def window_indices(data, positions):
    positions = np.asarray(positions, np.int64)
    assert positions.ndim == 1 and np.all(positions >= 168)
    assert np.all(positions < len(data['frame']))
    return positions[:, None] + np.arange(-168, 1)[None, :]


def window_temperatures(data, positions, physical_temperature=None):
    indices = window_indices(data, positions)
    if physical_temperature is None:
        values = data['metadata'].temperature.to_numpy(np.float64)
    else:
        values = np.asarray(physical_temperature, np.float64)
        if values.shape == indices.shape:
            return values.copy()
        assert values.shape == (len(data['frame']),)
    return values[indices]


def derived(data, positions, params=None, physical_temperature=None):
    params = data['params'] if params is None else params
    raw = raw_features(window_temperatures(data, positions, physical_temperature), params)
    spec = params['scaler']
    values = ((raw - np.asarray(spec['mean'])) / np.asarray(spec['scale'])).astype(np.float32)
    assert np.isfinite(values).all()
    return values[:, :168], values[:, 168:]


def cv_design(fold, prefix_days=(181, 227, 273, 319)):
    start, end = map(pd.Timestamp, BOUNDS[fold])
    boundaries = [start + pd.Timedelta(days=int(day)) for day in prefix_days] + [end]
    return {'fold': fold, 'grid': [{'heat': h, 'cool': c} for h in HEAT_GRID for c in COOL_GRID if h <= c],
            'folds': [{'id': i + 1, 'train_start': str(start), 'train_end_exclusive': str(boundaries[i]),
                       'validation_start': str(boundaries[i]), 'validation_end_exclusive': str(boundaries[i + 1])}
                      for i in range(4)],
            'support_min_each_side_each_prefix': 100,
            'support_definition': 'jointly observed LOAD and temperature targets; strict < and >; fitted Ridge retains observed LOAD targets with prefix-median fill',
            'score': 'equal mean of four Mean Daily WAPE values',
            'tie_rule': 'exact score ties: smaller heat, then smaller cool',
            'no_eligible_pair': 'hard failure before neural training; no fallback',
            'ridge': {'alpha': 10., 'fit_intercept': True},
            'controls': ['load_context_lag1', 'load_context_lag24', 'load_context_lag168', 'all22calendar'],
            'preprocessing': 'prefix weather median and StandardScaler fit on observed-target prefix; LOAD controls in physical units',
            'validation_support': 'report all regimes, no sample exclusion'}


def _daily_wape(actual, prediction, timestamps):
    frame = pd.DataFrame({'day': pd.DatetimeIndex(timestamps).normalize(),
                          'ae': np.abs(prediction - actual), 'ay': np.abs(actual)})
    summed = frame.groupby('day')[['ae', 'ay']].sum()
    if not summed.ay.gt(0).all():
        raise RuntimeError('Zero denominator in inner CV daily WAPE')
    return float((100 * summed.ae / summed.ay).mean())


def threshold_cv(data, directory):
    """Only small Ridge fits; main manager authorizes through independent design review."""
    from .inputs import CALENDAR
    fold = fold_name(data)
    contract = cv_design(fold)
    frame, metadata = data['frame'], data['metadata']
    start, end = map(pd.Timestamp, BOUNDS[fold])
    positions = np.asarray(data['train_positions'], np.int64)
    timestamps = frame.index[positions]
    assert timestamps.min() >= start and timestamps.max() < end
    actual = frame.LOAD.to_numpy(float)[positions]
    physical_context = frame.load_context.to_numpy(float)
    controls = np.column_stack([physical_context[positions - lag] for lag in (1, 24, 168)] +
                               [frame[CALENDAR].to_numpy(float)[positions]])
    original_t = metadata.temperature_raw.to_numpy(float)
    trials, predicted, prefixes = [], [], []
    for cv in contract['folds']:
        fit_end = pd.Timestamp(cv['train_end_exclusive'])
        val_end = pd.Timestamp(cv['validation_end_exclusive'])
        train = timestamps < fit_end
        validation = (timestamps >= fit_end) & (timestamps < val_end)
        assert train.any() and validation.any()
        fit_weather = (frame.index >= start) & (frame.index < fit_end) & np.isfinite(original_t)
        median = float(np.median(original_t[fit_weather]))
        temperature = np.where(np.isfinite(original_t[positions]), original_t[positions], median)
        prefixes.append({**cv, 'train_targets': int(train.sum()), 'validation_targets': int(validation.sum()),
                         'weather_fit_rows': int(fit_weather.sum()), 'temperature_median': median,
                         'weather_fit_start': str(frame.index[fit_weather].min()),
                         'weather_fit_last': str(frame.index[fit_weather].max())})
        for pair in contract['grid']:
            heat, cool = pair['heat'], pair['cool']
            support = {}
            for label, mask in [('train', train), ('validation', validation)]:
                mask = mask & np.isfinite(original_t[positions])
                for name, threshold in [('heat', heat), ('cool', cool)]:
                    support[f'{label}_below_{name}'] = int((temperature[mask] < threshold).sum())
                    support[f'{label}_above_{name}'] = int((temperature[mask] > threshold).sum())
            adequate = min(v for key, v in support.items() if key.startswith('train')) >= 100
            row = {'inner_fold': cv['id'], **pair, **support, 'eligible_prefix': adequate,
                   'mean_daily_wape': np.nan}
            if adequate:
                x = np.column_stack([controls, np.maximum(heat - temperature, 0.), np.maximum(temperature - cool, 0.)])
                scaler = StandardScaler().fit(x[train])
                model = Ridge(alpha=10., fit_intercept=True).fit(scaler.transform(x[train]), actual[train])
                prediction = model.predict(scaler.transform(x[validation]))
                row['mean_daily_wape'] = _daily_wape(actual[validation], prediction, timestamps[validation])
                row['heating_coefficient_physical'] = float(model.coef_[-2] / scaler.scale_[-2])
                row['cooling_coefficient_physical'] = float(model.coef_[-1] / scaler.scale_[-1])
                predicted.append(pd.DataFrame({'inner_fold': cv['id'], **pair, 'timestamp': timestamps[validation],
                                               'actual': actual[validation], 'prediction': prediction}))
            trials.append(row)
    trial_table = pd.DataFrame(trials)
    trial_table.to_csv(directory / 'cv_trials.csv', index=False)
    pd.DataFrame(prefixes).to_csv(directory / 'cv_prefixes.csv', index=False)
    if predicted:
        pd.concat(predicted, ignore_index=True).to_parquet(directory / 'cv_predictions.parquet', index=False)
    summary = []
    for (heat, cool), group in trial_table.groupby(['heat', 'cool']):
        eligible = len(group) == 4 and bool(group.eligible_prefix.all())
        summary.append({'heat': heat, 'cool': cool, 'eligible': eligible,
                        'mean_daily_wape': group.mean_daily_wape.mean() if eligible else np.nan,
                        'std_inner_fold_wape': group.mean_daily_wape.std(ddof=0) if eligible else np.nan})
    summary = pd.DataFrame(summary).sort_values(['mean_daily_wape', 'heat', 'cool'], na_position='last')
    summary.to_csv(directory / 'cv_summary.csv', index=False)
    eligible = summary[summary.eligible]
    if eligible.empty:
        raise RuntimeError(f'No threshold pair has support in all four prefixes: {fold}')
    best = eligible.iloc[0]
    winners = trial_table[trial_table.mean_daily_wape.notna()].sort_values(
        ['inner_fold', 'mean_daily_wape', 'heat', 'cool']).groupby('inner_fold').first().reset_index()
    winners.to_csv(directory / 'cv_inner_winners.csv', index=False)
    selection = {'heat_threshold': float(best.heat), 'cool_threshold': float(best.cool),
                 'mean_cv_daily_wape': float(best.mean_daily_wape),
                 'runner_up_gap_pp': float(eligible.iloc[1].mean_daily_wape - best.mean_daily_wape) if len(eligible) > 1 else None,
                 'boundary_optimum': best.heat in (min(HEAT_GRID), max(HEAT_GRID)) or best.cool in (min(COOL_GRID), max(COOL_GRID)),
                 'eligible_pairs': len(eligible), 'ridge_fits': int(trial_table.eligible_prefix.sum()),
                 'inner_winners': winners[['inner_fold', 'heat', 'cool', 'mean_daily_wape']].to_dict('records'),
                 'interpretation': 'predictive thresholds; not causal or physical balance-temperature estimates'}
    write_json(directory / 'threshold_selection.json', selection)
    return selection


def fit_parameters(data, directory, selection=None):
    """Train-only weighted occurrence scaler; selection override is smoke-only."""
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    selection = threshold_cv(data, directory) if selection is None else selection
    positions = np.asarray(data['train_positions'], np.int64)
    indices = window_indices(data, positions)
    raw = raw_features(window_temperatures(data, positions), selection)
    start, end = map(pd.Timestamp, data['bounds']['train'])
    times = data['frame'].index.to_numpy()[indices]
    keep = (times >= start.to_datetime64()) & (times < end.to_datetime64())
    fit = raw[keep]
    scaler = StandardScaler().fit(fit)
    median = np.median(fit, axis=0)
    params = {'fold': data['fold'], **selection,
              'identity': {'data_fingerprint': data['fingerprint'], 'code_sha256': sha(__file__)},
              'scaler': {'features': F2, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                         'var': scaler.var_.tolist(), 'median_raw': median.tolist(),
                         'median_scaled': ((median - scaler.mean_) / scaler.scale_).tolist(),
                         'n_samples_seen': int(scaler.n_samples_seen_)},
              'fit_scope': {'start_inclusive': str(start), 'end_exclusive': str(end),
                            'observed_train_origins': len(positions),
                            'excluded_pretrain_occurrences': int((~keep).sum()),
                            'fitted_occurrences': int(keep.sum()),
                            'overlapping_windows_multiplicity_retained': True}}
    write_json(directory / 'parameters.json', params)
    data['params'] = params
    return params

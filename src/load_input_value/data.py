"""Fresh, fold-specific I14 reconstruction; prior studies are text-only sources.

The calendar, causal LOAD hierarchy, schema, and standardization definitions are
adapted from I11 stage1_core.py and I13 study_data.py with SHA256 provenance.
No prior module, scaler, registry, processed data, or model is imported or loaded.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PAST = ['temperature', 'humidity', 'temperature_missing', 'humidity_missing',
        'load_observed', 'load_imputed', 'causal_gap_age',
        'recent_observed_fraction', 'long_gap_flag']
FUTURE = ['temperature', 'humidity', 'temperature_missing', 'humidity_missing',
          'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
          'hour_of_week_sin', 'hour_of_week_cos',
          *[f'shamsi_month_{m:02d}' for m in range(1, 13)],
          'working_day', 'holiday', 'pre_holiday', 'post_holiday']
CALENDAR = FUTURE[4:]
WEATHER = ['temperature', 'humidity']
FOLDS = {
    'warm_late_summer': {'train': ('2024-08-04', '2025-08-04'),
                         'validation': ('2025-08-04', '2025-09-01'),
                         'evaluation': ('2025-09-01', '2025-09-29')},
    'cold': {'train': ('2024-11-03', '2025-11-03'),
             'validation': ('2025-11-03', '2025-12-01'),
             'evaluation': ('2025-12-01', '2025-12-29')},
}
EXPECTED_COUNTS = {'warm_late_summer': [7088, 611, 626], 'cold': [7196, 662, 669]}
_MEMORY = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding='utf-8')


def gregorian_to_jalali(gy, gm, gd):
    """Local arithmetic conversion, transcribed from I11 and checked to raw."""
    md = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy, gy = 979, gy - 1600
    else:
        jy, gy = 0, gy - 621
    gy2 = gy + 1 if gm > 2 else gy
    days = 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 - 80 + gd + md[gm - 1]
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    return (jy, 1 + days // 31, 1 + days % 31) if days < 186 else (jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30)


def make_calendar(grid, raw):
    import holidays
    from holidays.countries.iran import Iran
    index = grid.index
    days = pd.date_range(index.min().normalize() - pd.Timedelta(days=1),
                         index.max().normalize() + pd.Timedelta(days=1), freq='D')
    iran = holidays.country_holidays('IR', years=sorted(set(days.year)), language='en_US')
    daily = pd.DataFrame(index=days)
    daily['holiday_name'] = [iran.get(day.date(), '') for day in days]
    daily['holiday'] = [int(day.dayofweek == 4 or day.date() in iran) for day in days]
    daily['pre_holiday'] = daily.holiday.shift(-1).fillna(0).astype(int)
    daily['post_holiday'] = daily.holiday.shift(1).fillna(0).astype(int)
    raw_daily = raw.groupby(raw.index.normalize()).HOLIDAY.agg(['first', 'nunique'])
    assert raw_daily['nunique'].eq(1).all()
    daily['csv_holiday'] = raw_daily['first']
    daily['csv_differs'] = daily.csv_holiday.notna() & daily.csv_holiday.ne(daily.holiday)
    calendar = pd.DataFrame(index=index)
    for name, values, period in (('hour', index.hour, 24), ('weekday', index.dayofweek, 7),
                                 ('hour_of_week', index.dayofweek * 24 + index.hour, 168)):
        calendar[name + '_sin'] = np.sin(2 * np.pi * values / period)
        calendar[name + '_cos'] = np.cos(2 * np.pi * values / period)
    month = (grid.shamsi_date // 100) % 100
    for number in range(1, 13):
        calendar[f'shamsi_month_{number:02d}'] = (month == number).astype(float)
    for name in ('holiday', 'pre_holiday', 'post_holiday'):
        calendar[name] = daily.loc[index.normalize(), name].to_numpy()
    calendar['working_day'] = 1 - calendar.holiday
    meta = {'policy': 'Iran national public schedule OR Friday; no unarchived emergency closure',
            'library_version': holidays.__version__, 'library_source_sha256': sha256(inspect.getsourcefile(Iran)),
            'csv_schedule_disagreement_days': int(daily.csv_differs.sum()),
            'future_schedule_for_adjacent_holiday_flags': 'Known national calendar, never target/weather/LOAD data'}
    return calendar, daily, meta


def make_raw_fold(fold, raw_path, bounds=None):
    """Parse only permitted values; timestamp-only pass identifies skipped rows."""
    spec = bounds or FOLDS[fold]
    train_start, train_end = map(pd.Timestamp, spec['train'])
    start = train_start - pd.Timedelta(hours=168)
    end = pd.Timestamp(spec['evaluation'][1])
    assert end <= pd.Timestamp('2026-01-08')
    timestamps = pd.read_csv(raw_path, usecols=['TIMESTAMP'], parse_dates=['TIMESTAMP']).TIMESTAMP
    assert timestamps.notna().all() and timestamps.eq(timestamps.dt.floor('h')).all()
    assert timestamps.is_unique and timestamps.is_monotonic_increasing
    allowed = timestamps.ge(start) & timestamps.lt(end)
    skip = set((np.flatnonzero(~allowed.to_numpy()) + 1).tolist())
    raw = pd.read_csv(raw_path, usecols=['TIMESTAMP', 'LOAD', 'TEMP_HOUR', 'HUMIDITY', 'HOLIDAY', 'SHAMSI'],
                      parse_dates=['TIMESTAMP'], skiprows=lambda number: number in skip).set_index('TIMESTAMP')
    assert len(raw) == int(allowed.sum()) and raw.index.is_unique
    assert raw.index.min() >= start and raw.index.max() < end
    index = pd.date_range(start, end, freq='h', inclusive='left', name='timestamp')
    grid = raw.reindex(index).rename(columns={'TEMP_HOUR': 'temperature', 'HUMIDITY': 'humidity'})
    jalali = np.array([gregorian_to_jalali(t.year, t.month, t.day) for t in index])
    grid['shamsi_date'] = jalali[:, 0] * 10000 + jalali[:, 1] * 100 + jalali[:, 2]
    present = grid.SHAMSI.notna()
    assert np.array_equal(grid.loc[present, 'shamsi_date'], grid.loc[present, 'SHAMSI'])
    grid['row_present'] = index.isin(raw.index)
    for column in ['LOAD'] + WEATHER:
        assert not np.isinf(grid[column].to_numpy(float)).any()
    calendar, daily, calendar_meta = make_calendar(grid, raw)
    return {'fold': fold, 'spec': spec, 'grid': grid, 'raw': raw, 'calendar': calendar,
            'calendar_daily': daily, 'calendar_metadata': calendar_meta,
            'train_start': train_start, 'train_end': train_end, 'context_start': start,
            'evaluation_end': end}


def causal_load(grid, calendar):
    """Past observed donors only; bounded-prefix empty history uses zero sentinel.

    The sentinel never becomes a donor. Existing missingness/quality channels
    distinguish it; it affects 31 initial cold context hours, not target values.
    """
    values = grid.LOAD.to_numpy(float)
    observed = np.isfinite(values)
    context = values.copy()
    methods = np.full(len(values), 'observed', dtype=object)
    gap_age = np.zeros(len(values), dtype=int)
    source_max = np.arange(len(values), dtype=int)
    source_min = np.arange(len(values), dtype=int)
    counts = np.ones(len(values), dtype=int)
    donor_age = np.zeros(len(values), dtype=float)
    hour_daytype, hour_week, prior, lineage = {}, {}, [], []
    for i, timestamp in enumerate(grid.index):
        daytype_key = (timestamp.hour, int(calendar.working_day.iloc[i]))
        week_key = (timestamp.dayofweek, timestamp.hour)
        if observed[i]:
            hour_daytype.setdefault(daytype_key, []).append(i)
            hour_week.setdefault(week_key, []).append(i)
            prior.append(i)
            continue
        gap_age[i] = (gap_age[i - 1] if i else 0) + 1
        donors = [i - lag for lag in (168, 336, 504, 672) if i >= lag and observed[i - lag]]
        if donors:
            method = 'observed_weekly_lag_median'
        elif hour_daytype.get(daytype_key):
            donors, method = hour_daytype[daytype_key], 'past_observed_same_hour_daytype_median'
        elif hour_week.get(week_key):
            donors, method = hour_week[week_key], 'past_observed_hour_of_week_median'
        elif prior:
            donors, method = prior, 'past_observed_global_median'
        else:
            donors, method = [], 'no_prior_observation_zero_sentinel'
        assert all(d < i and observed[d] for d in donors)
        context[i] = float(np.median(values[donors])) if donors else 0.0
        source_max[i] = max(donors) if donors else -1
        source_min[i] = min(donors) if donors else -1
        donor_age[i] = i - max(donors) if donors else np.nan
        counts[i], methods[i] = len(donors), method
        lineage.append({'context_timestamp': str(timestamp), 'imputed_value': context[i], 'method': method,
                        'donor_count': len(donors), 'donor_timestamps': [str(grid.index[d]) for d in donors],
                        'donor_values': values[donors].tolist(),
                        'source_max_timestamp': str(grid.index[max(donors)]) if donors else None,
                        'source_min_timestamp': str(grid.index[min(donors)]) if donors else None,
                        'newest_donor_age_hours': float(donor_age[i]) if donors else None})
    recent = pd.Series(observed.astype(float)).rolling(168, min_periods=1).mean().to_numpy()
    info = pd.DataFrame({'load_context': context, 'load_observed': observed.astype(float),
                         'load_imputed': (~observed).astype(float), 'load_imputation_method': methods,
                         'causal_gap_age_raw': gap_age, 'recent_observed_fraction_raw': recent,
                         'long_gap_flag': (gap_age >= 24).astype(float),
                         'load_source_max_position': source_max, 'load_source_min_position': source_min,
                         'load_donor_count': counts, 'load_newest_donor_age_hours': donor_age}, index=grid.index)
    assert all(row['source_max_timestamp'] is None or pd.Timestamp(row['source_max_timestamp']) < pd.Timestamp(row['context_timestamp']) for row in lineage)
    return info, lineage


def fit_preprocessor(grid, info, fit_start, fit_end):
    """Fresh scalers on [fit_start, fit_end); pre-Train context never fit."""
    fit_start, fit_end = pd.Timestamp(fit_start), pd.Timestamp(fit_end)
    fit = (grid.index >= fit_start) & (grid.index < fit_end)
    assert fit.any()
    medians = {column: float(grid.loc[fit, column].median()) for column in WEATHER}
    assert all(np.isfinite(list(medians.values())))
    filled = grid[WEATHER].fillna(medians)
    load = StandardScaler().fit(grid.loc[fit & grid.LOAD.notna(), ['LOAD']].to_numpy())
    weather = StandardScaler().fit(filled.loc[fit, WEATHER].to_numpy())
    quality_values = np.column_stack([np.log1p(info.loc[fit, 'causal_gap_age_raw']), info.loc[fit, 'recent_observed_fraction_raw']])
    derived = StandardScaler().fit(quality_values)
    metadata = {'fit_start_inclusive': str(fit_start), 'fit_end_exclusive': str(fit_end),
                'load_observed_rows': int((fit & grid.LOAD.notna()).sum()), 'weather_rows': int(fit.sum()),
                'context_before_fit_rows': 0, 'validation_or_evaluation_fit_rows': 0,
                'weather_imputation': 'per-variable fixed fit-prefix median; not future interpolation or observed weather',
                'weather_medians': medians, 'binary_and_cyclic_standardized': False,
                'parameters': {name: {'mean': scale.mean_.tolist(), 'scale': scale.scale_.tolist(),
                                     'var': scale.var_.tolist(), 'n_samples_seen': int(scale.n_samples_seen_)}
                               for name, scale in [('load', load), ('weather', weather), ('derived', derived)]}}
    return {'load': load, 'weather': weather, 'derived': derived, 'weather_medians': medians, 'metadata': metadata}


def transform(grid, calendar, info, pp):
    frame = pd.concat([grid[['LOAD', 'shamsi_date', 'row_present']], calendar, info], axis=1)
    for column in WEATHER:
        frame[column + '_missing'] = grid[column].isna().astype(float)
    frame[WEATHER] = pp['weather'].transform(grid[WEATHER].fillna(pp['weather_medians']).to_numpy())
    quality = pp['derived'].transform(np.column_stack([np.log1p(info.causal_gap_age_raw), info.recent_observed_fraction_raw]))
    frame['causal_gap_age'], frame['recent_observed_fraction'] = quality[:, 0], quality[:, 1]
    frame['context_scaled'] = pp['load'].transform(info[['load_context']].to_numpy()).ravel()
    frame['target_scaled'] = np.nan
    obs = grid.LOAD.notna()
    frame.loc[obs, 'target_scaled'] = pp['load'].transform(grid.loc[obs, ['LOAD']].to_numpy()).ravel()
    metadata = pd.concat([grid[['LOAD', 'temperature', 'humidity', 'row_present', 'shamsi_date']], calendar, info], axis=1)
    for column in WEATHER:
        metadata[column + '_raw'] = grid[column]
        metadata[column + '_observed'] = grid[column].notna()
        metadata[column + '_missing'] = grid[column].isna()
        metadata[column] = grid[column].fillna(pp['weather_medians'][column])
    metadata['context_observed_fraction'] = info.load_observed.rolling(168).mean().shift(1)
    assert np.isfinite(frame[['context_scaled'] + PAST + FUTURE].to_numpy(float)).all()
    return frame, metadata


def _registry(frame, fold, partition, bounds=None):
    start, end = map(pd.Timestamp, (bounds or FOLDS[fold])[partition])
    positions = np.flatnonzero((frame.index >= start) & (frame.index < end))
    rows = []
    for p in positions:
        t = frame.index[p]
        ctx = frame.iloc[p - 168:p]
        assert len(ctx) == 168 and ctx.index[0] == t - pd.Timedelta(hours=168)
        sources = ctx.load_source_max_position.to_numpy(int)
        assert np.all((sources == -1) | (sources < p))
        valid = sources[sources >= 0]
        rows.append({'fold': fold, 'partition': partition, 'position': int(p), 'target_timestamp': t,
                     'observed': bool(np.isfinite(frame.LOAD.iloc[p])), 'context_start': ctx.index[0],
                     'context_end_exclusive': t, 'context_observed_fraction': float(ctx.load_observed.mean()),
                     'context_imputed_hours': int(ctx.load_imputed.sum()),
                     'context_no_prior_sentinel_hours': int((sources == -1).sum()),
                     'max_context_source_timestamp': frame.index[valid.max()] if len(valid) else pd.NaT,
                     'context_newest_donor_age_max_hours': float(ctx.load_newest_donor_age_hours.max()),
                     'holiday': int(frame.holiday.iloc[p]), 'working_day': int(frame.working_day.iloc[p])})
    return pd.DataFrame(rows)


def _assemble(fold, frame, metadata, scaler, fingerprint, cache, audit, bounds=None):
    bounds = bounds or FOLDS[fold]
    arrays = {'load': frame.context_scaled.to_numpy(np.float32)[:, None],
              'past': frame[PAST].to_numpy(np.float32), 'future': frame[FUTURE].to_numpy(np.float32),
              'target_scaled': frame.target_scaled.to_numpy(np.float32)[:, None],
              'actual': frame.LOAD.to_numpy(float), 'index': frame.index}
    parts = {}
    for name in bounds:
        registry = _registry(frame, fold, name, bounds)
        positions = registry.position.to_numpy(int)
        observed = registry.observed.to_numpy(bool)
        parts[name] = {'partition': name, 'positions': positions, 'observed_positions': positions[observed],
                       'timestamps': frame.index[positions], 'actual': frame.LOAD.iloc[positions].to_numpy(float),
                       'observed': observed, 'registry': registry, 'metadata': metadata.iloc[positions],
                       'arrays': arrays, 'frame': frame, 'scaler': scaler, 'fingerprint': fingerprint}
    train_positions = parts['train']['observed_positions']
    return {'fold': fold, 'bounds': bounds, 'train_start': pd.Timestamp(bounds['train'][0]),
            'train_end': pd.Timestamp(bounds['train'][1]), 'frame': frame, 'metadata': metadata,
            'scaler': scaler, 'arrays': arrays, 'partitions': parts,
            'train_positions': train_positions, 'validation_positions': parts['validation']['positions'],
            'evaluation_positions': parts['evaluation']['positions'],
            'train_registry': parts['train']['registry'].loc[parts['train']['observed']].reset_index(drop=True),
            'fingerprint': fingerprint, 'cache': cache, 'audit': audit}



def validate_input(path):
    """Validate the contract without returning private observations to the console."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Private input CSV not found. Supply --data PATH; see docs/data-contract.md.")
    required = ['TIMESTAMP', 'LOAD', 'TEMP_HOUR', 'HUMIDITY', 'HOLIDAY', 'SHAMSI']
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise ValueError('Missing required CSV columns: ' + ', '.join(missing))
    times = pd.read_csv(path, usecols=['TIMESTAMP'], parse_dates=['TIMESTAMP']).TIMESTAMP
    if not pd.api.types.is_datetime64_any_dtype(times.dtype) or times.dt.tz is not None:
        raise ValueError('TIMESTAMP must contain timezone-naive local hourly timestamps.')
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing or not times.eq(times.dt.floor('h')).all():
        raise ValueError('TIMESTAMP must be unique, sorted, finite, and aligned to whole hours.')
    return path


def load_fold(fold, raw_path, output=None, strict_counts=True, bounds=None):
    """Rebuild I14 preprocessing from the supplied CSV; no historical caches used."""
    raw_path = validate_input(raw_path)
    bounds = bounds or FOLDS[fold]
    raw = make_raw_fold(fold, raw_path, bounds)
    info, lineage = causal_load(raw['grid'], raw['calendar'])
    scaler = fit_preprocessor(raw['grid'], info, raw['train_start'], raw['train_end'])
    frame, metadata = transform(raw['grid'], raw['calendar'], info, scaler)
    identity = {'raw_sha256': sha256(raw_path), 'code_sha256': sha256(__file__),
                'fold': fold, 'bounds': bounds, 'history_hours': 168}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    audit = {'identity': identity, 'scalers': scaler['metadata'], 'calendar': raw['calendar_metadata']}
    cache = Path(output) / 'preprocessing' / fold if output else None
    data = _assemble(fold, frame, metadata, scaler, fingerprint, cache, audit, bounds)
    counts = [int(p['observed'].sum()) for p in data['partitions'].values()]
    audit['observed_targets'] = counts
    if strict_counts and counts != EXPECTED_COUNTS[fold]:
        raise ValueError(f'Observed-target counts differ from the paper for {fold}: {counts}; use a separate exploratory configuration explicitly.')
    if cache:
        cache.mkdir(parents=True, exist_ok=False)
        _write_json(cache / 'audit.json', audit)
        _write_json(cache / 'load_lineage.json', {'rows': lineage})
        frame.to_parquet(cache / 'frame.parquet')
        metadata.to_parquet(cache / 'metadata.parquet')
    return data


def inverse_load(values, data):
    scale = data['scaler']['load']
    return np.asarray(values) * float(scale.scale_[0]) + float(scale.mean_[0])

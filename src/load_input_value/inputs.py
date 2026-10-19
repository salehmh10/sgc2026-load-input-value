"""Name-based, shape-preserving seven-arm inputs and observed-history weather proxy."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .protocol import ARMS
from .features import F2, BOUNDS, derived, window_temperatures, cv_design

PAST = ['temperature', 'humidity', 'temperature_missing', 'humidity_missing',
        'load_observed', 'load_imputed', 'causal_gap_age', 'recent_observed_fraction', 'long_gap_flag']
CALENDAR = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'hour_of_week_sin', 'hour_of_week_cos'] + [
    f'shamsi_month_{month:02d}' for month in range(1, 13)] + ['working_day', 'holiday', 'pre_holiday', 'post_holiday']
FUTURE = ['temperature', 'humidity', 'temperature_missing', 'humidity_missing'] + CALENDAR
FUTURE_F2 = FUTURE + F2
GROUP_NAMES = {
    'LOAD': ['LOAD'] + PAST[4:], 'calendar': CALENDAR,
    'temperature': ['temperature', 'temperature_missing'],
    'humidity': ['humidity', 'humidity_missing'], 'derived': F2,
}
ARM_MASKS = {
    'FULL': [], 'NO_DERIVED': F2, 'NO_TEMP': GROUP_NAMES['temperature'] + F2,
    'NO_HUMIDITY': GROUP_NAMES['humidity'], 'NO_CALENDAR': CALENDAR,
    'NO_WEATHER': GROUP_NAMES['temperature'] + GROUP_NAMES['humidity'] + F2,
    'FULL_PROXY': [],
}


def group_slots():
    history = ['LOAD'] + PAST + FUTURE_F2
    groups = {group: {'history': [i for i, feature in enumerate(history) if feature in names],
                      'target': [i for i, feature in enumerate(FUTURE_F2) if feature in names]}
              for group, names in GROUP_NAMES.items()}
    assert sorted(i for slots in groups.values() for i in slots['history']) == list(range(42))
    assert sorted(i for slots in groups.values() for i in slots['target']) == list(range(32))
    return groups


def mask_slots(arm):
    names = set(ARM_MASKS[arm])
    return {'past': [i for i, name in enumerate(PAST) if name in names],
            'historic_future': [i for i, name in enumerate(FUTURE_F2) if name in names],
            'target_future': [i for i, name in enumerate(FUTURE_F2) if name in names]}


def train_references(data, params):
    start, end = map(pd.Timestamp, data['bounds']['train'])
    frame = data['frame']
    train = frame.loc[(frame.index >= start) & (frame.index < end)]
    past = train[PAST].median().to_numpy(np.float32)
    future = np.r_[train[FUTURE].median().to_numpy(np.float32),
                   np.asarray(params['scaler']['median_scaled'], np.float32)].astype(np.float32)
    return {'past': past, 'future': future}


def proxy_lineage(data, positions, physical_temperature_delta=0.):
    """Independent donors; no actual weather or missingness at target is accessed."""
    index = data['frame'].index
    rows = []
    medians = data['scaler']['weather_medians']
    for position in np.asarray(positions, np.int64):
        assert 168 <= position < len(index)
        record = {'position': int(position), 'timestamp': index[position]}
        for variable in ('temperature', 'humidity'):
            past = data['metadata'][variable + '_raw'].iloc[position - 168:position].to_numpy(float)
            observed = np.flatnonzero(np.isfinite(past))
            if len(observed):
                local = int(observed[-1])
                donor = int(position - 168 + local)
                value = float(past[local])
                if variable == 'temperature':
                    value += float(physical_temperature_delta)
                record.update({variable: value, variable + '_donor_position': donor,
                               variable + '_donor_timestamp': index[donor], variable + '_age_hours': int(position - donor),
                               variable + '_fallback': False, variable + '_missing': 0.})
            else:
                record.update({variable: float(medians[variable]), variable + '_donor_position': -1,
                               variable + '_donor_timestamp': pd.NaT, variable + '_age_hours': np.nan,
                               variable + '_fallback': True, variable + '_missing': 1.})
        rows.append(record)
    return pd.DataFrame(rows)


class ModelInputRows:
    """No target LOAD is read by inputs(); arrays have fixed native Darts layout."""
    def __init__(self, data, positions, arm='FULL', params=None, physical_temperature_delta=0.):
        assert arm in ARMS
        self.data, self.a, self.arm = data, data['arrays'], arm
        self.positions = np.asarray(positions, np.int64)
        assert self.positions.ndim == 1 and np.all(self.positions >= 168)
        self.params = data['params'] if params is None else params
        self.references = train_references(data, self.params)
        self.slots = mask_slots(arm)
        self.temperature_delta = float(physical_temperature_delta)
        temperatures = window_temperatures(data, self.positions) + self.temperature_delta
        self.proxy = None
        if arm == 'FULL_PROXY':
            self.proxy = proxy_lineage(data, self.positions, self.temperature_delta)
            temperatures[:, -1] = self.proxy.temperature.to_numpy(float)
            weather = self.proxy[['temperature', 'humidity']].to_numpy(float)
            self.target_weather = data['scaler']['weather'].transform(weather).astype(np.float32)
        self.dh, self.dt = derived(data, self.positions, params=self.params, physical_temperature=temperatures)

    def __len__(self):
        return len(self.positions)

    def inputs(self, p, i=None):
        if i is None:
            matches = np.flatnonzero(self.positions == int(p))
            assert len(matches) == 1
            i = int(matches[0])
        assert int(self.positions[i]) == int(p)
        a = self.a
        pt = a['load'][p - 168:p].copy()
        pc = a['past'][p - 168:p].copy()
        hf = np.concatenate([a['future'][p - 168:p], self.dh[i]], axis=1)
        ff = np.concatenate([a['future'][p:p + 1], self.dt[i]], axis=1)
        if self.temperature_delta:
            delta = self.temperature_delta / float(self.data['scaler']['weather'].scale_[0])
            pc[:, PAST.index('temperature')] += delta
            hf[:, FUTURE_F2.index('temperature')] += delta
            ff[:, FUTURE_F2.index('temperature')] += delta
        if self.arm == 'FULL_PROXY':
            for j, variable in enumerate(('temperature', 'humidity')):
                ff[:, FUTURE_F2.index(variable)] = self.target_weather[i, j]
                ff[:, FUTURE_F2.index(variable + '_missing')] = self.proxy[variable + '_missing'].iloc[i]
        if self.slots['past']:
            pc[:, self.slots['past']] = self.references['past'][self.slots['past']]
        if self.slots['historic_future']:
            hf[:, self.slots['historic_future']] = self.references['future'][self.slots['historic_future']]
        if self.slots['target_future']:
            ff[:, self.slots['target_future']] = self.references['future'][self.slots['target_future']]
        assert pt.shape == (168, 1) and pc.shape == (168, 9)
        assert hf.shape == (168, 32) and ff.shape == (1, 32)
        assert all(np.isfinite(value).all() for value in (pt, pc, hf, ff))
        return pt, pc, hf, ff


def build_model_inputs(data, positions, arm='FULL', params=None, physical_temperature_delta=0.):
    rows = ModelInputRows(data, positions, arm, params, physical_temperature_delta)
    history = np.empty((len(rows), 168, 42), np.float32)
    target = np.empty((len(rows), 1, 32), np.float32)
    for i, position in enumerate(rows.positions):
        pt, pc, hf, ff = rows.inputs(int(position), i)
        history[i] = np.concatenate([pt, pc, hf], axis=1)
        target[i] = ff
    return history, target


def rebuild_with_temperature_shift(data, positions, arm, delta, params=None):
    return build_model_inputs(data, positions, arm, params, physical_temperature_delta=delta)

def build_matrix(data, partition, arm='FULL', batch_size=256, observed_only=None):
    """Exact chronological C-order flatten: 168 x 42 then 1 x 32 = 7088."""
    observed_only = partition == 'train' if observed_only is None else observed_only
    part = data['partitions'][partition]
    positions = part['observed_positions'] if observed_only else part['positions']
    matrix = np.empty((len(positions), 7088), np.float32)
    for start in range(0, len(positions), batch_size):
        subset = positions[start:start + batch_size]
        history, target = build_model_inputs(data, subset, arm)
        matrix[start:start + len(subset), :7056] = history.reshape(len(subset), 7056)
        matrix[start:start + len(subset), 7056:] = target.reshape(len(subset), 32)
    actual = data['arrays']['actual'][positions]
    meta = pd.DataFrame({'timestamp': data['frame'].index[positions], 'actual': actual,
                         'observed': np.isfinite(actual), 'position': positions})
    return matrix, meta

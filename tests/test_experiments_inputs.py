import copy
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
from load_input_value.data import causal_load, fit_preprocessor, transform, _assemble, CALENDAR
from load_input_value.features import fit_parameters, raw_features
from load_input_value.inputs import build_model_inputs, build_matrix, ARM_MASKS, mask_slots, train_references


@pytest.fixture
def data(tmp_path):
    index = pd.date_range('2025-01-01', periods=288, freq='h', name='timestamp')
    t = np.arange(len(index))
    grid = pd.DataFrame({'LOAD': 100. + t / 10, 'temperature': 18 + 12 * np.sin(t / 20),
                         'humidity': 50 + 5 * np.sin(t), 'shamsi_date': 14031012,
                         'row_present': True}, index=index)
    grid.loc[index[[0, 1, 40, 180, 240]], 'LOAD'] = np.nan
    grid.loc[index[[5, 80, 220]], 'temperature'] = np.nan
    calendar = pd.DataFrame(0., index=index, columns=CALENDAR)
    calendar['working_day'] = 1.
    info, _ = causal_load(grid, calendar)
    bounds = {'train': ['2025-01-08', '2025-01-10'], 'validation': ['2025-01-10', '2025-01-11'],
              'evaluation': ['2025-01-11', '2025-01-13']}
    scaler = fit_preprocessor(grid, info, *bounds['train'])
    frame, meta = transform(grid, calendar, info, scaler)
    result = _assemble('warm_late_summer', frame, meta, scaler, 'synthetic', None, {}, bounds)
    fit_parameters(result, tmp_path / 'features', {'heat_threshold': 18., 'cool_threshold': 22.})
    return result


def test_causal_donors_never_use_reconstructions_or_future():
    index = pd.date_range('2025-01-01', periods=400, freq='h')
    values = np.full(400, np.nan)
    values[[10, 34, 178, 200]] = [2., 4., 8., 100000.]
    grid = pd.DataFrame({'LOAD': values}, index=index)
    calendar = pd.DataFrame({'working_day': np.ones(400)}, index=index)
    info, lineage = causal_load(grid, calendar)
    assert info.load_context.iloc[0] == 0 and info.load_source_max_position.iloc[0] == -1
    assert info.load_context.iloc[346] == 5.  # observed weekly donors at178 and10
    for row in lineage:
        for timestamp in row['donor_timestamps']:
            assert pd.Timestamp(timestamp) < pd.Timestamp(row['context_timestamp'])
            assert np.isfinite(grid.loc[timestamp, 'LOAD'])
    before = info.load_context.iloc[:200].copy()
    grid.iloc[200:, 0] = -99999.
    changed, _ = causal_load(grid, calendar)
    np.testing.assert_array_equal(before, changed.load_context.iloc[:200])


def test_current_and_future_load_cannot_enter_inputs(data):
    position = data['partitions']['evaluation']['positions'][0]
    original = build_model_inputs(data, [position])
    changed = copy.deepcopy(data)
    changed['arrays']['load'][position:] = 999999.
    changed['arrays']['actual'][position:] = 999999.
    changed['arrays']['target_scaled'][position:] = 999999.
    changed['frame'].iloc[position:, changed['frame'].columns.get_loc('LOAD')] = 999999.
    for actual, expected in zip(build_model_inputs(changed, [position]), original):
        np.testing.assert_array_equal(actual, expected)


def test_flatten_order_and_mask_duplicate_paths(data):
    matrix, meta = build_matrix(data, 'validation')
    h, f = build_model_inputs(data, meta.position.to_numpy(int))
    np.testing.assert_array_equal(matrix[:, :7056], h.reshape(len(h), -1))
    np.testing.assert_array_equal(matrix[:, 7056:], f.reshape(len(f), -1))
    refs = train_references(data, data['params'])
    for arm in ARM_MASKS:
        history, target = build_model_inputs(data, meta.position.to_numpy(int), arm)
        slots = mask_slots(arm)
        assert history.shape[1:] == (168, 42) and target.shape[1:] == (1, 32)
        for slot in slots['past']:
            assert np.all(history[:, :, 1 + slot] == refs['past'][slot])
        for slot in slots['historic_future']:
            assert np.all(history[:, :, 10 + slot] == refs['future'][slot])
        for slot in slots['target_future']:
            assert np.all(target[:, :, slot] == refs['future'][slot])


def test_proxy_does_not_use_target_weather_or_missingness(data):
    position = data['partitions']['evaluation']['positions'][0]
    original = build_model_inputs(data, [position], 'FULL_PROXY')
    changed = copy.deepcopy(data)
    changed['arrays']['future'][position, :4] = [9999., -9999., 1., 1.]
    for col in ['temperature', 'temperature_raw', 'humidity', 'humidity_raw']:
        changed['metadata'].iloc[position, changed['metadata'].columns.get_loc(col)] = 9999.
    for actual, expected in zip(build_model_inputs(changed, [position], 'FULL_PROXY'), original):
        np.testing.assert_array_equal(actual, expected)


def test_thermal_accumulation_excludes_current_and_resets_window():
    spec = {'heat_threshold': 18., 'cool_threshold': 22.}
    temp = np.array([[10., 12., 30., 14.]])
    raw = raw_features(temp, spec)
    np.testing.assert_array_equal(raw[0, :, 2], [0., 8., 14., 14.])
    np.testing.assert_array_equal(raw[0, :, 3], [0., 0., 0., 8.])
    assert np.all(raw[:, 0, 2:] == 0.)


def test_feature_scaler_excludes_pretrain_but_retains_occurrences(data):
    from load_input_value.features import window_indices, window_temperatures
    positions = data['train_positions']
    index = window_indices(data, positions)
    raw = raw_features(window_temperatures(data, positions), data['params'])
    times = data['frame'].index.to_numpy()[index]
    keep = (times >= np.datetime64('2025-01-08')) & (times < np.datetime64('2025-01-10'))
    fit = raw[keep]
    np.testing.assert_allclose(data['params']['scaler']['mean'], fit.mean(axis=0), rtol=0, atol=1e-12)
    assert data['params']['scaler']['n_samples_seen'] == keep.sum()


def test_ridge_centered_solver_matches_normal_equations():
    from load_input_value.baselines import ridge_fit, ridge_statistics
    X = np.array([[1., 5.], [2., 1.], [7., 3.], [8., 10.]], np.float32)
    y = np.array([1., 3., 2., 6.])
    (coef, intercept), _ = ridge_fit(ridge_statistics(X, y), 10.)
    centered = X.astype(float) - X.mean(axis=0)
    expected = np.linalg.solve(centered.T @ centered + 10 * np.eye(2), centered.T @ (y - y.mean()))
    np.testing.assert_allclose(coef, expected, rtol=1e-12)
    assert intercept == pytest.approx(y.mean() - X.mean(axis=0) @ expected)


def test_float32_inverse_scaling_survives_json_parameter_roundtrip():
    import json
    from sklearn.preprocessing import StandardScaler
    from load_input_value.training import inverse_scaled_load
    scaler = StandardScaler().fit(np.array([[12345.67], [32200.87], [54400.123]]))
    values = np.random.default_rng(500).normal(size=2000).astype(np.float32)
    parameters = json.loads(json.dumps({'scale': float(scaler.scale_[0]), 'mean': float(scaler.mean_[0])}))
    result = inverse_scaled_load(values, **parameters)
    expected = scaler.inverse_transform(values[:, None]).ravel()
    np.testing.assert_array_equal(result, expected)


def test_train_scalers_do_not_fit_context_or_future(data):
    grid = data['metadata'][['LOAD', 'temperature_raw', 'humidity_raw']].rename(
        columns={'temperature_raw': 'temperature', 'humidity_raw': 'humidity'})
    info = data['frame']
    original = fit_preprocessor(grid, info, *data['bounds']['train'])
    changed = grid.copy()
    outside = (grid.index < data['train_start']) | (grid.index >= data['train_end'])
    changed.loc[outside] = 999999.
    other = fit_preprocessor(changed, info, *data['bounds']['train'])
    for name in ('load', 'weather', 'derived'):
        np.testing.assert_array_equal(original[name].mean_, other[name].mean_)
        np.testing.assert_array_equal(original[name].scale_, other[name].scale_)
    assert original['weather_medians'] == other['weather_medians']


def test_proxy_fallback_uses_train_median_and_own_flag(data):
    from load_input_value.inputs import proxy_lineage
    position = data['partitions']['evaluation']['positions'][0]
    for variable in ('temperature', 'humidity'):
        data['metadata'].iloc[position - 168:position, data['metadata'].columns.get_loc(variable + '_raw')] = np.nan
    proxy = proxy_lineage(data, [position]).iloc[0]
    for variable in ('temperature', 'humidity'):
        assert proxy[variable] == data['scaler']['weather_medians'][variable]
        assert proxy[variable + '_donor_position'] == -1
        assert proxy[variable + '_missing'] == 1.

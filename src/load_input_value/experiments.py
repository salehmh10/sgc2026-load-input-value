"""Explicit private-data execution and a small, clearly synthetic model smoke test."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_fold, validate_input, gregorian_to_jalali
from .features import fit_parameters
from .inputs import build_matrix, build_model_inputs
from .io import write_json, sha256, utc
from .evaluation import metrics, add_or_check_ensembles, tables_from_predictions, daily_metrics


def _config(path, synthetic=False):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported experiment configuration schema.')
    if not synthetic:
        from .protocol import PAPER_CONFIG_SHA256
        scientific = {key: value for key, value in config.items() if key != 'label'}
        digest = hashlib.sha256(json.dumps(scientific, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != PAPER_CONFIG_SHA256:
            raise ValueError('Scientific configuration differs from the frozen paper protocol. This runner does not silently accept unsupported method changes; use the supplied configs/paper.json.')
    return config


def _prepare_output(output, config, synthetic):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory must be new or empty; previous runs are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(output / 'cache')
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[name] = str(output / 'cache/tmp')
    (output / 'cache/tmp').mkdir(parents=True, exist_ok=True)
    import tempfile
    tempfile.tempdir = str(output / 'cache/tmp')
    write_json(output / 'execution.json', {
        'created_utc': utc(), 'synthetic_smoke_only': synthetic,
        'historical_execution': False, 'configuration': config,
        'python': sys.version.split()[0],
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scipy', 'scikit-learn', 'holidays', 'xgboost', 'darts', 'torch', 'pytorch-lightning')},
        'scientific_code': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}})
    return output


def _prediction_table(data, meta, prediction, family, arm, seed, partition):
    table = meta[['timestamp', 'actual', 'observed']].copy()
    table['prediction'] = np.asarray(prediction, float)
    table['family'], table['fold'], table['arm'] = family, data['fold'], arm
    table['seed'], table['partition'] = str(seed), partition.capitalize()
    source = data['metadata'].iloc[meta.position.to_numpy(int)]
    for field in ('context_observed_fraction', 'temperature_raw', 'humidity_raw', 'working_day', 'holiday'):
        table[field] = source[field].to_numpy()
    return table


def _reload(directory, family, Xv, Xe, mv, me, pv, pe, data):
    directory = Path(directory)
    # Keep validation/evaluation as separate prediction batches, as in I14.
    # Both sizes in the paper are multiples of neither64; joining changes the
    # second partition batching. Separate fresh-process probes preserve layout.
    reports = []
    for name, X, meta, expected in [('validation', Xv, mv, pv), ('evaluation', Xe, me, pe)]:
        np.savez(directory / 'reload_inputs.npz', X=X, timestamps=meta.timestamp.to_numpy(), expected=expected)
        write_json(directory / 'reload.json', {'family': family, 'scale': float(data['scaler']['load'].scale_[0]),
                                               'mean': float(data['scaler']['load'].mean_[0])})
        process = subprocess.run([sys.executable, '-m', 'load_input_value.cli', '_reload', str(directory)],
                                 capture_output=True, text=True, check=False)
        (directory / f'reload_{name}.log').write_text(process.stdout + '\n' + process.stderr, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'Fresh-process {family} reload failed; inspect the local reload log.')
        report = json.loads((directory / 'reload_report.json').read_text(encoding='utf-8'))
        reports.append({'partition': name, **report})
    # The larger matrices are reproducible from the saved private input contract.
    (directory / 'reload_inputs.npz').unlink()
    write_json(directory / 'reload_report.json', reports)
    return reports


def _fit_baseline(family, X, y, Xv, mv, params, seed, directory, stats=None):
    from .baselines import ridge_fit, predict
    import xgboost as xgb
    directory.mkdir(parents=True, exist_ok=False)
    if family == 'Ridge-ARX':
        model, diagnostics = ridge_fit(stats, params['alpha'])
        np.savez(directory / 'model.npz', coef=model[0], intercept=model[1])
    else:
        observed = mv.observed.to_numpy(bool)
        model = xgb.XGBRegressor(**params, random_state=int(seed))
        model.fit(X, y, eval_set=[(Xv[observed], mv.loc[observed, 'actual'].to_numpy(float))], verbose=False)
        model.save_model(directory / 'model.ubj')
        curve = model.evals_result()['validation_0']['mae']
        pd.DataFrame({'validation_mae': curve}).to_csv(directory / 'learning_curve.csv', index=False)
        diagnostics = {'best_iteration': int(model.best_iteration), 'boosting_rounds': len(curve)}
    pv = predict(family, model, Xv)
    score = float(np.abs(pv[mv.observed] - mv.loc[mv.observed, 'actual'].to_numpy(float)).mean())
    write_json(directory / 'fit.json', {'family': family, 'params': params, 'seed': seed,
                                       'validation_mae_kw': score, 'diagnostics': diagnostics})
    return model, pv, score


def _tune(data, config, output, smoke):
    from .baselines import ridge_statistics
    X, mt = build_matrix(data, 'train')
    Xv, mv = build_matrix(data, 'validation')
    y = mt.actual.to_numpy(float)
    selected = {}
    for family in ('Ridge-ARX', 'XGBoost'):
        stats = ridge_statistics(X, y) if family == 'Ridge-ARX' else None
        candidates = ([{'alpha': v} for v in config['ridge']['alphas']] if family == 'Ridge-ARX' else
                      [{**config['xgboost']['fixed'], **v} for v in config['xgboost']['candidates']])
        if smoke:
            candidates = candidates[:1]
            if family == 'XGBoost':
                candidates[0] = {**candidates[0], 'n_estimators': 2, 'early_stopping_rounds': 1}
        scores = []
        for number, params in enumerate(candidates):
            model, _, score = _fit_baseline(family, X, y, Xv, mv, params,
                                             'deterministic' if family == 'Ridge-ARX' else 42,
                                             output / 'tuning' / data['fold'] / family / f'candidate_{number:02d}', stats)
            scores.append((score, number, params))
            del model
        score, number, params = min(scores, key=lambda row: (row[0], row[1]))
        selected[family] = params
        write_json(output / 'selections' / f'{data["fold"]}__{family}.json',
                   {'candidate_index': number, 'params': params, 'observed_validation_mae_kw': score,
                    'selection_condition': 'FULL', 'refit_train_validation': False})
        del stats
        gc.collect()
    return selected


def _fit_fold(data, config, output, selected, smoke):
    from .baselines import ridge_statistics, predict
    from .training import fit_tide, predict_tide
    tables, model_records, initial_states = [], [], {}
    arms = ['FULL'] if smoke else config['conditions']
    seeds = [42] if smoke else config['seeds']
    for arm in arms:
        X, mt = build_matrix(data, 'train', arm)
        Xv, mv = build_matrix(data, 'validation', arm)
        Xe, me = build_matrix(data, 'evaluation', arm)
        y = mt.actual.to_numpy(float)
        for family in ('Ridge-ARX', 'XGBoost', 'TiDE'):
            stats = ridge_statistics(X, y) if family == 'Ridge-ARX' else None
            for seed in (['deterministic'] if family == 'Ridge-ARX' else seeds):
                print(f'Fitting {data["fold"]} {family} {arm} seed={seed}', flush=True)
                directory = output / 'models' / f'{data["fold"]}__{family}__{arm}__s{seed}'
                if family == 'TiDE':
                    observed = mv.observed.to_numpy(bool)
                    ys = data['arrays']['target_scaled'][mt.position.to_numpy(int), 0]
                    yvs = data['arrays']['target_scaled'][mv.loc[observed, 'position'].to_numpy(int), 0]
                    model, report = fit_tide(X, ys, Xv[observed], yvs, seed, directory, config['tide'], smoke)
                    key = (data['fold'], seed)
                    initial_states.setdefault(key, report['initial_state_sha256'])
                    assert initial_states[key] == report['initial_state_sha256'], 'Paired initialization mismatch'
                    scale, mean = data['scaler']['load'].scale_[0], data['scaler']['load'].mean_[0]
                    pv = predict_tide(model, Xv, mv.timestamp, scale, mean)
                    pe = predict_tide(model, Xe, me.timestamp, scale, mean)
                else:
                    model, pv, _ = _fit_baseline(family, X, y, Xv, mv, selected[family], seed, directory, stats)
                    pe = predict(family, model, Xe)
                assert np.isfinite(pv).all() and np.isfinite(pe).all()
                reload_report = _reload(directory, family, Xv, Xe, mv, me, pv, pe, data)
                for part, meta, pred in [('validation', mv, pv), ('evaluation', me, pe)]:
                    tables.append(_prediction_table(data, meta, pred, family, arm, seed, part))
                model_records.append({'family': family, 'fold': data['fold'], 'arm': arm, 'seed': seed,
                                      'reload': reload_report,
                                      'files': {str(p.relative_to(output)): sha256(p) for p in directory.rglob('*') if p.is_file()}})
                del model
                gc.collect()
            del stats
        del X, Xv, Xe
        gc.collect()
    return tables, model_records


def supplemental_tables(predictions, data_by_fold):
    """Current-paper descriptive references and TiDE-only sensitivities."""
    references, regimes = [], []
    for fold, data in data_by_fold.items():
        for part in ('validation', 'evaluation'):
            positions = data['partitions'][part]['positions']
            actual = data['arrays']['actual'][positions]
            base = pd.DataFrame({'timestamp': data['frame'].index[positions], 'actual': actual,
                                 'observed': np.isfinite(actual)})
            for lag in (1, 24, 168):
                base['prediction'] = data['frame'].load_context.to_numpy(float)[positions - lag]
                references.append({'fold': fold, 'partition': part.capitalize(), 'lag_hours': lag, **metrics(base)})
        train = data['metadata'].loc[data['train_start']:data['train_end'] - pd.Timedelta(hours=1)]
        tq = np.quantile(train.temperature_raw.dropna(), [.25, .5, .75])
        hq = train.humidity_raw.quantile(.75)
        for (family, part, arm, seed), table in predictions.loc[predictions.fold == fold].groupby(['family', 'partition', 'arm', 'seed']):
            masks = {'low_context': table.context_observed_fraction < .98,
                     'complete_context': table.context_observed_fraction == 1,
                     'working_day': table.working_day == 1, 'holiday': table.holiday == 1,
                     'high_humidity': table.humidity_raw >= hq}
            for j in range(4):
                masks[f'temperature_Q{j+1}'] = table.temperature_raw.notna() & (np.searchsorted(tq, table.temperature_raw, side='left') == j)
            for label, mask in masks.items():
                if mask.any():
                    row = metrics(table.loc[mask])
                    regimes.append({'family': family, 'fold': fold, 'partition': part, 'arm': arm,
                                    'seed': seed, 'regime': label, **row,
                                    'low_support': row['ObservedHours'] < 24 or row['ScoredDays'] < 5})
    tide = predictions.loc[(predictions.family == 'TiDE') & (predictions.partition == 'Evaluation')]
    sensitivity = []
    for fold, group in tide.groupby('fold'):
        variants = {f'leave_out_seed_{s}': group.loc[group.seed.isin([str(v) for v in (42,314,2718) if v != s])]
                    for s in (42,314,2718)}
        if fold == 'warm_late_summer':
            variants['omit_2025_09_23_through_27_posthoc'] = group.loc[
                ~group.timestamp.between('2025-09-23', '2025-09-28', inclusive='left')]
        for label, subset in variants.items():
            per_seed = {}
            for arm, arm_table in subset.groupby('arm'):
                if label.startswith('leave_out'):
                    numeric = arm_table.groupby('timestamp', sort=True).agg(
                        actual=('actual', 'first'), observed=('observed', 'first'), prediction=('prediction', 'mean')).reset_index()
                    per_seed[(arm, 'two_seed_mean')] = numeric
                else:
                    for seed, table in arm_table.groupby('seed'):
                        per_seed[(arm, seed)] = table
            for (arm, seed), table in per_seed.items():
                ref = per_seed[('FULL', seed)]
                d = daily_metrics(table).set_index('date').DailyWAPE - daily_metrics(ref).set_index('date').DailyWAPE
                sensitivity.append({'family': 'TiDE', 'fold': fold, 'sensitivity': label, 'arm': arm, 'seed': seed,
                                    'posthoc_no_refit': True, **metrics(table),
                                    'delta_percentage_points': d.mean()})
    return {'lag_references': pd.DataFrame(references), 'regime_metrics': pd.DataFrame(regimes),
            'tide_sensitivity': pd.DataFrame(sensitivity)}


def run(data_path, output, config_path, smoke=False):
    config = _config(config_path, synthetic=smoke)
    validate_input(data_path)
    output = _prepare_output(output, config, smoke)
    tables, records, data_by_fold = [], [], {}
    try:
        with threadpool_limits(limits=4):
            for fold, bounds in config['folds'].items():
                data = load_fold(fold, data_path, output, config['strict_observed_counts'], bounds)
                fit_parameters(data, output / 'features' / fold,
                               selection={'heat_threshold': 18., 'cool_threshold': 22., 'synthetic_override': True} if smoke else None)
                for arm in config['conditions']:
                    h, t = build_model_inputs(data, data['train_positions'][:2], arm)
                    assert h.shape == (2, 168, 42) and t.shape == (2, 1, 32)
                selected = _tune(data, config, output, smoke)
                new_tables, new_records = _fit_fold(data, config, output, selected, smoke)
                tables.extend(new_tables)
                records.extend(new_records)
                data_by_fold[fold] = data
        # No Evaluation metric is computed until every requested fitted model and
        # its fresh-process reload evidence is hashed in the frozen manifest.
        write_json(output / 'models_frozen.json', {'created_utc': utc(), 'synthetic_smoke_only': smoke, 'models': records})
        predictions = pd.concat(tables, ignore_index=True)
        if smoke:
            result_tables = {'smoke_metrics': pd.DataFrame([
                dict(zip(['family', 'partition'], keys), **metrics(table))
                for keys, table in predictions.groupby(['family', 'partition'])])}
        else:
            predictions = add_or_check_ensembles(predictions)
            result_tables = tables_from_predictions(predictions)
            result_tables.update(supplemental_tables(predictions, data_by_fold))
        predictions.to_parquet(output / 'private_predictions.parquet', index=False)
        (output / 'metrics').mkdir()
        for name, table in result_tables.items():
            table.to_csv(output / 'metrics' / f'{name}.csv', index=False)
        result = {'completed': True, 'synthetic_smoke_only': smoke, 'full_retraining': not smoke,
                  'models_fitted': len(records), 'prediction_rows': len(predictions),
                  'model_lock_sha256': sha256(output / 'models_frozen.json')}
        write_json(output / 'completed.json', result)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', {'exception_type': type(error).__name__, 'message': str(error),
                                            'created_utc': utc(), 'partial_outputs_preserved': True})
        raise


def smoke(output, config_path):
    """Synthetic data and reduced budget: validates plumbing, never paper findings."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Smoke output directory must be new or empty.')
    output.mkdir(parents=True, exist_ok=True)
    index = pd.date_range('2025-01-01', periods=168 + 48 + 24 + 24, freq='h')
    t = np.arange(len(index))
    temperature = 18 + 12 * np.sin(2 * np.pi * t / 24)
    frame = pd.DataFrame({'TIMESTAMP': index, 'LOAD': 1000 + 100 * np.sin(2 * np.pi * t / 24) + 5 * np.maximum(temperature - 22, 0),
                          'TEMP_HOUR': temperature, 'HUMIDITY': 50 + 10 * np.cos(2 * np.pi * t / 24),
                          'HOLIDAY': (index.dayofweek == 4).astype(int),
                          'SHAMSI': [y * 10000 + m * 100 + d for y, m, d in [gregorian_to_jalali(v.year, v.month, v.day) for v in index]]})
    frame.loc[[0, 1, 90, 180, 220], 'LOAD'] = np.nan
    frame.loc[[10, 80, 150, 240], 'TEMP_HOUR'] = np.nan
    frame.loc[[5, 40, 100, 230], 'HUMIDITY'] = np.nan
    data_path = output / 'SYNTHETIC_NOT_PAPER_DATA.csv'
    frame.to_csv(data_path, index=False)
    config = _config(config_path)
    config['label'] = 'synthetic-smoke-NOT-paper-reproduction'
    config['strict_observed_counts'] = False
    config['folds'] = {'warm_late_summer': {'train': ['2025-01-08', '2025-01-10'],
                                         'validation': ['2025-01-10', '2025-01-11'],
                                         'evaluation': ['2025-01-11', '2025-01-12']}}
    synthetic_config = output / 'synthetic-config.json'
    write_json(synthetic_config, config)
    return run(data_path, output / 'execution', synthetic_config, smoke=True)

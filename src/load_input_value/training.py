"""Faithful model fitting and fixed-weight inference; imported only on demand."""
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .io import write_json


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cap_threads():
    import torch
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def inverse_scaled_load(values, scale, mean):
    """Preserve StandardScaler float32 replay after numeric parameters cross JSON."""
    result = np.asarray(values).copy()
    result *= np.float64(scale)
    result += np.float64(mean)
    return result


def tide_datasets():
    """Construct Darts dataset types lazily; matrix order is the I14 native order."""
    from darts import TimeSeries
    from darts.utils.data import TorchTrainingDataset, TorchInferenceDataset

    class TrainRows(TorchTrainingDataset):
        def __init__(self, X, y):
            super().__init__()
            self.X, self.y = X, np.asarray(y, np.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            h = self.X[i, :7056].reshape(168, 42)
            f = self.X[i, 7056:].reshape(1, 32)
            return h[:, :1], h[:, 1:10], h[:, 10:], f, None, None, self.y[i:i+1, None]

    class InferenceRows(TorchInferenceDataset):
        def __init__(self, X, times):
            super().__init__()
            self.X, self.times = X, pd.DatetimeIndex(times)
            self.schema = TimeSeries.from_times_and_values(
                pd.date_range('2000-01-01', periods=3, freq='h'),
                np.zeros((3, 1), np.float32), columns=['LOAD']).schema(copy=False)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            h = self.X[i, :7056].reshape(168, 42)
            f = self.X[i, 7056:].reshape(1, 32)
            return h[:, :1], h[:, 1:10], None, h[:, 10:], f, None, self.schema, self.times[i]

    return TrainRows, InferenceRows


def predict_tide(model, X, times, scale, mean):
    import torch
    _, InferenceRows = tide_datasets()
    before = state_hash(model.model)
    with torch.inference_mode():
        values, _, returned_times = model.predict_from_dataset(
            n=1, dataset=InferenceRows(X, times), batch_size=64, num_samples=1,
            verbose=False, values_only=True,
            dataloader_kwargs={'num_workers': 0, 'pin_memory': False})
    assert list(returned_times) == list(pd.DatetimeIndex(times))
    values = np.asarray(values)
    assert values.shape == (len(X), 1, 1, 1)
    # StandardScaler.inverse_transform preserves float32 and performs in-place
    # multiplication followed by addition; preserve that original rounding.
    prediction = inverse_scaled_load(values[:, 0, 0, 0], scale, mean)
    assert np.isfinite(prediction).all() and state_hash(model.model) == before
    return prediction


def fit_tide(X, y_scaled, Xv, yv_scaled, seed, directory, protocol, smoke=False):
    import torch
    from pytorch_lightning.callbacks import Callback, EarlyStopping
    from .tide import FixedTiDEModel, assert_fixed, EpochEvidence

    cap_threads()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = protocol['training']
    epochs = 1 if smoke else cfg['max_epochs']

    evidence = EpochEvidence(directory, cfg['max_fit_seconds'])
    early = EarlyStopping(monitor='val_loss', mode='min', patience=cfg['early_stopping']['patience'],
                          min_delta=cfg['early_stopping']['min_delta'], check_on_train_epoch_end=False)
    name = 'fit'
    model = FixedTiDEModel(
        input_chunk_length=168, output_chunk_length=1, output_chunk_shift=0,
        n_epochs=epochs, batch_size=64, random_state=seed,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        loss_fn=torch.nn.L1Loss(), use_reversible_instance_norm=False,
        save_checkpoints=True, log_tensorboard=False, model_name=name,
        work_dir=str(directory / 'training'), force_reset=False,
        pl_trainer_kwargs={'accelerator': 'cpu', 'devices': 1, 'max_epochs': epochs,
                          'logger': False, 'enable_checkpointing': True, 'enable_progress_bar': False,
                          'enable_model_summary': False, 'num_sanity_val_steps': 0, 'deterministic': True,
                          'gradient_clip_val': 1., 'callbacks': [evidence, early]}, **protocol['architecture'])
    TrainRows, _ = tide_datasets()
    model.fit_from_dataset(TrainRows(X, y_scaled), val_dataset=TrainRows(Xv, yv_scaled),
                           verbose=False, load_best=False,
                           dataloader_kwargs={'num_workers': 0, 'pin_memory': False, 'shuffle': False})
    checkpoint = Path(model.trainer.checkpoint_callback.best_model_path)
    best = FixedTiDEModel.load_from_checkpoint(name, work_dir=str(directory / 'training'),
                                              file_name=checkpoint.name, best=True,
                                              map_location='cpu', strict=True)
    assert_fixed(best.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for key, value in best.model.state_dict().items():
        assert torch.equal(value.cpu(), state['state_dict'][key].cpu())
    best.save(str(directory / 'model.pt'), clean=True)
    report = {'initial_state_sha256': evidence.initial_hash, 'best_state_sha256': state_hash(best.model),
              'best_epoch': int(state['epoch']) + 1, 'stopped_epoch': len(evidence.rows),
              'time_limited': evidence.limited, 'parameter_count': 477491, 'smoke': smoke}
    write_json(directory / 'fit.json', report)
    return best, report


def reload_check(directory):
    """Runs in a fresh Python process against only newly generated local artifacts."""
    import xgboost as xgb
    directory = Path(directory)
    import json
    spec = json.loads((directory / 'reload.json').read_text(encoding='utf-8'))
    probe = np.load(directory / 'reload_inputs.npz', allow_pickle=False)
    X, expected = probe['X'], probe['expected']
    if spec['family'] == 'Ridge-ARX':
        from .baselines import predict
        saved = np.load(directory / 'model.npz', allow_pickle=False)
        actual = predict('Ridge-ARX', (saved['coef'], saved['intercept'].item()), X)
    elif spec['family'] == 'XGBoost':
        model = xgb.XGBRegressor()
        model.load_model(directory / 'model.ubj')
        actual = model.predict(X)
    else:
        from .tide import FixedTiDEModel, assert_fixed
        cap_threads()
        model = FixedTiDEModel.load(str(directory / 'model.pt'), map_location='cpu', strict=True,
                                    pl_trainer_kwargs={'accelerator': 'cpu', 'devices': 1, 'logger': False,
                                                       'enable_progress_bar': False, 'enable_model_summary': False})
        assert_fixed(model.model)
        actual = predict_tide(model, X, probe['timestamps'], spec['scale'], spec['mean'])
    tolerance = .001 if spec['family'] == 'TiDE' else 1e-6
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-12)
    report = {'passed': True, 'rows': len(X), 'max_absolute_difference_kw': float(np.max(abs(actual - expected))),
              'absolute_tolerance_kw': tolerance, 'fresh_process': True}
    write_json(directory / 'reload_report.json', report)
    return report

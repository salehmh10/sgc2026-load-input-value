"""I14 local TiDE correction; adapted from audited I13, without monkey patching."""
import torch
from darts.models import TiDEModel
from darts.models.forecasting.tide_model import _TideModule
from pytorch_lightning.callbacks import Callback
import pandas as pd
import time

ARCHITECTURE = dict(hidden_size=128, num_encoder_layers=2, num_decoder_layers=1,
                    decoder_output_dim=16, temporal_width_past=4, temporal_width_future=4,
                    temporal_hidden_size_past=16, temporal_hidden_size_future=16,
                    temporal_decoder_hidden=32, use_layer_norm=True, dropout=0.1)


class FixedTideModule(_TideModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layer = self.temporal_decoder.layer_norm
        assert isinstance(layer, torch.nn.LayerNorm) and tuple(layer.normalized_shape) == (1,)
        self.temporal_decoder.layer_norm = torch.nn.Identity()
        assert self.output_chunk_length == 1 and self.input_chunk_length == 168


class FixedTiDEModel(TiDEModel):
    def _create_model(self, train_sample):
        pt, pc, hf, ff, static, ft = train_sample
        assert pt.shape == (168, 1) and pc.shape == (168, 9)
        assert hf.shape == (168, 32) and ff.shape == (1, 32) and ft.shape == (1, 1)
        return FixedTideModule(
            input_dim=pt.shape[1] + pc.shape[1] + hf.shape[1],
            output_dim=ft.shape[1], future_cov_dim=ff.shape[1],
            static_cov_dim=static.shape[0] * static.shape[1] if static is not None else 0,
            nr_params=1 if self.likelihood is None else self.likelihood.num_parameters,
            num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim, hidden_size=self.hidden_size,
            temporal_width_past=self.temporal_width_past, temporal_width_future=self.temporal_width_future,
            temporal_hidden_size_past=self.temporal_hidden_size_past,
            temporal_hidden_size_future=self.temporal_hidden_size_future,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm, dropout=self.dropout, **self.pl_module_params)


def assert_fixed(module):
    assert isinstance(module, FixedTideModule)
    assert isinstance(module.temporal_decoder.layer_norm, torch.nn.Identity)
    assert module.input_chunk_length == 168 and module.output_chunk_length == 1
    assert sum(isinstance(layer, torch.nn.LayerNorm) for layer in module.modules()) == 5
    assert sum(parameter.numel() for parameter in module.parameters()) == 477491
    assert not any(key.startswith('temporal_decoder.layer_norm.') for key in module.state_dict())


class EpochEvidence(Callback):
    """Top-level callback so Darts can serialize the complete model parameters."""
    def __init__(self, directory, time_limit):
        self.directory, self.time_limit = directory, time_limit
        self.rows, self.limited = [], False

    def on_fit_start(self, trainer, module):
        from .training import state_hash
        self.started = time.monotonic()
        assert_fixed(module)
        self.initial_hash = state_hash(module)

    def on_train_epoch_start(self, trainer, module):
        self.total, self.count = 0., 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        n = int(batch[-1].shape[0])
        self.total += float(loss.detach()) * n
        self.count += n
        if time.monotonic() - self.started >= self.time_limit:
            self.limited = True
            trainer.should_stop = True

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        self.rows.append({'epoch': int(trainer.current_epoch) + 1,
                          'train_loss': self.total / self.count,
                          'validation_loss': float(trainer.callback_metrics['val_loss']),
                          'elapsed_seconds': time.monotonic() - self.started})
        pd.DataFrame(self.rows).to_csv(self.directory / 'learning_curve.csv', index=False)

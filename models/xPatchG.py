import torch.nn as nn

from layers.component_network import ComponentAwareNetwork
from layers.decomp import DECOMP
from layers.revin import RevIN


class Model(nn.Module):
    """Component-aware xPatch for multivariate dependency modeling.

    xPatchG keeps the xPatch decomposition idea but no longer appends a generic
    graph after forecasting. It injects variable interactions into the two
    decomposed streams differently: dynamic patch-level dependencies for the
    seasonal stream and a stable global mixer for the trend stream.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta

        dep_hidden = getattr(configs, 'dep_hidden', 64)
        dep_dropout = getattr(configs, 'dep_dropout', 0.0)
        dep_temperature = getattr(configs, 'dep_temperature', 1.0)
        dep_alpha = getattr(configs, 'dep_alpha', 0.8)
        seasonal_alpha = getattr(configs, 'seasonal_dep_alpha', dep_alpha)
        trend_alpha = getattr(configs, 'trend_dep_alpha', 0.9)
        seasonal_scale = getattr(configs, 'seasonal_dep_scale', 0.1)
        trend_scale = getattr(configs, 'trend_dep_scale', 0.1)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = ComponentAwareNetwork(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            channels=c_in,
            dep_hidden=dep_hidden,
            dep_dropout=dep_dropout,
            dep_temperature=dep_temperature,
            seasonal_alpha=seasonal_alpha,
            trend_alpha=trend_alpha,
            seasonal_scale=seasonal_scale,
            trend_scale=trend_scale,
        )

    def forward(self, x):
        # x: [Batch, Input, Channel]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x

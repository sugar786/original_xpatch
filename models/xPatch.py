import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN
from cross_correlation_embedding import CrossCorrelationEmbedding


class Model(nn.Module):
    """
    xPatch + CCE（仅趋势分支；无分支内RevIN/无门控残差）
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # ------------ 基础参数 ------------
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # ------------ 全局 RevIN(保留) ------------
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # ------------ 分解 ------------
        self.ma_type = configs.ma_type    # 'ema'/'dema'/'reg'
        alpha = configs.alpha
        beta = configs.beta
        self.decomp = DECOMP(self.ma_type, alpha, beta)

        # ------------ 仅趋势分支使用 CCE ------------
        self.use_cce_seasonal = False
        self.use_cce_trend    = True

        self.cce_kernel = getattr(configs, "cce_kernel", 3)  # 建议默认3更稳
        self.cce_alpha  = getattr(configs, "cce_alpha", 1.0)

        # 季节分支：不使用 CCE
        self.cce_seasonal = nn.Identity()

        # 趋势分支：使用 CCE
        if self.use_cce_trend:
            self.cce_trend = CrossCorrelationEmbedding(
                in_channels=c_in,
                kernel_size=self.cce_kernel,
                alpha=self.cce_alpha
            )
        else:
            self.cce_trend = nn.Identity()

        # ------------ 主干 ------------
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)

    def forward(self, x):
        # x: [B, L, C]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # 分解
        if self.ma_type == 'reg':
            seasonal_init, trend_init = x, x
        else:
            seasonal_init, trend_init = self.decomp(x)  # [B, L, C]

        # 仅趋势分支做 CCE（无分支内RevIN/无门控/无残差）
        seasonal_feat = self.cce_seasonal(seasonal_init)  # Identity
        trend_feat = self.cce_trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 双流预测
        y = self.net(seasonal_feat, trend_feat)

        if self.revin:
            y = self.revin_layer(y, 'denorm')

        return y

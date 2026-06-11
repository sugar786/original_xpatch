import torch
import torch.nn as nn


class CrossCorrelationEmbedding(nn.Module):
    """
    Conservative Gated Cross-Correlation Embedding (CG-CCE)

    Input:
        x: [B, C, T]
    Output:
        out: [B, C, T]

    Difference from the previous Gate CCE:
        Previous:
            adaptive_out = gate * correlated + (1 - gate) * residual
            out = alpha * residual + (1 - alpha) * adaptive_out

        This version:
            delta = correlated - residual
            out = residual + scale * gate * delta

    The module only learns a small residual correction instead of reconstructing
    the whole branch representation. This is safer for strong baselines such as
    xPatch, especially on Solar and short-horizon settings.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = 3,
        gate_type: str = "channel",
        reduction: int = 4,
        dropout: float = 0.1,
        init_scale: float = -5.0,
        use_depthwise: bool = False,
    ):
        super(CrossCorrelationEmbedding, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gate_type = gate_type
        self.use_depthwise = use_depthwise

        if use_depthwise and in_channels == out_channels:
            self.corr = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=in_channels,
                    bias=True,
                ),
                nn.GELU(),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.Dropout(dropout),
            )
        else:
            self.corr = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                ),
                nn.Dropout(dropout),
            )

        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        hidden = max(in_channels // reduction, 1)

        if gate_type == "scalar":
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        elif gate_type == "channel":
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        elif gate_type == "time":
            self.gate = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(
                f"Unknown gate_type: {gate_type}. Choose from ['scalar', 'channel', 'time']."
            )

        # Conservative residual scale. sigmoid(-5) is about 0.0067.
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Start close to identity: the correlation branch initially outputs
        # a small correction around the residual feature.
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Make the last conv in corr near zero so delta is initially small.
        last_conv = None
        for m in self.corr.modules():
            if isinstance(m, nn.Conv1d):
                last_conv = m
        if last_conv is not None:
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        residual = self.residual_proj(x)
        correlated = self.corr(x)

        gate = self.gate(x)
        if self.gate_type == "scalar":
            # [B, 1, 1], broadcast to [B, C, T]
            pass
        elif self.gate_type == "channel":
            # [B, C, 1], broadcast to [B, C, T]
            pass
        else:
            # [B, C, T]
            pass

        delta = correlated - residual
        scale = torch.sigmoid(self.res_scale)
        out = residual + scale * gate * delta
        return out

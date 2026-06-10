import torch
import torch.nn as nn


class CrossCorrelationEmbedding(nn.Module):
    # 
    # Cross-Correlation Embedding (CCE)
    # ---------------------------------
    # 来自 CrossLinear 模型，用于建模多变量时间序列中变量间的静态依赖关系。

    # 功能:
    #     - 通过 1D 卷积在不同变量间建立“互相关”特征；
    #     - 通过可学习系数 alpha 控制原始信号与相关性信号的融合比例。

    # 输入:
    #     x: Tensor, 形状 [batch, variables, time]

    # 输出:
    #     out: Tensor, 与输入形状相同 [batch, variables, time]

    # 参数:
    #     in_channels: 输入变量数量 (即 dec_in)
    #     out_channels: 输出变量数量 (默认与输入相同)
    #     kernel_size: 卷积核大小 (默认 3)
    #     alpha: 原始信号与相关性信号的融合系数初值 (0~1)

    # 公式:
    #     out = α * x + (1 - α) * Conv1d(x)
    # 

    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 3, alpha: float = 0.5):
        super(CrossCorrelationEmbedding, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        # 可学习融合系数 α
        self.alpha = nn.Parameter(torch.ones(1) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 前向传播:
        # x: [batch, variables, time]
         
        correlated = self.conv(x)
        out = self.alpha * x + (1 - self.alpha) * correlated
        return out

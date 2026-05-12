import math

import torch
from torch import nn


class ComponentAwareNetwork(nn.Module):
    """xPatch temporal backbone with component-specific variable modeling.

    The seasonal and trend components have different dependency structures after
    decomposition. Seasonal residuals are often local, phase-shifted, and
    sample-dependent, so they are mixed with dynamic attention over patch
    embeddings. Trend components are smoother and usually governed by more
    stable long-term co-movement, so they use a static global channel mixer
    before the MLP trend stream.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        channels,
        dep_hidden=64,
        dep_dropout=0.0,
        dep_temperature=1.0,
        seasonal_alpha=0.7,
        trend_alpha=0.9,
        seasonal_scale=0.1,
        trend_scale=0.1,
    ):
        super(ComponentAwareNetwork, self).__init__()

        if channels <= 0:
            raise ValueError('channels must be positive')
        if dep_temperature <= 0:
            raise ValueError('dep_temperature must be positive')

        self.pred_len = pred_len
        self.channels = channels
        self.dep_temperature = dep_temperature
        self.seasonal_alpha = min(max(seasonal_alpha, 0.0), 1.0)
        self.trend_alpha = min(max(trend_alpha, 0.0), 1.0)

        # Non-linear seasonal stream
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        dep_hidden = max(1, dep_hidden)
        self.seasonal_node = nn.Sequential(
            nn.Linear(self.dim, dep_hidden),
            nn.GELU(),
            nn.Dropout(dep_dropout),
        )
        self.seasonal_query = nn.Linear(dep_hidden, dep_hidden, bias=False)
        self.seasonal_key = nn.Linear(dep_hidden, dep_hidden, bias=False)
        self.seasonal_projection = nn.Linear(self.dim, self.dim)
        self.seasonal_dropout = nn.Dropout(dep_dropout)
        self.seasonal_scale = nn.Parameter(torch.tensor([seasonal_scale]))

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear trend stream
        self.trend_logits = nn.Parameter(torch.zeros(channels, channels))
        self.trend_scale = nn.Parameter(torch.tensor([trend_scale]))
        self.trend_dropout = nn.Dropout(dep_dropout)

        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        self.fc8 = nn.Linear(pred_len * 2, pred_len)

        self.register_buffer('identity', torch.eye(channels), persistent=False)

    def _seasonal_dependency(self, s):
        # s: [Batch, Channel, Patch_num, Dim]
        nodes = s.mean(dim=2)
        nodes = self.seasonal_node(nodes)
        query = self.seasonal_query(nodes)
        key = self.seasonal_key(nodes)
        scale = math.sqrt(query.shape[-1]) * self.dep_temperature
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / scale, dim=-1)
        adjacency = self.seasonal_alpha * self.identity.unsqueeze(0) + (1.0 - self.seasonal_alpha) * attention
        mixed = torch.einsum('bij,bjpd->bipd', adjacency, s)
        delta = self.seasonal_projection(mixed - s)
        return s + torch.tanh(self.seasonal_scale) * self.seasonal_dropout(delta)

    def _trend_dependency(self, t):
        # t: [Batch, Channel, Input]
        attention = torch.softmax(self.trend_logits, dim=-1)
        adjacency = self.trend_alpha * self.identity + (1.0 - self.trend_alpha) * attention
        mixed = torch.einsum('ij,bjt->bit', adjacency, t)
        delta = self.trend_dropout(mixed - t)
        return t + torch.tanh(self.trend_scale) * delta

    def forward(self, s, t):
        # s: seasonal component [Batch, Input, Channel]
        # t: trend component [Batch, Input, Channel]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]

        s = torch.reshape(s, (B*C, I))
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        if C > 1:
            s = torch.reshape(s, (B, C, self.patch_num, self.dim))
            s = self._seasonal_dependency(s)
            s = torch.reshape(s, (B*C, self.patch_num, self.dim))

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        if C > 1:
            t = self._trend_dependency(t)
        t = torch.reshape(t, (B*C, I))

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x

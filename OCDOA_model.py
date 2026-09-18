import torch
import torch.nn as nn
import torch.nn.init as init


# 通道注意力层（保持原实现）
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class OrthoBasisCNN1D(nn.Module):
    def __init__(self,input_length=120,output_dim=32):
        super().__init__()
        self.input_length = input_length
        self.fc_hidden=240


        # ========== 前置线性层（按需初始化） ==========

        self.pre_linear = nn.Linear(2 * self.input_length, self.fc_hidden)  # 无激活函数

        # 卷积层结构（保持原实现）
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )



        self.attention = ChannelAttention(256)

        # 公共层（保持不变）

        self.fc_layers = nn.Sequential(
            nn.Linear(2560, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1280, output_dim),
        )

        # 正交权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # 输入维度：[B, 2, 1, 120] → 挤压为[B, 2, 120]
        x = x.squeeze(2)
        B, C, L = x.shape  # B=批次，C=2通道，L=120长度


        x_flatten = x.reshape(B, -1)
        x_fc = self.pre_linear(x_flatten)
        x = x_fc.reshape(B, C, self.fc_hidden // C)

        # 卷积层（核心逻辑不变）
        x = self.conv_layers(x)

        x = self.attention(x)

        # 后续公共逻辑（保持不变）
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x



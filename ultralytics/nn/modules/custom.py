import torch
from torch import nn

from ultralytics.nn.modules import Conv


class EMAttention(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMAttention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class PreProcessorFold(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.fold = Conv(input_channel, output_channel)

    def forward(self, x):
        return self.fold(x)


class IceFusion(nn.Module):
    def __init__(self, c1, c2, kernels=[5, 7], reduction=8, group=1, L=32):
        super(IceFusion, self).__init__()
        channel = min(c1, c2)
        self.channel_down_conv = nn.Conv2d(c2, channel, kernel_size=1)
        self.hw_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d = max(L, channel // reduction)
        # 将全局向量降维
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x[0].size()
        # Do conv and upsample for second input
        x[1] = self.channel_down_conv(x[1])
        if x[0].size() != x[1].size():
            x[1] = self.hw_upsample(x[1])
        feats = torch.stack(x, 0)  # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U = sum(x)  # (K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S = U.mean(-1).mean(-1)
        # 降低通道数,提高计算效率: (B,C)-->(B,d)
        Z = self.fc(S)

        # 将紧凑特征Z通过K个全连接层得到K个尺度对应的通道描述符表示, 然后基于K个通道描述符计算注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # 恢复预输入相同的通道数: (B,d)-->(B,C)
            weights.append(weight.view(B, C, 1, 1))  # (B,C)-->(B,C,1,1)
        scale_weight = torch.stack(weights, 0)  # 将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight = self.softmax(scale_weight)  # 在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V = (scale_weight * feats).sum(
            0)  # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V

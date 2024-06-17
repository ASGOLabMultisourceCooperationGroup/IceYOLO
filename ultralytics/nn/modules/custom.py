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


class IceCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(IceCBAM, self).__init__()
        self.channel_attention = IceChannelAttention(channels, reduction)
        self.spatial_attention = IceSKAttention(channels, reduction=reduction)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x)
        return x


class IceChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(IceChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class IceSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(IceSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * input


class IceSKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[5, 7], reduction=8, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        # 有几个卷积核,就有几个尺度, 每个尺度对应的卷积层由Conv-bn-relu实现
        for k in kernels:
            self.convs.append(
                IceSpatialAttention(k)
            )
        # 将全局向量降维
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs = []
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  # 每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats = torch.stack(conv_outs, 0)  # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U = sum(conv_outs)  # (K,B,C,H,W)-->(B,C,H,W)
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

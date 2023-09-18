import torch
import torch.nn as nn

class LiteConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(LiteConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LiteRASPP(nn.Module):
    def __init__(self, in_channels):
        super(LiteRASPP, self).__init__()
        atrous_rates = [6, 12, 18]
        inter_channels = 64

        # 下采样模块
        self.conv1 = LiteConv(in_channels, 64, 1, stride=1, padding=0)

        # ASPP模块
        self.aspp0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LiteConv(in_channels, inter_channels, 1, stride=1, padding=0)
        )
        self.aspp1 = LiteConv(in_channels, inter_channels, 1, stride=1, padding=0)
        self.aspp2 = LiteConv(in_channels, inter_channels, 3, stride=1, padding=6, dilation=6)
        self.aspp3 = LiteConv(in_channels, inter_channels, 3, stride=1, padding=12, dilation=12)
        self.aspp4 = LiteConv(in_channels, inter_channels, 3, stride=1, padding=18, dilation=18)

        # Refinement模块，主干网
        self.project = LiteConv(5*inter_channels, inter_channels, 1, stride=1, padding=0)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.refinement = LiteConv(inter_channels+48, inter_channels, 3, stride=1, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = LiteConv(inter_channels, 256, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.aspp0(x)
        x2 = nn.functional.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.aspp1(x)
        x4 = self.aspp2(x)
        x5 = self.aspp3(x)
        x6 = self.aspp4(x)
        x_concat = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x_concat = self.project(x_concat)
        x_concat = self.up1(x_concat)
        x_cat = torch.cat([x_concat, x1], dim=1)
        x_cat = self.refinement(x_cat)
        x_cat = self.up2(x_cat)
        x_cat = self.conv2(x_cat)
        return x_cat
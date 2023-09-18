# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        # if mask is not None:
        #     ## mask:  [N, T_k] --> [h, N, T_q, T_k]
        #     mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
        #     scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        # self.v = nn.Linear(c, c, bias=False)
        self.ma = MultiHeadAttention(query_dim=c,key_dim=c,num_units=c,num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        # x=self.ma(x)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c2, num_heads, num_layers):
        super().__init__()
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1
                ),
        nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )  # 四个1x1卷积用来减小channel为原来的1/N
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1
                ),
        nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1
                ),
        nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1
                ),
        nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels*2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
                ),
        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )  # 最后的1x1卷积缩小为原来的channel
        self.tf1=TransformerBlock(in_channels//4,8,1)
        self.tf2=TransformerBlock(in_channels//4,8,1)
        self.tf3=TransformerBlock(in_channels//4,8,1)
        self.tf4=TransformerBlock(in_channels//4,8,1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.tf1(self.conv1(self.pool(x, 1))), size)
        feat2 = self.upsample(self.tf2(self.conv2(self.pool(x, 2))), size)
        feat3 = self.upsample(self.tf3(self.conv3(self.pool(x, 3))), size)
        feat4 = self.upsample(self.tf4(self.conv4(self.pool(x, 6))), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention=self.sigmoid(out)
        x=x.mul(attention)
        return x



class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes,momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.origin=nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 *inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            # BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            # BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            # BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.bn=nn.BatchNorm2d(out_planes,momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        # self.se=ChannelAttention(out_planes,8)
    def forward(self, x):
        origin=self.origin(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((origin,x0, x1, x2), 1)
        out=self.ConvLinear(out)
        # attention = self.se(out)
        # out = out.mul(attention)
        # out = self.ConvLinear(out)
        short = self.shortcut(x)
        short = self.bn(short)
        out = out * self.scale + short
        out = self.relu(out)

        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # dw: depth-wise convolution
        # pw: point-wise convolution
        # pw-linear: point-wise convolution without activation
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, hidden_channels, 3, stride, 1, groups = hidden_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # dw
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups = hidden_channels, bias = False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class Down(nn.Module):

    def __init__(self, inplanes, planes):
        super(Down, self).__init__()
        self.down=nn.Sequential(
            nn.Conv2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.down(x)
class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.satt=SelfAttention(1)
    def forward(self, x):
        z = self.squeeze(x)
        z=self.relu(z)
        z=self.satt(z)
        z = self.sigmoid(z)
        return x * z

# 通道注意力模块，和SE类似
class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, in_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x
# 空间+通道，双管齐下
class DownConv(nn.Module):

    def __init__(self, inplanes, planes):
        super(DownConv, self).__init__()
        self.down=nn.Sequential(
            nn.Conv2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.down(x)
class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        # return self.satt(x) + self.catt(x)
        return self.catt(x)



class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        # self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        # self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        # return self.gamma * out + input
        return self.gamma * out


# class SelfAttention(nn.Module):
#     "Self attention layer for `n_channels`."
#     def __init__(self, n_channels):
#         self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
#         self.gamma = nn.Parameter(torch.tensor([0.]))
#
#     def _conv(self,n_in,n_out):
#         return DownConv(n_in,n_out)
#
#     def forward(self, x):
#         #Notation from the paper.
#         size = x.size()
#         x = x.view(*size[:2],-1)
#         f,g,h = self.query(x),self.key(x),self.value(x)
#         beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
#         o = self.gamma * torch.bmm(h, beta) + x
#         return o.view(*size).contiguous()

#LXH-----------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class InPlaceUp(nn.Module):
    def __init__(self, in_channels):
        super(InPlaceUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        # InvertedResidual blocks
        self.conv2 = InvertedResidual(64, 24, 1, 1)
        self.conv3 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
        )
        self.conv4 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
        )
        self.conv5 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        self.conv6 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.conv7 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
        )
        self.conv8 = InvertedResidual(160, 320, 1, 6)
        # Last Conv
        # self.fc23 = nn.Sequential(
        #
        #     nn.Conv2d(128, num_output,
        #               kernel_size=1, stride=1, padding=0))
        # self.fc24 = nn.Sequential(
        #
        #     nn.Conv2d(128, num_output,
        #               kernel_size=1, stride=1, padding=0))
        # self.fc34 = nn.Sequential(
        #
        #     nn.Conv2d(128, num_output,
        #               kernel_size=1, stride=1, padding=0))

        self.stage2_de=nn.Sequential(
            DownConv(128, 64),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
        nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )
        self.stage3_de1=nn.Sequential(
            DownConv(256, 128),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
        nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
        )
        self.stage3_de2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage4_de1=nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage4_de2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage4_de3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.down=nn.Sequential(
            # ChannelAttention(512),
            nn.Conv2d(128, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.stage33_down=nn.Sequential(
            # ChannelAttention(256),
            nn.Conv2d(256, 64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        # self.stage35_se=ChannelAttention(256)
        self.stage43_down=nn.Sequential(
            # ChannelAttention(512),
            nn.Conv2d(512, 128,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage45_down=nn.Sequential(
            # ChannelAttention(512),
            nn.Conv2d(256, 64,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.down23=Down(128,64)
        self.down35=Down(128,64)
        self.down47 = Down(192, 64)
        self.up1=DownConv(24,64)
        self.up2=DownConv(32,128)
        self.up3=DownConv(96,256)
        self.up4=DownConv(320,512)
        self.se1 = SCse(64)
        # self.se1 = SelfAttention(64)
        # self.se2 = SelfAttention(128)
        # self.se3 = SelfAttention(256)
        self.se2 = SCse(128)
        self.se3 = SCse(256)
        self.spp = PyramidPooling(512, 256)

        self.up5 = InPlaceUp(64)
        self.up6 = InPlaceUp(64)
        self.outc = OutConv(64, 1)
        # self.selfatt=nn.Sequential(SelfAttention(512),
        #                            DownConv(512,256))
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv2(x)  # out: B * 16 * 112 * 112  256
        stage1 = self.se1(self.up1(x))

        x = self.conv4(x)  # out: B * 32 * 28 * 28    64
        stage21 = self.se2(self.up2(x))
        stage22 = self.stage2_de(stage21)
        stage23 = torch.cat((stage1, stage22), dim=1)
        stage23 = self.down23(stage23)
        x = self.conv5(x)  # out: B * 64 * 14 * 14
        x = self.conv6(x)  # out: B * 96 * 14 * 14     32
        stage31 = self.se3(self.up3(x))
        stage32 = self.stage3_de1(stage31)
        stage33 = torch.cat((stage21, stage32), dim=1)
        stage33 = self.stage33_down(stage33)
        stage34 = self.stage3_de2(stage33)
        stage35 = torch.cat((stage34, stage1), dim=1)
        stage35 = self.down35(stage35)
        x = self.conv7(x)  # out: B * 160 * 7 * 7
        x = self.conv8(x)  # out: B * 320 * 7 * 7
        stage41 = self.spp(self.up4(x))

        stage42 = self.stage4_de1(stage41)
        stage43 = torch.cat((stage42, stage31), dim=1)
        stage43 = self.stage43_down(stage43)
        stage44 = self.stage4_de2(stage43)
        stage45 = torch.cat((stage44, stage21), dim=1)
        stage45 = self.stage45_down(stage45)
        stage46 = self.stage4_de3(stage45)
        stage46 = self.down(torch.cat((stage1, stage46), 1))

        stage_LXH = torch.cat([stage46, stage35, stage23], dim=1)
        stage47 = self.down47(torch.cat([stage46,stage35, stage23], dim=1))

        xLXH = self.up5(stage47)
        xLXH = self.up6(xLXH)
        logits = self.outc(xLXH)
            # else:
            #     ret[head] = self.__getattr__(head)(stage35)
            # if head=='hm':
            #     ret[head]=ret[head]+stage1_det+stage2_det+stage3_det
            # if head=='wh':
            #     ret[head] = ret[head] +stage1_wh+stage2_wh+stage3_wh
            # if head=='reg':
            #     ret[head] = ret[head] + stage1_reg + stage2_reg + stage3_reg
        return logits

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            # for _, m in self.deconv_layers.named_modules():
            #     if isinstance(m, nn.ConvTranspose2d):
            #         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
            #         # print('=> init {}.bias as 0'.format(name))
            #         nn.init.normal_(m.weight, std=0.001)
            #         if self.deconv_with_bias:
            #             nn.init.constant_(m.bias, 0)
            #     elif isinstance(m, nn.BatchNorm2d):
            #         # print('=> init {}.weight as 1'.format(name))
            #         # print('=> init {}.bias as 0'.format(name))
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for _, m in self.stage2_de.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.stage3_de1.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.stage3_de2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.stage4_de1.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.stage4_de2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.stage4_de3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            fc_layers=[self.fc21,self.fc31]
            for f in fc_layers:
                for i, m in enumerate(f.modules()):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == 2:
                            # if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                            # else:
                            #     nn.init.normal_(m.weight, std=0.001)
                            #     nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            # url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = torch.load('/Pinchao_model/mobilenetv2_pretrain.pth')
            # print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=64):
  block_class, layers = resnet_spec[18]

  model = PoseResNet(block_class, layers, heads, head_conv=64)
  #model.init_weights(18, pretrained=True)
  return model

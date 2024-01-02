# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.ops import DeformConv2d
from ..builder import NECKS
from collections import OrderedDict
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCM(nn.Module): #AGIEN
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.score = nn.Conv2d(out_channel*4, 3, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.convert(x)))
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)
        return x

class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if (init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if (self.temprature > 1):
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)

class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                   init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k
        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

class RA(nn.Module): #DFRM
    def __init__(self, in_channel,out_channel):
        super(RA, self).__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.convs2=DynamicConv(out_channel, out_channel, 1)
        self.high_lateral_conv_attention = nn.Sequential(
            DynamicConv(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3,padding=1))
        self.DynamicConv=DynamicConv(out_channel,out_channel,1)
        self.channel = out_channel
        self.convert = nn.Conv2d(in_channel,out_channel,1)
    def forward(self, x, y):
        y=self.DynamicConv(y)
        a = torch.sigmoid(-y)
        x = self.relu(self.bn(x))
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        y = y + self.convs(x)
        y= self.convert(y)
        return y

class RAS(BaseModule):
    def __init__(self,channel=256):
        super(RAS, self).__init__()
        self.mscm = MSCM(2048, channel)
        self.ra2 = RA(256, channel)
        self.ra3 = RA(512, channel)
        self.ra4 = RA(1024, channel)

        self.initialize()

    def forward(self, outs):
        x0_size = outs[0].size()[2:]
        x1_size = outs[1].size()[2:]
        x2_size = outs[2].size()[2:]
        outs1 = []

        # y5 = self.ECAAttention3(outs[3])
        y5 = self.mscm(outs[3])
        # 通道降维优化
        outs1.append(y5)
        # y5提取全局信息，输出y5
        y5_4 = F.interpolate(outs[3], x2_size, mode='bilinear', align_corners=True)
        # y5_4 = self.convert1(y5_4)
        # 改变y5尺寸适合y4
        y4 = self.ra1(outs[2], y5_4)
        # y4 = self.ECAAttention2(y4)
        outs1.append(y4)
        # y4加上反转后的y5，输出y4
        y4_3 = F.interpolate(y5_4, x1_size, mode='bilinear', align_corners=True)
        # y4_3 = self.convert2(y4_3)
        # 改变y4尺寸适合y3
        y3 = self.ra2(outs[1], y4_3)
        # y3 = self.ECAAttention1(y4_3)
        outs1.append(y3)
        # y3加上反转后的y4，输出y3
        y3_2 = F.interpolate(y4_3, x0_size, mode='bilinear', align_corners=True)
        # y3_2 = self.convert3(y3_2)
        # 改变y3尺寸适合y2
        y2 = self.ra3(outs[0], y3_2)
        # y2 = self.ECAAttention0(y3_2)
        outs1.append(y2)

        # y2加上反转后的y3，输出y2
        outs2 = []
        outs2.append(outs1[3])
        outs2.append(outs1[2])
        outs2.append(outs1[1])
        outs2.append(outs1[0])


        return tuple(outs2)
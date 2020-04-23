import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.efficientnet.utils import MemoryEfficientSwish as Swish


class ConvModule(nn.Module):
    """ Regular Convolution with BatchNorm """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWiseSeparableConvModule(nn.Module):
    """ DepthWise Separable Convolution with BatchNorm and ReLU activation """
    def __init__(self, in_channels, out_channels, bath_norm=True, relu=True, bias=False):
        super(DepthWiseSeparableConvModule, self).__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 padding=0, bias=bias)

        self.bn = None if not bath_norm else \
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.act = None if not relu else Swish()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MaxPool2dSamePad(nn.MaxPool2d):
    """ TensorFlow-like 2D Max Pooling with same padding """

    PAD_VALUE: float = -float('inf')

    def __init__(self, kernel_size: int, stride=1, padding=0,
                 dilation=1, ceil_mode=False, count_include_pad=True):
        assert padding == 0, 'Padding in MaxPool2d Same Padding should be zero'

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        super(MaxPool2dSamePad, self).__init__(kernel_size, stride, padding,
                                               dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        h, w = x.size()[-2:]

        pad_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] + \
                (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h
        pad_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] + \
                (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                          pad_h - pad_h // 2], value=self.PAD_VALUE)

        x = F.max_pool2d(x, self.kernel_size, self.stride,
                         self.padding, self.dilation, self.ceil_mode)
        return x


class ChannelAdjuster(nn.Module):
    """ Adjusts number of channels before BiFPN via 1x1 conv layers
    Creates P3, P4, P4_2, P5, P5_2, P6 and P7 feature maps  for use in BiFPN """
    def __init__(self, in_channels: list, out_channels: int):
        super(ChannelAdjuster, self).__init__()
        assert isinstance(in_channels, list), 'in_channels should be a list'
        assert isinstance(out_channels, int), 'out_channels should be an integer'

        self.convs = nn.ModuleList()
        for idx, n_channels in enumerate(in_channels):
            self.convs.append(ConvModule(n_channels, out_channels))
            if idx > 0:
                self.convs.append(ConvModule(n_channels, out_channels))

        self.p5_to_p6 = nn.Sequential(
            ConvModule(in_channels[-1], out_channels),
            MaxPool2dSamePad(3, 2)
        )
        self.p6_to_p7 = MaxPool2dSamePad(3, 2)

    def forward(self, features):
        """ param: features: a list of P3, P4, P5 feature maps from backbone
            returns: outs: P3, P4, P4_2, P5, P5_2, P6, P7 feature maps """
        outs = []
        conv_idx = 0
        for feature in features:
            outs.append(self.convs[conv_idx](feature))

            if conv_idx > 0:
                conv_idx += 1
                outs.append(self.convs[conv_idx](feature))

            conv_idx += 1

        outs.append(self.p5_to_p6(features[-1]))
        outs.append(self.p6_to_p7(outs[-1]))

        return outs

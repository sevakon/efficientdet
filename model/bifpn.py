import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import DepthWiseSeparableConvModule as DWSConv


class BiFPN(nn.Module):
    """
    BiFPN block.
    Depending on its order, it either accepts
    seven feature maps (if this block is the first block in FPN)
    or five feature maps (otherwise)
    """

    eps: float = 1e-04

    def __init__(self, n_channels):
        super(BiFPN, self).__init__()

        self.conv_4_td = DWSConv(n_channels, n_channels)
        self.conv_5_td = DWSConv(n_channels, n_channels)
        self.conv_6_td = DWSConv(n_channels, n_channels)

        self.weights_4_td = nn.Parameter(torch.ones(2))
        self.weights_5_td = nn.Parameter(torch.ones(2))
        self.weights_6_td = nn.Parameter(torch.ones(2))

        self.conv_3_out = DWSConv(n_channels, n_channels)
        self.conv_4_out = DWSConv(n_channels, n_channels)
        self.conv_5_out = DWSConv(n_channels, n_channels)
        self.conv_6_out = DWSConv(n_channels, n_channels)
        self.conv_7_out = DWSConv(n_channels, n_channels)

        self.weights_3_out = nn.Parameter(torch.ones(2))
        self.weights_4_out = nn.Parameter(torch.ones(3))
        self.weights_5_out = nn.Parameter(torch.ones(3))
        self.weights_6_out = nn.Parameter(torch.ones(3))
        self.weights_7_out = nn.Parameter(torch.ones(2))

    def forward(self, features):
        if len(features) == 5:
            p_3, p_4, p_5, p_6, p_7 = features
            p_4_2, p_5_2 = None, None
        else:
            p_3, p_4, p_4_2, p_5, p_5_2, p_6, p_7 = features

        # Top Down Path
        p_6_td = self.conv_6_td(
            self._fuse_features(
                weights=self.weights_6_td,
                features=[p_6, F.interpolate(p_7, scale_factor=2)]
            )
        )
        p_5_td = self.conv_5_td(
            self._fuse_features(
                weights=self.weights_5_td,
                features=[p_5, F.interpolate(p_6_td, scale_factor=2)]
            )
        )
        p_4_td = self.conv_4_td(
            self._fuse_features(
                weights=self.weights_4_td,
                features=[p_4, F.interpolate(p_5_td, scale_factor=2)]
            )
        )

        p_4_in = p_4 if p_4_2 is None else p_4_2
        p_5_in = p_5 if p_5_2 is None else p_5_2

        # Out
        p_3_out = self.conv_3_out(
            self._fuse_features(
                weights=self.weights_3_out,
                features=[p_3, F.interpolate(p_4_td, scale_factor=2)]
            )
        )
        p_4_out = self.conv_4_out(
            self._fuse_features(
                weights=self.weights_4_out,
                features=[p_4_in, p_4_td, F.max_pool2d(p_3_out, 2)]
            )
        )
        p_5_out = self.conv_5_out(
            self._fuse_features(
                weights=self.weights_5_out,
                features=[p_5_in, p_5_td, F.max_pool2d(p_4_out, 2)]
            )
        )
        p_6_out = self.conv_6_out(
            self._fuse_features(
                weights=self.weights_6_out,
                features=[p_6, p_6_td, F.max_pool2d(p_5_out, 2)]
            )
        )
        p_7_out = self.conv_7_out(
            self._fuse_features(
                weights=self.weights_7_out,
                features=[p_7, F.max_pool2d(p_6, 2)]
            )
        )

        return [p_3_out, p_4_out, p_5_out, p_6_out, p_7_out]

    def _fuse_features(self, weights, features):
        num = sum([F.relu(w) * f for w, f in zip(weights, features)])
        det = sum(weights) + self.eps
        return num / det

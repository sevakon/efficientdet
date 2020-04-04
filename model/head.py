import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import DepthWiseSeparableConvModule as DWSConv


class Regresser(nn.Module):
    def __init__(self, in_channels, n_repeats, n_anchors=9, n_features=88):
        super(Regresser, self).__init__()
        layers = [DWSConv(in_channels, n_features)] + \
                 [DWSConv(n_features, n_features) for _ in range(n_repeats - 1)]

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1, groups=n_features),
            nn.Conv2d(n_features, n_anchors * 4, 1)
        )

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.head(inputs)
        out = inputs
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, n_repeats, n_anchors=9, n_features=88, n_classes=90):
        super(Classifier, self).__init__()
        layers = [DWSConv(in_channels, n_features)] + \
                 [DWSConv(n_features, n_features) for _ in range(n_repeats - 1)]

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1, groups=n_features),
            nn.Conv2d(n_features, n_anchors * n_classes, 1)
        )

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.head(inputs)
        out = torch.sigmoid(inputs)
        return out

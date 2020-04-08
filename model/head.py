import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import DepthWiseSeparableConvModule as DWSConv


class Regresser(nn.Module):
    def __init__(self, n_features, n_repeats, n_anchors=9):
        super(Regresser, self).__init__()
        layers = [DWSConv(n_features, n_features) for _ in range(n_repeats)]

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
    def __init__(self, n_features, n_repeats, n_anchors=9, n_classes=90):
        super(Classifier, self).__init__()
        layers = [DWSConv(n_features, n_features) for _ in range(n_repeats)]

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1, groups=n_features),
            nn.Conv2d(n_features, n_anchors * n_classes, 1)
        )

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.head(inputs)
        out = inputs
        return out

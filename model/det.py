import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import EfficientNet
from model.bifpn import BiFPN
from model.utils import efficientdet_params, check_model_name
from model.head import Classifier, Regresser
from model.module import ChannelAdjuster


class EfficientDet(nn.Module):
    def __init__(self, name):
        super(EfficientDet, self).__init__()
        check_model_name(name)
        self.params = efficientdet_params(name)
        self.backbone = EfficientNet(self.params['backbone'])

        self.adjuster = ChannelAdjuster(self.backbone.get_channels_list(),
                                        self.params['W_bifpn'])
        self.bifpn = nn.Sequential(*[BiFPN(self.params['W_bifpn'])
                                     for _ in range(self.params['D_bifpn'])])

        self.regresser = Regresser(self.params['W_bifpn'], self.params['D_class'])
        self.classifier = Classifier(self.params['W_bifpn'], self.params['D_class'])

    def forward(self, x):
        features = self.backbone(x)
        features = self.adjuster(features)
        features = self.bifpn(features)

        box_outputs, cls_outputs = [], []
        for f_map in features:
            box_outputs.append(self.regresser(f_map))
            cls_outputs.append(self.classifier(f_map))

        return box_outputs, cls_outputs

    def load_backbone(self, path):
        self.backbone.load_state_dict(torch.load(path), strict=True)

    @staticmethod
    def load_from_name(name):
        pass


if __name__ == '__main__':
    """ Quick test on parameters number """
    true_params = ['3.9M', '6.6M', '8.1M', '12.0M', '20.7M', '34.3M', '51.9M']

    for phi in [0, 1, 2, 3, 4, 5, 6]:
        model_name = 'efficientdet-d' + str(phi)
        model = EfficientDet(model_name)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Phi: {}, params: {}M, true params: {}'.format(phi, params / 1000000, true_params[phi]))

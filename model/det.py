import torch
import numpy as np
import torch.nn as nn
from itertools import chain

import config as cfg
from model.backbone import EfficientNet
from model.bifpn import BiFPN
from model.head import HeadNet
from model.module import ChannelAdjuster
from model.utils import efficientdet_params, check_model_name, download_model_weights


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

        self.regresser = HeadNet(n_features=self.params['W_bifpn'],
                                 out_channels=cfg.NUM_ANCHORS * 4,
                                 n_repeats=self.params['D_class'])

        self.classifier = HeadNet(n_features=self.params['W_bifpn'],
                                  out_channels=cfg.NUM_ANCHORS * cfg.NUM_CLASSES,
                                  n_repeats=self.params['D_class'])

    def forward(self, x):
        features = self.backbone(x)

        features = self.adjuster(features)
        features = self.bifpn(features)

        box_outputs = self.regresser(features)
        cls_outputs = self.classifier(features)

        return box_outputs, cls_outputs

    def initialize_weights(self):
        """ Initialize Model Weights before training from scratch """
        for module in chain(self.adjuster.modules(),
                            self.bifpn.modules(),
                            self.regresser.modules(),
                            self.classifier.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.regresser.head.conv_pw.bias)
        nn.init.constant_(self.classifier.head.conv_pw.bias, -np.log((1 - 0.01) / 0.01))

    def load_backbone(self, path):
        self.backbone.model.load_state_dict(torch.load(path), strict=True)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def from_pretrained(name=cfg.MODEL_NAME):
        check_model_name(name)

        if not cfg.MODEL_WEIGHTS.exists():
            download_model_weights(name, cfg.MODEL_WEIGHTS)

        model_to_return = EfficientDet(name)
        model_to_return.load_weights(cfg.MODEL_WEIGHTS)
        return model_to_return


if __name__ == '__main__':
    """ Quick test on parameters number """
    true_params = ['3.9M', '6.6M', '8.1M', '12.0M', '20.7M', '34.3M', '51.9M']

    for phi in [0, 1, 2, 3, 4, 5, 6]:
        model_name = 'efficientdet-d' + str(phi)
        model = EfficientDet(model_name)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Phi: {}, params: {}M, true params: {}'.format(phi, params / 1000000, true_params[phi]))

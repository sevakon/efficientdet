from itertools import chain

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from log.logger import logger
from model.backbone import EfficientNet
from model.bifpn import BiFPN
from model.head import HeadNet
from model.module import ChannelAdjuster
from utils.utils import check_model_name, download_model_weights
from utils.tools import variance_scaling_


class EfficientDet(nn.Module):
    def __init__(self, name):
        super(EfficientDet, self).__init__()
        check_model_name(name)

        self.backbone = EfficientNet(cfg.MODEL.BACKBONE)

        self.adjuster = ChannelAdjuster(self.backbone.get_channels_list(),
                                        cfg.MODEL.W_BIFPN)
        self.bifpn = nn.Sequential(*[BiFPN(cfg.MODEL.W_BIFPN)
                                     for _ in range(cfg.MODEL.D_BIFPN)])

        self.regresser = HeadNet(n_features=cfg.MODEL.W_BIFPN,
                                 out_channels=cfg.NUM_ANCHORS * 4,
                                 n_repeats=cfg.MODEL.D_CLASS)

        self.classifier = HeadNet(n_features=cfg.MODEL.W_BIFPN,
                                  out_channels=cfg.NUM_ANCHORS * cfg.NUM_CLASSES,
                                  n_repeats=cfg.MODEL.D_CLASS)

    def forward(self, x):
        features = self.backbone(x)

        features = self.adjuster(features)
        features = self.bifpn(features)

        cls_outputs = self.classifier(features)
        box_outputs = self.regresser(features)

        return cls_outputs, box_outputs

    @staticmethod
    def from_name(name):
        """ Interface for model prepared to train on COCO """
        cfg.MODEL.choose_model(name)

        model_to_return = EfficientDet(name)

        if not cfg.MODEL.BACKBONE_WEIGHTS.exists():
            logger('Downloading backbone {}...'.format(cfg.MODEL.BACKBONE))
            download_model_weights(cfg.MODEL.BACKBONE, cfg.MODEL.BACKBONE_WEIGHTS)

        model_to_return._load_backbone(cfg.MODEL.BACKBONE_WEIGHTS)
        model_to_return._initialize_weights()
        return model_to_return

    @staticmethod
    def from_pretrained(name):
        """ Interface for pre-trained model """
        cfg.MODEL.choose_model(name)

        model_to_return = EfficientDet(name)

        if not cfg.MODEL.WEIGHTS.exists():
            logger('Downloading pre-trained {}...'.format(cfg.MODEL.NAME))
            download_model_weights(name, cfg.MODEL.WEIGHTS)

        model_to_return._load_weights(cfg.MODEL.WEIGHTS)
        return model_to_return

    def _initialize_weights(self):
        """ Initialize Model Weights before training from scratch """
        for module in chain(self.adjuster.modules(),
                            self.bifpn.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in chain(self.regresser.modules(),
                            self.classifier.modules()):
            if isinstance(module, nn.Conv2d):
                variance_scaling_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.regresser.head.conv_pw.bias)
        nn.init.constant_(self.classifier.head.conv_pw.bias, -np.log((1 - 0.01) / 0.01))

    def _load_backbone(self, path):
        self.backbone.model.load_state_dict(torch.load(path), strict=False)
        logger('Loaded backbone checkpoint {}'.format(path))

    def _load_weights(self, path):
        self.load_state_dict(torch.load(path))
        logger('Loaded checkpoint {}'.format(path))


if __name__ == '__main__':
    """ Quick test on parameters number """
    true_params = ['3.9M', '6.6M', '8.1M', '12.0M', '20.7M', '34.3M', '51.9M']

    for phi in [0, 1, 2, 3, 4, 5, 6]:
        model_name = 'efficientdet-d' + str(phi)
        model = EfficientDet(model_name)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Phi: {}, params: {}M, true params: {}'.format(phi, params / 1000000, true_params[phi]))

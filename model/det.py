import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import EfficientNet
from model.utils import efficientdet_params, check_model_name


class EfficientDet(nn.Module):
    def __init__(self, name):
        super(EfficientDet, self).__init__()
        check_model_name(name)
        self.params = efficientdet_params(name)
        self.backbone = EfficientNet(self.params['backbone'])
        self.bifpn = None
        self.regresser = None
        self.classifier = None

    def forward(self, x):
        pass

    def load_backbone(self, path):
        self.backbone.load_state_dict(torch.load(path), strict=True)

    @staticmethod
    def load_from_name(name):
        pass


if __name__ == '__main__':
    """ Quick test on parameters number """
    for phi in [0, 1, 2, 3, 4, 5, 6]:
        model_name = 'efficientdet-d' + str(phi)
        model = EfficientDet(model_name)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Phi: {}, params: {}M'.format(phi, params / 1000000))

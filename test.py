import time
import torch
from model.backbone import EfficientNet
import numpy as np
from model.det import EfficientDet
from model.utils import efficientdet_params, count_parameters


""" Quick test on parameters number """

# for phi in [0, 1, 2, 3, 4, 5, 6]:
for phi in [0, 1, 2, 3, 4, 5]:
    model_name = 'efficientdet-d' + str(phi)
    model = EfficientDet(model_name).to('cpu')

    model.train()
    params = count_parameters(model)

    print('Phi: {}, params: {:.6f}M, params in paper: {}'.format(phi, params / 1e6,
                                                         efficientdet_params(model_name)['params']))
    print('   Backbone: {:.6f}M'.format(count_parameters(model.backbone) / 1e6))
    print('   Adjuster: {:.6f}M'.format(count_parameters(model.adjuster) / 1e6))
    print('      BiFPN: {:.6f}M'.format(count_parameters(model.bifpn) / 1e6))
    print('       Head: {:.6f}M'.format((count_parameters(model.classifier) +
                                        count_parameters(model.regresser)) / 1e6))

    # image_size = efficientdet_params(model_name)['R_input']
    # x = torch.rand(1, 3, image_size, image_size)
    # model(x)

import torch
from model.backbone import EfficientNet
import numpy as np
from model.det import EfficientDet
from model.utils import efficientdet_params

""" Quick test on parameters number """

for phi in [0, 1, 2, 3, 4, 5, 6]:
    model_name = 'efficientdet-d' + str(phi)
    model = EfficientDet(model_name).to('cpu')

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    image_size = efficientdet_params(model_name)['R_input']
    x = torch.rand(1, 3, image_size, image_size).to('cpu')
    features = model(x)

    print(' Input: {}'.format(x.shape))

    print(model.backbone.get_channels_list())
    for idx, p in enumerate(features):
        print(' P{}: {}'.format(idx + 1, p.shape))

    print('Phi: {}, params: {}M, params in paper: {}'.format(phi, params / 1000000,
                                                         efficientdet_params(model_name)['params']))


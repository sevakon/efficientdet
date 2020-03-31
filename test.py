import torch
from model.backbone import EfficientNet
import numpy as np
from model.det import EfficientDet

""" Quick test on parameters number """
for phi in [0, 1, 2, 3, 4, 5, 6]:
    model_name = 'efficientdet-d' + str(phi)
    model = EfficientDet(model_name)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    if phi == 0:
        x = torch.rand(1, 3, 512, 512)
        features = model.backbone(x)

        for idx, p in enumerate(features):
            print(p.shape)


    print('Phi: {}, params: {}M'.format(phi, params / 1000000))

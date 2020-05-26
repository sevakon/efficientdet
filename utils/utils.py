import math
import random

import numpy as np
import torch
from torch.hub import download_url_to_file


def check_model_name(model_name):
    model_zoo = ['efficientdet-d' + str(i) for i in range(7)]
    if model_name == 'efficientdet-d6':
        raise ValueError('Sorry! EfficientDet D-6 is not yet supported :( ')
    if model_name not in model_zoo:
        raise ValueError('Name {} not in {}'.format(model_name, model_zoo))


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def is_valid_number(x):
    is_valid = not (math.isnan(x) or math.isinf(x) or x > 1e4)
    return is_valid


def get_lr(optimizer, group=0):
    lr = optimizer.param_groups[group]['lr']
    return lr


def get_gradnorm(optimizer, group=0):
    norms = [torch.norm(p.grad).item() for p in optimizer.param_groups[group]['params']]
    gradnorm = np.mean(norms) if norms else 0
    return gradnorm


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def efficientdet_params(model_name):
    """ Map EfficientDet model name to parameter coefficients. """
    params_dict = {

        'efficientdet-d0': {
            'compound_coef': 0,
            'backbone': 'efficientnet-b0',
            'R_input': 512,
            'W_bifpn': 64,
            'D_bifpn': 3,
            'D_class': 3,
            'params': '3.9M'
        },

        'efficientdet-d1': {
            'compound_coef': 1,
            'backbone': 'efficientnet-b1',
            'R_input': 640,
            'W_bifpn': 88,
            'D_bifpn': 4,
            'D_class': 3,
            'params': '6.6M'
        },

        'efficientdet-d2': {
            'compound_coef': 2,
            'backbone': 'efficientnet-b2',
            'R_input': 768,
            'W_bifpn': 112,
            'D_bifpn': 5,
            'D_class': 3,
            'params': '8.1M'
        },

        'efficientdet-d3': {
            'compound_coef': 3,
            'backbone': 'efficientnet-b3',
            'R_input': 896,
            'W_bifpn': 160,
            'D_bifpn': 6,
            'D_class': 4,
            'params': '12.0M'
        },

        'efficientdet-d4': {
            'compound_coef': 4,
            'backbone': 'efficientnet-b4',
            'R_input': 1024,
            'W_bifpn': 224,
            'D_bifpn': 7,
            'D_class': 4,
            'params': '20.7M'
        },

        'efficientdet-d5': {
            'compound_coef': 5,
            'backbone': 'efficientnet-b5',
            'R_input': 1280,
            'W_bifpn': 288,
            'D_bifpn': 7,
            'D_class': 4,
            'params': '33.7M'
        },

        'efficientdet-d6': {
            'compound_coef': 6,
            'backbone': 'efficientnet-b6',
            'R_input': 1280,
            'W_bifpn': 384,
            'D_bifpn': 8,
            'D_class': 5,
            'params': '51.9M'
        }

    }

    return params_dict[model_name]


def download_model_weights(model_name, filename):
    model_to_url = {
        'efficientdet-d0': 'https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d0.pth',
        'efficientdet-d1': 'https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d1.pth',
        'efficientdet-d2': 'https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d2.pth',
        'efficientdet-d3': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d3.pth',
        'efficientdet-d4': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d4.pth',
        'efficientdet-d5': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d5.pth',
        'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
    }
    download_url_to_file(model_to_url[model_name], filename)

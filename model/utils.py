import torch
import numpy as np
from torch.hub import download_url_to_file


def check_model_name(model_name):
    possibles = ['efficientdet-d' + str(i) for i in range(7)]
    if model_name == 'efficientdet-d6':
        raise ValueError('Sorry! EfficientDet D-6 is not yet supported :( ')
    if model_name not in possibles:
        raise ValueError('Name {} not in {}'.format(model_name, possibles))


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def download_model_weights(model_name, filename):
    model_to_url = {
        'efficientdet-d0': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d0.pth',
        'efficientdet-d1': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d1.pth',
        'efficientdet-d2': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d2.pth',
        'efficientdet-d3': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d3.pth',
        'efficientdet-d4': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d4.pth',
        'efficientdet-d5': 'https://github.com/sevakon/efficientdet/releases/download/v1.0/efficientdet-d5.pth'
    }
    download_url_to_file(model_to_url[model_name], filename)

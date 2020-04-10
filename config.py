import torch
from pathlib import Path


MODEL_ZOO = ['efficientdet-d' + str(phi) for phi in range(7)]
MODEL_NAME = 'efficientdet-d0'
assert MODEL_NAME in MODEL_ZOO, '{} not in model zoo'.format(MODEL_NAME)

BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / 'data'
COCO_PATH = DATA_PATH / 'coco'
WEIGHTS_PATH = BASE_PATH / 'weights'
MODEL_WEIGHTS = WEIGHTS_PATH / '{}.pth'.format(MODEL_NAME)

ASPECT_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
NUM_SCALES = 3
ANCHOR_SCALE = 4.0

NUM_ANCHORS = len(ASPECT_RATIOS) * NUM_SCALES
NUM_CLASSES = 90

MIN_LEVEL = 3
MAX_LEVEL = 7
NUM_LEVELS = MAX_LEVEL - MIN_LEVEL + 1

MAX_DETECTION_POINTS = 5000
MAX_DETECTIONS_PER_IMAGE = 100


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


COMPOUND_COEF = efficientdet_params(MODEL_NAME)['compound_coef']
BACKBONE = efficientdet_params(MODEL_NAME)['backbone']
IMAGE_SIZE = efficientdet_params(MODEL_NAME)['R_input']
W_BIFPN = efficientdet_params(MODEL_NAME)['W_bifpn']
D_BIFPN = efficientdet_params(MODEL_NAME)['D_bifpn']
D_CLASS = efficientdet_params(MODEL_NAME)['D_class']
PARAMS = efficientdet_params(MODEL_NAME)['params']

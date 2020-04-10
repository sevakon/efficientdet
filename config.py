import torch
from pathlib import Path

MODEL_NAME = 'efficientdet-d0'

BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / 'data'
WEIGHTS_PATH = BASE_PATH / 'weights'
MODEL_WEIGHTS = WEIGHTS_PATH / '{}.pth'.format(MODEL_NAME)

ASPECT_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
NUM_SCALES = 3
ANCHOR_SCALE = 4.0

NUM_ANCHORS = len(ASPECT_RATIOS) * NUM_SCALES
NUM_CLASSES = 90

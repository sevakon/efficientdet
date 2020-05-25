import math

from pathlib import Path
from utils.utils import check_model_name, efficientdet_params


BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / 'data'
WEIGHTS_PATH = BASE_PATH / 'weights'
LOG_PATH = BASE_PATH / 'log'
LOG_FILE = LOG_PATH / 'output'

COCO_PATH = DATA_PATH / 'coco'
TRAIN_SET = COCO_PATH / 'train2017'
VAL_SET = COCO_PATH / 'val2017'
COCO_RESULTS = COCO_PATH / 'results.json'

ANNOTATIONS_PATH = COCO_PATH / 'annotations'
TRAIN_ANNOTATIONS = ANNOTATIONS_PATH / 'instances_train2017.json'
VAL_ANNOTATIONS = ANNOTATIONS_PATH / 'instances_val2017.json'
TRAIN_SCALE_MIN = 0.1
TRAIN_SCALE_MAX = 2.0

SEED = 1234

# Original batch size = 128, assuming linear correlation with lr:
# ${YOUR BASE LR} = ${YOUR BATCH SIZE} * 0.016 / 128

BATCH_SIZE = 32
NUM_EPOCHS = 300
NUM_EXAMPLES_PER_EPOCH = 117266
STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / BATCH_SIZE)
TOTAL_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
VAL_DELAY = 50
VAL_INTERVAL = 10

OPT = 'SGD'
MOMENTUM = 0.9
BASE_LR = 0.04
WARMUP_LR = 0.004
WEIGHT_DECAY = 4e-5
MOVING_AVERAGE_DECAY = 0.9998
CLIP_GRADIENTS_NORM = 10.0

# classification loss
ALPHA = 0.25
GAMMA = 1.5
# localization loss
DELTA = 0.1
BOX_LOSS_WEIGHT = 50.0

ASPECT_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
NUM_SCALES = 3
ANCHOR_SCALE = 4.0

NUM_ANCHORS = len(ASPECT_RATIOS) * NUM_SCALES
NUM_CLASSES = 90

MIN_LEVEL = 3
MAX_LEVEL = 7
NUM_LEVELS = MAX_LEVEL - MIN_LEVEL + 1

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000
# The maximum number of detections per image.
MAX_DETECTIONS_PER_IMAGE = 100
# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5
# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0


class ModelInfo:

    NAME: str
    COMPOUND_COEF: int
    BACKBONE: str
    IMAGE_SIZE: int
    W_BIFPN: int
    D_BIFPN: int
    D_CLASS: int
    PARAMS: str
    WEIGHTS: Path
    BACKBONE_WEIGHTS: Path
    SAVE_PATH: Path

    def choose_model(self, model_name):
        check_model_name(model_name)

        self.NAME = model_name
        self.COMPOUND_COEF = efficientdet_params(self.NAME)['compound_coef']
        self.BACKBONE = efficientdet_params(self.NAME)['backbone']
        self.IMAGE_SIZE = efficientdet_params(self.NAME)['R_input']
        self.W_BIFPN = efficientdet_params(self.NAME)['W_bifpn']
        self.D_BIFPN = efficientdet_params(self.NAME)['D_bifpn']
        self.D_CLASS = efficientdet_params(self.NAME)['D_class']
        self.PARAMS = efficientdet_params(self.NAME)['params']

        self.WEIGHTS = WEIGHTS_PATH / '{}.pth'.format(self.NAME)
        self.SAVE_PATH = WEIGHTS_PATH / 'trained-{}.pth'.format(self.NAME)
        self.BACKBONE_WEIGHTS = WEIGHTS_PATH / '{}.pth'.format(self.BACKBONE)


MODEL = ModelInfo()

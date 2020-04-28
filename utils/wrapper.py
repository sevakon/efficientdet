import torch
import torch.nn as nn

import config as cfg
from utils.anchors import Anchors, generate_detections
from utils.processing import postprocess, preprocess


class DetectionWrapper(nn.Module):
    """ Wrapper on top of the model. Pre-process and postprocess raw data """
    def __init__(self, model, device):
        super(DetectionWrapper, self).__init__()
        self.model = model
        self.device = device
        self.anchors = Anchors(
            cfg.MIN_LEVEL, cfg.MAX_LEVEL,
            cfg.NUM_SCALES, cfg.ASPECT_RATIOS,
            cfg.ANCHOR_SCALE, cfg.MODEL.IMAGE_SIZE, device)
        self._anchor_cache = None

    def forward(self, image_paths):
        x, img_scales = preprocess(image_paths)
        cls_outs, box_outs = self.model(x.to(self.device))
        cls_outs, box_outs, indices, classes = postprocess(cls_outs, box_outs)
        batch_detections = []
        if self._anchor_cache is None:
            anchor_boxes = self.anchors.boxes
            self._anchor_cache = anchor_boxes
        else:
            anchor_boxes = self._anchor_cache

        for i in range(x.shape[0]):
            detections = generate_detections(
                cls_outs[i], box_outs[i], anchor_boxes,
                indices[i], classes[i], img_scales[i])
            batch_detections.append(detections)

        return batch_detections

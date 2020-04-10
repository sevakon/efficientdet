import torch
import torch.nn as nn

import config as cfg
from utils.anchors import Anchors, generate_detections
from utils.processing import post_process


class DetectionWrapper(nn.Module):
    def __init__(self, model):
        super(DetectionWrapper, self).__init__()
        self.model = model
        self.anchors = Anchors(
            cfg.MIN_LEVEL, cfg.MAX_LEVEL,
            cfg.NUM_SCALES, cfg.ASPECT_RATIOS,
            cfg.ANCHOR_SCALE, cfg.IMAGE_SIZE)
        self._anchor_cache = None

    def forward(self, x, image_ids, image_scales):
        cls_outs, box_outs = self.model(x)
        cls_outs, box_outs, indices, classes = post_process(cls_outs, box_outs)

        batch_detections = []
        cls_outs = cls_outs.cpu().numpy()
        box_outs = box_outs.cpu().numpy()
        if self._anchor_cache is None:
            anchor_boxes = self.anchors.boxes.cpu().numpy()
            self._anchor_cache = anchor_boxes
        else:
            anchor_boxes = self._anchor_cache
        indices = indices.cpu().numpy()
        classes = classes.cpu().numpy()
        image_ids = image_ids.cpu().numpy()
        image_scale = image_scales.cpu().numpy()
        for i in range(x.shape[0]):
            detections = generate_detections(
                cls_outs[i], box_outs[i], anchor_boxes, indices[i], classes[i],
                image_ids[i], image_scale[i], cfg.NUM_CLASSES)
            batch_detections.append(detections)

        return batch_detections

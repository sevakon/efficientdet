import torch
import torch.nn as nn

import config as cfg
from utils.anchors import Anchors, AnchorsLabeler, generate_detections
from utils.processing import postprocess, preprocess


class DetectionTrainWrapper(nn.Module):
    """ Train Wrapper on top of the model. Labels anchors, calculates loss """

    def __init__(self, model, device, criterion):
        super(DetectionTrainWrapper, self).__init__()
        self.model = model
        self.device = device
        anchors = Anchors(
            cfg.MIN_LEVEL, cfg.MAX_LEVEL,
            cfg.NUM_SCALES, cfg.ASPECT_RATIOS,
            cfg.ANCHOR_SCALE, cfg.MODEL.IMAGE_SIZE, device)
        self.anchor_labeler = AnchorsLabeler(anchors, cfg.NUM_CLASSES)
        self.criterion = criterion

    def forward(self, x, gt_labels, gt_boxes):
        batch_size = x.shape[0]
        cls_outputs, box_outputs = self.model(x.to(self.device))
        gt_labels = gt_labels.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        cls_targets, box_targets, num_positives = [], [], []
        # Iterating over batch since labels length is different for each image
        for i in range(batch_size):
            gt_class, gt_box, num_positive = \
                self.anchor_labeler.label_anchors(gt_labels[i], gt_boxes[i])
            cls_targets.append(gt_class)
            box_targets.append(gt_box)
            num_positives.append(num_positive)

        total_loss, cls_loss, box_loss = self.criterion(
            cls_outputs, box_outputs, cls_targets, box_targets, num_positives)
        return total_loss, cls_loss, box_loss


class DetectionEvalWrapper(nn.Module):
    """ Eval Wrapper on top of the model. Preprocess & postprocess raw data """

    def __init__(self, model, device):
        super(DetectionEvalWrapper, self).__init__()
        self.model = model
        self.device = device
        self.anchor_boxes = Anchors(
            cfg.MIN_LEVEL, cfg.MAX_LEVEL,
            cfg.NUM_SCALES, cfg.ASPECT_RATIOS,
            cfg.ANCHOR_SCALE, cfg.MODEL.IMAGE_SIZE, device).boxes

    def forward(self, image_paths):
        x, img_scales = preprocess(image_paths)
        cls_outs, box_outs = self.model(x.to(self.device))
        cls_outs, box_outs, indices, classes = postprocess(cls_outs, box_outs)

        batch_detections = generate_detections(
            cls_outs, box_outs, self.anchor_boxes, indices, classes, img_scales)

        return batch_detections

import config as cfg
import numpy as np
import torch

from torchvision.ops.boxes import batched_nms


def decode_box_outputs(rel_codes, anchors):
    """Transforms relative regression coordinates to absolute positions.
    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.
    Args:
        rel_codes: box regression targets.
        anchors: anchors on all feature levels.
    Returns:
        outputs: bounding boxes.
    """
    ycenter_a = (anchors[0] + anchors[2]) / 2
    xcenter_a = (anchors[1] + anchors[3]) / 2
    ha = anchors[2] - anchors[0]
    wa = anchors[3] - anchors[1]
    ty, tx, th, tw = rel_codes

    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return torch.stack([ymin, xmin, ymax, xmax], dim=1)


def generate_detections(
        cls_outputs, box_outputs, anchor_boxes, indices, classes, image_scale):
    """Generates detections with RetinaNet model outputs and anchors.
    Args:
        cls_outputs: a numpy array with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)
        box_outputs: a numpy array with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels. (k being MAX_DETECTION_POINTS)
        anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.
        indices: a numpy array with shape [N], which is the indices from top-k selection.
        classes: a numpy array with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.
        image_id: an integer number to specify the image id.
        image_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.
        num_classes: a integer that indicates the number of classes.
    Returns:
        detections: detection results in a tensor with each row representing
            [image_id, x, y, width, height, score, class]
    """
    anchor_boxes = anchor_boxes[indices, :]
    scores = cls_outputs.sigmoid().squeeze(1).float()

    # apply bounding box regression to anchors
    boxes = decode_box_outputs(box_outputs.T.float(), anchor_boxes.T)
    boxes = boxes[:, [1, 0, 3, 2]]

    top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)
    # keep only topk scoring predictions
    top_detection_idx = top_detection_idx[:cfg.MAX_DETECTIONS_PER_IMAGE]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx]
    classes = classes[top_detection_idx]
    scores = scores.view(-1, 1)
    classes = classes.view(-1, 1)

    # xyxy to xywh & rescale to original image
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes *= image_scale

    classes += 1  # back to class idx with background class = 0

    # stack em and pad out to MAX_DETECTIONS_PER_IMAGE if necessary
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if len(top_detection_idx) < cfg.MAX_DETECTIONS_PER_IMAGE:
        detections = torch.cat([
            detections,
            torch.zeros(
                (cfg.MAX_DETECTIONS_PER_IMAGE - len(top_detection_idx), 6), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections


class Anchors(object):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales,
                 aspect_ratios, anchor_scale, image_size, device):
        """Constructs multiscale RetinaNet anchors.
        Args:
            min_level: integer number of minimum level of the output feature pyramid.
            max_level: integer number of maximum level of the output feature pyramid.
            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.
            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.
            image_size: integer number of input image size. The input image has the
                same dimension for width and height. The image_size should be divided by
                the largest feature stride 2^max_level.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.device = device
        self.config = self._generate_configs()
        self.boxes = self._generate_boxes()

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        anchor_configs = {}
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(
                        (2 ** level, scale_octave / float(self.num_scales), aspect))
        return anchor_configs

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = []
            for config in configs:
                stride, octave_scale, aspect = config
                if self.image_size % stride != 0:
                    raise ValueError(
                        "input size must be divided by the stride.")
                base_anchor_size = self.anchor_scale * stride * 2 ** octave_scale
                anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
                anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

                x = np.arange(stride / 2, self.image_size, stride)
                y = np.arange(stride / 2, self.image_size, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        boxes = torch.from_numpy(anchor_boxes).float().to(self.device)
        return boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)

import numpy as np
import torch

from torchvision.ops.boxes import batched_nms


def encode_boxes_to_anchors(boxes, anchors, eps=1e-8):
    """ Create anchors regression target based on anchors
    Args:
        boxes: ground truth boxes.
        anchors: anchor boxes on all feature levels.
        eps: small number for stability
    Returns:
        outputs: anchors w.r.t. ground truth
    """

    def corner_to_center(rects):
        h, w = rects[:, 2] - rects[:, 0], rects[:, 3] - rects[:, 1]
        y_ctr, x_ctr = rects[:, 0] + 0.5 * h, rects[:, 1] + 0.5 * w
        return y_ctr, x_ctr, h, w

    ycenter_a, xcenter_a, ha, wa = corner_to_center(anchors)
    ycenter, xcenter, h, w = corner_to_center(boxes)
    ha, wa, h, w = ha + eps, wa + eps, h + eps, w + eps
    dy = (ycenter - ycenter_a) / ha
    dx = (xcenter - xcenter_a) / wa
    dh = torch.log(h / ha)
    dw = torch.log(w / wa)
    outputs = torch.stack([dy, dx, dh, dw]).T

    return outputs


def decode_box_outputs(rel_codes, anchors):
    """Transforms relative regression coordinates to absolute positions.
    Args:
        rel_codes: batched box regression targets.
        anchors: batched anchors on all feature levels.
    Returns:
        outputs: batched bounding boxes.
    """
    y_center_a = (anchors[:, :, 0] + anchors[:, :, 2]) / 2
    x_center_a = (anchors[:, :, 1] + anchors[:, :, 3]) / 2
    ha = anchors[:, :, 2] - anchors[:, :, 0]
    wa = anchors[:, :, 3] - anchors[:, :, 1]
    ty, tx = rel_codes[:, :, 0], rel_codes[:, :, 1]
    th, tw = rel_codes[:, :, 2], rel_codes[:, :, 3]

    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    y_center = ty * ha + y_center_a
    x_center = tx * wa + x_center_a
    y_min = y_center - h / 2.
    x_min = x_center - w / 2.
    y_max = y_center + h / 2.
    x_max = x_center + w / 2.
    outputs = torch.stack([y_min, x_min, y_max, x_max], dim=-1)

    return outputs


def clip_boxes_(boxes, size):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size)
    return boxes


def generate_detections(
        cls_outputs, box_outputs, anchor_boxes, indices, classes,
        image_sizes, image_scales, max_detections_per_image):
    """Generates batched detections with RetinaNet model outputs and anchors.
    Args: (B - batch size, N - top-k selection length)
        cls_outputs: a numpy array with shape [B, N, 1], which has the
            highest class scores on all feature levels.
        box_outputs: a numpy array with shape [B, N, 4], which stacks
            box regression outputs on all feature levels.
        anchor_boxes: a numpy array with shape [B, N, 4], which stacks
            anchors on all feature levels.
        indices: a numpy array with shape [B, N], which is the indices from top-k selection.
        classes: a numpy array with shape [B, N], which represents
            the class prediction on all selected anchors from top-k selection.
        image_sizes: a list of tuples representing size of incoming images
        image_scales: a list representing the scale between original images
            and input images for the detector.
    Returns:
        detections: detection results in a tensor of shape [B, N, 6],
            where [:, :, 0:4] are boxes, [:, :, 5] are scores,
    """
    batch_size = indices.shape[0]
    device = indices.device
    anchor_boxes = anchor_boxes[indices, :]
    scores = cls_outputs.sigmoid().squeeze(2).float()

    # apply bounding box regression to anchors
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes)
    boxes = boxes[:, :, [1, 0, 3, 2]]

    batched_boxes, batched_scores, batched_classes = [], [], []
    # iterate over batch since we need non-max suppression for each image
    for batch_idx in range(batch_size):
        batch_boxes = boxes[batch_idx, :, :]
        batch_scores = scores[batch_idx, :]
        batch_classes = classes[batch_idx, :]
        # clip boxes outputs
        boxes_max_size = [image_sizes[batch_idx][0] / image_scales[batch_idx],
                          image_sizes[batch_idx][1] / image_scales[batch_idx]]
        boxes_max_size = torch.FloatTensor(boxes_max_size).to(batch_boxes.device)
        batch_boxes = clip_boxes_(batch_boxes, boxes_max_size)
        # perform non-maximum suppression
        top_detection_idx = batched_nms(
            batch_boxes, batch_scores, batch_classes, iou_threshold=0.5)
        # keep only topk scoring predictions
        top_detection_idx = top_detection_idx[:max_detections_per_image]
        batch_boxes = batch_boxes[top_detection_idx]
        batch_scores = batch_scores[top_detection_idx]
        batch_classes = batch_classes[top_detection_idx]
        # fill zero predictions to match MAX_DETECTIONS_PER_IMAGE
        detections_diff = len(top_detection_idx) - max_detections_per_image
        if detections_diff < 0:
            add_boxes = torch.zeros(
                (-detections_diff, 4), device=device, dtype=batch_boxes.dtype)
            batch_boxes = torch.cat([batch_boxes, add_boxes], dim=0)
            add_scores = torch.zeros(
                (-detections_diff, 1), device=device, dtype=batch_scores.dtype)
            batch_scores = torch.cat([batch_scores, add_scores], dim=0)
            add_classes = torch.zeros(
                (-detections_diff, 1), device=device, dtype=batch_classes.dtype)
            batch_classes = torch.cat([batch_classes, add_classes], dim=0)

        batch_scores = batch_scores.view(-1, 1)
        batch_classes = batch_classes.view(-1, 1)
        # stack them together
        batched_boxes.append(batch_boxes)
        batched_scores.append(batch_scores)
        batched_classes.append(batch_classes)

    boxes = torch.stack(batched_boxes)
    scores = torch.stack(batched_scores)
    classes = torch.stack(batched_classes)

    # xyxy to xywh & rescale to original image
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]
    boxes_scaler = torch.FloatTensor(image_scales).to(boxes.device)
    boxes = boxes * boxes_scaler[:, None, None]

    classes += 1  # back to class idx with background class = 0
    detections = torch.cat([boxes, scores, classes.float()], dim=2)

    return detections


def calc_iou(a, b):
    """ Calculate Intersection-over-Union of two samples
    Args:
        a (torch.Tensor): set of boxes with shape [N, 4] in yxyx format
        b (torch.Tensot): set of boxes with shape [M, 4] in yxyx format
    """
    area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])

    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) \
            - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) \
            - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    ih = torch.clamp(ih, min=0)
    iw = torch.clamp(iw, min=0)

    ua = torch.unsqueeze((a[:, 3] - a[:, 1])
                         * (a[:, 2] - a[:, 0]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    iou = intersection / ua

    return iou


class Anchors:
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


class AnchorsLabeler:
    """ Labeler for multiscale anchor boxes. """

    def __init__(self, anchors, num_classes, threshold=0.5):
        """Constructs anchor labeler to assign labels to anchors.
        Args:
            anchors: an instance of class Anchors.
            num_classes: integer number representing number of classes in the dataset.
            threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        self.anchors = anchors
        self.num_classes = num_classes
        self.threshold = threshold

    def _unpack_labels(self, labels):
        labels_unpacked = []
        anchors = self.anchors
        count = 0
        for level in range(anchors.min_level, anchors.max_level + 1):
            feat_size = int(anchors.image_size / 2 ** level)
            steps = feat_size ** 2 * anchors.get_anchors_per_location()
            indices = torch.arange(count, count + steps, device=labels.device)
            count += steps
            labels_unpacked.append(
                torch.index_select(labels, 0, indices).view([feat_size, feat_size, -1]))
        return labels_unpacked

    def label_anchors(self, gt_labels, gt_boxes):
        device = gt_boxes.device
        indices = gt_labels != -1
        labels = gt_labels[indices]
        boxes = gt_boxes[indices]
        iou = calc_iou(self.anchors.boxes, boxes)

        cls_target = torch.zeros(
            self.anchors.boxes.shape[0], self.num_classes, device=device)
        box_target = torch.zeros(
            self.anchors.boxes.shape[0], 4, device=device)
        num_positive_anchors = 0

        if iou.nelement() != 0:
            iou_max, iou_argmax = torch.max(iou, dim=1)
            positive_indices = torch.ge(iou_max, self.threshold)
            num_positive_anchors = positive_indices.sum()
            assigned_boxes = boxes[iou_argmax, :]
            assigned_labels = labels[iou_argmax]

            cls_target[positive_indices, assigned_labels[positive_indices].long()] = 1
            box_target[positive_indices, :] = encode_boxes_to_anchors(
                assigned_boxes[positive_indices], self.anchors.boxes[positive_indices])

        cls_target -= 1
        cls_target = cls_target.long()

        cls_target = self._unpack_labels(cls_target)
        box_target = self._unpack_labels(box_target)

        return cls_target, box_target, num_positive_anchors

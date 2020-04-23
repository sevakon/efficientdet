import torch
import config as cfg

from PIL import Image
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def postprocess(cls_outputs, box_outputs):
    """Selects top-k predictions.
    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.
    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, cfg.NUM_CLASSES])
        for level in range(cfg.NUM_LEVELS)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(cfg.NUM_LEVELS)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=cfg.MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all / cfg.NUM_CLASSES
    classes_all = cls_topk_indices_all % cfg.NUM_CLASSES

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, cfg.NUM_CLASSES))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


def resize(pil_img, target_image, interpolate=Image.BILINEAR):
    width, height = pil_img.size
    if height > width:
        scale = target_image / height
        scaled_height = target_image
        scaled_width = int(width * scale)
    else:
        scale = target_image / width
        scaled_height = int(height * scale)
        scaled_width = target_image

    new_img = Image.new("RGB", (target_image, target_image))
    pil_img = pil_img.resize((scaled_width, scaled_height), interpolate)
    new_img.paste(pil_img)

    scale = 1. / scale

    return new_img, scale


def preprocess(img_paths, img_ids=None):
    images, scales = [], []

    if img_ids is None:
        img_ids = [0 for _ in range(len(img_paths))]

    for img_path, img_id in zip(img_paths, img_ids):
        pil_img = Image.open(img_path).convert('RGB')
        pil_img, scale = resize(pil_img, cfg.MODEL.IMAGE_SIZE)

        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)

        normalized_np_img = (np_img / 255 - IMAGENET_MEAN) / IMAGENET_STD
        normalized_np_img = np.rollaxis(normalized_np_img, 2)
        images.append(torch.from_numpy(normalized_np_img).float())
        scales.append(scale)

    batch_x = torch.stack(images)

    return batch_x, img_ids, scales

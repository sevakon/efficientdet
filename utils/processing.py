import torch
import config as cfg

from PIL import Image
from utils.transforms import Normalizer, NumpyToTensor, Resizer, ImageToNumpy


def postprocess(cls_outputs, box_outputs):
    """Selects top-k predictions.
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

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1),
                                         dim=1, k=cfg.MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all.floor_divide(cfg.NUM_CLASSES)
    classes_all = cls_topk_indices_all % cfg.NUM_CLASSES

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, cfg.NUM_CLASSES))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


def preprocess(img_paths: list):
    """ Preprocess: image paths to input batch
        Args:
            img_paths (list): list of image paths to be formed into a batch
    """
    images, sizes, scales = [], [], []
    resizer = Resizer(cfg.MODEL.IMAGE_SIZE)
    to_numpy = ImageToNumpy()
    normalizer = Normalizer()
    to_tensor = NumpyToTensor()

    for img_path in img_paths:
        pil_img = Image.open(img_path).convert('RGB')
        sizes.append(pil_img.size)
        pil_img, annos = resizer(pil_img, {})
        scale = annos['scale']

        np_img, _ = to_numpy(pil_img)
        normalized_np_img, _ = normalizer(np_img)
        torch_tensor, _ = to_tensor(normalized_np_img)

        images.append(torch_tensor)
        scales.append(scale)

    batch_x = torch.stack(images)
    return batch_x, sizes, scales

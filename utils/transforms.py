import torch
from PIL import Image
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageToNumpy:
    """ PIL Image to NumPy array"""

    def __call__(self, pil_img, annotations: dict = None):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        return np_img, annotations


class NumpyToTensor:
    """ NumPy array to Torch tensor """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, np_img, annotations: dict = None):
        np_img = np_img.transpose(2, 0, 1)
        torch_img = torch.from_numpy(np_img).to(dtype=self.dtype)
        return torch_img, annotations


class Resizer:
    """ Scales image to the target size by the bigger side """

    def __init__(self, target_size: int, interpolation: str = 'bilinear'):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img, annotations: dict = None):
        width, height = img.size
        if height > width:
            scale = self.target_size / height
            scaled_height = self.target_size
            scaled_width = int(width * scale)
        else:
            scale = self.target_size / width
            scaled_height = int(height * scale)
            scaled_width = self.target_size

        new_img = Image.new("RGB", (self.target_size, self.target_size))
        img = img.resize((scaled_width, scaled_height), Image.BILINEAR)
        new_img.paste(img)

        if 'bbox' in annotations:
            bbox = annotations['bbox']
            bbox[:, :4] *= scale
            annotations['bbox'] = bbox

        annotations['scale'] = 1. / scale

        # from PIL import ImageDraw
        # d = ImageDraw.Draw(new_img)
        # for bbox in annotations['bbox']:
        #     d.rectangle((bbox[1], bbox[0], bbox[3], bbox[2]), outline=5)
        # new_img.show()

        return new_img, annotations


class Normalizer:
    """ Z-Score on an Image """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, annotations: dict = None):
        normalized_image = (image / 255 - self.mean) / self.std
        return normalized_image, annotations


class Compose:
    """ Compose augmentations on both image and bbox """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict = None):
        for transform in self.transforms:
            img, annotations = transform(img, annotations)
        return img, annotations

import cv2
import numpy as np
from PIL import Image


def resize_with_aspect(
        image: np.ndarray,
        size: int = 512
) -> np.ndarray:
    h, w, c = image.shape
    resized = np.zeros((size, size, c))

    scale = size / max(h, w)

    image = cv2.resize(image, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_NEAREST)

    nh, nw, _ = image.shape
    px, py = (size - nw) // 2, (size - nh) // 2

    resized[py:py + nh, px:px + nw, :] = image

    return resized


def resize(img: Image, size=280) -> Image:
    width, height = img.size

    new_height = size
    new_width = int(new_height * width / height)

    img = img.resize((new_width, new_height), Image.BILINEAR)
    return img


def apply_paddings(img: Image, padding: int) -> Image:
    image_copy = np.asarray(img, np.uint8)
    h, w, c = image_copy.shape
    box = np.zeros((h + 2 * padding, w + 2 * padding, min(c, 3)))

    box[padding: padding + h, padding: padding + w, :] = image_copy[:, :, :3]
    return box


def resize_source(img: Image) -> np.ndarray:
    img = resize(img, 512)
    box = apply_paddings(img, 0)
    return box


def resize_target(img: Image) -> np.ndarray:
    img = resize(img, 512)
    box = apply_paddings(img, 75)
    return box

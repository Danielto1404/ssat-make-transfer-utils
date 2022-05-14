import time

import torch
from PIL import Image

from .dataset import MakeupTransferData
from .model import MakeupGAN
from face_parsing import FaceParser
from utils.image_utils import resize_target, resize_source


@torch.no_grad()
def transfer(
        ssat_model: MakeupGAN,
        face_parser: FaceParser,
        source: Image,
        target: Image,
        speed=False,
) -> torch.Tensor:
    target = resize_target(target)
    source = resize_source(source)
    target_parsing = face_parser(target)
    source_parsing = face_parser(source)

    data = MakeupTransferData(source, target, source_parsing, target_parsing).get()

    result = ssat_model.test_pair(data).cpu()[0]
    if speed:
        start = time.time()

        for _ in range(20):
            _ = ssat_model.test_pair(data)

        end = time.time()

        print(f'Finished 20 iterations in: {end - start}')

    result = result / 2 + 0.5

    return result

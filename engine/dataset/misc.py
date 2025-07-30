import importlib.metadata
from torch import Tensor

import torchvision
from torchvision.transforms.v2 import SanitizeBoundingBoxes
from torchvision.tv_tensors import (
    BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
_boxes_keys = ['format', 'canvas_size']

def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"

    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)

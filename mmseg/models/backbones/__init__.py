# Copyright (c) OpenMMLab. All rights reserved.
#from .bisenetv1 import BiSeNetV1
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .erfnet import ERFNet

from .bscnet import bscnet



__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ERFNet', 'bscnet'
]

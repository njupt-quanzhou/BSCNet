# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ...cv_core import kaiming_init, constant_init
from ..utils import brick as vn_layer
from ..builder import BACKBONES


@BACKBONES.register_module()
class RRDarknet53(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.Stage, vn_layer.Stage.custom_layers)

    def __init__(self, pretrained=None, input_channels=32):
        super().__init__()
        stage_cfg = {'stage_2': 2, 'stage_3': 3, 'stage_4': 9, 'stage_5': 9, 'stage_6': 5}

        # Network
        layer_list = [
            # first scale, smallest
            OrderedDict([
                ('stage_1', vn_layer.Conv2dBatchLeaky(3, input_channels, 3, 1)),
                ('stage_2', vn_layer.Stage(input_channels, stage_cfg['stage_2'])),
                ('stage_3', vn_layer.Stage(input_channels * (2 ** 1), stage_cfg['stage_3'])),
                ('stage_4', vn_layer.Stage(input_channels * (2 ** 2), stage_cfg['stage_4'])),
            ]),

            # second scale
            OrderedDict([
                ('stage_5', vn_layer.Stage(input_channels * (2 ** 3), stage_cfg['stage_5'])),
            ]),

            # third scale, largest
            OrderedDict([
                ('stage_6', vn_layer.Stage(input_channels * (2 ** 4), stage_cfg['stage_6'])),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, RRDarknet53.custom_layers)):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            weights = vn_layer.WeightLoader(pretrained)
            for module in self.__modules_recurse():
                try:
                    weights.load_layer(module)
                    print(f'Layer loaded: {module}')
                    if weights.start >= weights.size:
                        print(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                        break
                except NotImplementedError:
                    print(f'Layer skipped: {module.__class__.__name__}')
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        stage_4 = self.layers[0](x)
        stage_5 = self.layers[1](stage_4)
        stage_6 = self.layers[2](stage_5)
        return [stage_6, stage_5, stage_4]  # 由小到大特征图输出

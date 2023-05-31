import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm_str, atomic=False):
    assert not atomic
    if norm_str is None or norm_str.lower() == 'none':
        return nn.Identity
    if norm_str.lower() in ['batch', 'batch_norm', 'bn']:
        return nn.BatchNorm2d
    if re.search('group|gn|group_norm', norm_str.lower()):
        groups = re.search(r'\d+$', norm_str.lower())
        if groups is not None:
            groups = int(groups.group(0))
        else:
            groups = 8
        return lambda num_channels: nn.GroupNorm(groups, num_channels)

    raise ValueError(f"Unknown norm_str: {norm_str}")


###############################################################################################
# Weight Normalized Conv from: https://github.com/joe-siyuan-qiao/WeightStandardization
# Added fan-in factor from normalization free networks
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class WNConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1, 2, 3], keepdim=True) + 1e-5
        fan_in = weight.size(1) * weight.size(2) * weight.size(3)
        weight = weight / (math.sqrt(fan_in) * std.expand_as(weight))
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Weight Normalization
###############################################################################################

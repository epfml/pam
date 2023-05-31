from collections import OrderedDict
from pprint import pprint
import re

import torch
import torch.nn as nn

from shared.modules.factories import get_conv_factory, get_norm_factory, get_linear_factory, get_activation_factory

from ._registry import register_model


class ScalarScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale


def conv3x3(conv_factory, in_channels, out_channels, *, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return conv_factory(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        stride=stride,
        groups=groups,
        bias=bias,
    )


def conv1x1(conv_factory, in_channels, out_channels, *, stride=1, bias=False):
    """1x1 convolution"""
    return conv_factory(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias
    )


class BasicBlock(nn.Module):
    def __init__(self, *,
                 pre_activation,
                 channels_in,
                 channels_out,
                 stride=1,
                 conv_factory=None,
                 norm_factory=None,
                 activation_factory=None,
                 dropout=0.0,
                 skip_init=False,
                 conv_biases=False,
                 **kwargs):  # Extra ignored arguments like groups
        super().__init__()

        if conv_factory is None:
            conv_factory = nn.Conv2d
        if norm_factory is None:
            norm_factory = nn.BatchNorm2d
        if activation_factory is None:
            activation_factory = lambda num_features: nn.ReLU

        self.pre_activation = pre_activation
        self.downsample = None

        if pre_activation:
            self.norm1 = norm_factory(num_features=channels_in)
            self.act1 = activation_factory(num_features=channels_in)
            self.conv1 = conv3x3(conv_factory, channels_in, channels_out,
                                 stride=stride, bias=conv_biases)

            self.norm2 = norm_factory(num_features=channels_out)
            self.act2 = activation_factory(num_features=channels_out)
            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = None
            self.conv2 = conv3x3(conv_factory, channels_out, channels_out, bias=conv_biases)

            if skip_init:
                self.residual_scale = ScalarScale()
            else:
                self.residual_scale = None

            if stride != 1 or channels_in != channels_out:
                self.downsample = nn.Sequential(OrderedDict(
                    # What is the effect of this norm for SkipInit?
                    norm=norm_factory(num_features=channels_in),
                    conv=conv1x1(conv_factory, channels_in, channels_out,
                                 stride=stride, bias=conv_biases),
                ))
        else:
            self.conv1 = conv3x3(conv_factory, channels_in, channels_out,
                                 stride=stride, bias=conv_biases)
            self.norm1 = norm_factory(num_features=channels_out)
            self.act1 = activation_factory(num_features=channels_out)

            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = None

            self.conv2 = conv3x3(conv_factory, channels_out, channels_out, bias=conv_biases)
            self.norm2 = norm_factory(num_features=channels_out)

            if stride != 1 or channels_in != channels_out:
                self.downsample = nn.Sequential(OrderedDict(
                    conv=conv1x1(conv_factory, channels_in, channels_out,
                                 stride=stride, bias=conv_biases),
                    norm=norm_factory(num_features=channels_out),
                ))

            assert not skip_init, "SkipInit not implemented for v1"

            self.act2 = activation_factory(num_features=channels_out)

    def forward(self, x):
        if self.pre_activation:
            out = self.norm1(x)
            out = self.act1(out)
            out = self.conv1(out)

            out = self.norm2(out)
            out = self.act2(out)
            if self.dropout is not None:
                # TODO: Verify this placement for WRN
                out = self.dropout(out)
            out = self.conv2(out)

            if self.residual_scale is not None:
                out = self.residual_scale(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x

            out = out + identity
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)

            if self.dropout is not None:
                # TODO: Verify this placement for WRN
                out = self.dropout(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x

            out = out + identity
            out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    # NOTE that this is v1.5, i.e. the stride is on the 3x3
    def __init__(self, *,
                 pre_activation,
                 channels_in,
                 channels_out,
                 stride=1,
                 cardinality=1,
                 conv_factory=None,
                 norm_factory=None,
                 activation_factory=None,
                 dropout=None,
                 skip_init=False,
                 expansion=4,
                 conv_biases=False,
                 **kwargs):
        super().__init__()

        if conv_factory is None:
            conv_factory = nn.Conv2d
        if norm_factory is None:
            norm_factory = nn.BatchNorm2d
        if activation_factory is None:
            activation_factory = lambda num_features: nn.ReLU

        self.pre_activation = pre_activation
        self.downsample = None
        self.expansion = expansion

        mid_width = cardinality * channels_out // self.expansion

        if pre_activation:
            self.norm1 = norm_factory(num_features=channels_in)
            self.act1 = activation_factory(num_features=channels_in)
            self.conv1 = conv1x1(conv_factory, channels_in, mid_width, bias=conv_biases)

            self.norm2 = norm_factory(num_features=mid_width)
            self.act2 = activation_factory(num_features=mid_width)
            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = None
            self.conv2 = conv3x3(conv_factory,
                                 mid_width,
                                 mid_width,
                                 stride=stride,  # v1.5
                                 groups=cardinality,
                                 bias=conv_biases)

            self.norm3 = norm_factory(num_features=mid_width)
            self.act3 = activation_factory(num_features=mid_width)
            self.conv3 = conv1x1(conv_factory, mid_width, channels_out, bias=conv_biases)

            if skip_init:
                self.residual_scale = ScalarScale()
            else:
                self.residual_scale = None

            if stride != 1 or channels_in != channels_out:
                self.downsample = nn.Sequential(
                    norm_factory(channels_in),
                    conv1x1(conv_factory,
                            channels_in,
                            channels_out,
                            stride=stride,
                            bias=conv_biases),
                )
        else:
            self.conv1 = conv1x1(conv_factory, channels_in, mid_width, bias=conv_biases)
            self.norm1 = norm_factory(num_features=mid_width)
            self.act1 = activation_factory(num_features=mid_width)

            self.conv2 = conv3x3(conv_factory,
                                 mid_width,
                                 mid_width,
                                 stride=stride,  # v1.5
                                 groups=cardinality,
                                 bias=conv_biases)
            self.norm2 = norm_factory(num_features=mid_width)
            self.act2 = activation_factory(num_features=mid_width)
            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)
            else:
                self.dropout = None

            self.conv3 = conv1x1(conv_factory, mid_width, channels_out, bias=conv_biases)
            self.norm3 = norm_factory(num_features=channels_out)

            assert not skip_init, "SkipInit not implemented for v1"

            if stride != 1 or channels_in != channels_out:
                self.downsample = nn.Sequential(
                    conv1x1(conv_factory,
                            channels_in,
                            channels_out,
                            stride=stride,
                            bias=conv_biases),
                    norm_factory(num_features=channels_out),
                )

            self.act3 = activation_factory(num_features=channels_out)

    def forward(self, x):
        if self.pre_activation:
            out = self.norm1(x)
            out = self.act1(out)
            out = self.conv1(out)

            out = self.norm2(out)
            out = self.act2(out)
            if self.dropout is not None:
                # TODO: Verify this placement for WRN
                out = self.dropout(out)
            out = self.conv2(out)

            out = self.norm3(out)
            out = self.act3(out)
            out = self.conv3(out)

            if self.residual_scale is not None:
                out = self.residual_scale(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x

            out += identity
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.act2(out)
            if self.dropout is not None:
                # TODO: Verify this placement for WRN
                out = self.dropout(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x

            out += identity
            out = self.act3(out)

        return out


class ResNet(nn.Module):
    def __init__(self, *,
                 pre_activation=True,
                 block_name='Bottleneck',
                 stage_depths=(3, 4, 6, 3),
                 input_block='imagenet',
                 input_block_channels=64,
                 base_channels=64, # The output channels of the first stage
                 in_chans=3,  # Naming consistent with TIMM
                 num_classes=1000,
                 cardinality=1,
                 zero_init_residual=False,
                 conv_cfg='standard',
                 linear_cfg='standard',
                 norm_cfg='bn2d',
                 activation_cfg='relu',
                 block_dropout=0.0,
                 skip_init=False,
                 conv_biases=False,
                 final_dropout=0.0,
    ):
        super().__init__()
        conv_factory = get_conv_factory(conv_cfg)
        linear_factory = get_linear_factory(linear_cfg)
        norm_factory = get_norm_factory(norm_cfg)
        activation_factory = get_activation_factory(activation_cfg)

        block_cls = {
            'bottleneck': Bottleneck,
            'basicblock': BasicBlock,
        }[block_name.lower()]

        self.pre_activation = pre_activation
        self.cardinality = cardinality

        if input_block == 'imagenet':
            input_modules = OrderedDict(
                conv=conv_factory(
                    in_channels=in_chans,
                    out_channels=input_block_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=conv_biases
                ),
                norm=norm_factory(num_features=input_block_channels),
            )
            if not pre_activation:
                # Commutes with the maxpool for ReLU
                input_modules['act'] = activation_factory(num_features=input_block_channels)

            input_modules['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.input_block = torch.nn.Sequential(input_modules)
        elif input_block == 'cifar':
            input_modules = OrderedDict(
                conv=conv_factory(
                    in_channels=in_chans,
                    out_channels=input_block_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=conv_biases
                ),
            )
            if not pre_activation:
                input_modules['norm'] = norm_factory(num_features=input_block_channels)
                input_modules['act'] = activation_factory(num_features=input_block_channels)
            self.input_block = torch.nn.Sequential(input_modules)
        else:
            raise ValueError(f"Unknown input block: {input_block}")

        stages = []
        channels_in = input_block_channels
        channels_out = base_channels
        for stage_idx, num_blocks in enumerate(stage_depths):
            blocks = []
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                block = block_cls(
                    pre_activation=pre_activation,
                    channels_in=channels_in,
                    channels_out=channels_out,
                    stride=stride,
                    cardinality=cardinality,
                    dropout=block_dropout,
                    skip_init=skip_init,
                    conv_biases=conv_biases,
                    conv_factory=conv_factory,
                    norm_factory=norm_factory,
                    activation_factory=activation_factory,
                )
                blocks.append(block)
                channels_in = channels_out

            stages.append(nn.Sequential(*blocks))
            channels_out = 2 * channels_in
        self.stages = torch.nn.Sequential(*stages)

        output_modules = OrderedDict()
        if pre_activation:
            output_modules['norm'] = norm_factory(num_features=channels_in)
            output_modules['act'] = activation_factory(num_features=channels_in)
        output_modules['pool'] = nn.AdaptiveAvgPool2d((1, 1))
        output_modules['flatten'] = nn.Flatten(1)
        if final_dropout > 0:
            output_modules['dropout'] = nn.Dropout(final_dropout)
        self.output_block = torch.nn.Sequential(output_modules)

        self.out_fc = linear_factory(
            in_features=channels_in,
            out_features=num_classes
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                # Expect other norms to set these by default
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.norm3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.norm2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.stages(x)
        x = self.output_block(x)
        x = self.out_fc(x)
        return x


i1k_depth_to_stages = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

def get_resnet(cfg, verbose=False):
    model_cfg = {}

    if arch := cfg.get('name'):
        # (cifar|i1k)(_pre)?_(wrn|rn)($DEPTH)(_$WIDTH)?
        pre_activation = 'pre' in arch
        wide = 'wrn' in arch

        if wide:
            match = re.search(r'wrn(\d+)_(\d+)', arch)
            depth = int(match.group(1)) # Includes skip convs
            width = int(match.group(2))
        else:
            match = re.search(r'(rn|resnet)(\d+)', arch)
            depth = int(match.group(2))
            width = 1

        dataset_type = arch.split('_')[0]
        if dataset_type == 'i1k':
            depth_adjustment = 3 if wide else 0
            stage_depths = i1k_depth_to_stages[depth - depth_adjustment]
            input_block = 'imagenet'
            num_classes = 1000
            base_width = 64
        elif dataset_type == 'cifar':
            depth_adjustment = 2 if wide else 0
            stage_depths = [(depth - 2 - depth_adjustment) // 6] * 3
            input_block = 'cifar'
            num_classes = 10
            base_width = 16
        else:
            raise ValueError(f'Unknown dataset type for ResNet: {dataset_type}')

        block_name = 'BasicBlock' if depth < 50 or dataset_type == 'cifar' else 'Bottleneck'
        expansion = 4 if block_name == 'Bottleneck' else 1

        model_cfg.update({
            'pre_activation': pre_activation,
            'block_name': block_name,
            'stage_depths': stage_depths,
            'input_block': input_block,
            'num_classes': num_classes,
            'input_block_channels': base_width*width,
            'base_channels': base_width*width*expansion,
        })

    cfg.pop('name', None)  # This is not a argument for the ResNet class
    model_cfg.update(cfg or {})
    model = ResNet(**model_cfg)
    if verbose:
        print("Model config:")
        pprint(model_cfg)

    return model


@register_model
def cifar_resnet(pretrained=False, **kwargs):
    assert not pretrained
    # Remove extra TIMM arguments that we don't support
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('drop_rate', None)
    return get_resnet(kwargs, verbose=False)

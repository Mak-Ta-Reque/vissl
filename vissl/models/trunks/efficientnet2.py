#vissl/models/trunks/efficientnet2.py
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch
import torch.nn as nn
from classy_vision.models.efficientnet import (
    MODEL_PARAMS,
    EfficientNet as ClassyEfficientNet,
)

from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, Wrap, parse_out_keys_arg

from vissl.models.trunks import register_model_trunk


import torch
import torch.nn as nn
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
#EffNet2
class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
            

MODEL_PARAMS = {
    "s":[# t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ],
    "m":[# t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ],
    "l":[# t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ],
    "xl":[# t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
}


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    print(kwargs)
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
model_options ={
    "s": effnetv2_s,
    "m": effnetv2_m

}


@register_model_trunk("efficientnet2")
class EfficientNet2(nn.Module):
    """
    Wrapper for ClassyVision EfficientNet2 model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(EfficientNet2, self).__init__()
        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"

        trunk_config = model_config.TRUNK.EFFICIENT_NETS
        assert "model_version" in trunk_config, "Please specify EfficientNet2 version"
        model_version = trunk_config["model_version"]
        model_params = MODEL_PARAMS[model_version]
        trunk_config["model_params"] = model_params
        trunk_config.pop("model_version")
        # we don't use the FC constructed with num_classes. This param is required
        # to build the model in Classy Vision hence we pass the default value.
        trunk_config["num_classes"] = 1000
        logging.info(f"Building model: EfficientNet-{model_version}")
        model = model_options[model_version]()
        self.drop_connect_rate = model.drop_connect_rate
        self.num_blocks = len(model.blocks)
        self.dropout = model.dropout
        self.activation = Wrap(model.relu_fn)  # using swish, not relu actually

        # We map the layers of model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by out_feat_keys argument in the
        # forward() call.
        # - Stem
        feature_blocks = [
            ["conv1", nn.Sequential(model.conv_stem, model.bn0, self.activation)]
        ]

        # - Mobile Inverted Residual Bottleneck blocks
        feature_blocks.extend(
            [[f"block{i}", v] for i, v in enumerate(model.blocks.children())]
        )

        # - Conv Head + Pooling
        feature_blocks.extend(
            [
                [
                    "conv_final",
                    nn.Sequential(model.conv_head, model.bn1, self.activation),
                ],
                ["avgpool", model.avg_pooling],
                ["flatten", Flatten(1)],
            ]
        )

        if model.dropout:
            feature_blocks.append(["dropout", model.dropout])

        # Consolidate into one indexable trunk
        self._feature_blocks = nn.ModuleDict(feature_blocks)
        self.all_feat_names = list(self._feature_blocks.keys())

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)
        feat = x

        # Walk through the EfficientNet, block by block
        blocks = iter(self._feature_blocks.named_children())

        # - First block is always the stem
        stem_name, stem_block = next(blocks)
        feat = stem_block(feat)
        if stem_name in out_feat_keys:
            out_feats[out_feat_keys.index(stem_name)] = feat

        # - Next go through all the MIRB, then the eventual conv and pooling
        for i, (feature_name, feature_block) in enumerate(blocks):
            if "block" in feature_name:
                # -- MIRB blocks (needs ad-hoc drop connect rate)
                drop_connect_rate = self.drop_connect_rate
                if self.drop_connect_rate:
                    drop_connect_rate *= float(i) / self.num_blocks
                feat = feature_block(feat, drop_connect_rate=drop_connect_rate)
            else:
                # -- Conv, Pooling (simple forward)
                feat = feature_block(feat)

            # If requested, store the feature
            if feature_name in out_feat_keys:
                out_feats[out_feat_keys.index(feature_name)] = feat

            # Early exit if all the features have been collected
            if i == max_out_feat:
                break

        return out_feats

if __name__ == "__main__":
    model = EfficientNet2(*{'model_config':"B", 'model_name':"Eff2" })
    print(model)
    config = effnetv2_s()
    print("Efficient net 2")
    print(config.features)
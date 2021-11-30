""" PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple
from functools import partial
import numpy as np
from config import params

# from efficientDet.backbone.efficientnet_builder import efficientnet


from timm import create_model
from timm.models.layers import create_conv2d, create_pool2d, Swish, get_act_layer
import timm
from effdet.config_bifpn import (
    get_efficientdet_config,
    default_detection_model_configs,
    set_config_readonly,
    get_fpn_config,
)

# import config

# from config.fpn_config import get_fpn_config
# from config.config_utils import set_config_writeable, set_config_readonly

_DEBUG = False

_ACT_LAYER = Swish


class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        channel_multiplier=1.0,
        pw_kernel_size=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class HeadNet(nn.Module):
    def __init__(self, config, num_outputs):
        super(HeadNet, self).__init__()
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, "head_bn_level_first", False)
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_type = (
            config.head_act_type
            if getattr(config, "head_act_type", None)
            else config.act_type
        )
        act_layer = get_act_layer(act_type) or _ACT_LAYER

        # Build convolution repeats
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        conv_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=config.fpn_channels,
            kernel_size=3,
            padding=config.pad_type,
            bias=config.redundant_bias,
            act_layer=None,
            norm_layer=None,
        )
        self.conv_rep = nn.ModuleList(
            [conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats)]
        )

        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        # This can be organized with repeats first or feature levels first in module lists, the original models
        # and weights were setup with repeats first, levels first is required for efficient torchscript usage.
        self.bn_rep = nn.ModuleList()
        if self.bn_level_first:
            for _ in range(self.num_levels):
                self.bn_rep.append(
                    nn.ModuleList(
                        [
                            norm_layer(config.fpn_channels)
                            for _ in range(config.box_class_repeats)
                        ]
                    )
                )
        else:
            for _ in range(config.box_class_repeats):
                self.bn_rep.append(
                    nn.ModuleList(
                        [
                            nn.Sequential(
                                OrderedDict([("bn", norm_layer(config.fpn_channels))])
                            )
                            for _ in range(self.num_levels)
                        ]
                    )
                )

        self.act = act_layer(inplace=True)

        # Prediction (output) layer. Has bias with special init reqs, see init fn.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=num_outputs * num_anchors,
            kernel_size=3,
            padding=config.pad_type,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )
        self.predict = conv_fn(**predict_kwargs)

    @torch.jit.ignore()
    def toggle_bn_level_first(self):
        """ Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        with torch.no_grad():
            new_bn_rep = nn.ModuleList()
            for i in range(len(self.bn_rep[0])):
                bn_first = nn.ModuleList()
                for r in self.bn_rep.children():
                    m = r[i]
                    # NOTE original rep first model def has extra Sequential container with 'bn', this was
                    # flattened in the level first definition.
                    bn_first.append(
                        m[0]
                        if isinstance(m, nn.Sequential)
                        else nn.Sequential(OrderedDict([("bn", m)]))
                    )
                new_bn_rep.append(bn_first)
            self.bn_level_first = not self.bn_level_first
            self.bn_rep = new_bn_rep

    @torch.jit.ignore()
    def _forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)  # this is not allowed in torchscript
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def _forward_level_first(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(
            self.bn_rep
        ):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x)


def _init_weight(m, n=""):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, (fan_in + fan_out) / 2.0)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if "box_net" in n or "class_net" in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if "class_net.predict" in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if "box_net" in n or "class_net" in n:
            m.conv.weight.data.normal_(std=0.01)
            if m.conv.bias is not None:
                if "class_net.predict" in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(m, n=""):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if "class_net.predict" in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ["size", "scale_factor", "mode", "align_corners", "name"]
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == "nearest" else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )


class ResampleFeatureMap(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        reduction_ratio=1.0,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_bn=False,
        conv_after_downsample=False,
        redundant_bias=False,
    ):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or "max"
        upsample = upsample or "nearest"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias,
                act_layer=None,
            )

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module("conv", conv)
            if downsample in ("max", "avg"):
                stride_size = int(reduction_ratio)
                downsample = create_pool2d(
                    downsample,
                    kernel_size=stride_size + 1,
                    stride=stride_size,
                    padding=pad_type,
                )
            else:
                downsample = Interpolate2d(
                    scale_factor=1.0 / reduction_ratio, mode=downsample
                )
            self.add_module("downsample", downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module("conv", conv)
        else:
            if conv is not None:
                self.add_module("conv", conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module(
                    "upsample", Interpolate2d(scale_factor=scale, mode=upsample)
                )


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class FpnCombine(nn.Module):
    def __init__(
        self,
        feature_info,
        fpn_config,
        fpn_channels,
        inputs_offsets,
        target_reduction,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_resample_bn=False,
        conv_after_downsample=False,
        redundant_bias=False,
        weight_method="attn",
    ):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]["num_chs"]
                input_reduction = feature_info[offset]["reduction"]
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]["reduction"]
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels,
                fpn_channels,
                reduction_ratio=reduction_ratio,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias,
            )

        if weight_method == "attn" or weight_method == "fastattn":
            self.edge_weights = nn.Parameter(
                torch.ones(len(inputs_offsets)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == "attn":
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == "fastattn":
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [
                    (nodes[i] * edge_weights[i]) / (weights_sum + 0.0001)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == "sum":
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError("unknown weight_method {}".format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [
            dict(num_chs=f["num_chs"], reduction=f["reduction"])
            for i, f in enumerate(backbone.feature_info())
        ]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=["num_chs", "reduction"])
    return feature_info


class BiFpnLayer(nn.Module):
    def __init__(
        self,
        feature_info,
        fpn_config,
        fpn_channels,
        num_levels=5,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
        apply_resample_bn=False,
        conv_after_downsample=True,
        conv_bn_relu_pattern=False,
        separable_conv=True,
        redundant_bias=False,
    ):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug("fnode {} : {}".format(i, fnode_cfg))
            reduction = fnode_cfg["reduction"]
            combine = FpnCombine(
                feature_info,
                fpn_config,
                fpn_channels,
                tuple(fnode_cfg["inputs_offsets"]),
                target_reduction=reduction,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg["weight_method"],
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if not conv_bn_relu_pattern:
                conv_kwargs["bias"] = redundant_bias
                conv_kwargs["act_layer"] = None
                after_combine.add_module("act", act_layer(inplace=True))
            after_combine.add_module(
                "conv",
                SeparableConv2d(**conv_kwargs)
                if separable_conv
                else ConvBnAct2d(**conv_kwargs),
            )

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels : :]


class BiFpn(nn.Module):
    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level
        )

        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]["num_chs"]
                reduction = feature_info[level]["reduction"]
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    apply_bn=config.apply_resample_bn,
                    conv_after_downsample=config.conv_after_downsample,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug("building cell {}".format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                conv_after_downsample=config.conv_after_downsample,
                conv_bn_relu_pattern=config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


class EfficientDet(nn.Module):
    def __init__(self, config, n_classes_seg, n_classes_depth, batchnorm = False, pretrained_backbone=True, alternate_init=False):
        super(EfficientDet, self).__init__()
        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name,
            features_only=True,
            out_indices=(2, 3, 4),
            pretrained=pretrained_backbone,
            **config.backbone_args
        )
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(self.config, feature_info)
        self.decoder1 = Decoder(n_classes_seg, batchnorm)
        self.decoder2 = Decoder(n_classes_depth, batchnorm)

        self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
        self.box_net = HeadNet(self.config, num_outputs=4)

        self.attn = multi_attention()

        for n, m in self.named_modules():
            if "backbone" not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def forward(self, x):
        x = self.backbone(x)
        """
        TODO: put attention module here
        # x = self.attn(x)
        """

        x = self.attn(x)
        
        [x1, x2, x3, x4, x5] = self.fpn(x)
        x = self.decoder1(x1, x2, x3, x4, x5)
        y = self.decoder2(x1, x2, x3, x4, x5)
        z = self.class_net([x1, x2, x3, x4, x5])
        zz = self.box_net([x1, x2, x3, x4, x5])
        return x, y, z, zz

class Decoder(nn.Module):
    def __init__(self, n_classes, batchnorm = False):
        super(Decoder, self).__init__()
        self.up1 = up(128, 64, batchnorm)
        self.up2 = up(128, 64, batchnorm)
        self.up3 = up(128, 64, batchnorm)
        self.up4 = up(128, 64, batchnorm)
        self.outc = outconv(64, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, bn):
        super(double_conv, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bn, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class attention(nn.Module):

    """
    TODO: create a attention module 
    x1 : [1, C, H, W] / [C, H*W]
    input: x (list of [x1, x2, x3], x1 has the biggest height x width) 
    x1 : [batch, 40, H/8, W/8] / [C, H*W]
    x2 : [batch, 112, H/16, W/16]
    x3 : [batch, 320, H/32, W/32]    output: x
    """

    def __init__(self, in_feature, head_num, bias=True, activation=F.relu):
        super(attention, self).__init__()

        self.in_feature = in_feature
        self.head_num = head_num
        self.activation = activation

        self.linear_q = nn.Linear(in_feature, in_feature, device = params.device)
        self.linear_k = nn.Linear(in_feature, in_feature, device = params.device)
        self.linear_v = nn.Linear(in_feature, in_feature, device = params.device)
        self.linear_o = nn.Linear(in_feature, in_feature, device = params.device)
    
    def forward(self, q, k, v, mask=None):
        
        B, C, H, W = q.shape

        q_p = q.permute(0,2,3,1)
        k_p = k.permute(0,2,3,1)
        v_p = v.permute(0,2,3,1)


        q_p, k_p, v_p = self.linear_q(q_p.reshape(B,-1, C)), self.linear_k(k_p.reshape(B,-1, C)), self.linear_v(v_p.reshape(B,-1, C))

        q_p = torch.cat(torch.tensor_split(q_p, self.head_num, dim=-1), dim=0)  #[B*num_head, H*W, C/num_head]
        k_p = torch.cat(torch.tensor_split(k_p, self.head_num, dim=-1), dim=0)  #[B*num_head, H*W, C/num_head]
        v_p = torch.cat(torch.tensor_split(v_p, self.head_num, dim=-1), dim=0)  #[B*num_head, H*W, C/num_head]
        
        attention, attention_score = self.scaled_dot_product_attention(q_p, k_p, v_p, mask)
        #attention score : [B*num_head, H*W, H*W]

        output = torch.cat(torch.tensor_split(attention, self.head_num, dim=0), dim=-1) #[B, H*W, C]
        output = self.linear_o(output)

        output_re = output.permute(0,2,1).reshape(B,C,H,W)

        return output_re


    def scaled_dot_product_attention(self, q, k, v, mask):
        
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2,-1)) / math.sqrt(dk) #[B*num_head, H*W, H*W]

        attention = F.softmax(scores, dim=-1)

        return attention.matmul(v), attention #[B*num_head, H*W, C/num_head]

class multi_attention(nn.Module):
    def __init__(self):
        super(multi_attention, self).__init__()
        """
        x1 : [batch, 40, H/8, W/8] / [C, H*W]
        x2 : [batch, 112, H/16, W/16]
        x3 : [batch, 320, H/32, W/32]
        """
        
        feature_size = [40, 112, 320]
        self.attention_net = [attention(feature_size[i], head_num=4) for i in range(len(feature_size))]

    def forward(self, x):
        
        # return [self.attention_net[i](x[i],x[i],x[i]) for i in range(len(x))]
        return [x[0], x[1], self.attention_net[2](x[2], x[2], x[2])]

        
if __name__ == "__main__":
    inp = torch.randn(1, 40, 64, 64)
    resample = ResampleFeatureMap(in_channels=40, out_channels=112, reduction_ratio=0.5)
    out = resample(inp)

    config = get_efficientdet_config("efficientdet_d0")
    # print(config)
    print(inp.shape, out.shape)
    backbone = timm.create_model(
        config.backbone_name,
        features_only=True,
        out_indices=(2, 3, 4),
        pretrained=True,
        **config.backbone_args
    )
    # print(backbone)
    feature_info = get_feature_info(backbone)
    print(feature_info, "\n")

    efficientdet = EfficientDet(config, 34, 1)
    output = efficientdet(torch.randn(1, 3, 512, 512))

    print("output shape:", np.shape(output[0]), np.shape(output[1]))


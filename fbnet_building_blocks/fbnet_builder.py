# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from .layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
    _NewEmptyTensorOp
)
from . import fbnet_modeldef
import importlib

import sys

sys.path.append('../')

from general_functions.utils import count_conv_flop

logger = logging.getLogger(__name__)

def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


PRIMITIVES = {
    "skip": lambda C_in, C_out, expansion, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, **kwargs
    ),
    "ir_k5": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=5, **kwargs
    ),
    "ir_k7": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=7, **kwargs
    ),
    "ir_k1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=1, **kwargs
    ),
    "shuffle": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "basic_block": lambda C_in, C_out, expansion, stride, **kwargs: CascadeConv3x3(
        C_in, C_out, stride
    ),
    "shift_5x5": lambda C_in, C_out, expansion, stride, **kwargs: ShiftBlock5x5(
        C_in, C_out, expansion, stride
    ),
    # layer search 2
    "ir_k3_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, **kwargs
    ),
    "ir_k3_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, **kwargs
    ),
    "ir_k3_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, **kwargs
    ),
    "ir_k3_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
    ),
    "ir_k5_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, **kwargs
    ),
    "ir_k5_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, **kwargs
    ),
    "ir_k5_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, **kwargs
    ),
    "ir_k5_s4": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
    ),
    # layer search se
    "ir_k3_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
    ),
    "ir_k3_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    "ir_k5_e1_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e3_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_e6_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
    ),
    "ir_k5_s4_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        4,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=4,
        se=True,
        **kwargs
    ),
    # layer search 3 (in addition to layer search 2)
    "ir_k3_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k5_s2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
    ),
    "ir_k3_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=3,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    "ir_k5_s2_se": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in,
        C_out,
        1,
        stride,
        kernel=5,
        shuffle_type="mid",
        pw_group=2,
        se=True,
        **kwargs
    ),
    # layer search 4 (in addition to layer search 3)
    "ir_k3_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=3, cdw=True, **kwargs
    ),
    "ir_k33_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=3, cdw=True, **kwargs
    ),
    # layer search 5 (in addition to layer search 4)
    "ir_k7_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, **kwargs
    ),
    "ir_k7_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, **kwargs
    ),
    "ir_k7_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, **kwargs
    ),
    "ir_k7_sep": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 1, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e3": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 3, stride, kernel=7, cdw=True, **kwargs
    ),
    "ir_k7_sep_e6": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, 6, stride, kernel=7, cdw=True, **kwargs
    ),
    # for SimpleNet
    "s_k3": lambda C_in, C_out, expansion, stride, **kwargs: Simple(
        C_in, C_out, stride, kernel=3
    ),
    "s_k5": lambda C_in, C_out, expansion, stride, **kwargs: Simple(
        C_in, C_out, stride, kernel=5
    ),
    "s_k7": lambda C_in, C_out, expansion, stride, **kwargs: Simple(
        C_in, C_out, stride, kernel=7
    ),
    # for ResNet
    "r_k3": lambda C_in, C_out, expansion, stride, **kwargs: ResidualPreAct(
        C_in, C_out, stride, kernel=3
    ),
    "r_k5": lambda C_in, C_out, expansion, stride, **kwargs: ResidualPreAct(
        C_in, C_out, stride, kernel=5
    ),
    "r_k7": lambda C_in, C_out, expansion, stride, **kwargs: ResidualPreAct(
        C_in, C_out, stride, kernel=7
    ),
}


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class DownSample(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(DownSample, self).__init__()
        self.conv = Conv2d(C_in, C_out, kernel_size=1, stride=stride, bias=False)
        #self.conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, bias=False)
        self.bn = BatchNorm2d(C_out)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        return out
    

class ResidualPreAct(nn.Module):
    def __init__(self, C_in, C_out, stride, kernel):
        super(ResidualPreAct, self).__init__()
        self.output_depth = C_out # ANNA's code here
        self.stride = stride 

        if self.stride == 2:
            input_channel = int(C_in/2)
        else:
            input_channel = C_in

        self.conv1 = (
            BNReluConv(
                input_channel,
                C_in,
                kernel=kernel,
                stride=stride,
                pad=(kernel//2),
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
        )
        self.conv2 = (
            BNReluConv(
                C_in,
                C_in,
                kernel=kernel,
                stride=1,
                pad=(kernel//2),
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
        )
        
        if self.stride == 2:
            self.downsample = DownSample(
                    input_channel,
                    C_in,
                    stride=2
                    )


    def forward(self, x):
        if self.stride == 2:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + shortcut 
        
        return out

    def get_flops(self, x):
        self.flops = count_conv_flop(self.conv, x)
        out = self.conv(x)
        return self.flops, out



class Simple(nn.Module):
    def __init__(self, C_in, C_out, stride, kernel):
        super(Simple, self).__init__()
        self.output_depth = C_out # ANNA's code here
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out

    def get_flops(self, x):
        if self.conv:
            # self.flops = 0
            self.flops = count_conv_flop(self.conv, x)
            out = self.conv(x)
        else:
            self.flops = 0
            out = x
        return self.flops, out


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out # ANNA's code here
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out

    def get_flops(self, x):

        if self.conv:
            # self.flops = 0
            self.flops = count_conv_flop(self.conv, x)
            out = self.conv(x)
        else:
            self.flops = 0
            out = x
        return self.flops, out


class CascadeConv3x3(nn.Sequential):
    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [
            Conv2d(C_in, C_in, 3, stride, 1, bias=False),
            BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            BatchNorm2d(C_out),
        ]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = (stride == 1) and (C_in == C_out)

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):
    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx : ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(
                x,
                self.kernel,
                self.bias,
                (self.stride, self.stride),
                (self.padding, self.padding),
                self.dilation,
                self.C,  # groups
            )

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                (self.padding, self.dilation),
                (self.dilation, self.dilation),
                (self.kernel_size, self.kernel_size),
                (self.stride, self.stride),
            )
        ]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ShiftBlock5x5(nn.Sequential):
    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = (stride == 1) and (C_in == C_out)

        C_mid = _get_divisible_by(C_in * expansion, 8, 8)

        ops = [
            # pw
            Conv2d(C_in, C_mid, 1, 1, 0, bias=False),
            BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            # shift
            Shift(C_mid, 5, stride, 2),
            # pw-linear
            Conv2d(C_mid, C_out, 1, 1, 0, bias=False),
            BatchNorm2d(C_out),
        ]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )



class BNReluConv(nn.Sequential):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(BNReluConv, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]
        # for flops calculation

        self.stride = (stride, stride)
        self.in_channels = input_depth
        self.out_channels = output_depth
        self.kernel_size = (kernel, kernel)
        self.groups = group

        self.bn = None
        if bn_type == "bn":
            self.bn = BatchNorm2d(input_depth)
        elif bn_type == "gn":
            self.bn = nn.GroupNorm(num_groups=gn_group, num_channels=input_depth)
        elif bn_type == "af":
            self.bn = FrozenBatchNorm2d(input_depth)
        # if bn_type is not None:
            # self.add_module("bn", self.bn_op)

        self.relu = None
        if use_relu == "relu":
            self.relu = nn.ReLU(inplace=True)
            # self.add_module("relu",  self.activation)

        self.conv = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        # self.add_module("conv", self.op)


    def get_flops(self, x, only_flops=False):
        flops1 = count_conv_flop(self.conv, x)

        if only_flops:
            return flops1

        if self.bn != None:
            y = self.bn(x)

        if self.relu != None:
            y = self.relu(y)
        
        y = self.conv(y)

        return flops1, y


class ConvBNRelu(nn.Sequential):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]
        # for flops calculation

        self.stride = (stride, stride)
        self.in_channels = input_depth
        self.out_channels = output_depth
        self.kernel_size = (kernel, kernel)
        self.groups = group


        self.conv = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        # self.add_module("conv", self.op)

        self.bn = None
        if bn_type == "bn":
            self.bn = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            self.bn = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            self.bn = FrozenBatchNorm2d(output_depth)
        # if bn_type is not None:
            # self.add_module("bn", self.bn_op)

        self.relu = None
        if use_relu == "relu":
            self.relu = nn.ReLU(inplace=True)
            # self.add_module("relu",  self.activation)

    def get_flops(self, x, only_flops=False):
        flops1 = count_conv_flop(self.conv, x)

        if only_flops:
            return flops1

        y = self.conv(x)

        if self.bn != None:
            y = self.bn(y)

        if self.relu != None:
            y = self.relu(y)

        return flops1, y



class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
        stride in [1, 2, 4]
        or stride in [-1, -2, -4]
        or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


class IRFBlock(nn.Module):
    def __init__(
        self,
        input_depth,
        output_depth,
        expansion,
        stride,
        bn_type="bn",
        kernel=3,
        width_divisor=1,
        shuffle_type=None,
        pw_group=1,
        se=False,
        cdw=False,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu="relu",
            bn_type=bn_type,
            group=pw_group,
        )

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # self.dw_kernel == 1
        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu",
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=None,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y

    def get_flops(self, x):
        flops1, y = self.pw.get_flops(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)

        flops2, y = self.dw.get_flops(y)
        flops3, y = self.pwl.get_flops(y)

        flops4 = 0


        if self.use_res_connect:
            y += x

            flops4 = 0
            # TODO - update for skip connection flops
            # ProxylessNas doesn't add skip connection flops
        y = self.se4(y)
        return (flops1 + flops2 + flops3 + flops4), y


def _expand_block_cfg(block_cfg):
    assert isinstance(block_cfg, list)
    ret = []
    for idx in range(block_cfg[2]):
        cur = copy.deepcopy(block_cfg)
        cur[2] = 1
        cur[3] = 1 if idx >= 1 else cur[3]
        ret.append(cur)
    return ret


def expand_stage_cfg(stage_cfg):
    """ For a single stage """
    assert isinstance(stage_cfg, list)
    ret = []
    for x in stage_cfg:
        ret += _expand_block_cfg(x)
    return ret


def expand_stages_cfg(stage_cfgs):
    """ For a list of stages """
    assert isinstance(stage_cfgs, list)
    ret = []
    for x in stage_cfgs:
        ret.append(expand_stage_cfg(x))
    return ret


def _block_cfgs_to_list(block_cfgs):
    assert isinstance(block_cfgs, list)
    ret = []
    for stage_idx, stage in enumerate(block_cfgs):
        stage = expand_stage_cfg(stage)
        for block_idx, block in enumerate(stage):
            cur = {"stage_idx": stage_idx, "block_idx": block_idx, "block": block}
            ret.append(cur)
    return ret


def _add_to_arch(arch, info, name):
    """ arch = [{block_0}, {block_1}, ...]
        info = [
            # stage 0
            [
                block0_info,
                block1_info,
                ...
            ], ...
        ]
        convert to:
        arch = [
            {
                block_0,
                name: block0_info,
            },
            {
                block_1,
                name: block1_info,
            }, ...
        ]
    """
    assert isinstance(arch, list) and all(isinstance(x, dict) for x in arch)
    assert isinstance(info, list) and all(isinstance(x, list) for x in info)
    idx = 0
    for stage_idx, stage in enumerate(info):
        for block_idx, block in enumerate(stage):
            assert (
                arch[idx]["stage_idx"] == stage_idx
                and arch[idx]["block_idx"] == block_idx
            ), "Index ({}, {}) does not match for block {}".format(
                stage_idx, block_idx, arch[idx]
            )
            assert name not in arch[idx]
            arch[idx][name] = block
            idx += 1


def unify_arch_def(arch_def):
    """ unify the arch_def to:
        {
            ...,
            "arch": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    ...
                },
                {}, ...
            ]
        }
    """
    ret = copy.deepcopy(arch_def)

    assert "block_cfg" in arch_def and "stages" in arch_def["block_cfg"]
    assert "stages" not in ret
    # copy 'first', 'last' etc. inside arch_def['block_cfg'] to ret
    ret.update({x: arch_def["block_cfg"][x] for x in arch_def["block_cfg"]})
    ret["stages"] = _block_cfgs_to_list(arch_def["block_cfg"]["stages"])
    del ret["block_cfg"]

    assert "block_op_type" in arch_def
    _add_to_arch(ret["stages"], arch_def["block_op_type"], "block_op_type")
    del ret["block_op_type"]
    
    print('--jieun:(type):', type(ret))
    print('--jieun: ret:', ret)

    return ret


def get_num_stages(arch_def):
    ret = 0
    for x in arch_def["stages"]:
        ret = max(x["stage_idx"], ret)
    ret = ret + 1
    return ret


def get_blocks(arch_def, stage_indices=None, block_indices=None):
    ret = copy.deepcopy(arch_def)
    ret["stages"] = []
    for block in arch_def["stages"]:
        keep = True
        if stage_indices not in (None, []) and block["stage_idx"] not in stage_indices:
            keep = False
        if block_indices not in (None, []) and block["block_idx"] not in block_indices:
            keep = False
        if keep:
            ret["stages"].append(block)
    return ret


class FBNetBuilder(object):
    def __init__(
        self,
        width_ratio,
        bn_type="bn",
        width_divisor=1,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.bn_type = bn_type
        self.width_divisor = width_divisor
        self.dw_skip_bn = dw_skip_bn
        self.dw_skip_relu = dw_skip_relu

    def add_first(self, stage_info, dim_in=3, pad=True):
        # stage_info: [c, s, kernel]
        assert len(stage_info) >= 2
        channel = stage_info[0]
        stride = stage_info[1]
        out_depth = self._get_divisible_width(int(channel * self.width_ratio))
        kernel = 3
        if len(stage_info) > 2:
            kernel = stage_info[2]

        out = ConvBNRelu(
            dim_in,
            out_depth,
            kernel=kernel,
            stride=stride,
            pad=kernel // 2 if pad else 0,
            no_bias=1,
            use_relu="relu",
            bn_type=self.bn_type,
        )
        self.last_depth = out_depth
        return out
    
    def add_first_resnet(self, stage_info, dim_in=3, pad=True):
        # stage_info: [c, s, kernel]
        assert len(stage_info) >= 2
        channel = stage_info[0]
        stride = stage_info[1]
        out_depth = self._get_divisible_width(int(channel * self.width_ratio))
        kernel = 7
        if len(stage_info) > 2:
            kernel = stage_info[2]

        out = ConvBNRelu(
            dim_in,
            out_depth,
            kernel=kernel,
            stride=stride,
            pad=3,
            no_bias=1,
            use_relu="relu",
            bn_type=self.bn_type,
        )

        self.last_depth = out_depth
        return out

    def add_maxpool_resnet(self):
        out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return out

    def add_blocks(self, blocks):
        """ blocks: [{}, {}, ...]
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            #print("stage_idx =", stage_idx)
            
            block_idx = block["block_idx"]
            block_op_type = block["block_op_type"]
            tcns = block["block"]
            n = tcns[2]
            assert n == 1
            nnblock = self.add_ir_block(tcns, [block_op_type])
            nn_name = "xif{}_{}".format(stage_idx, block_idx)
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        return ret

    # def add_final_pool(self, model, blob_in, kernel_size):
    #     ret = model.AveragePool(blob_in, "final_avg", kernel=kernel_size, stride=1)
    #     return ret

    def _add_ir_block(
        self, dim_in, dim_out, stride, expand_ratio, block_op_type, **kwargs
    ):
        ret = PRIMITIVES[block_op_type](
            dim_in,
            dim_out,
            expansion=expand_ratio,
            stride=stride,
            bn_type=self.bn_type,
            width_divisor=self.width_divisor,
            dw_skip_bn=self.dw_skip_bn,
            dw_skip_relu=self.dw_skip_relu,
            **kwargs
        )
        return ret, ret.output_depth

    def add_ir_block(self, tcns, block_op_types, **kwargs):
        t, c, n, s = tcns
        assert n == 1
        out_depth = self._get_divisible_width(int(c * self.width_ratio))
        dim_in = self.last_depth
        op, ret_depth = self._add_ir_block(
            dim_in,
            out_depth,
            stride=s,
            expand_ratio=t,
            block_op_type=block_op_types[0],
            **kwargs
        )
        self.last_depth = ret_depth
        return op

    def _get_divisible_width(self, width):
        ret = _get_divisible_by(int(width), self.width_divisor, self.width_divisor)
        return ret

    def add_last_states(self, cnt_classes, dropout_ratio=0.2):
        assert cnt_classes >= 1
        op = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(self.last_depth, 1280, kernel_size = 1)),
            ("dropout", nn.Dropout(dropout_ratio)),
            ("avg_pool_k7", nn.AdaptiveAvgPool2d((1,1))),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1280, out_features=cnt_classes)),
        ]))
        self.last_depth = cnt_classes
        return op
    
    def add_last_states_resnet(self, cnt_classes, dropout_ratio=0.2):
        assert cnt_classes >= 1
        op = nn.Sequential(OrderedDict([
            ("avg_pool_k7", nn.AdaptiveAvgPool2d((1,1))),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=512, out_features=cnt_classes)),
        ]))
        self.last_depth = cnt_classes
        return op


def _get_trunk_cfg(arch_def):
    num_stages = get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    ret = get_blocks(arch_def, stage_indices=trunk_stages)
    return ret

class FBNet(nn.Module):
    def __init__(
        self, builder, arch_def, dim_in, cnt_classes=1000, supernet_type='mobilenetv2'
    ):
        super(FBNet, self).__init__()
        
        self.supernet_type = supernet_type

        if self.supernet_type == 'resnet':
            self.first = builder.add_first_resnet(arch_def["first"], dim_in=dim_in)
            self.maxpool = builder.add_maxpool_resnet()
        else:
            self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])
        if self.supernet_type == 'resnet':
            self.last_stages = builder.add_last_states_resnet(cnt_classes)
        else:
            self.last_stages = builder.add_last_states(cnt_classes)
        self.cnt_classes = cnt_classes
    
    def forward(self, x):
        y = self.first(x)

        if self.supernet_type == 'resnet':
            y = self.maxpool(y)
            pass
        
        y = self.stages(y)
        y = self.last_stages(y)
        return y
        
    def get_flops(self, x):
        accumlated_flops = 0
        flops2 = 0

        # first conv
        flops1, y = self.first.get_flops(x)
        accumlated_flops += flops1

        # searched space
        for i in range(len(self.stages)):
            flops, y = self.stages[i].get_flops(y)
            flops2 += flops

        accumlated_flops += flops2

        # last stage
        # if backbone changed, need, code change!!
        last_conv_temp = ConvBNRelu(input_depth=320, output_depth=1280, kernel=1,
                                    stride=1,
                                    pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        # if stride changed, need code change!!
        data_shape = [1, 320, 4, 4]
        x = torch.torch.zeros(data_shape).cuda()

        flops3 = last_conv_temp.get_flops(x, only_flops=True) + nn.Linear(in_features=1280, out_features=self.cnt_classes).weight.numel()

        accumlated_flops += flops3

        print('first stage : ', flops1)
        print('search space : ', flops2)
        print('last stage : ', flops3)
        return accumlated_flops

def get_model(arch, cnt_classes):
    # for reload updated arch
    importlib.reload(fbnet_modeldef)
    assert arch in fbnet_modeldef.MODEL_ARCH
    arch_def = fbnet_modeldef.MODEL_ARCH[arch]
    arch_def = unify_arch_def(arch_def)
    builder = FBNetBuilder(width_ratio=1.0, bn_type="bn", width_divisor=8, dw_skip_bn=True, dw_skip_relu=True)
    model = FBNet(builder, arch_def, dim_in=3, cnt_classes=cnt_classes)
    return model

def get_model_simple(arch, cnt_classes):
    # for reload updated arch
    importlib.reload(fbnet_modeldef)
    assert arch in fbnet_modeldef.MODEL_ARCH
    arch_def = fbnet_modeldef.MODEL_ARCH[arch]
    arch_def = unify_arch_def(arch_def)
    builder = FBNetBuilder(width_ratio=1.0, bn_type="bn", width_divisor=8, dw_skip_bn=True, dw_skip_relu=True)
    model = FBNet(builder, arch_def, dim_in=3, cnt_classes=cnt_classes, supernet_type='resnet')
    return model


##################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #if pretrained:
    #    state_dict = load_state_dict_from_url(model_urls[arch],
    #                                          progress=progress)
    #    model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


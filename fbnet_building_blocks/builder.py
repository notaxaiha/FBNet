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
}

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

class Builder(object):
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


        # nn.init.kaiming_normal_(conv0.weight, mode="fan_out", nonlinearity="relu")

        op = nn.Sequential(OrderedDict([
            ("conv_0", nn.Conv2d(dim_in, out_depth, kernel_size=kernel, stride=stride, padding=kernel // 2 if pad else 0, bias = False)),
            ("bn_0", nn.BatchNorm2d(out_depth)),
            ("act_0", nn.ReLU(inplace=True)),
        ]))

        self.last_depth = out_depth
        return op

    def add_blocks(self, blocks):
        """ blocks: [{}, {}, ...]
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            # print("stage_idx =", stage_idx)

            block_idx = block["block_idx"]
            block_op_type = block["block_op_type"]
            tcns = block["block"]
            n = tcns[2]
            assert n == 1


            # TODO : Need fix
            nnblock = self.add_ir_block(tcns, [block_op_type])


            nn_name = "xif{}_{}".format(stage_idx, block_idx)
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        return ret

    # def add_final_pool(self, model, blob_in, kernel_size):
    #     ret = model.AveragePool(blob_in, "final_avg", kernel=kernel_size, stride=1)
    #     return ret

    # TODO : Need fix
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
            ("conv_k1", nn.Conv2d(self.last_depth, 1280, kernel_size=1)),
            ("dropout", nn.Dropout(dropout_ratio)),
            ("avg_pool_k7", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1280, out_features=cnt_classes)),
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
        self, builder, arch_def, dim_in, cnt_classes=1000
    ):
        super(FBNet, self).__init__()
        self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])
        self.last_stages = builder.add_last_states(cnt_classes)

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = self.last_stages(y)
        return y

def get_model(arch, cnt_classes):
    # for reload updated arch
    importlib.reload(fbnet_modeldef)
    assert arch in fbnet_modeldef.MODEL_ARCH
    arch_def = fbnet_modeldef.MODEL_ARCH[arch]
    arch_def = unify_arch_def(arch_def)
    builder = FBNetBuilder(width_ratio=1.0, bn_type="bn", width_divisor=8, dw_skip_bn=True, dw_skip_relu=True)
    model = FBNet(builder, arch_def, dim_in=3, cnt_classes=cnt_classes)
    return model

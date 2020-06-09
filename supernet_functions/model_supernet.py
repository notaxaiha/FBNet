import torch
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

class MixedOperation(nn.Module):
    
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, flops):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.flops = [flops[op_name] for op_name in ops_names]
        # self.flops = [1 for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))

    def forward(self, x, temperature, flops_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)

        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        # latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))

        flops = sum(m * flop for m, flop in zip(soft_mask_variables, self.flops))
        flops_to_accumulate = flops_to_accumulate + flops
        return output, flops_to_accumulate

    # update get Flops for data
    def get_flops(self, x, temperature):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)

        # print(self.ops)
        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        # get flops
        self.flops = [op.get_flops(x)[0] for op in self.ops]

        return output

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table, cnt_classes=1000):
        super(FBNet_Stochastic_SuperNet, self).__init__()

        data_shape = [1, 3, 32, 32]
        # data_shape = [1, 3, 224, 224]
        x = torch.torch.zeros(data_shape).cuda()

        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        # imagenet
        # self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=1,
        #                         pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=1,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.first_flops = self.first.get_flops(x, only_flops=True)
        print(self.first_flops)

        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_flops[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])
        self.last_stages = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(lookup_table.layers_parameters[-1][1], 1280, kernel_size = 1)),
            ("avg_pool_k7", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1280, out_features=cnt_classes)),
        ]))

        # conv and fc flops
        # fc flops == weights (1280 x 10)

        last_conv_temp = ConvBNRelu(input_depth=lookup_table.layers_parameters[-1][1], output_depth=1280, kernel=1, stride=1,
                                pad=0, no_bias=1, use_relu="relu", bn_type="bn")

        # if stride change, need change!!
        data_shape = [1, lookup_table.layers_parameters[-1][1], 4, 4]
        x = torch.torch.zeros(data_shape).cuda()

        self.last_stages_flops = last_conv_temp.get_flops(x, only_flops=True) + nn.Linear(in_features=1280, out_features=cnt_classes).weight.numel()
        print(self.last_stages_flops)
        del data_shape, x, last_conv_temp

    
    def forward(self, x, temperature, flops_to_accumulate):
        y = self.first(x)
        # add flops from first layer
        flops_to_accumulate += self.first.get_flops(x, only_flops=True)

        for mixed_op in self.stages_to_search:
            y, flops_to_accumulate = mixed_op(y, temperature, flops_to_accumulate)
        y = self.last_stages(y)

        # add flops from last stage
        flops_to_accumulate += self.last_stages_flops

        return y, flops_to_accumulate

    # TODO -
    def get_flops(self, x, temperature):
        flops_list = []

        y = self.first(x)
        for mixed_op in self.stages_to_search:
            y = mixed_op.get_flops(y, temperature)

        for mixed_op in self.stages_to_search:

            flops_list.append(mixed_op.flops)
        y = self.last_stages(y)
        return y, flops_list
    
class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.reg_lambda = CONFIG_SUPERNET['loss']['reg_lambda']
        self.weight_criterion = nn.CrossEntropyLoss()
        self.reg_loss_type= CONFIG_SUPERNET['loss']['reg_loss_type']
        # self.ref_value = 300 * 1e6
        self.ref_value = 300 * 1e6 * 0.1

    
    def forward(self, outs, targets, flops_to_accumulate, losses_ce, losses_flops, flops , N):
        
        ce_loss = self.weight_criterion(outs, targets)
        # TODO - FLops loss

        # print(flops_to_accumulate)

        losses_ce.update(ce_loss.item(), N)
        flops.update(flops_to_accumulate.item(), N)

        if self.reg_loss_type == 'mul#log':
            alpha = self.alpha
            beta = self.beta
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(flops_to_accumulate) / math.log(self.ref_value)) ** beta

            losses_flops.update(reg_loss.item(), N)

            return alpha * ce_loss * reg_loss

        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_lambda
            reg_loss = reg_lambda * (flops_to_accumulate - self.ref_value) / self.ref_value

            # print(reg_loss)
            losses_flops.update(reg_loss.item(), N)

            return ce_loss + reg_loss

        # loss = ce # self.alpha * ce * lat
        # return loss #.unsqueeze(0)


# ProxylessNAS search/imagenet_arch_search.py
# ref_values = {
#     'flops': {
#         '0.35': 59 * 1e6,
#         '0.50': 97 * 1e6,
#         '0.75': 209 * 1e6,
#         '1.00': 300 * 1e6,
#         '1.30': 509 * 1e6,
#         '1.40': 582 * 1e6,
#     },
#     # ms
#     'mobile': {
#         '1.00': 80,
#     },
#     'cpu': {},
#     'gpu8': {},
# }

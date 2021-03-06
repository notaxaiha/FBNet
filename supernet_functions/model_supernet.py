import torch
import random
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

params_list = []

class MixedOperation(nn.Module):

    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, flops, params):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.flops = [flops[op_name] for op_name in ops_names]
        # self.flops = [1 for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))

        # self.params= self.get_params()
        self.params= [params[op_name] for op_name in ops_names]


    def forward(self, x, temperature, flops_to_accumulate, params_to_accumulate, sampling_mode=None):
        # old_gumbel
        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        
        # new_gumbel
        if sampling_mode == 'sampling_valid':
            # argmax_one_hot mask (for valid)
            soft_mask_variables = torch.zeros(len(self.thetas))
            soft_mask_variables[torch.argmax(self.thetas)] = 1
            soft_mask_variables = soft_mask_variables.cuda()
        elif sampling_mode == 'sampling_train':
            # argmax_one_hot mask (for train)
            soft_mask_variables = self.get_gumbel_prob(temperature)
            soft_mask_variables = torch.zeros(len(soft_mask_variables))
            soft_mask_variables[torch.argmax(soft_mask_variables)] = 1
            soft_mask_variables = soft_mask_variables.cuda()
        elif sampling_mode == 'random_sampling' :
            # random_one_hot mask
            soft_mask_variables = torch.zeros(len(self.thetas))
            soft_mask_variables[int(random.random() * 9)] = 1
            soft_mask_variables = soft_mask_variables.cuda()
        elif sampling_mode == 'sum' :
            # sum mask
            soft_mask_variables = torch.ones(len(self.thetas))
            soft_mask_variables = soft_mask_variables.cuda()
        else: 
            # weighted sum
            soft_mask_variables = self.get_gumbel_prob(temperature)
            
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))

        # latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))

        flops = sum(m * flop for m, flop in zip(soft_mask_variables, self.flops))
        flops_to_accumulate = flops_to_accumulate + flops
        
        params = sum(m * param for m, param in zip(soft_mask_variables, self.params))
        params_to_accumulate = params_to_accumulate + params
        
        return output, flops_to_accumulate, params_to_accumulate

    # update get Flops for data
    def get_flops(self, x, temperature):
        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)

        soft_mask_variables = self.get_gumbel_prob(temperature)

        # print(self.ops)
        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        # get flops
        self.flops = [op.get_flops(x)[0] for op in self.ops]

        return output

    def get_params(self):
        params_list = []

        # N canidate in Mixed Operation
        for op in self.ops:
            params_list.append(sum(p.numel() for p in op.parameters() if p.requires_grad))

        return(params_list)
            # print(x.parameters())
            # print(x.i)

    # gumbel softmax function
    def get_gumbel_prob(self, tau):
        gumbels = -torch.empty_like(self.thetas).exponential_().log()
        logits = (self.thetas.log_softmax(dim=-1) + gumbels) / tau
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs



class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table, params_lookup_table, cnt_classes=1000):
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
        self.first_params = sum(p.numel() for p in self.first.parameters() if p.requires_grad)
        
        # print(self.first_flops)
        # print(self.first_params)

        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_flops[layer_id],
                                                   params_lookup_table.lookup_table_flops[layer_id])
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


        self.last_stages_params = sum(p.numel() for p in self.last_stages.parameters() if p.requires_grad)
        
        # print(self.last_stages_flops)
        # print(self.last_stages_params)
        del data_shape, x, last_conv_temp


    
    def forward(self, x, temperature, flops_to_accumulate, params_to_accumulate, sampling_mode=None):
        y = self.first(x)
        # add flops from first layer
        flops_to_accumulate += self.first.get_flops(x, only_flops=True)
        params_to_accumulate = self.first_params

        for mixed_op in self.stages_to_search:
            y, flops_to_accumulate, params_to_accumulate = mixed_op(y, temperature, flops_to_accumulate, params_to_accumulate, sampling_mode)
        y = self.last_stages(y)

        # add flops from last stage
        flops_to_accumulate += self.last_stages_flops

        params_to_accumulate += self.last_stages_params

        return y, flops_to_accumulate, params_to_accumulate

    # TODO -
    def get_flops(self, x, temperature):
        flops_list = []
        params_list = []

        y = self.first(x)
        for mixed_op in self.stages_to_search:
            y = mixed_op.get_flops(y, temperature)

        for mixed_op in self.stages_to_search:

            flops_list.append(mixed_op.flops)
            params_list.append(mixed_op.get_params())
            print('flops', mixed_op.flops)
            print('params', mixed_op.get_params())

        y = self.last_stages(y)
        return y, flops_list, params_list
    
class SupernetLoss(nn.Module):
    def __init__(self, alpha, beta, reg_lambda, reg_loss_type, ref_value = 30 * 1e6, apply_flop_loss="False") :
        super(SupernetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reg_lambda = reg_lambda
        self.weight_criterion = nn.CrossEntropyLoss()
        self.reg_loss_type= reg_loss_type
        # self.ref_value = 300 * 1e6
        self.ref_value = ref_value
        self.apply_flop_loss = apply_flop_loss

    
    def forward(self, outs, targets, flops_to_accumulate, params_to_accumulate, losses_ce, losses_flops, flops, params, N):
        
        ce_loss = self.weight_criterion(outs, targets)

        # value update to tb
        losses_ce.update(ce_loss.item(), N)
        flops.update(flops_to_accumulate.item(), N)
        params.update(params_to_accumulate.item(), N)
        
        if self.apply_flop_loss == "False":
            return ce_loss

        assert self.apply_flop_loss == "True", 'apply_flop_loss must be "True" or "Fasle".'

        # print(flops_to_accumulate)

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

import distiller
import torch
import torch.optim
import torch.nn as nn
from distiller.quantization.clipped_linear import PACTQuantizer
from fbnet_building_blocks.layers.misc import Conv2d

def convert_model_to_quant(model, yaml_path, optimizer=None):
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=0.0002)

    with open(yaml_path, 'r') as stream:
        sched_dict = distiller.utils.yaml_ordered_load(stream)

    scheduler = distiller.CompressionScheduler(model)
    quantizers = distiller.config.__factory('quantizers', model, sched_dict, optimizer=optimizer)

    lr_policies = []
    for policy_def in sched_dict['policies']:
        policy = None
        if 'quantizer' in policy_def:
            instance_name, args = distiller.config.__policy_params(policy_def, 'quantizer')
            quantizer = quantizers[instance_name]
            apply_weight_qbit_in_FBNet_Custom_conv2D(quantizer)
            policy = distiller.QuantizationPolicy(quantizer)

        elif 'lr_scheduler' in policy_def:
            lr_policies.append(policy_def)
            continue

        distiller.config.add_policy_to_scheduler(policy, policy_def, scheduler, False)

    resumed_epoch = None
    lr_schedulers = distiller.config.__factory('lr_schedulers', model, sched_dict, optimizer=optimizer,
                                               last_epoch=(resumed_epoch if resumed_epoch is not None else -1))

    for policy_def in lr_policies:
        instance_name, args = distiller.config.__policy_params(policy_def, 'lr_scheduler')
        lr_scheduler = lr_schedulers[instance_name]
        policy = distiller.LRPolicy(lr_scheduler)
        distiller.config.add_policy_to_scheduler(policy, policy_def, scheduler, False)

    return scheduler, optimizer


def apply_weight_qbit_in_FBNet_Custom_conv2D(quantizer):
    qbits = quantizer.default_qbits
    model = quantizer.model

    for module_full_name, module in model.named_modules():
        if type(module) == Conv2d :
            quantizer.module_qbits_map[module_full_name] = qbits


def delete_float_w(model):
    for module_full_name, module in model.named_modules():
        if type(module) == Conv2d :
            if 'float_weight' in dir(module):
            # print(dir(module))
                del(module.float_weight)

def mobilenet_relu6_to_relu(model):
    model.features[0][2] = nn.ReLU(inplace=True)

    model.features[1].conv[0][2] = nn.ReLU(inplace=True)

    for i in range(2, 18):
        model.features[i].conv[0][2] = nn.ReLU(inplace=True)
        model.features[i].conv[1][2] = nn.ReLU(inplace=True)

    model.features[18][2] = nn.ReLU(inplace=True)

    return model
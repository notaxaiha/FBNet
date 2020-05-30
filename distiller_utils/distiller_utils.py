import distiller
import torch
import torch.optim
from distiller.quantization.clipped_linear import PACTQuantizer
from fbnet_building_blocks.layers.misc import Conv2d

def convert_model_to_pact(model, yaml_path, optimizer=None):
    if optimizer == None :
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=0.0002)

    with open(yaml_path, 'r') as stream:
        sched_dict = distiller.utils.yaml_ordered_load(stream)

    scheduler = distiller.CompressionScheduler(model)
    quantizers = distiller.config.__factory('quantizers', model, sched_dict, optimizer=optimizer)

    for policy_def in sched_dict['policies']:
        if 'quantizer' in policy_def:
            instance_name, args = distiller.config.__policy_params(policy_def, 'quantizer')
            quantizer = quantizers[instance_name]
            print(quantizer.act_clip_decay)
            apply_weight_qbit_in_FBNet_Custom_conv2D(quantizer)
            policy = distiller.QuantizationPolicy(quantizer)
            distiller.config.add_policy_to_scheduler(policy, policy_def,scheduler)

    return scheduler


def apply_weight_qbit_in_FBNet_Custom_conv2D(quantizer):
    qbits = quantizer.default_qbits
    model = quantizer.model

    for module_full_name, module in model.named_modules():
        if type(module) == Conv2d :
            quantizer.module_qbits_map[module_full_name] = qbits

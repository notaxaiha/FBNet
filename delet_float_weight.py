import distiller
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import distiller.apputils.image_classifier as classifier
import fbnet_building_blocks.fbnet_builder as fbnet_builder
import logging
import random
import numpy as np
import distiller_utils.distiller_utils as du

from examples.classifier_compression import parser
from distiller.data_loggers import *
from distiller.apputils import image_classifier
from distiller.apputils import load_data
from fbnet_building_blocks.layers.misc import Conv2d

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

yaml_path = './yaml/FBNet_DoReFa_int2.yaml'
model_path = 'FBNet_DoReFa_int2.pth'

def delete_float_w(model):
    for module_full_name, module in model.named_modules():
        if type(module) == Conv2d :
            if 'float_weight' in dir(module):
            # print(dir(module))
                del(module.float_weight)
            

def main():

    # freeze seed value
    manual_seed = 472
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset load
    loader = load_data('cifar10', '../data.cifar10', batch_size=128, workers=1)
    (train_loader, val_loader, test_loader) = (loader[0], loader[1], loader[2])

    # mobilenet v2 define and change for FBNet
    #### Model
    #### Model
    arch = "FBNet_DoReFa_w2a2"
    model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    model = torch.nn.DataParallel(model)
    compression_scheduler, optimizer = du.convert_model_to_quant(model.module.stages,yaml_path)
    
    model.load_state_dict(torch.load(model_path), strict=False)
    
    delete_float_w(model)
    
    torch.save(model.state_dict(), "./FBNet_DoReFa_int2_delete_float_weight.pth")
    
main()
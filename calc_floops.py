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
from torchsummary import summary

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

yaml_path = './yaml/FBNet_DoReFa_int2.yaml'
model_path = 'FBNet_DoReFa_int2_delete_float_weight.pth'


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

    # mobilenet v2 define and change for FBNet
    #### Model
    #### Model
    arch = "FBNet_DoReFa_w2a2"
    model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    # model = torch.nn.DataParallel(model)
    # compression_scheduler, optimizer = du.convert_model_to_quant(model.module.stages,yaml_path)
    
    model.load_state_dict(torch.load(model_path), strict=False)
    
    data_shape = [1, 3, 32, 32]
    # data_shape = [1, 3, 224, 224]
    input_var = torch.zeros(data_shape).cuda()

    model = model.train()
    input_var = input_var.cuda(non_blocking=True)

    flops = model.get_flops(input_var)

    print('Model\'s Flops : ', flops)

    print(summary(model, input_size=(data_shape[1], data_shape[2], data_shape[3])))
    
main()
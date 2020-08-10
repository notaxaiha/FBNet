import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import os

from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, create_directories_from_list
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from architecture_functions.training_functions import TrainerArch
from architecture_functions.config_for_arch import CONFIG_ARCH
from distiller_utils.distiller_utils import convert_model_to_quant

from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


parser = argparse.ArgumentParser("architecture")

parser.add_argument('--architecture_name', type=str, default='', \
                    help='You can choose architecture from the fbnet_building_blocks/fbnet_modeldef.py')
parser.add_argument('--quantization', type=str, default='', \
                    help='quantization yaml file path')
parser.add_argument('--fnp', type=bool, default=False, \
                    help='if this value is true, only return Flops & Params')
args = parser.parse_args()

def main():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    create_directories_from_list([CONFIG_ARCH['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_ARCH['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_ARCH['logging']['path_to_tensorboard_logs'])

    #### DataLoading
    train_loader = get_loaders(1.0, CONFIG_ARCH['dataloading']['batch_size'],
                               CONFIG_ARCH['dataloading']['path_to_save_data'],
                               logger)
    valid_loader = get_test_loader(CONFIG_ARCH['dataloading']['batch_size'],
                                   CONFIG_ARCH['dataloading']['path_to_save_data'])
    
    #### Model
    arch = args.architecture_name
    yaml_path = args.quantization
    
    # flops & param
    fnp = args.fnp
    
    model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    model = model.apply(weights_init)
    
    # only calculate flops and params
    if fnp == True:
        #### Training Loop
        data_shape = [1, 3, 32, 32]
        # data_shape = [1, 3, 224, 224]
        input_var = torch.zeros(data_shape).cuda()

        model = model.train()
        input_var = input_var.cuda(non_blocking=True)

        flops = model.get_flops(input_var)

        print('Model\'s Flops : ', flops)

        print(summary(model, input_size=(data_shape[1], data_shape[2], data_shape[3])))

        return
    
    model = nn.DataParallel(model, [0])
    
    #### Loss and Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CONFIG_ARCH['optimizer']['lr'],
                                momentum=CONFIG_ARCH['optimizer']['momentum'],
                                weight_decay=CONFIG_ARCH['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss().cuda()
    
    # convert model quantization
    if yaml_path :
        compression_scheduler, optimizer = convert_model_to_quant(model.module.stages,yaml_path)
    else :
        compression_scheduler = None
    print(model)
    #### Scheduler
    if CONFIG_ARCH['train_settings']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=CONFIG_ARCH['train_settings']['milestones'],
                                                    gamma=CONFIG_ARCH['train_settings']['lr_decay'])  
    elif CONFIG_ARCH['train_settings']['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG_ARCH['train_settings']['cnt_epochs'],
                                                               eta_min=0.001, last_epoch=-1)
    else:
        logger.info("Please, specify scheduler in architecture_functions/config_for_arch")
        
    
    #### Training Loop
    trainer = TrainerArch(criterion, optimizer, scheduler, logger, writer, comp_scheduler=compression_scheduler)   
    trainer.train_loop(train_loader, valid_loader, model) 
    
if __name__ == "__main__":
    main()

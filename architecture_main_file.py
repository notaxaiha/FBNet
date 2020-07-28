import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import os
import random
from os.path import join, curdir, isdir, exists

from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, create_directories_from_list
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from architecture_functions.training_functions import TrainerArch
from architecture_functions.config_for_arch import CONFIG_ARCH
from distiller_utils.distiller_utils import convert_model_to_quant

from torchsummary import summary

from shutil import copytree, copy, rmtree


parser = argparse.ArgumentParser("architecture")

parser.add_argument('--architecture_name', type=str, default='', \
                    help='You can choose architecture from the fbnet_building_blocks/fbnet_modeldef.py')
parser.add_argument('--quantization', type=str, default='', \
                    help='quantization yaml file path')
parser.add_argument('--fnp', type=bool, default=False, \
                    help='if this value is true, only return Flops & Params')
parser.add_argument('--seed', type=int, default=1, \
                    help='seed for python and pytorch')
parser.add_argument('--gpu', type=str, default='0', \
                    help='gpu number to use')
parser.add_argument('--dataset', type=str, default='cifar10', \
                    help='using dataset')

# SGD optimizer - weight
parser.add_argument('--lr', type=float, default=0.1, \
                    help="weight optimizer's lr")
parser.add_argument('--momentum', type=float, default=0.9, \
                    help="weight optimizer's momentum")
parser.add_argument('--decay', type=float, default=1e-4, \
                    help="weight optimizer's decay")

# training setting
parser.add_argument('--epoch', type=int, default=360, \
                    help="train epoch")
parser.add_argument('--print_freq', type=int, default=50, \
                    help="set train step printing frequency")
parser.add_argument('--batch', type=int, default=100, \
                    help="set batch size")
parser.add_argument('--data_split', type=float, default=0.8, \
                    help="split dataset for weight, theta (x : 1-x)")

# training setting - scheduler
parser.add_argument('--scheduler', type=str, default="CosineAnnealingLR", \
                    help="set scheduler option 'CosineAnnealingLR' or MultiStepLR")
parser.add_argument('--eta_min', type=float, default=1e-3, \
                    help="lr cosine anneling eta_min value")

args = parser.parse_args()

code_list = ['architecture_functions', 'architecture_main_file.py', 'fbnet_building_blocks']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    manual_seed = args.seed

    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path = join(curdir, 'searched_result', args.architecture_name, 'architecture_function_logs')

    # copy code for search architecture
    for file in code_list:
        if isdir(join(curdir, file)):
            if exists(join(save_path, 'code', file)):
                rmtree(join(save_path, 'code', file))

            copytree(join(curdir, file), join(save_path, 'code', file))
        else:
            if exists(join(save_path, 'code', file)):
                os.remove(join(save_path, 'code', file))

            copy(join(curdir, file), join(save_path, 'code', file))

    create_directories_from_list([join(save_path, 'tb'),
                                  join(save_path, 'code')])

    logger = get_logger(join(save_path, 'logger'))
    writer = SummaryWriter(log_dir=join(save_path, 'tb'))

    #### DataLoading
    train_loader = get_loaders(1.0, args.batch,
                               CONFIG_ARCH['dataloading']['path_to_save_data'],
                               dataset=args.dataset)
    valid_loader = get_test_loader(args.batch,
                                   CONFIG_ARCH['dataloading']['path_to_save_data'],
                                   dataset=args.dataset)

    #### Model
    arch = args.architecture_name
    yaml_path = args.quantization

    # flops & param
    fnp = args.fnp
    if args.dataset == 'cifar10':
        model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    elif args.dataset == 'cifar100':
        model = fbnet_builder.get_model(arch, cnt_classes=100).cuda()

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
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().cuda()

    # convert model quantization
    if yaml_path:
        compression_scheduler, optimizer = convert_model_to_quant(model.module.stages, yaml_path)
    else:
        compression_scheduler = None
    print(model)
    #### Scheduler
    if args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=CONFIG_ARCH['train_settings']['milestones'],
                                                         gamma=CONFIG_ARCH['train_settings']['lr_decay'])
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epoch,
                                                               eta_min=args.eta_min, last_epoch=-1)
    else:
        logger.info("Please, specify scheduler in architecture_functions/config_for_arch")

    #### Training Loop
    trainer = TrainerArch(criterion, optimizer, scheduler, logger, writer, epoch=args.epoch, print_freq=args.print_freq,
                          comp_scheduler=compression_scheduler,
                          path_to_save_model=join(save_path, 'best_model.pth'))

    trainer.train_loop(train_loader, valid_loader, model)


if __name__ == "__main__":
    main()

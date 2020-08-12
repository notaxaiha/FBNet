import numpy as np
import os
from os.path import join, curdir, isdir, exists
import random
import torch
from torch import nn
from tensorboardX import SummaryWriter
from scipy.special import softmax
import argparse
import json
from shutil import copytree, copy, rmtree

from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH
from distiller_utils.distiller_utils import convert_model_to_quant

import fbnet_building_blocks.fbnet_builder as fbnet_builder

parser = argparse.ArgumentParser("action")
parser.add_argument('--train_or_sample', type=str, default='', \
                    help='train means training of the supernet, sample means sample from supernet\'s results')
parser.add_argument('--architecture_name', type=str, default='', \
                    help='name of an architecture to be sampled')
parser.add_argument('--hardsampling_bool_value', type=str, default='True', \
                    help='if not False or 0 -> do hardsampling, else - softmax sampling')
parser.add_argument('--quantization', type=str, default='', \
                    help='quantization yaml file path')
parser.add_argument('--seed', type=int, default=472, \
                    help='seed for python and pytorch')
parser.add_argument('--gpu', type=str, default='0', \
                    help='gpu number to use')
parser.add_argument('--dataset', type=str, default='cifar10', \
                    help='using dataset')

# SGD optimizer - weight
parser.add_argument('--w_lr', type=float, default=0.1, \
                    help="weight optimizer's lr")
parser.add_argument('--w_momentum', type=float, default=0.9, \
                    help="weight optimizer's momentum")
parser.add_argument('--w_decay', type=float, default=1e-4, \
                    help="weight optimizer's decay")

# Adam optimizer - theta
parser.add_argument('--thetas_lr', type=float, default=0.01, \
                    help="theatas optimizer's lr")
parser.add_argument('--thetas_decay', type=float, default=5 * 1e-4, \
                    help="theatas optimizer's decay")

# loss
parser.add_argument('--apply_flop_loss', type=bool, default=False, \
                    help="apply flops loss or not")
parser.add_argument('--reg_loss_type', type=str, default="add#linear", \
                    help="choose flop loss type 'add#linear' or 'mul#log'")
parser.add_argument('--ref_value', type=float, default=30 * 1e6, \
                    help="flop loss's ref value")
parser.add_argument('--reg_lambda', type=float, default=1e-1, \
                    help="flop loss's (reg)lambda value")
parser.add_argument('--alpha', type=float, default=0.2, \
                    help="(mul#log)flop loss's alpha value")
parser.add_argument('--beta', type=float, default=0.3, \
                    help="(mul#log)flop loss's beta value")

# training setting
parser.add_argument('--epoch', type=int, default=180, \
                    help="train epoch")
parser.add_argument('--warm_up', type=int, default=10, \
                    help="set warm_up epoch (if you don't want wram_up, set 0)")
parser.add_argument('--print_freq', type=int, default=50, \
                    help="set train step printing frequency")
parser.add_argument('--batch', type=int, default=128, \
                    help="set batch size")
parser.add_argument('--data_split', type=float, default=0.8, \
                    help="split dataset for weight, theta (x : 1-x)")

# evalation setting
parser.add_argument('--eval_mode', type=str, default=None, \
                    help="select evalution method. (default) None(same with training) / sampling")

# TODO : warmup stage and gumbel scheduling
parser.add_argument('--tau_scheduling', type=str, default='exp', \
                    help="tau scheduling mode - choose 'exp' (default) or 'cos'.")
parser.add_argument('--eta_max', type=float, default=5, \
                    help="max gumbel tau value")
parser.add_argument('--eta_min', type=float, default=None, \
                    help="min gumbel tau value")
parser.add_argument('--exp_anneal_rate', type=float, default=np.exp(-0.045), \
                    help="flops loss (reg)lambda value")

# Lookup Table - 
parser.add_argument('--params_LUT_path', type=str, default='./supernet_functions/params_lookup_table.txt', \
                    help="saved params lookup table path")
parser.add_argument('--flops_LUT_path', type=str, default='./supernet_functions/lookup_table.txt', \
                    help="saved flops lookup table path")

# dataset
parser.add_argument('--dataset_path', type=str, default='./cifar10_data', \
                    help="saved dataset path")
# dataset , lambda, warmup steps, epochs,
args = parser.parse_args()

yaml_path = ''

copy_list = ['supernet_functions', 'supernet_main_file.py', 'general_functions', 'yaml', 'distiller_utils',
             'fbnet_building_blocks']


def train_supernet():
    save_path = join(curdir, 'searched_result', args.architecture_name, 'supernet_function_logs')

    create_directories_from_list([join(save_path, 'tb'),
                                  join(save_path, 'code')])

    # copy code for search architecture
    for file in copy_list:
        if isdir(join(curdir, file)):
            if exists(join(save_path, 'code', file)):
                rmtree(join(save_path, 'code', file))

            copytree(join(curdir, file), join(save_path, 'code', file))
        else:
            if exists(join(save_path, 'code', file)):
                os.remove(join(save_path, 'code', file))

            copy(join(curdir, file), join(save_path, 'code', file))

    # train setting
    with open(join(save_path, 'code', 'search_hyperparmeter.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger = get_logger(join(save_path, 'logger'))
    writer = SummaryWriter(log_dir=join(save_path, 'tb'))

    #### lookup table consists all information about layers
    lookup_table = LookUpTable(calulate_latency=False, path=args.flops_LUT_path)
    # params_lookup_table = LookUpTable(calulate_latency=False, path=args.params_LUT_path)

    #### dataloading
    train_w_loader, train_thetas_loader = get_loaders(args.data_split,
                                                      args.batch,
                                                      args.dataset_path,
                                                      dataset=args.dataset)
    test_loader = get_test_loader(args.batch,
                                  args.dataset_path,
                                  dataset=args.dataset)

    #### model

    if args.dataset == 'cifar10':
        model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).cuda()
    elif args.dataset == 'cifar100':
        model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=100).cuda()

    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.SGD(params=params_except_thetas,
                                  lr=args.w_lr,
                                  momentum=args.w_momentum,
                                  weight_decay=args.w_decay)

    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=args.thetas_lr,
                                       weight_decay=args.thetas_decay)

    if args.quantization:
        yaml_path = args.quantization
        comp_scheduler, w_optimizer = convert_model_to_quant(model.stages_to_search, yaml_path, optimizer=w_optimizer)
    else:
        comp_scheduler = None

    model = model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=[0])
    # print(model)
    #### loss, optimizer and scheduler
    criterion = SupernetLoss(reg_loss_type=args.reg_loss_type, alpha=args.alpha, beta=args.beta,
                             reg_lambda=args.reg_lambda, ref_value=args.ref_value).cuda()
    criterion.apply_flop_loss = args.apply_flop_loss

    # thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    # params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=args.epoch,
                                                             last_epoch=last_epoch)

    #### training loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer,
                              tau_scheduling=args.tau_scheduling ,temperature=args.eta_max, min_temperature=args.eta_min, exp_anneal_rate=args.exp_anneal_rate, epoch=args.epoch,
                              train_thetas_from_the_epoch=args.warm_up, print_freq=args.print_freq,
                              comp_scheduler=comp_scheduler, path_to_save_model=join(save_path, 'best_model.pth'))
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model, args.eval_mode)


# arguments:
# hardsampling=True means get operations with the largest weights
#             =False means apply softmax to weights and sample from the distribution
# unique_name_of_arch - name of architecture. will be written into fbnet_building_blocks/fbnet_modeldef.py
#                       and can be used in the training by train_architecture_main_file.py
def sample_architecture_from_the_supernet(unique_name_of_arch, hardsampling=True):
    logger = get_logger(join(curdir, 'searched_result', args.architecture_name, 'supernet_function_logs', 'logger'))

    lookup_table = LookUpTable()

    if args.dataset == 'cifar10':
        model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).cuda()
    elif args.dataset == 'cifar100':
        model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=100).cuda()

    if args.quantization:
        w_optimizer = None
        yaml_path = args.quantization
        comp_scheduler, w_optimizer = convert_model_to_quant(model.stages_to_search, yaml_path, optimizer=w_optimizer)
    else:
        comp_scheduler = None
    model = nn.DataParallel(model)

    load(model, join(curdir, 'searched_result', args.architecture_name,
                                                      'supernet_function_logs', 'best_model.pth'))

    ops_names = [op_name for op_name in lookup_table.lookup_table_operations]
    cnt_ops = len(ops_names)

    arch_operations = []
    if hardsampling:
        for layer in model.module.stages_to_search:
            arch_operations.append(ops_names[np.argmax(layer.thetas.detach().cpu().numpy())])
    else:
        rng = np.linspace(0, cnt_ops - 1, cnt_ops, dtype=int)
        for layer in model.module.stages_to_search:
            distribution = softmax(layer.thetas.detach().cpu().numpy())
            arch_operations.append(ops_names[np.random.choice(rng, p=distribution)])

    logger.info("sampled architecture: " + " - ".join(arch_operations))
    writh_new_ARCH_to_fbnet_modeldef(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
    logger.info("congratulations! new architecture " + unique_name_of_arch \
                + " was written into fbnet_building_blocks/fbnet_modeldef.py")


def check_flops():
    #### lookup table consists all information about layers
    lookup_table = LookUpTable(calulate_latency=False)

    #### dataloading
    data_shape = [1, 3, 32, 32]
    # data_shape = [1, 3, 224, 224]
    input_var = torch.zeros(data_shape).cuda()

    #### model
    model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).cuda()
    model = model.apply(weights_init)

    #### training loop
    trainer = TrainerSupernet(None, None, None, None, None, None, check_flops=True,
                              tau_scheduling=args.tau_scheduling, temperature=args.eta_max, min_temperature=args.eta_min, exp_anneal_rate=args.exp_anneal_rate, epoch=args.epoch,
                              train_thetas_from_the_epoch=args.warm_up, print_freq=args.print_freq,
                              path_to_save_model=join(curdir, 'searched_result', args.architecture_name,
                                                      'supernet_function_logs', 'best_model.pth'))
    flops_list = trainer.train_loop(None, None, input_var, model)

    lookup_table.write_lookup_table_to_file(path_to_file=args.flops_LUT_path,
                                            flops_list=flops_list)


if __name__ == "__main__":

    # set gpu number to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seed number for python and pytorch
    manual_seed = args.seed

    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert args.train_or_sample in ['train', 'sample', 'flops']
    if args.train_or_sample == 'train':
        assert args.architecture_name != '' or args.architecture_name in MODEL_ARCH
        train_supernet()

    elif args.train_or_sample == 'sample':
        assert args.architecture_name != '' and args.architecture_name not in MODEL_ARCH
        hardsampling = False if args.hardsampling_bool_value in ['False', '0'] else True
        sample_architecture_from_the_supernet(unique_name_of_arch=args.architecture_name, hardsampling=hardsampling)


    elif args.train_or_sample == 'flops':
        check_flops()

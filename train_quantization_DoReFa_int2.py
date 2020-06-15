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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

yaml_path = "./yaml/FBNet_DoReFa_int2.yaml"

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
    arch = "FBNet_DoReFa_w2a2"
    model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    model = torch.nn.DataParallel(model)
    compression_scheduler, optimizer = du.convert_model_to_quant(model.module.stages,yaml_path)
    print(model)
    
    criterion = nn.CrossEntropyLoss()

    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    args.device = 'cuda'
    args.print_freq = 50
    args.seed = 472
    args.dataset = 'cifar10'

    msglogger = logging.getLogger()
    image_classifier._init_logger(args, '/home/dwkim/Project/FBNet')
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)
    loggers = [tflogger, pylogger]

    activations_collectors = image_classifier.create_activation_stats_collectors(model, args.activation_stats)
    for epoch in range(0, 200):
        with collectors_context(activations_collectors["train"]) as collectors:
            top1, top5, loss = image_classifier.train(train_loader, model, criterion, optimizer, epoch=epoch, compression_scheduler=compression_scheduler, loggers=loggers, args=args)
            image_classifier.validate(val_loader, model, criterion, loggers, args, epoch)
            compression_scheduler.on_epoch_end(epoch, optimizer, metrics={'min': loss, 'max': top1})
    torch.save(model.state_dict(), "./FBNet_DoReFa_int2.pth")

if __name__ == '__main__':
    main()
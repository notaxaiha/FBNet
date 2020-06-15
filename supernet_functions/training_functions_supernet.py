import torch
from torch.autograd import Variable
import time
import math
from general_functions.utils import AverageMeter, save, accuracy
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
# from tensorboardX import SummaryWriter
import pandas as pd
import logging
import argparse
import distiller.apputils.image_classifier as classifier
from distiller.apputils import image_classifier
from examples.classifier_compression import parser
from distiller.data_loggers import *
import distiller
import distiller.models as models
from distiller.utils import float_range_argparse_checker as float_range

def init_classifier_compression_arg_parser(include_ptq_lapq_args=False):
    '''Common classifier-compression application command-line arguments.
    '''
    SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']

    parser = argparse.ArgumentParser(description='Distiller image classification model compression')
    # parser.add_argument('data', metavar='DATASET_DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', type=lambda s: s.lower(),
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, metavar='N', default=90,
                        help='number of total epochs to run (default: 90')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    load_checkpoint_group_exc.add_argument('--resume', dest='deprecated_resume', default='', type=str,
                        metavar='PATH', help=argparse.SUPPRESS)
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path',
                        default='', type=str, metavar='PATH',
                        help='path to checkpoint to load weights from (excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true',
                        help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser.add_argument('--activation-histograms', '--act-hist',
                        type=float_range(exc_min=True),
                        metavar='PORTION_OF_TEST_SET',
                        help='Run the model in evaluation mode on the specified portion of the test dataset and '
                             'generate activation histograms. NOTE: This slows down evaluation significantly')
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES, action='append',
                        help='print a summary of the model, and exit - options: | '.join(SUMMARY_CHOICES))
    parser.add_argument('--export-onnx', action='store', nargs='?', type=str, const='model.onnx', default=None,
                        help='export model to ONNX format')
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(), help='test the sensitivity of layers to pruning')
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing '
                             'providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')
    parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')
    parser.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=include_ptq_lapq_args)
    return parser

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, comp_scheduler=None):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']

        self.comp_scheduler = comp_scheduler

        # custom scalar
        # https://tensorboardx.readthedocs.io/en/latest/tensorboard.html?highlight=custom#tensorboardX.SummaryWriter.add_custom_scalars
        # https://github.com/lanpa/tensorboardX/blob/master/examples/demo_custom_scalars.py
        # self.custom_dict = dict()
        # self.thetas_dict = dict()
        # self.thetas_gs_dict = dict()
        #
        # # for i in range(len(self.theta_optimizer.param_groups[0]['params'])):
        # for i in range(22):
        #     self.thetas_dict[f"layer_{i}"] = ['Multiline', list()]
        #     # self.thetas_gs_dict[f"layer_{i}"] = ['Multiline', list()]
        #
        #     self.thetas_dict[f"layer_{i}"][1] = [f"layer_{i}/block_{j}" for j in range(9)]
        #     # self.thetas_gs_dict[f"layer_{i}"][1] = [f"gumbel_thetas/layer_{i}/block_{j}" for j in range(9)]
        #
        # self.custom_dict['thetas'] = self.thetas_dict
        # # self.custom_dict['gumbel_thetas'] = self.thetas_gs_dict
        #
        # self.writer.add_custom_scalars(self.custom_dict)

    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):

        best_top1 = 0.0
        
        # firstly, train weights only
        # for epoch in range(self.train_thetas_from_the_epoch):
        #     self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
        #
        #     self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
        #     self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
        #     self.w_scheduler.step()

        all_theta_list = []

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            top1, losses = self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            # self.w_scheduler.step()
            self.comp_scheduler.on_epoch_end(epoch, self.w_optimizer, metrics={'min': losses, 'max': top1})

            self.logger.info("Start to train theta for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch, info_for_logger="_theta_step_")


            theta_list = []
            for i in range(17):

                temp_list = self.theta_optimizer.param_groups[0]['params'][i].tolist()
                theta_list.append(temp_list)

            all_theta_list.append([theta_list, self.temperature])

            top1_avg = self._validate(model, test_loader, epoch)
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)

            # self.writer.add_scalar('temperature', self.temperature, epoch)


            self.temperature = self.temperature * self.exp_anneal_rate
            
            
        pd.DataFrame(all_theta_list).to_csv('./supernet_functions/logs/theatas.csv')


       
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model = model.train()
        start_time = time.time()

        total_samples = len(loader.sampler)
        batch_size = loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)
        
        
        for step, (X, y) in enumerate(loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]
            

            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()

            self.comp_scheduler.on_minibatch_begin(epoch, step, steps_per_epoch, optimizer)

            outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
            loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)
            
            agg_loss = self.comp_scheduler.before_backward_pass(epoch, step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss                                                      
            '''
            print("loss : ", loss)
            
            print("agg_loss : ", agg_loss)
            
            print("overall_loss : ", loss)
            '''

            optimizer.zero_grad()

            # loss.backward(retain_graph=True)
            loss.backward()

            # torch.nn.utils.clip_grad_norm(model.parameters(), 1)


            self.comp_scheduler.before_parameter_optimization(epoch, step, steps_per_epoch, optimizer)

            # clip val update
            optimizer.step()
            # print("[wook] optimizer", optimizer)
            self.comp_scheduler.on_minibatch_end(epoch, step, steps_per_epoch, optimizer)
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')

        #for avg in [self.top1, self.top3, self.losses]:
        #    avg.reset()
            
        return self.top1, self.losses
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]
                
                latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
                outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
                loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,3) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg()))
        

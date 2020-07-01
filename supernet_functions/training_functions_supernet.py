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
from distiller_utils.distiller_utils import delete_float_w

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, check_flops=False, comp_scheduler=None):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_flops = AverageMeter()
        self.losses_ce  = AverageMeter()
        self.flops = AverageMeter()
        
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        self.check_flops = check_flops
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']

        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']

        self.comp_scheduler = comp_scheduler

    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):

        best_top1 = 0.0

        all_theta_list = []

        if self.check_flops == True:
            flops_list = self._check_flops(model, test_loader)
            return flops_list

        else :
            for epoch in range(self.train_thetas_from_the_epoch):

                self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)

                self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
                self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
                self.w_scheduler.step()

            for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
                self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)

                self.logger.info("Start to train weights for epoch %d" % (epoch))
                top1, losses = self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
                
                if self.comp_scheduler :
                    self.comp_scheduler.on_epoch_end(epoch, self.w_optimizer, metrics={'min': losses, 'max': top1})
                else :
                    self.w_scheduler.step()

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

            
            if self.comp_scheduler :
                self.comp_scheduler.on_minibatch_begin(epoch, step, steps_per_epoch, optimizer)
            
            flops_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            outs, flops_to_accumulate = model(X, self.temperature, flops_to_accumulate)
            loss = self.criterion(outs, y, flops_to_accumulate, self.losses_ce, self.losses_flops, self.flops, N)
            
            if self.comp_scheduler:
                agg_loss = self.comp_scheduler.before_backward_pass(epoch, step, steps_per_epoch, loss,
                                                                      optimizer=optimizer, return_loss_components=True)
                loss = agg_loss.overall_loss                                                      

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)

            loss.backward()

            # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            if self.comp_scheduler :
                self.comp_scheduler.before_parameter_optimization(epoch, step, steps_per_epoch, optimizer)
                self.comp_scheduler.on_minibatch_end(epoch, step, steps_per_epoch, optimizer)

            optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')
        
        top1_acc = self.top1
        losses = self.losses
        
        for avg in [self.top1, self.top3, self.losses, self.losses_flops, self.losses_ce, self.flops]:
            avg.reset()
        
        return top1_acc, losses

    def _check_flops(self, model, input_var):
        model = model.train()
        input_var = input_var.cuda(non_blocking=True)

        # X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        #optimizer.zero_grad()
        _, flops_list = model.get_flops(input_var, self.temperature)

        return flops_list
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]
                
                flops_to_accumulate = torch.Tensor([[0.0]]).cuda()
                outs, flops_to_accumulate = model(X, self.temperature, flops_to_accumulate)
                loss = self.criterion(outs, y, flops_to_accumulate, self.losses_ce, self.losses_flops, self.flops, N)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses, self.losses_flops, self.losses_ce, self.flops]:
            avg.reset()
        return top1_avg
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_\'+info_for_logger, self.losses_flops.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/' + val_or_train + '_flops' + info_for_logger,
                               self.flops.get_avg(), epoch)
        
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
                   self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_flops.get_avg()))
        

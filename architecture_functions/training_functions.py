import torch
import time
import math

from general_functions.utils import AverageMeter, save, accuracy
from architecture_functions.config_for_arch import CONFIG_ARCH
from distiller_utils.distiller_utils import delete_float_w

class TrainerArch:
    def __init__(self, criterion, optimizer, scheduler, logger, writer, comp_scheduler, path_to_save_model, epoch, print_freq):
        self.top1   = AverageMeter()
        self.top3   = AverageMeter()
        self.losses = AverageMeter()
        
        self.logger = logger
        self.writer = writer
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # self.path_to_save_model = CONFIG_ARCH['train_settings']['path_to_save_model']
        self.path_to_save_model = path_to_save_model

        self.cnt_epochs = epoch
        self.print_freq = print_freq
        
        self.comp_scheduler = comp_scheduler
        
    def train_loop(self, train_loader, valid_loader, model):
        best_top1 = 0.0

        for epoch in range(self.cnt_epochs):
            
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            #if epoch and epoch % self.lr_decay_period == 0:
            #    self.optimizer.param_groups[0]['lr'] *= self.lr_decay

            # training
            top1, losses = self._train(train_loader, model, epoch)
            # validation
            top1_avg = self._validate(valid_loader, model, epoch)

            if self.comp_scheduler :
                self.comp_scheduler.on_epoch_end(epoch, self.optimizer, metrics={'min': losses, 'max': top1})
            else :
                self.scheduler.step()

            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 accuracy by now. Save model")
                save(model, self.path_to_save_model)
            
        
    
    def _train(self, loader, model, epoch):
        start_time = time.time()
        model = model.train()

        total_samples = len(loader.sampler)
        batch_size = loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        for step, (X, y) in enumerate(loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            #X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]
            
            if self.comp_scheduler :
                self.comp_scheduler.on_minibatch_begin(epoch, step, steps_per_epoch, self.optimizer)

            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            
            if self.comp_scheduler:
                agg_loss = self.comp_scheduler.before_backward_pass(epoch, step, steps_per_epoch, loss,
                                                                      optimizer=self.optimizer, return_loss_components=True)
                loss = agg_loss.overall_loss
            
            loss.backward()
            
            if self.comp_scheduler :
                self.comp_scheduler.before_parameter_optimization(epoch, step, steps_per_epoch, self.optimizer)
                self.comp_scheduler.on_minibatch_end(epoch, step, steps_per_epoch, self.optimizer)
            
            self.optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='train')
        
        top1_acc = self.top1
        losses = self.losses
        
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
            
        return top1_acc, losses

    def _validate(self, loader, model, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                outs = model(X)
                loss = self.criterion(outs, y)
                
                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
        
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg

    def _epoch_stats_logging(self, start_time, epoch, val_or_train):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss', self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1', self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3', self.top3.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 3))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,3) ({:.1%}, {:.1%})".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top3.get_avg()))
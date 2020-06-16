CONFIG_ARCH = {
    'logging' : {
        'path_to_log_file' : './architecture_functions/logs/logger/',
        'path_to_tensorboard_logs' : './architecture_functions/logs/tb'
    },
    'dataloading' : {
        'batch_size' : 256,
        'train_portion' : 0.8,
        'path_to_save_data' : './cifar10_data'
    },
    'optimizer' : {
        'lr' : 0.1,
        'momentum' : 0.9,
        'weight_decay' : 1 * 1e-4
    },
    'train_settings' : {
        'print_freq' : 50, # show logging information
        'path_to_save_model' : './architecture_functions/logs/best_model.pth',
        'cnt_epochs' : 360, #
        # YOU COULD USE 'CosineAnnealingLR' or 'MultiStepLR' scheduler
        'scheduler' : 'CosineAnnealingLR', #MultiStepLR',
        ## CosineAnnealingLR settings
        'eta_min' : 0.0001,
        ## MultiStepLR settings
        'milestones' : [90, 180, 270], # [90, 180, 270], # decay 10x at 90, 180, and 270 epochs
        'lr_decay' : 0.1
    }
}

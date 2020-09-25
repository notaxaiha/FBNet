import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

from general_functions.datasets import ImageFolder


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]

TINY_MEAN = [0.5, 0.5, 0.5]
TINY_STD = [0.5, 0.5, 0.5]

def get_loaders(train_portion, batch_size, path_to_save_data, dataset):

    if "cifar" in dataset:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
              ])

    elif "tiny" in dataset:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
              ])


    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=path_to_save_data, train=True,
                                      download=True, transform=train_transform)
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(root=path_to_save_data, train=True,
                                      download=True, transform=train_transform)
    elif dataset == 'tiny_imagenet':
        train_data = ImageFolder(os.path.join(path_to_save_data,'tiny-imagenet-200', 'train'), transform=train_transform, shuffle=True)

    num_train = len(train_data)                        # 50k
    indices = list(range(num_train))                   # 
    split = int(np.floor(train_portion * num_train))   # 40k
    
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=32)
    
    if train_portion == 1:
        return train_loader
    
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler,
        pin_memory=True, num_workers=16)
    
    return train_loader, val_loader
    
def get_test_loader(batch_size, path_to_save_data, dataset):

    if "cifar" in dataset:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
              ])

    elif "tiny" in dataset:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
              ])

    if dataset == 'cifar10':
        test_data = datasets.CIFAR10(root=path_to_save_data, train=False,
                                      download=True, transform=test_transform)
    elif dataset == 'cifar100':
        test_data = datasets.CIFAR100(root=path_to_save_data, train=False,
                                      download=True, transform=test_transform)
    elif dataset == 'tiny_imagenet':
        test_data = ImageFolder(os.path.join(path_to_save_data,'tiny-imagenet-200', 'val'), transform=test_transform, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    return test_loader

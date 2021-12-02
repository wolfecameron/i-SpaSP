import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import os
import copy

def get_dataset_ft(dset_name, batch_size, n_worker, data_root, istrick = 0, istrick2 = 0, downsample_size=None):
    print('=> Preparing data..')
    if dset_name == 'imagenet':
        # get dir
        traindir = os.path.join(data_root, 'train')

        valdir = os.path.join(data_root, 'val')

        # preprocessing
        input_size = 224
        if istrick:
            imagenet_tran_train = [
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        elif istrick2:
            imagenet_tran_train = [
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

        else:
            imagenet_tran_train = [
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]


        imagenet_tran_test = [
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(imagenet_tran_test)),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)
        n_class = 1000
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class

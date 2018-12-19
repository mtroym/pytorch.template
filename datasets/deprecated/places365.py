import os
import torch
import torchvision
import torchvision.transforms as transforms

def getInstance(info, opt, split):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_train_root = os.path.join(opt.data, 'train')
    transform_test_root = os.path.join(opt.data, 'val')

    if split == 'train':
        return torchvision.datasets.ImageFolder(root=transform_train_root, transform=transform_train)
    elif split == 'val':
        return torchvision.datasets.ImageFolder(root=transform_test_root, transform=transform_test)

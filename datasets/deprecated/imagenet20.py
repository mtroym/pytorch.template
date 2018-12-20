import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

    
def getInstance(info, opt, split):
    transformTrain = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])    

    transformVal = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])    

    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'val')
    if split == 'train':
        return datasets.ImageFolder(traindir, transformTrain)
    if split == 'val':
        return datasets.ImageFolder(valdir, transformVal)
    return None
    

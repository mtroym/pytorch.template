import torchvision
import torchvision.transforms as transforms

def getInstance(info, opt, split):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if split == 'train':
        return torchvision.datasets.CIFAR10(root=opt.data, train=True, download=True, transform=transform_train)
    elif split == 'val':
        return torchvision.datasets.CIFAR10(root=opt.data, train=False, download=True, transform=transform_test)

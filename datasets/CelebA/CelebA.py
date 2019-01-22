from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.file_input, self.file_target = imageInfo[split]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        label_image = Image.open(self.file_target[idx])  # PIL image, label
        train_image = Image.open(self.file_input[idx])  # train
        # transformation.
        label_image = self.transform(label_image)  # label
        train_image = self.transform(train_image)  # train
        return train_image, label_image

    def __len__(self):
        return len(self.file_input)


def getInstance(info, opt, split):
    myInstance = CelebA(info, opt, split)
    return myInstance
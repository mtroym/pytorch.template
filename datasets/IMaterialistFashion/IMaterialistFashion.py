import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2

class IMat(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['base_dir']
        self.files_path = self.imageInfo
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        train_path, label_path = self.files_path[idx]
        img = cv2.imread(train_path)
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        WIDTH, HEIGHT = label_image.shape
        train_image = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        train_image = train_image.transpose(2,0,1)
        # only for appeal object.
        # 0-26 classes, 27 - bg
        label_image[label_image >= 27] = 27

        # todo: real-time augmentations.
        return train_image, label_image

    def __len__(self):
        return len(self.files_path)


def getInstance(info, opt, split):
    myInstance = IMat(info, opt, split)
    opt.numClasses = 27
    return myInstance

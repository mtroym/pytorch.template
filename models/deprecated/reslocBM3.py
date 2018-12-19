import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class reslocNM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, dataset = "imagenetLOC"):

        #-------------------------------------Init Part--------------------------------------
        super(reslocNM, self).__init__()
        self.in_planes = 64
        self.num_types = 0

        if dataset == "imagenetLOC" or dataset == "imagenet" or dataset == "imagenetLOCm":
            self.num_types = 1000
        elif dataset == "VOC":
            self.num_types = 20
        else:
            print("[WARNING]: Unsupported datasets!")
        #-------------------------------------Fixed Part--------------------------------------

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)

        #-----------------------------------Reg VGG----------------------------------

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.classLinear = nn.Sequential(
            nn.Linear(self.num_types, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
        )
        VGGcfg = [256, 256, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'ME']
        self.VGG = self._make_VGG_layer(VGGcfg)
        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 4),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_VGG_layer(self, cfg):
        layers = []
        in_channels = 256
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'MF':
                layers += [nn.MaxPool2d(kernel_size=3, stride=3, padding=1)]
            elif v == 'ME':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=True)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, classVec, target):
        classVec = classVec.view(classVec.size(0), -1)
        classFeat = self.classLinear(classVec)  # 512  1  1
        classFeat = (nn.Softmax()(classFeat))

        inFeat = self.mp(F.relu(self.bn1(self.conv1(x))))
        inFeat = self.layer3(self.layer2(self.layer1(inFeat)))
        
        # maxidx = (classFeat.data.cpu().numpy().argmax(axis=1))

        classFeat[classFeat < 0.2] = 0 

        classFeat = classFeat.view(classFeat.size(0), classFeat.size(1), 1, 1)
        classFeat = classFeat.expand(classFeat.size(0), classFeat.size(1), 56, 56)
        inFeat = inFeat * classFeat
        # 512 56 56
        #-----------------------Heatmap--------------------------

        heat = self.upsample(inFeat)
        heat = torch.sum(heat, dim=1, keepdim=True)
        heat = F.sigmoid(heat)
        #-----------------------Reg VGG--------------------------

        out = self.VGG(inFeat)
        out = out.view(out.size(0), -1)
        reg = self.regressor(out)

        regCenLoss = nn.SmoothL1Loss()(reg[:, :2], target[:, :2])
        regResLoss = nn.SmoothL1Loss()(reg[:, 2:], target[:, 2:])

        regCenLoss = regCenLoss * 20
        regResLoss = regResLoss

        loss = regCenLoss + regResLoss

        return loss, reg, heat, (regCenLoss, regResLoss)

def freezeLayer(layer):
    for param in layer.parameters():
        param.requires_grad = False

def createModel(opt):
    rootPath = (os.path.dirname(os.path.abspath('.')))
    model = reslocNM(BasicBlock, [2, 2, 2], opt.numClasses, opt.dataset)
    restoreName = 'imagenet20_reslocNA_mse1_LR0.05/model_best.pth.tar'
    pretrained_dict = torch.load(os.path.join(rootPath, 'models', restoreName))
    model_dict = model.state_dict()
    print('=> Restoring weights from model: ' + restoreName)
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if opt.GPU:
        model = model.cuda()
    return model

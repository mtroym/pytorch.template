import os
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

class RegLayer(nn.Module):
    def __init__(self):
        super(RegLayer, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.heatConv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1_deconv1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2_deconv2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3_deconv3 = nn.BatchNorm2d(256)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        heat = self.heatConv(out)

        out = F.relu(self.bn1_deconv1(self.deconv1(out)))
        out = F.relu(self.bn2_deconv2(self.deconv2(out)))
        out = F.relu(self.bn3_deconv3(self.deconv3(out)))
        return out, heat


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

        #-----------------------------------New Class Part----------------------------------

        self.classLinear = nn.Linear(self.num_types, 256)
        self.regLinear1 = nn.Linear(128 * 7 * 7, 512)
        self.regDrop = nn.Dropout()
        self.regLinear2 = nn.Linear(512, num_classes)
        self.regLayer1 = RegLayer()
        self.regLayer2 = RegLayer()
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, classVec, target):
        classVec = classVec.view(classVec.size(0), -1)
        classFeat = self.classLinear(classVec)


        inFeat = self.mp(F.relu(self.bn1(self.conv1(x))))
        inFeat = self.layer3(self.layer2(self.layer1(inFeat)))

        classFeat = classFeat.view(classFeat.size(0), classFeat.size(1), 1, 1)
        classFeat = classFeat.expand(classFeat.size(0), classFeat.size(1), 56, 56)
        inFeat = inFeat * classFeat
        #-----------------------reg1--------------------------

        out, heat1 = self.regLayer1(inFeat)
        regHeat1 = self.upsample(heat1)
        regHeat1 = torch.sum(regHeat1, dim=1, keepdim=True)
        regHeat1 = F.sigmoid(regHeat1)
        reg1 = heat1.view(-1, 128 * 7 * 7)
        reg1 = self.regDrop(self.regLinear1(reg1))
        reg1 = self.regLinear2(reg1)

        out = inFeat + out * regHeat1

        #---------------------reg2----------------------------

        _, heat2 = self.regLayer2(out)
        regHeat2 = self.upsample(heat2)
        regHeat2 = torch.sum(regHeat2, dim=1, keepdim=True)
        regHeat2 = F.sigmoid(regHeat2)
        reg2 = heat2.view(-1, 128 * 7 * 7)
        reg2 = self.regDrop(self.regLinear1(reg2))
        reg2 = self.regLinear2(reg2)

        #---------------------loss----------------------------

        # reg1CenLoss = nn.SmoothL1Loss()(reg1[:, :2], target[:, :2])
        # reg1ResLoss = nn.SmoothL1Loss()(reg1[:, 2:], target[:, 2:])
        reg2CenLoss = nn.SmoothL1Loss()(reg2[:, :2], target[:, :2])
        reg2ResLoss = nn.SmoothL1Loss()(reg2[:, 2:], target[:, 2:])

        regCenLoss = (reg2CenLoss) * 20
        regResLoss = reg2ResLoss

        loss = regCenLoss + regResLoss

        return loss, (reg1, reg2), (regHeat1, regHeat2), (regCenLoss, regResLoss)

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

    freezeLayer(model.conv1)
    freezeLayer(model.bn1)
    freezeLayer(model.mp)
    freezeLayer(model.layer1)
    freezeLayer(model.layer2)
    freezeLayer(model.layer3)

    if opt.GPU:
        model = model.cuda()
    return model

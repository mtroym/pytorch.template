import os
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [256, 256, 'M', 512, 512, 'M', 512, 512, 'ME']

def make_VGG_layers(cfg, batch_norm=True, expension = 1):
    layers = []
    in_channels = 512 * 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'ME':
            layers += [nn.MaxPool2d(kernel_size=3, stride=3, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
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
        self.num_types = 1000
        VGGcfg = [1024, 1024, 'M', 1024, 1024, 'M', 512, 512, 'ME']
        if dataset == "imagenetLOC" or dataset == "imagenet" or dataset == "imagenetLOCm":
            self.num_types = 1000
        elif dataset == "VOC":
            self.num_types = 20
        else:
            print("[WARNING]: Unsupported datasets!")
        #-------------------------------------Fixed Part--------------------------------------

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)

        self.VGG = make_VGG_layers(VGGcfg, True)
        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 4),
        )
        #-----------------------------------New Class Part----------------------------------

        self.classLinear = nn.Sequential(
            nn.Linear(self.num_types, 2048),
            nn.Sigmoid(),
        )
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
        classFeat1 = self.classLinear(classVec)

        inFeat = self.mp(F.relu(self.bn1(self.conv1(x))))
        inFeat = self.layer4(self.layer3(self.layer2(self.layer1(inFeat)))) # torch.Size([24, 2048, 28, 28])
        
        classFeat = classFeat1.view(classFeat1.size(0), classFeat1.size(1), 1, 1)
        classFeat = classFeat.expand(classFeat.size(0), classFeat.size(1), 28, 28)
        
        inFeat = inFeat * classFeat
        #-----------------------reg--------------------------
        out1 = self.VGG(inFeat)
        #VGG
        out1 = out1.view(out1.size(0), -1)

        reg = self.regressor(out1)
        regCenLoss = nn.SmoothL1Loss()(reg[:, :2], target[:, :2])
        regResLoss = nn.SmoothL1Loss()(reg[:, 2:], target[:, 2:])

        regCenLoss = (regCenLoss) * 20
        regResLoss = regResLoss

        loss = regCenLoss + regResLoss

        return loss, reg, (regCenLoss, regResLoss), classFeat1, inFeat

def freezeLayer(layer):
    for param in layer.parameters():
        param.requires_grad = False

def createModel(opt):
    rootPath = (os.path.dirname(os.path.abspath('.')))
    model = reslocNM(Bottleneck, [3, 4, 6, 3], opt.numClasses, opt.dataset)
    restoreName = 'imagenet20_res50locNA28V_mse1_LR0.001_Mar20_VU/model_best.pth.tar'
    pretrained_dict = torch.load(os.path.join(rootPath, 'models', restoreName))
    model_dict = model.state_dict()
    print('=> Restoring weights from model: ' + restoreName)
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('[WRANING!]: res50loc28B1EV is NOT A FROZEN VERSION!')
#     freezeLayer(model.conv1)
#     freezeLayer(model.bn1)
#     freezeLayer(model.mp)
#     freezeLayer(model.layer1)
#     freezeLayer(model.layer2)
#     freezeLayer(model.layer3)
#     freezeLayer(model.layer4)
    
    if opt.GPU:
        model = model.cuda()
    return model

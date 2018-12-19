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


class reslocNM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, dataset = "imagenetLOC"):
        
        #-------------------------------------Init Part--------------------------------------
        super(reslocNM, self).__init__()
        self.in_planes = 64
        self.num_types = 0
        print('=> trained on '+ dataset)
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
        self.linear_reg = nn.Linear(512, num_classes)
        
        #-----------------------------------New Class Part----------------------------------
        
        self.linear_class = nn.Linear(self.num_types, 512)
        
        #-----------------------------------regress Part1--------------------------------------
        
        self.reg1_conv1 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg1_bn1 = nn.BatchNorm2d(512)
        self.reg1_conv2 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg1_bn2 = nn.BatchNorm2d(512)
        
        self.reg1_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.reg1_bn1_deconv1 = nn.BatchNorm2d(256)
        self.reg1_deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.reg1_bn2_deconv2 = nn.BatchNorm2d(256)

        #-----------------------------------regress Part2--------------------------------------
        
        self.reg2_conv1 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg2_bn1 = nn.BatchNorm2d(512)
        self.reg2_conv2 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg2_bn2 = nn.BatchNorm2d(512)
        
        self.reg2_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.reg2_bn1_deconv1 = nn.BatchNorm2d(256)
        self.reg2_deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.reg2_bn2_deconv2 = nn.BatchNorm2d(256)
        
        #-----------------------------------regress Part3--------------------------------------
        
        self.reg3_conv1 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg3_bn1 = nn.BatchNorm2d(512)
        self.reg3_conv2 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.reg3_bn2 = nn.BatchNorm2d(512)
        
        #-----------------------------------------END--------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, classVec):
        classVec = classVec.view(classVec.size(0), -1)
        classFeat = self.linear_class(classVec)
        
        
        inFeat = self.mp(F.relu(self.bn1(self.conv1(x))))
        inFeat = self.layer3(self.layer2(self.layer1(inFeat)))
        
        #-----------------------reg1--------------------------
        
        out = F.relu(self.reg1_bn1(self.reg1_conv1(inFeat)))
        out = F.relu(self.reg1_bn2(self.reg1_conv2(out)))
        
        reg1 = F.avg_pool2d(out, kernel_size=14)
        reg1 = reg1.view(reg1.size(0), -1)
        reg1 += classFeat
        reg1 = self.linear_reg(reg1)
        
        out = F.relu(self.reg1_bn1_deconv1(self.reg1_deconv1(out)))
        out = F.relu(self.reg1_bn2_deconv2(self.reg1_deconv2(out)))
        out *= inFeat
        
        #---------------------reg2----------------------------
        
        out = F.relu(self.reg2_bn1(self.reg2_conv1(out)))
        out = F.relu(self.reg2_bn2(self.reg2_conv2(out)))
        
        reg2 = F.avg_pool2d(out, kernel_size=14)
        reg2 = reg2.view(reg1.size(0), -1)
        reg2 += classFeat
        reg2 = self.linear_reg(reg2)
        
        out = F.relu(self.reg2_bn1_deconv1(self.reg2_deconv1(out)))
        out = F.relu(self.reg2_bn2_deconv2(self.reg2_deconv2(out)))
        out *= inFeat        
        
        #---------------------reg3----------------------------
        
        out = F.relu(self.reg3_bn1(self.reg3_conv1(out)))
        out = F.relu(self.reg3_bn2(self.reg3_conv2(out)))
        
        reg3 = F.avg_pool2d(out, kernel_size=14)
        reg3 = reg3.view(reg3.size(0), -1)
        reg3 += classFeat
        reg3 = self.linear_reg(reg3)
        
        return reg1, reg2, reg3

def createModel(opt):
    rootPath = (os.path.dirname(os.path.abspath('.')))
    model = reslocNM(BasicBlock, [2, 2, 2, 2], opt.numClasses, opt.dataset)
    restoreName = 'imagenet_reslocN_mse1_LR0.01/model_best.pth.tar'
    pretrained_dict = torch.load(os.path.join(rootPath,'models',restoreName))
    model_dict = model.state_dict()
    print('=> restoring model weights from '+restoreName)
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  
    if opt.GPU:
        model = model.cuda()
    return model


def test():
    rootPath = (os.path.dirname(os.path.abspath('.')))
    model = reslocNM(BasicBlock, [2, 2, 2, 2], 1000, 'imagenet')
    restoreName = 'imagenet_reslocN_mse1_LR0.01/model_best.pth.tar'
    pretrained_dict = torch.load(os.path.join(rootPath,'models',restoreName))
    model_dict = model.state_dict()
    print('=> restoring model weights from '+restoreName)
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  


if __name__ == '__main__':
    test()

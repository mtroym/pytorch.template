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


class resloc(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, dataset = "imagenetLOC"):
        super(resloc, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        num_types = 0
        if dataset == "imagenetLOC":
            num_types = 1000
        elif dataset == "VOC":
            num_types = 20
        else:
            print("[WARNING]: unsupported datasets!")
            
        self.linear_class = nn.Linear(num_types, 256)    
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        
        self.bn3_forconv3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.bn2_forconv2 = nn.BatchNorm2d(64)     
        
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, clas):
        clas = clas.view(clas.size(0), -1)
        class_out = self.linear_class(clas)

        out_x = self.mp(F.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out_x)
        out = self.layer2(out)
        out = self.layer3(out)

        
        out1 = F.avg_pool2d(out, kernel_size=14)

        out1 = out1.view(out1.size(0), -1)
        out1 += class_out

        out1 = self.linear(out1) # (bz, 4)
        
        out = self.bn3_forconv3(self.deconv3(out))
        out = self.bn2_forconv2(self.deconv2(out))
        out += out_x
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out2 = F.avg_pool2d(out, kernel_size=14)
        out2 = out2.view(out2.size(0), -1)
        out2 += class_out

        out2 = self.linear(out2) # (bz, 4)
        '''
        out = self.deconv3(out)
        out = self.deconv2(out)
        out += out_x
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out3 = F.avg_pool2d(out, kernel_size=14)
        out3 = out3.view(out3.size(0), -1)
        out3 += class_out

        out3 = self.linear(out3) # (bz, 4)
        '''
        return out2

def createModel(opt):
    model = resloc(BasicBlock, [2, 2, 2, 2], opt.numClasses, opt.dataset)
    if opt.GPU:
        model = model.cuda()

    return model

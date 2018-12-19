import os
import cv2
import time
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from util.progbar import progbar
import torch
import util.visualize as vis

def unNormalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] * std[0]) + mean[0]
    ipt[:][:][1] = (ipt[:][:][1] * std[1]) + mean[1]
    ipt[:][:][2] = (ipt[:][:][2] * std[2]) + mean[2]
    return ipt

class resnetTrainer():
    def __init__(self, model, criterion, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        self.opt = opt

        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening, weight_decay=opt.weightDecay)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.momentum, 0.999), eps=1e-8, weight_decay=opt.weightDecay)

        if self.optimState is not None:
            self.optimizer.load_state_dict(self.optimState)

        self.logger = {'train' : open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val' : open(os.path.join(opt.resume, 'test.log'), 'a+')}

    def train(self, trainLoader, epoch):
        self.model.train()

        print('=> Training epoch # ' + str(epoch))

        avgLoss = 0
        correct = 0
        total = 0

        self.progbar = progbar(len(trainLoader), width=self.opt.barwidth)

        
        for i, (inputData, targetData) in enumerate(trainLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()

            inputData, targetData = Variable(inputData), Variable(targetData)
            self.optimizer.zero_grad()
            if self.opt.GPU:
                inputData = inputData.cuda()
                targetData = targetData.cuda()
            dataTime = time.time() - start

            output, heats = self.model.forward(inputData)
            loss = self.criterion(output, targetData)
            loss.backward()
            self.optimizer.step()
            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            _, predicted = torch.max(output.data, 1)
            total += targetData.size(0)
            correct += predicted.eq(targetData.data).cpu().sum()
            acc = 100 * predicted.eq(targetData.data).cpu().sum() / targetData.size(0)
            
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f Acc %1.4f%%\n' % (epoch, i, len(trainLoader), runTime, dataTime, loss.data[0], acc)
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0]), ('Acc', acc)])


        log = '* Finished training epoch # %d     Loss: %1.4f | Acc: %1.4f %% (%d/%d)\n' % (epoch, avgLoss, 100. * correct / total, correct, total)
        self.logger['train'].write(log)
        print(log)

        return avgLoss

    def test(self, valLoader, epoch):
        self.model.eval()
        avgLoss = 0
        correct = 0
        total = 0
        
        visImg = []
        
        self.progbar = progbar(len(valLoader), width=self.opt.barwidth)

        for i, (inputData, targetData) in enumerate(valLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()
            inputData, targetData = Variable(inputData), Variable(targetData)
            if self.opt.GPU:
                inputData = inputData.cuda()
                targetData = targetData.cuda()
                
            dataTime = time.time() - start

            output, heats = self.model.forward(inputData)
            loss = self.criterion(output, targetData)
            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            _, predicted = torch.max(output.data, 1)
            total += targetData.size(0)
            correct += predicted.eq(targetData.data).cpu().sum()
            acc = 100 * predicted.eq(targetData.data).cpu().sum() / targetData.size(0)
            
            maxnum = 3
            max3index = torch.LongTensor(np.max(np.max(heats.data.cpu().numpy(),2),2).argsort(axis=1)[:,-1-maxnum:-1]).contiguous()
            max3index = max3index.view(max3index.size(0), max3index.size(1), 1, 1)
            max3index = max3index.expand(max3index.size(0), max3index.size(1), 56, 56)      
            max_heat = torch.gather(heats.data.cpu(), 1, torch.LongTensor(max3index))
            
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f Acc %1.4f%%\n' % (epoch, i, len(valLoader), runTime, dataTime, loss.data[0], acc)
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0]), ('Acc', acc)])
                        
            if i <= self.opt.visTest:
                visImg.append(inputData.data.cpu())
                visImg.append(max_heat)
            if i == self.opt.visTest:
                self.visualize(visImg, epoch, 'test')

        log = '* Finished testing epoch # %d     Loss: %1.4f | Acc: %1.4f %% (%d/%d)\n' % (epoch, avgLoss, 100. * correct / total, correct, total)
        self.logger['val'].write(log)
        print(log)

        return avgLoss

    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()

    def visualize(self, visImg, epoch, split):
        outputImgs = []
        mean = [ 0.485, 0.456, 0.406 ]
        std = [ 0.229, 0.224, 0.225 ]
        thred = 0.01
        for i in range(len(visImg) // 2):
            for j in range(self.opt.batchSize):
                inputImg = visImg[2 * i][j].numpy()
                inputImg = unNormalize(inputImg, mean, std)
                inputImg = np.transpose(inputImg, (1, 2, 0))
                inputImg *= 256
                outputImgs.append(inputImg)
                heats = visImg[2 * i + 1][j]
                h, w = heats[0].shape
                for k in range(3):
                    heat = heats[k,:,:].view(1,h,w).numpy()
                    heat /= np.max(heat)
                    heat[heat > thred] = 0.8
                    heat += 0.2
                    heat *= 256
                    heat = np.transpose(heat, (1, 2, 0))
                    heat = cv2.resize(heat, (224,224), interpolation=cv2.INTER_LINEAR)
                    outputImgs.append(heat)
                thred += 0.01   
        vis.writeImgHTML(outputImgs, epoch, split, 4, self.opt)
def createTrainer(model, criterion, opt, optimState):
    return resnetTrainer(model, criterion, opt, optimState)

import os
import cv2
import time
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import util.visualize as vis
from util.progbar import progbar

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
        visImg = []
        self.progbar = progbar(len(trainLoader), width=self.opt.barwidth)

        for i, (inputData, targetData) in enumerate(trainLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()
            inputData, inputClassVec = inputData
            inputData, targetData, inputClassVec = Variable(inputData), Variable(targetData), Variable(inputClassVec)
            self.optimizer.zero_grad()

            if self.opt.GPU:
                inputClassVec = inputClassVec.cuda()
                inputData = inputData.cuda()
                targetData = targetData.cuda()
            dataTime = time.time() - start

            reg1, reg2, reg3 = self.model.forward(inputData, inputClassVec)
            
            loss3 = self.criterion(reg3, targetData)
            loss2 = self.criterion(reg2, targetData)
            loss1 = self.criterion(reg1, targetData)
            loss = 0.5*loss1 + 0.3*loss2 + 0.2*loss3
            
            loss.backward()
            self.optimizer.step()
            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(trainLoader), runTime, dataTime, loss.data[0])
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', avgLoss)])

            if i <= self.opt.visTrain:
                visImg.append(inputData.data.cpu())
                visImg.append(reg1.data.cpu())
                visImg.append(reg2.data.cpu())
                visImg.append(reg3.data.cpu())
                visImg.append(targetData.data.cpu())
            if i == self.opt.visTrain:
                self.visualize(visImg, epoch, 'train', trainLoader.dataset.postprocess, trainLoader.dataset.postprocessTarget)

        log = '\n* Finished training epoch # %d     Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['train'].write(log)
        print(log)

        return avgLoss

    def test(self, valLoader, epoch):
        self.model.eval()

        avgLoss = 0
        visImg = []
        self.progbar = progbar(len(valLoader), width=self.opt.barwidth)

        for i, (inputData, targetData) in enumerate(valLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()
            inputData, inputClassVec = inputData
            inputData, targetData, inputClassVec = Variable(inputData), Variable(targetData), Variable(inputClassVec)

            if self.opt.GPU:
                inputClassVec = inputClassVec.cuda()
                inputData = inputData.cuda()
                targetData = targetData.cuda()
            dataTime = time.time() - start

            reg1, reg2, reg3 = self.model.forward(inputData, inputClassVec)
            loss3 = self.criterion(reg3, targetData)
            loss2 = self.criterion(reg2, targetData)
            loss1 = self.criterion(reg1, targetData)
            loss = 0.5*loss1 + 0.3*loss2 + 0.2*loss3
            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(valLoader), runTime, dataTime, loss.data[0])
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', avgLoss)])

            if i <= self.opt.visTest:
                visImg.append(inputData.data.cpu())
                visImg.append(reg1.data.cpu())
                visImg.append(reg2.data.cpu())
                visImg.append(reg3.data.cpu())
                visImg.append(targetData.data.cpu())
            if i == self.opt.visTest:
                self.visualize(visImg, epoch, 'test', valLoader.dataset.postprocess, valLoader.dataset.postprocessTarget)

        log = '\n* Finished testing epoch # %d      Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['val'].write(log)
        print(log)

        return avgLoss

    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()

    def visualize(self, visImg, epoch, split, postprocess, postprocessTarget):
        outputImgs = []
        for i in range(len(visImg) // 5):
            for j in range(self.opt.batchSize):
                img = postprocess()(visImg[5 * i][j].numpy())
                outputImgs.append(img)
                for k in range(1, 5):
                    regResult = postprocessTarget()(visImg[5 * i + k][j].numpy())
                    outputImgs.append(self.drawBox(img, regResult))
        vis.writeImgHTML(outputImgs, epoch, split, 5, self.opt)

    def drawBox(self, img, box):
        x_min = self.regPos(box[0])
        y_min = self.regPos(box[1])
        x_max = self.regPos(box[2])
        y_max = self.regPos(box[3])
        img = img.astype(np.uint8).copy()
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return img

    def regPos(self, x):
        x = int(round(x))
        if x < 0:
            return 0
        elif x >= self.opt.imgDim:
            return self.opt.imgDim - 1
        else:
            return x

def createTrainer(model, criterion, opt, optimState):
    return resnetTrainer(model, criterion, opt, optimState)

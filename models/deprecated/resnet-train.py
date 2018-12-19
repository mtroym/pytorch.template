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
        visOut = []
        visGT = []

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

            output = self.model.forward(inputData)
            loss = self.criterion(output, targetData)
            loss.backward()
            self.optimizer.step()
            runTime = time.time() - start

            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(trainLoader), runTime, dataTime, loss.data[0])
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0])])

            if i <= self.opt.visTrain:
                visImg.append(inputData.data.cpu())
                visOut.append(output.data.cpu())
                visGT.append(targetData.data.cpu())
            if i == self.opt.visTrain:
                self.visualize(visImg, visOut, visGT, epoch, 'train', trainLoader.dataset.postprocess)

        log = '\n* Finished training epoch # %d     Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['train'].write(log)
        print(log)

        return avgLoss

    def test(self, valLoader, epoch):
        self.model.eval()

        avgLoss = 0
        visImg = []
        visOut = []
        visGT = []

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

            output = self.model.forward(inputData)
            loss = self.criterion(output, targetData)
            runTime = time.time() - start

            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(valLoader), runTime, dataTime, loss.data[0])
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0])])

            if i <= self.opt.visTest:
                visImg.append(inputData.data.cpu())
                visOut.append(output.data.cpu())
                visGT.append(targetData.data.cpu())
            if i == self.opt.visTest:
                self.visualize(visImg, visOut, visGT, epoch, 'test', valLoader.dataset.postprocess)

        log = '\n* Finished testing epoch # %d      Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['val'].write(log)
        print(log)

        return avgLoss

    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()

    def visualize(self, visInput, visOutput, visTarget, epoch, split, postprocess):
        inputImgs = []
        targetImgs = []
        outputImgs = []
        for i in range(len(visInput)):
            for j in range(self.opt.batchSize):
                img = postprocess()(visInput[i][j].numpy())
                inputImgs.append(img)
                targetImgs.append(self.drawBox(img, visTarget[i][j]))
                outputImgs.append(self.drawBox(img, visOutput[i][j]))
        vis.writeImgHTML(inputImgs, targetImgs, outputImgs, epoch, split, self.opt)

    def drawBox(self, img, box):
        x_min = round(box[0])
        y_min = round(box[1])
        x_max = round(box[2])
        y_max = round(box[3])
        img = img.astype(np.uint8).copy()
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return img

def createTrainer(model, criterion, opt, optimState):
    return resnetTrainer(model, criterion, opt, optimState)

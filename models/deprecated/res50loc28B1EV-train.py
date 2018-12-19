import os
import cv2
import time
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import util.visualize as vis
import util.evaluation as evaluation
from util.progbar import progbar
import torch


class resnetTrainer():
    def __init__(self, model, criterion, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        self.opt = opt

        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening, weight_decay=opt.weightDecay)
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

            loss, reg, (regCenLoss, regResLoss), classFeat, inFeat= self.model.forward(inputData, inputClassVec, targetData)

            loss.backward()
            self.optimizer.step()
            runTime = time.time() - start
            
            maxnum = 3
            max3index = torch.LongTensor((classFeat.data.cpu().numpy()).argsort(axis=1)[:,0:maxnum]).contiguous()
            max3index = max3index.view(max3index.size(0), max3index.size(1), 1, 1)
            max3index = max3index.expand(max3index.size(0), max3index.size(1), 28, 28)      
            max_heat = torch.gather(inFeat.data.cpu(), 1, torch.LongTensor(max3index))
#             max_heat =  torch.sum(inFeat, dim=1, keepdim=True).expand(8,3,56,56)
            
            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            iouScore = evaluation.IoU(reg.data.cpu().numpy(), targetData.data.cpu().numpy())
            precision = np.sum(iouScore >= 0.5) / inputData.shape[0]
            iouScore = np.average(iouScore)
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f IoU %1.3f Err %1.4f\n' % (epoch, i, len(trainLoader), runTime, dataTime, iouScore, loss.data[0])
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0]), ('CenLoss', regCenLoss.data[0]), ('ResLoss', regResLoss.data[0]), ('IoU', iouScore), ('Precision', precision)])

            if i <= self.opt.visTrain:
                visImg.append(inputData.data.cpu())
                visImg.append(max_heat)
                visImg.append(reg.data.cpu())
                visImg.append(targetData.data.cpu().numpy())
            if i == self.opt.visTrain:
                self.visualize(visImg, epoch, 'train', trainLoader.dataset.postprocess, trainLoader.dataset.postprocessTarget, trainLoader.dataset.postprocessHeat)

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
            # TOTAL TARGET    
#             totalTarget = targetData
            start = time.time()
            inputData, inputClassVec = inputData
            inputData, targetData, inputClassVec = Variable(inputData), Variable(targetData), Variable(inputClassVec)

            if self.opt.GPU:
                inputClassVec = inputClassVec.cuda()
                inputData = inputData.cuda()
                targetData = targetData.cuda()
            dataTime = time.time() - start

            
            # TODO: targetData Evaluation!
            # targetData B,50,4 => B,4
            
            loss, reg, (regCenLoss, regResLoss), classFeat, inFeat = self.model.forward(inputData, inputClassVec, targetData)
            runTime = time.time() - start

          
            
            maxnum = 3
            
            max3index = torch.LongTensor(classFeat.data.cpu().numpy().argsort(axis=0)[:,0:maxnum]).contiguous()
            max3index = max3index.view(max3index.size(0), max3index.size(1), 1, 1)
            max3index = max3index.expand(max3index.size(0), max3index.size(1), 28, 28)      
            max_heat = torch.gather(inFeat.data.cpu(),1,torch.LongTensor(max3index))
            
            
            
            
            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)
            iouScore = evaluation.IoU(reg.data.cpu().numpy(), targetData.data.cpu().numpy())
            precision = np.sum(iouScore >= 0.5) / inputData.shape[0]
            iouScore = np.average(iouScore)
            
            
            
            
            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f IoU %1.3f Err %1.4f\n' % (epoch, i, len(valLoader), runTime, dataTime, iouScore, loss.data[0])
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0]), ('CenLoss', regCenLoss.data[0]), ('ResLoss', regResLoss.data[0]), ('IoU', iouScore), ('Precision', precision)])

            if i <= self.opt.visTest:
                visImg.append(inputData.data.cpu())
                visImg.append(max_heat)
                visImg.append(reg.data.cpu())
                visImg.append(targetData.data.cpu().numpy())
            if i == self.opt.visTest:
                self.visualize(visImg, epoch, 'test', valLoader.dataset.postprocess, valLoader.dataset.postprocessTarget, valLoader.dataset.postprocessHeat)

        log = '\n* Finished testing epoch # %d      Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['val'].write(log)
        print(log)

        return avgLoss

    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()

    def visualize(self, visImg, epoch, split, postprocess, postprocessTarget, postprocessHeat):
        outputImgs = []
        for i in range(len(visImg) // 6):
            for j in range(self.opt.batchSize):
                img = postprocess()(visImg[4 * i][j].numpy())
                outputImgs.append(img)
                heats = visImg[4 * i + 1][j]
                h, w = heats[0].shape
                for k in range(3):
                    heat = postprocessHeat()(heats[k].view(1, h ,w).numpy())
                    heat = cv2.resize(heat,(422,422),interpolation=cv2.INTER_LINEAR)
                    outputImgs.append(heat)
                regResult = postprocessTarget()(visImg[4 * i + 2][j].numpy())
                outputImgs.append(self.drawBox(img, regResult))
                regResult = postprocessTarget()(visImg[4 * i + 3][j])
                outputImgs.append(self.drawBox(img, regResult))
        vis.writeImgHTML(outputImgs, epoch, split, 6, self.opt)

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

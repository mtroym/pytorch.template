import os
import cv2
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

import util.visualize as vis
from util.progbar import progbar

class myTrainer():
    def __init__(self, model, criterion, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        if self.optimState == None:
            self.optimState = { 'learningRate' : opt.LR,
                                'learningRateDecay' : opt.LRDParam,
                                'momentum' : opt.momentum,
                                'nesterov' : True,
                                'dampening'  : opt.dampening,
                                'weightDecay' : opt.weightDecay
                            }
        self.opt = opt
        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening, weight_decay=opt.weightDecay)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.momentum, 0.999), eps=1e-8, weight_decay=opt.weightDecay)
        if opt.LRDecay == 'exp':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.logger = {'train' : open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val' : open(os.path.join(opt.resume, 'test.log'), 'a+')}

    def train(self, trainLoader, epoch):
        self.model.train()

        print('=> Training epoch # ' + str(epoch))

        avgLoss = 0
        total = 0
        correct = 0

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

        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        self.model._modules.get('layer4').register_forward_hook(hook_feature)

        avgLoss = 0
        total = 100

        self.progbar = progbar(len(valLoader), width=self.opt.barwidth)

        maxFeatures = [None]
        imgList = np.zeros((total, 3, 224, 224))
        totalFeatures = np.zeros((total, 512, 7, 7))

        for i, (inputData, targetData) in enumerate(valLoader):
            del features_blobs[:]
            if self.opt.debug and i > 10:
                break

            start = time.time()

            inputData, targetData = Variable(inputData), Variable(targetData)
            if self.opt.GPU:
                inputData = inputData.cuda()
                targetData = targetData.cuda()
            dataTime = time.time() - start

            output = self.model.forward(inputData)

            if maxFeatures[0] is None:
                for j, feat in enumerate(features_blobs):
                    size_f = (total, feat.shape[1])
                    maxFeatures[j] = np.zeros(size_f)
            iStart = i * self.opt.batchSize
            iEnd = min((i + 1) * self.opt.batchSize, total)
            for j, feat in enumerate(features_blobs):
                maxFeatures[j][iStart:iEnd] = np.max(np.max(feat, 3), 2)
                totalFeatures[iStart:iEnd, :, :, :] = feat
                imgList[iStart:iEnd, :, :, :] = inputData.data.cpu().numpy()
            runTime = time.time() - start

            self.progbar.update(i, [('Time', runTime), ('Data Time', dataTime)])

        log = '\n* Finished testing epoch # %d\n' % (epoch)
        self.logger['val'].write(log)
        print(log)

        visImg = []
        for unitID in range(maxFeatures[0].shape[1]):
            actUnit = np.squeeze(maxFeatures[0][:, unitID])
            idxSorted = np.argsort(actUnit)[::-1]
            imgListSorted = [imgList[item] for item in idxSorted[:self.opt.visWidth]]
            featureSorted = [totalFeatures[item][unitID] for item in idxSorted[:self.opt.visWidth]]
            for i in range(self.opt.visWidth):
                feat = featureSorted[i]
                feat = feat / np.max(feat)
                mask = cv2.resize(feat, (112, 112))
                mask[mask < self.opt.visThres] = 0.0
                mask[mask > self.opt.visThres] = 1.0
                img = imgListSorted[i]
                img = valLoader.dataset.postprocess()(img)
                img = cv2.resize(img, (112, 112))
                img = np.multiply(img, mask[:, :, np.newaxis])
                img = np.uint8(img)
                visImg.append(img)
        vis.writeImgHTML(visImg, epoch, 'test', self.opt)

        return avgLoss


def createTrainer(model, criterion, opt, optimState):
    return myTrainer(model, criterion, opt, optimState)

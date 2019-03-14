import os
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from util.summaries import BoardX


# allAcc = {}
# for metric in trainAcc:
#     allAcc[metric + "_train"] = trainAcc[metric]
#     allAcc[metric + "_val"] = testAcc[metric]
#
# bb.writer.add_scalars(opt.suffix + '/scalar/Loss', {'Loss_train': trainLoss, 'Loss_val': testLoss}, epoch)
# bb.writer.add_scalars(opt.suffix + '/scalar/Acc', allAcc, epoch)
# bb.writer.add_scalars(opt.suffix + '/scalar/LR', {'LR': float(trainer.scheduler.get_lr()[0])}, epoch)


class Trainer:
    def __init__(self, model, criterion, metrics, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        self.opt = opt
        self.metrics = metrics
        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening,
                                       weight_decay=opt.weightDecay)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.momentum, 0.999), eps=1e-8,
                                        weight_decay=opt.weightDecay)

        if self.optimState is not None:
            self.optimizer.load_state_dict(self.optimState)

        self.logger = {'train': open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val': open(os.path.join(opt.resume, 'test.log'), 'a+')}

        self.bb = BoardX(opt, self.metrics, opt.hashKey, self.opt.logNum)
        self.bb_suffix = opt.hashKey
        self.log_num = opt.logNum

    def train(self, trainLoader, epoch):
        print("=> Training epoch")
        self.model.train()
        self.bb.start(len(trainLoader))
        for i, (inputs, target) in enumerate(trainLoader):
            if self.opt.debug and i > 1:  # check debug.
                break
            start = time.time()

            inputV, targetV = Variable(inputs), Variable(target)
            if self.opt.GPU:
                inputV, targetV = inputV.cuda(), targetV.cuda()

            output = self.model(inputV)

            loss, loss_record = self.criterion(output, targetV.long())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            _, preds = torch.max(output, 1)
            metrics = self.metrics(preds, targetV)
            runTime = time.time() - start
            log = self.bb.update(loss_record, runTime, metrics, 'train', i, epoch)
            self.logger['train'].write(log)

        log = self.bb.finish(epoch, 'train')
        self.logger['train'].write(log)
        return self.bb.avgLoss()['loss']

    def test(self, trainLoader, epoch):
        print("=> Validating epoch")
        self.model.eval()
        self.bb.start(len(trainLoader))
        for i, (inputs, target) in enumerate(trainLoader):
            if self.opt.debug and i > 1:  # check debug.
                break
            start = time.time()

            with torch.no_grad():
                inputV, targetV = Variable(inputs), Variable(target)
                if self.opt.GPU:
                    inputV, targetV = inputV.cuda(), targetV.cuda()
                output = self.model(inputV)
                _, preds = torch.max(output, 1)
                # Compute loss and preds
                _, loss_record = self.criterion(output, targetV.long())
                metrics = self.metrics(preds, targetV)


            runTime = time.time() - start
            log = self.bb.update(loss_record, runTime, metrics, 'val', i, epoch)
            self.logger['val'].write(log)

        log = self.bb.finish(epoch, 'val')
        self.logger['train'].write(log)
        return self.bb.avgLoss()['loss']


    def LRDecay(self, epoch):
        # poly_scheduler.adjust_lr(self.optimizer, epoch)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=epoch - 2)

    def LRDecayStep(self):
        self.scheduler.step()

class poly_scheduler:
    def __init__(self, opt):
        self.opt = opt

    def adjust_lr(self, optimizer, epoch):
        lr = self.opt.LR * (1 - epoch / self.opt.nEpochs) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def createTrainer(model, criterion, metric, opt, optimState):
    return Trainer(model, criterion, metric, opt, optimState)

import os
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from util.summaries import BoardX


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
        self.www = opt.www

    def processing(self, dataloader, epoch, split, eval):
        # use store utils to update output.
        print('=> {}ing epoch # {}'.format(split, epoch))
        is_train = split == 'train'
        is_eval = eval
        if is_train:
            self.model.train()
        else:  # VAL
            self.model.eval()
        self.bb.start(len(dataloader))
        for i, (inputs, target) in enumerate(dataloader):
            # check debug.
            if self.opt.debug and i > 2:
                break
            # store the patients processed in this phase.
            start = time.time()
            # * Data preparation *
            if self.opt.GPU:
                inputs, target= inputs.cuda(), target.cuda()
            inputV, targetV = Variable(inputs).float(), Variable(target)
            datatime = time.time() - start
            # * Feed in nets*
            if is_train:
                self.optimizer.zero_grad()
            output = self.model(inputV)
            loss, loss_record = self.criterion(output, targetV.long())
            if is_train:
                loss.mean().backward()
                self.optimizer.step()
            metrics = {}
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                if is_eval:
                    metrics = self.metrics(preds, targetV)

            runTime = time.time() - start - datatime
            log = self.bb.update(loss_record, {'TD': datatime, 'TR': runTime}, metrics, split, i, epoch)
            del loss, loss_record, output
            self.logger[split].write(log)
        self.logger[split].write(self.bb.finish(epoch, split))
        return self.bb.avgLoss()['loss']

    def train(self, dataLoader, epoch):
        loss = self.processing(dataLoader, epoch, 'train', True)
        return loss

    def test(self, dataLoader, epoch):
        loss = self.processing(dataLoader, epoch, 'val', True)
        return loss

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

import os
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy
from util.summaries import BoardX
from util.utils import StoreArray

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

    def processing(self, dataloader, epoch, split):
        store_array_pred = StoreArray(len(dataloader))
        # use store utils to update output.
        print('=> {}ing epoch # {}'.format(split, epoch))
        is_train = split == 'train'
        if is_train:
            self.model.train()
        else:  # VAL
            self.model.eval()
        self.bb.start(len(dataloader))
        for i, ((pid, sid), inputs, target) in enumerate(dataloader):
            # store_array_gt.update(pid, sid, target)
            if self.opt.debug and i > 1:  # check debug.
                break
            start = time.time()
            # * Data preparation *
            # inputs.requires_grad_(TRAIN)
            # target.requires_grad_(TRAIN)
            if self.opt.GPU:
                inputs, target = inputs.cuda(), target.cuda()
            inputV, targetV = Variable(inputs), Variable(target)
            datatime = time.time() - start
            # * Feed in nets*
            if is_train:
                self.optimizer.zero_grad()
            output = self.model(inputV)
            loss, loss_record = self.criterion(output, targetV.long())
            if is_train:
                loss.mean().backward()
                self.optimizer.step()
            # * Eval *
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                # if save_pred:
                #     name = 'val_pred{}_{}.npy'.format(epoch, i)
                #     numpy.save(name.replace('pred', 'gt'), target)
                metrics = self.metrics(preds, targetV)
            store_array_pred.update(pid, sid, preds.detach().cpu().numpy())
            runTime = time.time() - start - datatime
            log = self.bb.update(loss_record, {'TD': datatime, 'TR': runTime}, metrics, split, i, epoch)
            del loss, loss_record, output
            self.logger['train'].write(log)

        log = self.bb.finish(epoch, split)
        self.logger[split].write(log)
        store_array_pred.save(None, save_path='Pred_'+split, split_save=True)
        return self.bb.avgLoss()['loss']

    def train(self, trainLoader, epoch):
        return self.processing(trainLoader, epoch, 'train')

    def test(self, trainLoader, epoch):
        return self.processing(trainLoader, epoch, 'val')

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

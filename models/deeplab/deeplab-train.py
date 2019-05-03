import os
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from util.summaries import BoardX
from util.utils import StoreArray
import SimpleITK as sitk
from util.utils import RunningAverageDict
from util.evaluation import modelsize
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
        self.www = opt.www

    def processing(self, dataloader, epoch, split, eval):
        store_array_pred = StoreArray(len(dataloader), self.www + '/Pred_' + split)
        store_array_gt = StoreArray(len(dataloader), self.www + '/GT_' + split)
        # use store utils to update output.
        print('=> {}ing epoch # {}'.format(split, epoch))
        is_train = split == 'train'
        is_eval = eval
        if is_train:
            self.model.train()
        else:  # VAL
            self.model.eval()
        self.bb.start(len(dataloader))
        processing_set = []
        for i, ((pid, sid), inputs, target) in enumerate(dataloader):
            # check debug.
            if self.opt.debug and i > 2:
                break

            # store the patients processed in this phase.
            if pid not in processing_set:
                processing_set += [*pid]
            start = time.time()
            # * Data preparation *
            if self.opt.GPU:
                inputs, target = inputs.cuda(), target.cuda()
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
            # * Eval *
            metrics = {}
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                if is_eval:
                    metrics = self.metrics(preds, targetV)

            # save each slice ...
            store_array_pred.update(pid, sid, preds.detach().cpu().numpy())
            store_array_gt.update(pid, sid, targetV.detach().cpu().numpy())

            runTime = time.time() - start - datatime
            log = self.bb.update(loss_record, {'TD': datatime, 'TR': runTime}, metrics, split, i, epoch)
            del loss, loss_record, output
            self.logger[split].write(log)

        store_array_pred.save()
        store_array_gt.save()
        del store_array_pred, store_array_gt
        self.logger[split].write(self.bb.finish(epoch, split))

        set_ = sorted(list(set(processing_set)))
        output_path = self.www + '/Pred_' + split
        gt_path = self.www + '/GT_' + split

        hdf = sitk.HausdorffDistanceImageFilter()
        dicef = sitk.LabelOverlapMeasuresImageFilter()
        #  ------------ eval ------------
        HDdict_mean = RunningAverageDict()
        dicedict_mean = RunningAverageDict()
        for instance in set_:
            print(instance)
            pred = np.load(os.path.join(output_path, instance + '.npy'))
            gt = np.load(os.path.join(gt_path, instance + '.npy'))
            # Post Processing.
            # DenseCRF

            # simpleITK HD dice
            pred = sitk.GetImageFromArray(pred)
            gt = sitk.GetImageFromArray(gt)
            HDdict = {}
            dicedict = {}
            for i in range(self.opt.numClasses - 1):
                HD = np.nan
                dice = np.nan
                try:
                    hdf.Execute(pred == i + 1, gt == i + 1)
                    HD = hdf.GetHausdorffDistance()
                except:
                    pass
                try:
                    dicef.Execute(pred == i + 1, gt == i + 1)
                    dice = dicef.GetDiceCoefficient()
                except:
                    pass
                HDdict['HD#' + str(i)] = HD
                dicedict['Dice#' + str(i)] = dice
            HDdict_mean.update(HDdict)
            dicedict_mean.update(dicedict)
            del pred, gt
        # calculate mean
        self.bb.writer.add_scalars(self.opt.hashKey + '/scalar/HD_{}/'.format(split), HDdict_mean(), epoch)
        self.bb.writer.add_scalars(self.opt.hashKey + '/scalar/dice_{}/'.format(split), dicedict_mean(), epoch)
        return self.bb.avgLoss()['loss']

    def train(self, dataLoader, epoch):
        loss = self.processing(dataLoader, epoch, 'train', False)
        return loss

    def test(self, dataLoader, epoch):
        loss = self.processing(dataLoader, epoch, 'val', False)
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

import numpy as np
import tensorboardX

from util.progbar import progbar
from util.utils import RunningAverage

def updateLog(epoch, i, length, time, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f Err %1.4f   ' % (
        epoch, i, length, time, err)
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]()
    log += '\n'
    return log


class BoardX:
    def __init__(self, opt, metrics, suffix, log_num):
        self.writer = tensorboardX.SummaryWriter(log_dir=opt.logDir)
        self.opt = opt
        self.metrics = metrics
        self.suffix = suffix
        self.log_num = log_num

    def start(self, lenDS):
        self.lenDS = lenDS
        self.avgLoss = RunningAverage()
        self.avgAcces = {}
        # init all metrics
        for metric in self.metrics.name:
            if metric == 'IoU':
                for c in range(1, self.opt.numClasses):
                    self.avgAcces['IoU#{}'.format(c)] = RunningAverage()
            else:
                self.avgAcces[metric] = RunningAverage()
        self.progbar = progbar(self.lenDS, width=self.opt.barwidth)
        self.log_interval = int(lenDS / self.log_num)

    def update(self, runningLoss, time, preds, targets, split, i, epoch):
        # self.writer.add_scalar(//)
        logger_idx = i // self.log_interval + (epoch - 1) * self.log_num
        flag = 0
        if (i - 1) % self.log_interval == 0 or i == (self.lenDS - 1):
            flag = 1

        if not np.isnan(runningLoss) and flag == 1:
            self.writer.add_scalars(self.suffix + '/scalar/Loss', {'Loss_' + split: runningLoss}, logger_idx)
            self.avgLoss.update(float(runningLoss))
        self.logAcc = []
        for metric in self.metrics.name:
            meTra = self.metrics[metric](preds, targets)
            if metric == 'IoU':
                newTra = dict()
                for class_i in range(1, len(meTra)):
                    lower_key = 'IoU#{}'.format(class_i)
                    lower_val = meTra[class_i]

                    # update...
                    newTra['IoU#{}'.format(class_i)] = meTra[class_i]
                    self.avgAcces[lower_key].update(lower_val)
                    self.logAcc.append((lower_key, lower_val))
                if flag == 1:
                    self.writer.add_scalars(self.suffix + '/scalar/' + metric + "_" + split, newTra, logger_idx)
            else:
                if flag == 1:
                    self.writer.add_scalars(self.suffix + '/scalar/' + metric, {metric + '_' + split: meTra},
                                            logger_idx)
                self.avgAcces[metric].update(meTra)
                self.logAcc.append((metric, self.avgAcces[metric]()))

        log = updateLog(epoch, i, self.lenDS, time, self.avgLoss(), self.avgAcces)
        self.progbar.update(i + 1, [('Time', time), ('loss', self.avgLoss()), *self.logAcc])
        return log

    def finish(self, epoch):
        log = '\n* Finished training epoch # %d  Loss: %1.4f  ' % (epoch, self.avgLoss())
        for metric in self.avgAcces:
            log += "\n" + metric + " %1.4f  " % self.avgAcces[metric]()
        log += '\n'
        print(log)
        return log

    def close(self):
        self.writer.close()

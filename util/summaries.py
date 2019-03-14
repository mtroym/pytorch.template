import numpy as np
import tensorboardX

from util.progbar import progbar
from util.utils import RunningAverageDict


def updateLog(epoch, i, length, time, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f ' % (
        epoch, i, length, time)
    for loss in err:
        log += loss + " %1.4f  " % err[loss]
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]
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
        self.avgLoss = RunningAverageDict(10.0)
        self.avgAcces = RunningAverageDict(0.0)
        self.progbar = progbar(self.lenDS, width=self.opt.barwidth)
        self.log_interval = int(lenDS / self.log_num)

    def update(self, lossRecord, time, metrics, split, i, epoch):
        # metrics -> {'m1': val, 'm2': np.nan, ... }
        # maybe contain nan values.
        # lossRecord -> {'loss': val, 'combined_1': val, 'combined_2':val}
        # must have one val which key is `loss` !
        logger_idx = np.floor(i // self.log_interval) + (epoch - 1) * self.log_num
        flag = (i - 1) % self.log_interval == 0
        if self.log_num == 1:
            logger_idx = epoch
            flag = i == self.log_interval - 1
        self.avgLoss.update(lossRecord)
        self.avgAcces.update(metrics)

        if flag == 1:
            self.writer.add_scalars(self.suffix + '/scalar/Loss', self.avgLoss(split), logger_idx)
            self.writer.add_scalars(self.suffix + '/scalar/Acc' + '_' + split, self.avgAcces(split), logger_idx)
        log = updateLog(epoch, i, self.lenDS, time, self.avgLoss(split), self.avgAcces(split))
        self.progbar.update(i + 1,
                            [('Time', time)] + list(self.avgLoss(split).items()) + list(self.avgAcces(split).items()))
        return log

    def finish(self, epoch, split):
        log = '\n\n* Finished training epoch # %d *\n\n' % epoch
        log += '* METRICS:'
        log += str(self.avgAcces(split)) + ' *\n\n* LOSSES: ' + str(
            self.avgLoss(split)) + '\n\n'
        print(log)
        return log

    def close(self):
        self.writer.close()

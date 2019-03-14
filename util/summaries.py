import numpy as np
import tensorboardX

from util.progbar import progbar
from util.utils import RunningAverage, RunningAverageNaN


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
        self.avgLoss = RunningAverage(100, 1)
        self.avgAcces = {}
        # init all metrics
        for metric in self.metrics.name:
            if metric == 'IoU':
                for c in range(1, self.opt.numClasses):
                    self.avgAcces['IoU#{}'.format(c)] = RunningAverageNaN(0.0)
            else:  # mIoU
                self.avgAcces[metric] = RunningAverageNaN(0.0)
        self.progbar = progbar(self.lenDS, width=self.opt.barwidth)
        self.log_interval = int(lenDS / self.log_num)

    def update(self, lossRecord, time, preds, targets, split, i, epoch):
        # TODO: separate the metrics operation out of this function.
        # lossRecord -> {'loss': val, 'combined_1': val, 'combined_2':val}
        # must have one val which key is `loss` !
        logger_idx = np.floor(i // self.log_interval) + (epoch - 1) * self.log_num
        flag = 0
        if (i - 1) % self.log_interval == 0:
            flag = 1

        for loss in lossRecord:
            if not np.isnan(lossRecord[loss]) and flag == 1 and self.log_num != 1:
                self.writer.add_scalars(self.suffix + '/scalar/Loss', {loss + '_' + split: lossRecord[loss]}, logger_idx)
            if loss == 'loss':
                self.avgLoss.update(lossRecord['loss'])
        if self.log_num == 1:
            self.writer.add_scalars(self.suffix + '/scalar/Loss', {'loss_' + split: self.avgLoss()}, logger_idx)

        self.logAcc = []
        for metric in self.metrics.name:
            metric_value = self.metrics[metric](preds, targets)
            if metric == 'IoU':
                IoU_dict = dict()
                for class_i in range(1, len(metric_value)):
                    lower_key = 'IoU#{}'.format(class_i)
                    # metric have GPU tensors.
                    lower_val = float(metric_value[class_i])
                    # update...
                    IoU_dict['IoU#{}'.format(class_i)] = lower_val
                    self.avgAcces[lower_key].update(lower_val)
                    self.logAcc.append((lower_key, lower_val))
                if flag == 1:
                    self.writer.add_scalars(self.suffix + '/scalar/' + metric + "_" + split, IoU_dict, logger_idx)
            else:
                if flag == 1:
                    self.writer.add_scalars(self.suffix + '/scalar/' + metric, {metric + '_' + split: metric_value},
                                            logger_idx)
                self.avgAcces[metric].update(metric_value)
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

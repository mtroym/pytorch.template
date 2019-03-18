import copy

import numpy as np


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, len=10, default=0.0):
        self.total = [np.nan] * len
        self.index = 0
        self.len = len
        self.default = default
        self.last = default

    def update(self, val):
        self.total[self.index] = val
        self.index = (self.index + 1) % self.len

    def __call__(self):
        # if total is all nan, then return last valid value.
        # if not, then update last valid value to current mean.
        if not np.all(np.isnan(self.total)):
            self.last = np.nanmean(self.total)
        return self.last

    def __len__(self):
        return self.len


class RunningAverageNaN:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, default=0.0):
        self.total = 0.0
        self.default = default
        self.count = 0

    def update(self, val):
        if not np.isnan(val):
            self.total += val
            self.count += 1

    def __call__(self):
        return (self.total / self.count) if self.count > 0 else self.default


class RunningAverageDict:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverageDict(0.0)
    loss_avg.update({'name1': 2, 'name2':3, 'name3': np.nan})
    loss_avg.update({'name1': 6, 'name2':np.nan, 'name3':np.nan})
    loss_avg() = {'name1':4, 'name2':3, 'name3': np.nan}
    loss_avg('fuck', return_mean=False) = {'name1_fuck':4, 'name2_fuck':3, 'name3_fuck': np.nan}
    loss_avg('fuck') = {'name1_fuck':4, 'name2_fuck':3, 'name3_fuck': np.nan, 'mean':3.5}
    loss_avg('fuck', mean_key='m_name') = {'name1_fuck':4, 'name2_fuck':3, 'name3_fuck': np.nan, 'm_name':3.5}
    ```
    """

    def __init__(self, default=0.0):
        self.total = None
        self.default = default
        self.metrics_name = {}

    def update(self, val):
        '''val is a dict'''
        if self.total is None:
            self.total = copy.deepcopy(val)
            for keys in self.total:
                self.total[keys] = RunningAverageNaN(default=self.default)
        for keys in self.total:
            name = keys.split('#')[0]
            self.metrics_name[name] = [0.0, 0.0]
            self.total[keys].update(val[keys])

    def __call__(self, suffix='', return_mean=True, mean_key='mean'):
        return_dict = dict()
        for keys in self.total:
            val = '_'.join((keys, suffix)) if suffix != '' else keys
            return_dict[val] = self.total[keys]()
            name = keys.split('#')[0]
            self.metrics_name[name][0] += return_dict[val]
            self.metrics_name[name][1] += 1
        if return_mean:
            for keys in self.metrics_name:
                return_dict.update({'m' + keys: self.metrics_name[keys][0] / (self.metrics_name[keys][1] + 1e-10)})
        return return_dict


if __name__ == '__main__':
    loss_avg = RunningAverageDict(0.0)
    loss_avg.update({'name1': 2, 'name2': 3, 'name3': np.nan})
    print(loss_avg('train'))
    loss_avg.update({'name1': 6, 'name2': np.nan, 'name3': np.nan})
    # loss_avg() = {'name1':4, 'name2':3, 'name3': np.nan}
    print(loss_avg('val'))
    loss_avg.update({'name1': 1, 'name2': np.nan, 'name3': 1})
    # loss_avg() = {'name1':4, 'name2':3, 'name3': np.nan}
    print(loss_avg('train'))
    #

    # loss = RunningAverageNaN(0.0)
    # loss.update(10)
    # print(loss())
    # loss.update(np.nan)
    # print(loss())
    # loss.update(4)
    # print(loss())

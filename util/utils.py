import copy
import os

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


class StoreArray:
    def __init__(self, length=1):
        self.mem = [list() for _ in range(length)]
        self.len = length
        self.size = 0
        self.index_dict = {}

    def update(self, index, value_idx, value):
        for i in range(len(index)):
            self.update_single(index[i], int(value_idx[i]), value[i])

    def update_single(self, index, value_idx, value):
        data = (value_idx, value)
        if index in self.index_dict:
            self.mem[self.index_dict[index]].append(data)
        else:
            self.index_dict[index] = self.size
            self.size += 1
            self.mem[self.index_dict[index]].append(data)

    def __len__(self):
        return self.len

    def save(self, index=None, split_save=True, save_path='.'):
        save_all = index is None
        for index_instance in self.index_dict.keys():
            if save_all or (index_instance == index and not save_all):
                if index_instance not in self.index_dict:
                    continue
                real_idx = self.index_dict[index_instance]
                after_sorting = sorted(self.mem[real_idx], key=lambda x: x[0])
                real_val = np.array(list(map(lambda x: x[1], after_sorting)))
                path = os.path.join(save_path, str(index_instance))
                if not os.path.exists(os.path.join(save_path, str(index_instance))):
                    os.makedirs(os.path.join(save_path, str(index_instance)))
                if split_save:
                    for idx in range(len(after_sorting)):
                        i = after_sorting[idx][0]
                        name = '0' * (5 - len(str(i))) + str(i) + '.npy'
                        np.save(os.path.join(path, name), real_val[idx])
                else:
                    np.save(os.path.join(path, 'pred.npy'), real_val)


def test_store_array():
    sa = StoreArray(10)
    sa.update(3, 0, 'a')
    sa.update(3, 100, 'f')
    sa.update(3, 2, 'b')
    sa.update(3, 3, 'c')
    sa.update(3, 1500, 'g')
    sa.update(4, 9, 'd')
    sa.update(3, 78, 'e')
    sa.update(100, 0, '0')
    sa.update(100, 1, '1')
    sa.update(100, 3, '3')
    sa.update(100, 4, '4')
    sa.update(100, 2, '2')
    sa.update(100, 9, '9')
    sa.update(9, 1, 'great')
    sa.save(None, split_save=False, save_path='/Users/tony/PycharmProjects/Pred')


def test_running_average():
    loss_avg = RunningAverageDict(0.0)
    loss_avg.update({'name1': 2, 'name2': 3, 'name3': np.nan})
    print(loss_avg('train'))
    loss_avg.update({'name1': 6, 'name2': np.nan, 'name3': np.nan})
    # loss_avg() = {'name1':4, 'name2':3, 'name3': np.nan}
    print(loss_avg('val'))
    loss_avg.update({'name1': 1, 'name2': np.nan, 'name3': 1})
    # loss_avg() = {'name1':4, 'name2':3, 'name3': np.nan}
    print(loss_avg('train'))


if __name__ == '__main__':
    test_store_array()
    #

    # loss = RunningAverageNaN(0.0)
    # loss.update(10)
    # print(loss())
    # loss.update(np.nan)
    # print(loss())
    # loss.update(4)
    # print(loss())

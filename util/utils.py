import copy
import os

import numpy as np


DEBUG = True

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
            if name not in self.metrics_name:
                self.metrics_name[name] = [0.0, 0.0]
            self.total[keys].update(val[keys])

    def __call__(self, suffix='', return_mean=True, mean_key='mean'):
        return_dict = dict()
        # calculate each mean value..
        for keys in self.total:
            new_key = '_'.join((keys, suffix)) if suffix != '' else keys
            return_dict[new_key] = self.total[keys]()
            name = keys.split('#')[0]
            self.metrics_name[name][0] += return_dict[new_key]
            self.metrics_name[name][1] += 1
        # calculate the average of mean value.
        if return_mean:
            for keys in self.metrics_name:
                return_dict.update({'m' + keys: self.metrics_name[keys][0] / (self.metrics_name[keys][1] + 1e-10)})
        return return_dict


class StoreArray:
    """
    store, save the 2d slices by index. and combine when end of the inters.
    call self.save(), to save all slices of one instance.
    """
    def __init__(self, length=1, path='.'):
        """
        :param length: maybe length of the StoreArray.(num of instances.)
        :param path: where to store the numpy value.
        """
        self.mem = [list() for _ in range(length)]
        self.len = length
        self.size = 0
        self.index_dict = {}
        self.path = path

    def update(self, index, value_idx, value, axis='z'):
        """
        Save the slice value to file. save mem.
        Batch/single format.
        :param index: `index` of instance.
        :param value_idx: `index` of slice of (instance index).
        :param value: `value`(2d numpy array) of `value_idx`th slice of `index`th instance.
        :return: None.
        """
        # if DEBUG: print("===> update single!")
        if isinstance(index, int):
            self.save_single(index, value_idx, value, axis=axis)
            return
        for i in range(len(index)):
            self.save_single(index[i], int(value_idx[i]), value[i],  axis=axis)

    def save_single(self, index, value_idx, value, axis='z'):
        """
        handler of one slice.
        :param index: `index` of instance.
        :param value_idx: `index` of slice of (instance index).
        :param value: `value`(2d numpy array) of `value_idx`th slice of `index`th instance.
        :return: None.
        """
        name = '0' * (5 - len(str(value_idx))) + str(value_idx) + '.npy'
        path = os.path.join(self.path, str(index))
        # print(path)
        if not os.path.exists(path):
            os.makedirs(os.path.join(path))
        np.save(os.path.join(path, name), value)
        del value

    def __len__(self):
        return self.len

    def save(self, branch='z'):
        """
        combine each slice of one instance into one 3d-numpy array. (H, W, Z)
        Z: number of slices. TO same level of single paths.
        :return:None
        """
        print(self.path)
        folder_with_instance = os.listdir(self.path)
        for ins in folder_with_instance:
            # validate the `ins` is the dir of one instance.
            if not os.path.isdir(os.path.join(self.path, ins)):
                continue
            # read all slices/sequences of this `dir`.
            slices = sorted(os.listdir(os.path.join(self.path, ins)))
            if not DEBUG:
                assert int(slices[-1].split('.')[0]) == len(slices) - 1, 'the number of slice did not match!'
            compose = []
            for slice_ in slices:
                compose.append(np.load(os.path.join(self.path, ins, slice_))[..., np.newaxis])
            # combine them all, and save.
            compose = np.concatenate(compose, 2)
            np.save(os.path.join(self.path, ins + '.npy'), compose)
            del compose, slices


    def save_zzz(self, branch='z'):
        """
        combine each slice of one instance into one 3d-numpy array. (H, W, Z)
        Z: number of slices. TO same level of single paths.
        :return:None
        """
        print(self.path)
        folder_with_instance = os.listdir(self.path)
        for ins in folder_with_instance:
            # validate the `ins` is the dir of one instance.
            if not os.path.isdir(os.path.join(self.path, ins)):
                continue
            # read all slices/sequences of this `dir`.
            slices = sorted(os.listdir(os.path.join(self.path, ins)))
            if not DEBUG:
                assert int(slices[-1].split('.')[0]) == len(slices) - 1, 'the number of slice did not match!'
            compose = []
            for slice_ in slices:
                compose.append(np.load(os.path.join(self.path, ins, slice_)))
            # combine them all, and save.
            compose = np.stack(compose, 3)
            if branch == 'x':
                compose = compose.transpose([0, 3, 1, 2])
            elif branch == 'y':
                compose = compose.transpose([0, 1, 3, 2])
            np.save(os.path.join(self.path, ins + '.npy'), compose)
            del compose, slices

def test_store_array():
    sa = StoreArray(10, '/Users/tony/PycharmProjects/pytorch.template/output/')
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
    sa.save()

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

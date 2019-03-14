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
        if np.isnan(val):
            return
        self.total += val
        self.count += 1

    def __call__(self):
        return (self.total / self.count) if self.count > 0 else self.default

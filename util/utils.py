import numpy as np

class RunningAverage():
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

    def update(self, val):
        self.total[self.index] = val
        self.index = (self.index + 1) % self.len

    def __call__(self):
        if np.all(np.isnan(self.total)): return self.default
        return np.nanmean(self.total)
import random
from torch import manual_seed, cuda, backends
import numpy as np


class Meter:
    def __init__(self):
        self.values, self.avg, self.sum, self.cnt = [], 0, 0, 0

    def reset(self):
        self.values, self.avg, self.sum, self.cnt = [], 0, 0, 0

    def update(self, value, k=1):
        self.values.append(value)
        self.sum += value
        self.cnt += k
        self.avg = self.sum / self.cnt


def fix_random_seed(seed=1234):
    # Ref.: https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True

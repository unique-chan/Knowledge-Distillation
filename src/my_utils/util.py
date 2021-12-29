import random

import torch
from torch import manual_seed, cuda, backends
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


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


def store_txt(path, txt):
    with open(path, 'w') as f:
        f.write(str(txt))
        f.flush()

def createConfusionMatrix(net, loader, num_of_classes):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = (list(range(num_of_classes)))

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True).get_figure()
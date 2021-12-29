import random

import torch
from matplotlib import pyplot as plt
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


def createConfusionMatrix(net, loader, num_of_classes, labels=None):
    y_pred, y_true = [], [] # save prediction, ground truth

    # iterate over data
    with torch.no_grad():
        for _, inputs, labels in loader:
            output = net(inputs.cuda())  # Feed Network
            output = (torch.max(output, 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth

    if num_of_classes <= 15:
        if labels is None:
            labels = list(range(num_of_classes))
        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        new_matrix = cf_matrix/np.sum(cf_matrix) * num_of_classes
        p = sn.heatmap(np.array(new_matrix), annot=True, vmin=0, vmax=1, fmt='.2f', square='True', cmap="Blues")
    else:
        classes = list(range(num_of_classes))

        # constant for classes
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * num_of_classes,
                             index=classes, columns=classes)
        p = sn.heatmap(df_cm, vmin=0, vmax=1, square='True', cmap="Blues")

    plt.figure(figsize=(11, 8))

    p.set_xlabel('Ground Truth Class')
    p.set_ylabel('Predicted Class')

    p.axhline(y=0, color='k', linewidth=1)
    p.axhline(y=num_of_classes, color='k', linewidth=2)
    p.axvline(x=0, color='k', linewidth=1)
    p.axvline(x=num_of_classes, color='k', linewidth=2)

    plt.tight_layout()
    return p.get_figure()

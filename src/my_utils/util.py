import itertools
import random

from matplotlib import pyplot as plt
from torch import manual_seed, cuda, backends
import numpy as np
from sklearn.metrics import confusion_matrix


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


def create_confusion_matrix(y_trues, y_preds, num_of_classes,
                            class_names=None, threshold=15, figsize=(8, 6), cmap=plt.cm.Blues):
    cf_matrix = confusion_matrix(y_trues, y_preds)
    normalized_cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # normalized_cf_matrix = cf_matrix / np.sum(cf_matrix) * num_of_classes

    fig = plt.figure(figsize=figsize)
    plt.imshow(normalized_cf_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if num_of_classes <= threshold:
        labels = class_names if class_names else np.arange(num_of_classes)
        plt.xticks(np.arange(num_of_classes), labels, rotation=45)
        plt.yticks(np.arange(num_of_classes), labels)

        # plotting probabilities: p(prediction=i|ground_truth=j) for all classes i and j.
        txt_color_threshold = 0.5
        for i, j in itertools.product(np.arange(normalized_cf_matrix.shape[0]),
                                      np.arange(normalized_cf_matrix.shape[1])):
            plt.text(j, i, f'{normalized_cf_matrix[i, j]: .2f}',
                     horizontalalignment='center',
                     color='white' if normalized_cf_matrix[i, j] > txt_color_threshold else 'black')

    plt.tight_layout()
    plt.xlabel('Predicted Class' + ('' if class_names else ' Index'))
    plt.ylabel('Ground Truth Class' + ('' if class_names else ' Index'))

    return fig

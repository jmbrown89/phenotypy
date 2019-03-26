from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict
import torch
from ignite.metrics.accuracy import Accuracy


def mean_accuracy(y_true, y_pred):

    return np.mean(per_class_accuracy(y_true, y_pred))


def per_class_accuracy(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    diag = cm.diagonal()
    return diag


class PerClassAccuracy(Accuracy):
    """
    Calculates the per-class accuracy for multi-class data
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)

    """
    def reset(self):
        self._num_correct = defaultdict(int)
        self._num_examples = defaultdict(int)

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))
        indices = torch.max(y_pred, dim=1)[1]  # max returns the max value and its index...

        correct = torch.eq(indices, y).view(-1)  # TODO finish this off
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):

        accuracy = []
        for (k, correct), (_, examples) in zip(self._num_correct.items(), self._num_examples.items()):

            if examples != 0:
                accuracy.append(correct / examples)

        return np.mean(accuracy)

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import numpy as np
from collections import defaultdict
import torch
from pathlib import Path
import pandas as pd
from ignite.metrics.accuracy import Accuracy
from phenotypy.visualization.plotting import Plotter


curves = {'pr': precision_recall_curve, 'roc': roc_curve}


class Evaluator:

    def __init__(self, out_dir):

        self.out_dir = Path(out_dir)
        self.plotter = Plotter(self.out_dir / 'plots')

    def auc(self, y_true, y_pred, method='roc'):

        curve = curves.get(method, roc_curve)
        fpr, tpr, _ = curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def run(self):

        return NotImplementedError("Use a subclass like MultiClassEvaluator")


class MultiClassEvaluator(Evaluator):

    """
    A class that serves to evaluate multi-class classifiers in such as way that
    various metrics can be recorded and plotted easily.
    """
    def __init__(self, out_dir, encoding=None):

        super(MultiClassEvaluator, self).__init__(out_dir)
        self.encoding = encoding
        self.roc = defaultdict(list)
        self.y_pred, self.y_true = [], []
        self.accuracy = []
        self.legends = []

    def auc(self, y_true, y_pred, method='roc', legend=None):

        if legend:
            self.legends.append(legend)

        if len(y_true.shape) == 1:

            # Infuriating way to insert columns where there is no data
            diff = map(str, set(range(0, 8)).difference(set(y_true)))
            y_true = pd.get_dummies(y_true)
            # y_true = y_true.assign(**dict.fromkeys(diff, 0)).values

        for i, label in self.encoding.items():

            if i not in y_true.columns:
                continue
            fpr, tpr, area = super().auc(y_true[i], y_pred[:, i], method=method)
            self.roc[label].append({'fpr': fpr, 'tpr': tpr, 'area': area})

            # label_str = label if 'micro' not in label else 'micro.'
            # plt.plot(fpr, tpr, lw=2., label=f'{label_str} (AUC = {roc_auc:.2f})')

    def mean_accuracy(self, y_true, y_pred):

        per_class_accuracy(y_true, y_pred)
        result = np.mean(per_class_accuracy(y_true, y_pred))
        self.accuracy.append(result)

    def confusion(self, y_true, y_pred):

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def run(self):

        # Plot
        for label in self.roc.keys():
            aucs = [self.roc[label][i]['area'] for i in range(0, len(self.roc[label]))]
            print(f'{label}: {np.mean(aucs)}')

        self.plotter.plot_multiclass_roc(self.roc, legend=self.legends)

        cm = confusion_matrix(self.y_true, self.y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self.plotter.plot_confusion(cm, self.encoding)


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

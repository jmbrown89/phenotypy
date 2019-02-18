# TODO this will be used to evaluate performance of a model, using the results from predict_model.py

import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion(y_true, y_pred, labels):

    mat = confusion_matrix(y_true, y_pred)
    pd.DataFrame(mat, columns=labels, index=labels)
    print(mat)

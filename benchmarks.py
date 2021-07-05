from typing import Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def sample_mean(y, start, cv_outer) -> pd.DataFrame:
    """
    :param y: pd.DataFrame, used to calculate sample mean
    :param start: time string, time point we start to calculate sample mean
    :param cv_outer: rules for train test split
    :return: two lists: sample mean prediction and mse
    """
    sm = list()
    mse = list()
    for train_index, test_index in cv_outer.split(y):
        # Split train and test sets
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_hat = np.mean(y_train)
        sm.append(y_hat)
        mse.append(np.mean(y_hat - y_test) ** 2)
    return sm, mse


fig_size = (10, 5)
f = plt.figure(figsize=fig_size)


def plot_R2(y, model_pred, sm_pred, adjust=False, **kwargs):
    """
    :param y: real data
    :param model_pred: prediction obtained by benchmark model
    :param sm_pred: prediction obtained by our model
    :param adjust: whether adjust the R2 or not, default is False
    :return: R^2 measurement
    """
    R2_raw = [1 - np.sum(i - j) ** 2 / np.sum(i - k) ** 2 for i, j, k in zip(y, model_pred, sm_pred)]
    if adjust:
        R2 = 1 / (1 + np.exp(-np.array(R2_raw)))
        base = 0.5
    else:
        R2 = R2_raw
        base = 0
    plt.figure(figsize=(16, 8))
    plt.plot(y.index, R2, **kwargs)
    plt.axhline(y=base, color='r', linestyle='-')
    plt.title('$R^2$')
    # assert isinstance(f, object)
    return f

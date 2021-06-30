from typing import Any, Union

import numpy as np
import pandas as pd
import datetime as dt
import openpyxl
import os

from numpy import ndarray
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray

from sklearn.model_selection import TimeSeriesSplit


def read_data(name, sheet_name=0) -> pd.DataFrame:
    """
    :param name: file name with extension
    :param sheet_name: which sheet to read, return 1st one as default
    :return: file content
    """
    path = 'data/%s.xlsx' % name
    if os.path.exists(path):
        df = pd.read_excel(path, sheet_name, index_col=0)
        return df
    else:
        raise FileNotFoundError('Cannot find the file')


def data_clean(df, start, drop_cols=[]) -> pd.DataFrame:
    """
    :param df: dataframe to be cleaned
    :param start: starting time
    :param drop_cols: list of columns names, default is empty
    :return: subset dataframe
    """
    target = [col for col in df.columns if 'yyyy' in col]
    if 'm' in target[0]:
        df['time'] = pd.to_datetime(df[target[0]].astype(str), format='%Y%m')
    elif 'q' in target[0]:
        df['time'] = [pd.to_datetime(str([target[0]])[:4]) + pd.offsets.QuarterBegin(int(str(target[0])[4:])) for x in
                      df['yyyyq']]
    else:
        df['time'] = [pd.to_datetime(str(target[0])[:4])]

    df = df.set_index('time')
    df = df.drop(target[0], axis=1)
    df = df.drop(drop_cols, axis=1)
    df = df[start:]
    return df


def data_split(df: pd.DataFrame, y_variable: str, train_end: str, test_end: str) -> pd.DataFrame:
    """
    :param df: dataframe
    :param y_variable: name of dependent variable
    :param train_end: "yyyy-mm-dd", the end of training set
    :param test_end:"yyyy-mm-dd", the end of testing set
    :return: 4 split dataframe: X_train and X_test, y_train and y_test

    """
    # find the last index with "train_year"
    X_train = df[:train_end].drop([y_variable], axis=1)
    y_train = df[:train_end][y_variable]
    X_test = df[train_end:test_end].drop([y_variable], axis=1)[1:]
    y_test = df[train_end:test_end][y_variable][1:]
    return X_train, X_test, y_train, y_test

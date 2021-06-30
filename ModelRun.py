import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree

import warnings
warnings.filterwarnings("ignore")


def Nested_CV(X, y, model, cv_inner, cv_outer, search_method, space) -> list:
    """
    :param X: features
    :param y: targer to be forecasted
    :param model: model being used
    :param cv_inner: cross validation split for inner round
    :param cv_outer: cross validation split for outer round
    :param search_method GridSearchCV or RandomizedSearchCV
    :param space: hyper parameter space
    :return: 3 lists: best models, predictions and mse
    """

    best_models = list()
    prediction = list()
    mse_list = list()

    for train_index, test_index in cv_outer.split(X):
        # Split train and test sets
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Define search
        if search_method == 'Grid':
            search = GridSearchCV(model, space, scoring='neg_mean_squared_error', cv=cv_inner, refit=True)
        else:
            search = RandomizedSearchCV(model, space, scoring='neg_mean_squared_error', cv=cv_inner, refit=True)
        # Execute search
        result = search.fit(X_train, y_train)
        # Get the best model
        best_model = result.best_estimator_
        best_models.append(best_model)
        # Evaluate model on hold out dataset
        yhat = best_model.predict(X_test)
        prediction.append(yhat)
        mse = mean_squared_error(y_test, yhat)
        mse_list.append(mse)
        print('mse=%.4f, best=%.4f, cfg=%s' % (mse, result.best_score_, result.best_params_))

    return best_models, prediction, mse_list
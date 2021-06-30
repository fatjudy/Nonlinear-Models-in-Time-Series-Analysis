import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


# def sin_func(self, x):
#     l = x.shape[1]
#
#     def obj_func(params):
#         func = np.sin(self.single_index(x)(params[0:l]) + params[l])
#         return func
#
#     return obj_func

# def sin_func(x):
#     l = x.shape[1]
#
#     def obj_func(parameters):
#         func = np.sin(single_index(x)(parameters[0:l]) + parameters[l])
#         return func
#
#     return obj_func

class CLS_Estimator(BaseEstimator, RegressorMixin):

    def __init__(self, obj_func=None, x0=0, method='SLSQP'):
        # self.obj_func = obj_func
        self.x0 = x0
        self.method = method
        self.obj_func = obj_func
        self.params_ = None

    def single_index(self, x):
        if isinstance(x, (pd.DataFrame, np.ndarray)):
            if isinstance(x, pd.DataFrame):
                x_values = x.values
            else:
                pass
        else:
            raise Exception('wrong type')

        def u(theta):
            if len(theta) == x_values.shape[1]:
                sum_up = [x_values[:, i] * theta[i] for i in range(x_values.shape[1])]
                index = np.sum(sum_up, axis=0)
            else:
                raise Exception('wrong parameter dimension')
            return index

        return u

    def _loss(self, x, y):
        def loss_func(params):
            error = np.sum((y - self.obj_func(x)(params)) ** 2)
            return error
        return loss_func

    def fit(self, x, y):
        self._train_data = x
        self._train_target = y

        res = minimize(
            self._loss,
            x0=self.x0,
            method=self.method
        )

        # res = self.optimizer
        if res.success:
            self.params_ = res.x
        return self

    def predict(self, X):
        self.yhat=self.obj_func(X)(self.params_)
        return self.yhat

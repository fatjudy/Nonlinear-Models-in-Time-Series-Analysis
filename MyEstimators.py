import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


def single_index(x):
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


class CLS_Estimator(BaseEstimator, RegressorMixin):

    def __init__(self, obj_func=None, x0=0, method='SLSQP', constraints=(), tol=1e-3, options={'maxiter':50000}):
        self.obj_func = obj_func
        self.x0 = x0
        self.method = method
        self.constraints = constraints
        self.tol = tol
        self.options = options
        self.params_ = None

    # def constraint_func(self, x):
    #     def constraint(params):
    #         con = 0
    #         for j in np.arange(0, x.shape[1]):
    #             con += params[j] ** 2
    #             cons = con - 1
    #         return cons
    #     return {'type': 'eq', 'fun': constraint}

    def loss(self, x, y):
        def loss_func(params):
            error = np.sum((y - self.obj_func(x)(params)) ** 2)
            return error

        return loss_func

    def fit(self, x, y):
        self._train_data = x
        self._train_target = y

        res = minimize(
            self.loss(x, y),
            x0=self.x0,
            method=self.method,
            constraints=self.constraints,
            options=self.options
        )

        #         res = self.optimizer
        if res.success:
            self.params_ = res.x
        else:
            raise Exception('fit failed')
        return self

    def predict(self, X):
        self.yhat = self.obj_func(X)(self.params_)
        return self.yhat

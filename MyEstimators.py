import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

class CLS_Estimator(BaseEstimator, RegressorMixin):

    def __init__(self, obj_func=None, x0=0, method='SLSQP', constraints=(), options={'maxiter':50000}):
        self.obj_func = obj_func
        self.x0 = x0
        self.method = method
        self.constraints = constraints
        self.options = options
        self.params_ = None

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

import numpy as np
import pandas as pd

param_num = None

def single_index(x):
    """
    :param x: non-stationary independent variables in the model
    :return: a function of unknown parameter theta
    """
    if isinstance(x, (pd.DataFrame, np.ndarray)):
        pass
    else:
        raise Exception('wrong type')

    def u(theta):
        """
        :param theta: parameters in the single-index
        :return:
        """
        if len(theta) == x.shape[1]:
            sum_up = [x.iloc[:, i] * theta[i] for i in range(x.shape[1])]
            index = np.sum(sum_up, axis=0)
        else:
            raise Exception('wrong parameter dimension')
        return index

    return u

def sin_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = np.sin(single_index(x.iloc[:,:d1])(params[0:d1])+params[d1+d2+extra[0]])+np.dot(
            x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def cos_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = np.cos(single_index(x.iloc[:,:d1])(params[0:d1])+params[d1+d2+extra[0]])+np.dot(
            x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def scaled_sin_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = np.sin(params[d1+d2+extra[1]]*single_index(x.iloc[:,:d1])(
            params[0:d1])+params[d1+d2+extra[0]])+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def scaled_cos_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = np.cos(params[d1+d2+extra[1]]*single_index(x.iloc[:,:d1])(
            params[0:d1])+params[d1+d2+extra[0]])+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def exp_shift_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = 1 - np.exp(params[d1+d2+extra[1]]*((single_index(x.iloc[:,:d1])(
            params[0:d1]))-params[d1+d2+extra[0]])**2)+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def exp_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = params[d1+d2+extra[0]]*np.exp(-params[d1+d2+extra[1]]*(single_index(x.iloc[:,:d1])(
            params[0:d1]))**2)+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def poly_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):

        func = params[d1+d2+extra[0]]+params[d1+d2+extra[1]]*(single_index(x.iloc[:,:d1])(
            params[0:d1]))+params[d1+d2+extra[2]]*((single_index(x.iloc[:,:d1])(
            params[0:d1]))**2)+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func

def linear_func(x):
    def objective_func(params,
                       d1 = param_num['theta'],
                       d2 = param_num['beta'],
                       extra = range(0,param_num['gamma'])):
        func = params[d1+d2+extra[0]]+params[d1+d2+extra[1]]*(single_index(x.iloc[:,:d1])(
            params[0:d1]))+np.dot(x.iloc[:,d1:d1+d2], params[d1:d1+d2])
        return func
    return objective_func
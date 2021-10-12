import os
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

import Models


def series_generator(length, alpha, et):
    """
    generate AR series
    :param length: length of the series
    :param alpha: list of AR coefficients
    :param et: error terms
    :return: generated series
    """
    if isinstance(alpha, float):
        alpha = [alpha]

    j = 0
    z = np.array([[0] * (len(alpha))])
    while j < length - 1:
        _new = alpha * z[j] + et[j]
        z = np.append(z, [_new], axis=0)
        j += 1
    return z

def xy_generator(parameters, sample_size, function, **options):
    """

    :param parameters: a dictionary of parameters theta, beta and gamma
    :param sample_size:
    :param function:
    :param options: stationarity, constraints
    :return:
    """

    # Models.param_num = {'theta': parameters['theta'],
    #                     'beta': parameters['beta'],
    #                     'gamma': parameters['gamma']
    #                     }

    # error generation
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    _ep = np.random.multivariate_normal(mean, cov, sample_size + initial)
    epson = _ep[1:, :]
    epson_lag = _ep[:(sample_size + initial - 1), :]

    c = np.array([[-1, 4 / 3], [0, 0]])

    # initiate v
    v = np.array([[0] * len(parameters['theta'])])
    # generate x variables
    k = 0
    while k < sample_size + initial - 1:
        _new = epson[k]+np.dot(c, epson_lag[k])
        v = np.append(v, [_new], axis=0)
        k += 1

    if options.get('stationary'):
        rho = [0.5] * len(parameters['theta'])
    else:
        rho = [1] * len(parameters['theta'])

    # generate x variables
    x = series_generator(sample_size+initial, rho, v)

    # construct the stationary variable
    st = np.random.standard_normal(sample_size + initial)
    z = series_generator(sample_size+initial, parameters['beta'][1], st)

    # construct the single-index
    u = Models.single_index(x)(parameters['theta'])

    # initiate y
    Models.param_num = {'theta': len(parameters['theta']),
                        'beta': len(parameters['beta']),
                        'gamma': len(parameters['gamma'])
                        }
    i = 0
    y = np.array([[0]])
    while i < 1100 - 1:
        X_ = np.concatenate((x[i], y[i], z[i])).reshape(1, -1)
        _new = function(X_)(parameters['theta'] + parameters['beta'] + [0.2])
        y = np.append(y, [_new], axis=0)
        i += 1

    x = x[initial:]
    z = z[initial:]
    y_lag = y[initial - 1:sample_size + initial - 1]
    y = y[initial:]
    data = np.concatenate((x, y_lag, z,y), axis = 1)

    return data

parameters = {'theta': [0.6, -0.8],
              'beta': [0.8, 0.9],
              'gamma': [0.2]
              }

# Models.param_num = {'theta': 2,
#               'beta': 2,
#               'gamma': 1
#               }

dt = xy_generator(parameters=parameters, sample_size=1000, function=Models.cos_func, stationary=False, constraints=True)
dt



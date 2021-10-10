import os
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

import Models


def xy_generator(parameters, sample_size, function, **options):
    """

    :param parameters: a dictionary of parameters theta, beta and gamma
    :param sample_size:
    :param function:
    :param options: stationarity, constraints
    :return:
    """

    Models.param_num = {'theta': parameters['theta'],
                        'beta': parameters['beta'],
                        'gamma': parameters['gamma']
                        }

    # error generation
    initial = 100
    sample_size = 1000

    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    _ep = np.random.multivariate_normal(mean, cov, sample_size + initial)
    epson = _ep[1:, :]
    epson_lag = _ep[:(sample_size + initial-1), :]

    c = np.array([[-1, 4 / 3], [0, 0]])

    v = epson+np.dot(epson_lag,c)

    if options.get('stationary'):
        rho = [0.5] * len(parameters['theta'])
    else:
        rho = [1] * len(parameters['theta'])

    # initiate x
    x = np.array([[0] * len(parameters['theta'])])
    # generate x variables
    i = 0
    while i < sample_size + initial-1:
        _new = rho * x[i] + v[i]
        x = np.append(x, [_new], axis=0)
        i += 1
    x = x[initial:]

    # construct zt
    j = 0
    wt = np.random.standard_normal(sample_size + initial)
    z = np.array([[0] * (len(parameters['beta'])-1)])
    while j < sample_size + initial-1:
        _new = 0.8 * z[j] + wt[j]
        z = np.append(z, [_new], axis=0)
        j += 1
    z = z[initial:]

    # construct the single-index
    u = Models.single_index(x)(parameters['theta'])

    # construct y and y_{t-1}
    X_ = np.concatenate((x,z), axis=1)
    # y = Models.sin_func(X_)(parameters['theta']+parameters['beta']+[0.2])
    return X_


parameters = {'theta': [0.6, -0.8],
              'beta': [0.5, 0.9],
              'gamma': 1
              }

Models.param_num = {'theta': 2,
              'beta': 1,
              'gamma': 1
              }
parameters['theta']+parameters['beta']+[0.2]
y = Models.sin_func(X_)([0.6,-0.8,0.5,0.9])
y


X_ = xy_generator(parameters=parameters, sample_size=1000, function='a', stationary=False, constraints=True)


plt.plot(z)
plt.show()

test = Models.single_index(x)(parameters['theta'])
plt.plot(test)
plt.show()
x = np.array([[0] * len(parameters['theta'])])
# generate x variables
i = 0
v1 = np.random.standard_normal(1100)
v2 = np.random.standard_normal(1100)
v = np.concatenate((np.array([v1]).T, np.array([v2]).T), axis=1)

while i < 1100:
    _new = [0.5,0.5] * x[i] + v[i]
    x = np.append(x, [_new], axis=0)
    i += 1
x = x[100 + 1:]

x[:,1]

mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
ep = np.random.multivariate_normal(mean, cov, 1000)

C = np.array([[-1,4/3], [0,0]])


ep = np.random.multivariate_normal(mean, cov, 1101)
epson = ep[1:, :]
epson_lag = ep[:1100, :]
epson+np.dot(epson_lag,C)
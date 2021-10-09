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

    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    v = np.random.multivariate_normal(mean, cov, sample_size+initial)

    if options.get('stationary'):
        rho = [0.5] * len(parameters['theta'])
    else:
        rho = [1] * len(parameters['theta'])

    # initiate x
    x = np.array([[0] * len(parameters['theta'])])
    # generate x variables
    i = 0
    while i < sample_size + initial:
        _new = rho * x[i] + v[i]
        x = np.append(x, [_new], axis=0)
        i += 1
    x = x[initial+1:]

    # construct the single-index
    u = Models.single_index(x)(parameters['theta'])
    return u


parameters = {'theta': [0.8, -0.6],
              'beta': [0.5, 0.9],
              'gamma': [0.1, 0.2, 0.3]
              }

u = xy_generator(parameters=parameters, sample_size=1000, function='a', stationary=False, constraints=True)
plt.plot(u)
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
np.random.multivariate_normal(mean, cov, 1000)

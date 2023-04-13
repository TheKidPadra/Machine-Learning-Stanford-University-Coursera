import numpy as np
from sigmoid import *

def lr_cost_function(theta, X, y, lmd):
    m = y.size
    cost = 0
    grad = np.zeros(theta.shape)
    return cost, grad

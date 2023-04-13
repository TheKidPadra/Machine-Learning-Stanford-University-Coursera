import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf

def validation_curve(X, y, Xval, yval):
    lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)
    return lambda_vec, error_train, error_val

import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf

def learning_curve(X, y, Xval, yval, lmd):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    return error_train, error_val

import numpy as np


def feature_normalize(X):

    n = X.shape[1]  # the number of features
    X_norm = X
    mu = np.zeros(n)
    sigma = np.zeros(n)

    return X_norm, mu,

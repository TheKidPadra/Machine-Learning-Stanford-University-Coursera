import numpy as np


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros(m)
    X = np.c_[np.ones(m), X]

    return

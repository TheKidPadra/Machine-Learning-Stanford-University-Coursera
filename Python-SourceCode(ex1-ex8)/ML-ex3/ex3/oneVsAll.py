import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *

def one_vs_all(X, y, num_labels, lmd):
    (m, n) = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.c_[np.ones(m), X]

    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        print('Done')

    return

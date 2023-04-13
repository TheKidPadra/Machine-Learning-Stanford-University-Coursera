import numpy as np

def select_threshold(yval, pval):
    f1 = 0
    # You have to return these values correctly
    best_eps = 0
    best_f1 = 0
    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001):
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1

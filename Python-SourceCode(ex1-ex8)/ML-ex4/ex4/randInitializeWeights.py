import numpy as np

def rand_initialization(l_in, l_out):
    w = np.zeros((l_out, 1 + l_in))
    return w

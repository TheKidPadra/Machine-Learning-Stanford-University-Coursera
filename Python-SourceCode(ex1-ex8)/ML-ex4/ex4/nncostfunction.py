import numpy as np
from sigmoid import *

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):

    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    # Useful value
    m = y.size
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad

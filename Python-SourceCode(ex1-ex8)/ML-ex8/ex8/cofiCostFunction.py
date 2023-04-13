import numpy as np

def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))
    # You need to set the following values correctly.
    cost = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten()))

    return cost, grad

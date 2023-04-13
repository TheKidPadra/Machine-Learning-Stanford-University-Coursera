import matplotlib.pyplot as plt
import numpy as np
from featureNormalize import *
from gradientDescent import *
from normalEqn import *

plt.ion()

# ===================== Part 1: Feature Normalization =====================
print('Loading Data...')
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Print out some data points
print('First 10 examples from the dataset: ')
for i in range(0, 10):
    print('x = {}, y = {}'.format(X[i], y[i]))

input('Program paused. Press ENTER to continue')

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = feature_normalize(X)
X = np.c_[np.ones(m), X]  # Add a column of ones to X

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.03
num_iters = 400

# Init theta and Run Gradient Descent
theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent : \n{}'.format(theta))


price = 0


print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Normal Equations =====================

print('Solving with normal equations ...')

data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Add intercept term to X
X = np.c_[np.ones(m), X]

theta = normal_eqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations : \n{}'.format(theta))


price = 0


# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price))

input('ex1_multi Finished. Press ENTER to exit')

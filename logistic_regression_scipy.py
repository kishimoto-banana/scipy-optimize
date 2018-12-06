import numpy as np
import scipy as sp
import scipy.optimize

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
y = data[:, 0]
X = data[:, 1:]

intercept = np.ones(X.shape[0]).reshape(X.shape[0], 1)
X = np.concatenate((intercept, X), axis = 1)

n_samples = X.shape[0]
n_dims = X.shape[1]

# 検証用初期値
w = np.array([-.10296645, -.0332327, -.01209484, .44626211, .92554137, .53973828, 1.7993371, .7148045  ])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def negative_log_likelihood(w, X, y):
    z = np.dot(X, w)
    return -np.sum((y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))))

def gradient(w, X, y):
    z = np.dot(X, w)
    return np.dot(X.T, (sigmoid(z) - y))

opt = scipy.optimize.fmin_bfgs(negative_log_likelihood, x0 = np.array([-.1, -.03, -.01, .44, .92, .53, 1.8, .71]), args = (X, y), gtol = 1e-3)
print(opt)

opt = scipy.optimize.fmin_bfgs(negative_log_likelihood, x0 = np.array([-.1, -.03, -.01, .44, .92, .53, 1.8, .71]), fprime = gradient, args = (X, y), gtol = 1e-3)
print(opt)
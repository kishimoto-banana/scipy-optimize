import numpy as np
import scipy.optimize
from sklearn.datasets import make_classification

def sigmoid(x):
    '''シグモイド関数'''
    return 1 / (1 + np.exp(-x))

def negative_log_likelihood(w, X, y):
    '''負の対数尤度'''
    z = np.dot(X, w)
    return -np.sum((y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))))

def objective_function(w, X1, y1, X2, y2):
    '''目的関数'''
    dim1 = X1.shape[1]

    w1 = w[:dim1]
    w2 = w[dim1:]

    return  negative_log_likelihood(w1, X1, y1) + negative_log_likelihood(w2, X2, y2) + (np.linalg.norm(w1) + np.linalg.norm(w2)) / 2

def main():

    # X1, y1の作成
    X1, y1 = make_classification(n_samples=1000, n_features=10, n_informative=8, n_classes=2, random_state=1209)
    intercept = np.ones(X1.shape[0]).reshape(X1.shape[0], 1)
    X1 = np.concatenate((X1, intercept), axis = 1)

    # X2, y2の作成
    X2, y2 = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=1209)
    intercept = np.ones(X2.shape[0]).reshape(X2.shape[0], 1)
    X2 = np.concatenate((X2, intercept), axis = 1)

    # パラメータの初期化
    w1 = np.random.randn(X1.shape[1])
    w2 = np.random.randn(X2.shape[1])
    w = np.concatenate((w1, w2))

    w_opt = scipy.optimize.fmin_bfgs(f=objective_function, x0=w, args=(X1, y1, X2, y2))
    print(f'w1={w_opt[:w1.shape[0]]}\nw2={w_opt[w1.shape[0]:]}')

if __name__ == '__main__':
    main()
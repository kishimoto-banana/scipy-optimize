import numpy as np
import scipy.optimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def sigmoid(x):
    '''シグモイド関数'''
    return 1 / (1 + np.exp(-x))

def negative_log_likelihood(w, X, y):
    '''負の対数尤度'''
    z = np.dot(X, w)
    return -np.sum((y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))))

def objective_function(w, X, y):
    '''目的関数'''
    return negative_log_likelihood(w, X, y) + np.linalg.norm(w)

def gradient(w, X, y):
    '''勾配'''
    z = np.dot(X, w)
    return np.dot(X.T, (sigmoid(z) - y))

def main():

    # データの作成
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_classes=2, random_state=1209)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1209, stratify=y)

    # 切片を含めて行列演算を行うため全サンプルに1つ要素を追加（値は1）
    intercept = np.ones(X_train.shape[0]).reshape(X_train.shape[0], 1)
    X_train = np.concatenate((X_train, intercept), axis = 1)
    intercept = np.ones(X_test.shape[0]).reshape(X_test.shape[0], 1)
    X_test = np.concatenate((X_test, intercept), axis = 1)

    # パラメータの初期値を設定（平均0、標準偏差1の正規乱数）
    np.random.seed(1209)
    w = np.random.randn(X_train.shape[1])

    w_opt = scipy.optimize.fmin_bfgs(f=objective_function, x0=w, args=(X_train, y_train))

    print(f'scipy\n係数:{w_opt[:-1]}\n切片:{w_opt[-1]}')

    # 予測
    y_proba_scipy = sigmoid(np.dot(X_test, w_opt))
    y_pred_scipy = (y_proba_scipy >= 0.5) * 1
    print(confusion_matrix(y_test, y_pred_scipy))

    # scikit-learn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1209, stratify=y)
    model = LogisticRegression(random_state=1209)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'scikit-learn\n係数:{model.coef_}\n切片:{model.intercept_}')
    print(confusion_matrix(y_test, y_pred))



if __name__ == '__main__':
    main()

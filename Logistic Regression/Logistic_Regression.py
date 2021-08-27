from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data
path = "C:\\Users\\nphuc\\Desktop\\AI\\Logistic Regression\\data_logistic_regression.csv"
data = pd.read_csv(path)
data.head()

X = data['DiemThi1'].values
Y = data['DiemThi2'].values
Z = data['Pass'].values

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))
Z = Z.reshape((len(Z), 1))

K = np.concatenate([X, Y], axis=1)
positive = K[Z.reshape(-1) == 1, :]
negative = K[Z.reshape(-1) == 0, :]
plt.scatter(positive[:, 0], positive[:, 1], c='r')
plt.scatter(negative[:, 0], negative[:, 1], c='b')

ones = np.ones((K.shape[0], 1), dtype=np.float)
K = np.concatenate((ones, K), axis=1)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=200000):
    w = [w_init]
    N = X.shape[0]
    d = X.shape[1]

    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :].reshape(1, d)

            yi = y[i]

            zi = sigmoid(np.dot(xi, w[-1].T))

            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w


eta = .005
d = K.shape[1]
w_init = np.random.randn(1, d)

w = logistic_sigmoid_regression(K, Z, w_init, eta)

pass_predict = sigmoid(np.dot([[1, 7, 6]], w[-1].T))
print(pass_predict)

k_predict = []
# for i in range(0, X.shape[0], 1):
#
#     k_predict.append(sigmoid(np.dot([[1, X[i], Y[i]]], w[-1].T)))
weight = w[-1].reshape(-1)

x1 = np.min(K[:, 1])
x2 = np.max(K[:, 2])

x = np.array([x1, x2])
y_vals = -(weight[0] + np.dot(x, weight[1])) / weight[2]
# print(x)


plt.plot(x, y_vals)
plt.show()

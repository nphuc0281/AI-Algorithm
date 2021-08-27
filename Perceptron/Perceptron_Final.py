from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data
data = pd.read_csv('data_perceptron.csv')
data.head()

X = data['A'].values
Y = data['B'].values
Z = data['C'].values

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))

X = np.concatenate([X, Y], axis=1)
Y = Z.reshape((len(Z), 1))
positive = X[Y.reshape(-1) == 1, :]
negative = X[Y.reshape(-1) == -1, :]
plt.scatter(positive[:, 0], positive[:, 1], c='r')
plt.scatter(negative[:, 0], negative[:, 1], c='b')

Y = Y.reshape(-1)

ones = np.ones((X.shape[0], 1), dtype=np.float)

X = np.concatenate((ones, X), axis=1)
w = np.random.randn(X.shape[1], 1)

i = 0

while True:
    y_predict = np.dot(X, w)

    y_predict = y_predict.reshape(-1)

    loss_index = ((y_predict > 0) ^ (Y > 0))

    X_loss = X[loss_index]

    Y_loss = Y[loss_index]

    if X_loss.shape[0] > 0:
        x_check = X_loss[0]
        y_check = Y_loss[0]

        w += x_check.reshape(-1, 1) * y_check

    else:
        break

if __name__ == '__main__':
    a = 1
    k_predict = np.dot([[1, 4, 3]], w).reshape(-1)
    weight = w.reshape(-1)
    print(k_predict)

    x1 = np.min(negative[:, 0])
    x2 = np.max(positive[:, 0])

    x = np.array([x1, x2])



    y_vals = -(weight[0] + np.dot(x, weight[1])) / weight[2]

    print('A = 4')
    print('B = 3')

    plt.plot(x, y_vals)
    plt.show()
    if k_predict > 0:
        print('C = 1')

    else:
        print('C = -1')


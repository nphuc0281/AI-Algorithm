from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data
path = "C:\\Users\\nphuc\\Desktop\\AI\\Logistic Regression\\data_logistic_regression.csv"
data = pd.read_csv(path)

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
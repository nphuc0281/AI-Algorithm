import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\nphuc\\Desktop\\AI\\Linear Regression\\data1.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('Dien tich')
plt.ylabel('gia')
x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 1.]).reshape(-1, 1)
numOfIteration = 200
cost = np.zeros((numOfIteration, 1))

learning_rate = 0.000001
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5 * np.sum(r * r)
    w[0] = w[0] - learning_rate * np.sum(r)
    # correct the shape dimension
    w[1] = w[1] - learning_rate * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
    print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')
plt.show()
print('W0 = %s\t W1 = %s' % (w[0], w[1]))
x1 = 91
y1 = w[0] + w[1] * x1
print('Giá nhà cho 91m^2 là : ', y1)

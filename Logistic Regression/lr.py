import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\nphuc\\Desktop\\AI\\Logistic Regression\\data.csv', header=None)
# print(data.values)

true_x = []
true_y = []
false_x = []
false_y = []

for i in data.values:
    if i[2] == 1.:
        true_x.append(i[0])
        true_y.append(i[1])
    else:
        false_x.append(i[0])
        false_y.append(i[1])
# plt.scatter(true_x, true_y, marker='o', c='r')
# plt.scatter(false_x, false_y, marker='o', c='b')
# plt.show()


def sigmoid(z):
    return 1.0/(1 + np.exp(z))


def phan_chia(p):
    if p >= 0.5:
        return 1
    else:
        return 0


def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)


def cost_function(features, labels, weights):
    """
    :param features: 100 x 3
    :param labels: 100 x 1
    :param weights: 3 x 1
    :return: chi phi cost
    """
    n = len(labels)
    predictons = predict(features, weights)
    cost_class1 = -labels*np.log(predictons)
    cost_class2 = -(1 - labels)*np.log(1 - predictons)
    cost = cost_class2 + cost_class1
    return cost.sum()/n


def update_weight(features, labels, weights, learning_rate):
    """

    :param features: 100 x 3
    :param labels: 100 x 1
    :param weights: 3 x 1
    :param learning_rate: float
    :return: new weight : float
    """
    n = len(labels)
    # gia tri du doan cua all cac diem
    predictions = predict(features, weights)
    gd = np.dot(features.T, (predictions - labels))
    gd = gd/n
    gd = gd*learning_rate
    weights = weights - gd
    return weights


def train(features, labels, weights, learning_rate, iter): #iter so lan lap
    cost_hs = []
    for i in range(iter): # update lai weight va tinh chi phi
        weights = update_weight(features, labels, weights, learning_rate)
        cost = cost_function(weights, labels, weights)
        cost_hs.append(cost)
    return weights, cost_hs


ones =np.ones((data.values.shape[0],1))
Xbar = np.concatenate((ones, data.values[:,:2]), axis = 1)

d = Xbar.shape[1]
w_init = np.random.randn(1, d)

print(w_init)

weights, cost_history = train(Xbar, data.values[:, 2], w_init[0], 0.001, 100)
print(weights, cost_history)
print(cost_history)

iter = [m for m in range(100)]
plt.plot(iter, cost_history)
plt.show()
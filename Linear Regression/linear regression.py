import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'C:\\Users\\nphuc\\Desktop\\AI\\Linear Regression\\data.csv'
dataframe = pd.read_csv(path)

print(dataframe)

x = dataframe.values[:, 2]
y = dataframe.values[:, 4]

def predict(new_radio, weight, bias):
    return weight*new_radio + bias

def cost_function(x, y, weight, bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*x[i] + bias))**2
    return sum_error / n

def update_weight(x, y, weight, bias, learning_rate):
    n = len(x)
    temp_weight = np.sum([-2 * x[i] * (y[i] - predict(x[i], weight, bias)) for i in range(n)])
    temp_bias = np.sum([-2 * (y[i] - predict(x[i], weight, bias)) for i in range(n)])
    
    # for i in range(n):
    #     temp_weight += -2 * x[i] * (y[i] - (weight * x[i] + bias))
    #     temp_bias += -2 * (y[i] - (weight * x[i] + bias))
    
    weight -= (temp_weight / n) * learning_rate
    bias -= (temp_bias / n) * learning_rate

    return weight, bias

def train(x, y, weight, bias, learning_rate, iter):
    cost_history = []
    
    for i in range(iter):
        weight, bias = update_weight(x, y, weight, bias, learning_rate)
        cost = cost_function(x, y, weight, bias)
        cost_history.append(cost)

    return weight, bias, cost_history

weight, bias, cost_history = train(x, y, 0.03, 0.0014, 0.001, 60)

print("\nResult:")
print("Weight: ", weight)
print("Bias: ", bias)

print("\nCost history:")
print(cost_history)

iter = [i for i in range(60)]
plt.plot(iter, cost_history)
plt.show()
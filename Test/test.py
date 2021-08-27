import numpy as numpy
import math
 
def grad(x): 
    return 2*math.exp(-2*x)*(math.exp(4*x)-4)
 
def cost_func(x):
    return (math.exp(x)-math.exp(-x)*2)**2
 
def GradientDescent(theta,num_iterate,x0):
    x = [x0]
    for it in range(num_iterate):
        x_new = x[-1] - theta*grad(x[-1])   
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
        #print(x)
    return (x, it)
 
(x,it) = GradientDescent(0.001, 100, 0.5)
 
for i in range(it):
    print(f"After {i} iterate : value x : {x[i]}")
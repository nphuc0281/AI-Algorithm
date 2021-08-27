import numpy as np 

def grad(x): 
    return 2*np.exp(-2*x)*(np.exp(4*x) - 4)

def cost(x):
    return (np.exp(x) - 2/np.exp(x))**2

def GD(eta, x0):
    x = [x0]
    for it in range(200):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-4:
            break
        x.append(x_new)
    return x, it

x, it = GD(0.001, 0.5)

print('Solution x = %f, cost = %f, obtained after %d iterations'%(x[-1], cost(x[-1]), it))

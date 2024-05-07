# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:40:40 2024

@author: chewei
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#%% Least Square Estimation

np.random.seed(20)
N = 100     # sample size
k = 5       # 
beta0 = np.array([[0.9, 0.5, 0.2, -0.2, -0.7]]).T       # true beta
X = np.random.randn(N, k)
e = np.random.randn(N,1)
y = X@beta0+e

bhat = inv(X.T@X)@X.T@y

#%% Gradient Descent Method

bhat0 = np.zeros([k, 1])
cost = (y - X@bhat0[:, 0:1]).T @ (y - X@bhat0[:, 0:1]) / N      # loss function
eta = 0.05  # learning rate
criterion = 0.0000001   # 停止條件 e.g. (bhat-b)/b
max_iters = 500000     # 重複最多次數

ii = 0
dd = 999

while dd > criterion and ii < max_iters:
    g = -X.T @ (y-X@bhat0[:, ii:ii+1]) / N    # form 
    d = (g>0)*-1 + (g<0)*1
    bhat0 = np.append(bhat0, bhat0[:,ii:ii+1]-eta*g, axis=1)
    #bhat0 = np.append(bhat0, bhat0[:,ii:ii+1]+eta*d, axis=1)
    e2 = (y-X@bhat0[:,ii+1:ii+2]).T@(y-X@bhat0[:,ii+1:ii+2]) / N # new error, 用迭帶後產生的新beta hat
    cost = np.append(cost,e2, axis=1)   
    dd = abs((cost[0,ii+1]-cost[0,ii])/cost[0,ii])
    ii += 1
    
#%% dynamic path for beta
fig1 = plt.figure(figsize=(5,4), dpi=200)
ax = fig1.add_subplot(1,1,1)
ax.plot(bhat0[0,:])
ax.axhline(y=0.9, linewidth=1, color = 'r')
ax.tick_params(labelsize=16)
ax.set_xlabel('iterations',fontsize=16)
ax.set_ylabel(r"$\beta_1$",fontsize=16)
plt.show()

fig2 = plt.figure(figsize=(5,4), dpi=200)
ax = fig2.add_subplot(1,1,1)
ax.plot(bhat0[4,:])
ax.axhline(y=-0.7, linewidth=1, color = 'r')
ax.tick_params(labelsize=16)
ax.set_xlabel('iterations',fontsize=16)
ax.set_ylabel(r"$\beta_5$",fontsize=16)
plt.show()


#%% likelihood
from scipy.optimize import minimize
def nlikelihood(theta):
    """Normal distribution"""
    global X
    global y
    global N
    global k
    beta = theta[:k][:, np.newaxis]
    sigma = np.exp(theta[k])
    return -(-1/(2*N*sigma**2)*(y-X@beta).T@(y-X@beta)-np.log(sigma*(2*np.pi)**(0.5)))

theta0=np.random.randn(k+1)
res = minimize(nlikelihood, theta0, method='nelder-mead',options={'xtol': 1e-5, 'disp': True})
print(res.x)

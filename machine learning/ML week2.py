# -*- coding: utf-8 -*-
"""
2024/3/6
Machine Learning
    Linear function
    Quadratic function
    Least square estimation(polynomial case)
    
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

    
#%% Linear Function: y=1+x   
x = np.linspace(0, 2, 500)[:, np.newaxis]  # 在0~2之中，生成500個等差序列，並將為原(500,)的數組，變為(500,1)的向量
y = 1 + 1*x

fig1 = plt.figure(figsize=(5, 4), dpi=200) 
ax = fig1.add_subplot(1, 1, 1)
ax.plot(x, y)
ax.tick_params(labelsize=16)  # 設定軸刻度的參數
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_title(r"$y=1+x$ ($k=1$)", fontsize=16)
plt.show()

#%% Quadratic function: y=1+x-x^2
x = np.linspace(0, 2, 500)[:, np.newaxis]
y = 1 + 1*x -1*(x**2)

fig2 = plt.figure(figsize=(5, 4), dpi=200)
ax = fig2.add_subplot(1, 1, 1)
ax.plot(x, y)
ax.tick_params(labelsize=16)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_title(r"$y=1+x-x^2$ ($k=2$)", fontsize=16)
plt.show()

#%% Least square estimation(polynomial case): y=1+x-x^2+error term
np.random.seed(20)
N = 10
x = np.linspace(0, 2, N)[:, np.newaxis]
e = 0.5 * np.random.randn(N)[:, np.newaxis]
y = 1 + 1*x -1*(x**2) + e
plt.scatter(x, y)

M = 10  # different degrees (up to 10)
x0 = np.linspace(0, 2, 500)[:, np.newaxis]
yhat0 = np.zeros((500, M))

power = np.linspace(0, 9, M)
X = np.power(x, power)
X0 = np.power(x0, power)


Xk = X[:, :2]   # select all rows of first two column
bhat = inv(Xk.T@Xk)@Xk.T@y
yhat = Xk@bhat.flatten()


#%%
L = np.zeros((M, 1))

for i in range(0, M, 1):
    Xk = X[:, :i+1]     # select first of i+1 column
    bhat = inv(Xk.T@Xk)@Xk.T@y
    print(bhat)
    X0k = X0[:, :i+1]
    
    yhat = Xk@bhat.flatten()
    yhat0[:, i] = X0k@bhat.flatten()
    L[i, 0] = sum((y-yhat[:, np.newaxis]) ** 2) / N
    

    
c = (1, 3, 5, 9)    # m = 1, 3, 5, 9
for cc in c:
    fig2 = plt.figure(figsize=(5, 4), dpi=200)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x0, yhat0[:, cc], 'r')
    ax2.scarrter(x, y)
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r"$x$", fontsize=16)
    ax2.set_ylabel(r"$y$", fontsize=16)
    dd=L[cc, 0]
    ax2.set_title(r"$m=%i$" %cc + "," + r"$L=%1.2f$" %dd, fontsize=16)
    
#%% in class
Loss = np.zeros((M, 1))

for i in range(0, M, 1):
    Xk = X[:, :i+1]     # select first of i+1 column
    bhat = inv(Xk.T@Xk)@Xk.T@y
    X0k = X0[:, :i+1]

    residual = y - Xk@bhat
    Loss[i] = np.mean(residual**2)
    
c = (1, 3, 5, 9)    # m=1,3,5,9
for cc in c:
    fig2 = plt.figure(figsize=(5, 4), dpi=200)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x0, yhat0[:, cc], 'r')
    ax2.scatter(x, y)
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r"$x$", fontsize=16)
    ax2.set_ylabel(r"$y$", fontsize=16)
    dd=Loss[cc, 0]
    ax2.set_title(r"$m=%i$" %cc + "," + r"$L=%1.2f$" %dd, fontsize=16)
    
#%% LOOCV ----
CV1 = np.zeros((N, M))
for i in np.arange(N):
    idx1 = (x != x[i]).flatten()
    idx2 = (x == x[i]).flatten()
    X1 = X[idx1, :]
    y1 = y[idx1, :]
    for j in range(0, M, 1):
        X1k = X1[:, :j+1]
        bhat1 = inv(X1k.T@X1k)@X1k.T@y1
        CV1[i, j] = (y[idx2, :] - X[idx2, :j+1]@bhat1) ** 2
CVm =CV1.mean(axis=0)




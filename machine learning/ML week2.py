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

fig1 = plt.figure(figsize=(5, 4), dpi=200)  # 建立畫布
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

#%% Least square estimation(polynomial case): y=+x-x^2+error term
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


Xk = X[:, :2]
bhat = inv(Xk.T@Xk)@Xk.T@y
yhat = Xk@bhat.flatten()


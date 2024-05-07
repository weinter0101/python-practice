# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:18:43 2024

@author: chewei
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# create data
np.random.seed(20)
N = 100     # sample size
k = 5        
beta = np.array([[0.9, 0.5, 0.2, -0.2, -0.7]]).T       # true beta
X = np.random.randn(N, k)
e = np.random.randn(N,1)
y = X@beta+e

# OLS
bhat1 = inv(X.T@X)@X.T@y

# gradient descent
bhat2 = np.zeros([k, 1])
cost1 = (y-X@bhat2).T @ (y-X@bhat2) / N
eta = 0.05
criterion = 0.0000001   # 停止條件  e.g. (bhat-b)/b
maxIters = 500      # 最多尋找次數
count = 0
delta = 999

bhatHistory = []
deltaHistory = []

while delta > criterion and count < maxIters:
    gradient = -X.T @ (y-X@bhat2) / N    # form
  # d = (g>0)*-1 + (g<0)*1   # 將梯度轉換成 1 或 -1 的方向
    bhat2 = bhat2 - eta*gradient    
  # bhat2 = bhat2 + eta*d    # 以方向去迭代初 bhat2
    bhatHistory.append(bhat2.ravel())   # 畫圖用
    cost2 = (y-X@bhat2).T @ (y-X@bhat2) / N
    delta = abs((cost2-cost1) / cost1)
    deltaHistory.append(delta.ravel())  # 添加 delta 到歷史記錄
    cost1 = cost2
    count += 1
     
# dynamic path for beta
bhatHistory = np.array(bhatHistory)

fig, axes = plt.subplots(2, 3, figsize=(12, 8)) 
fig.suptitle('$Beta\: Values\: Over\: Gradient\: Descent\: Iterations\: and\: OLS$')

for i in range(k):
    ax = axes.flat[i]
    ax.plot(bhatHistory[:, i], label='$gradient\: descent$')
    ax.axhline(y=beta[i, 0], color='r', linestyle='-', label='$True\: Beta$')
    ax.axhline(y=bhat1[i, 0], color='g', linestyle='--', label='$OLS$')
    ax.set_title(r'$\beta_{' + str(i+1) + r'}$')
    ax.set_xlabel('$Iteration$')
    ax.legend()

ax_delta = axes.flat[-1]  # 使用最後一個 subplot 來畫 delta
ax_delta.plot(deltaHistory, label='$Delta$')
ax_delta.set_title('$Delta\: over\: Iterations$')
ax_delta.set_xlabel('Iteration')
ax_delta.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()

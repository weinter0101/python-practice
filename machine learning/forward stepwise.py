# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:51:51 2024

@author: chewei

forward stepwise model selection
"""

import numpy as np
import statsmodels.api as sm
from numpy.linalg import inv


#%% True model

np.random.seed(20)
    
N = 200
X = np.random.randn(N, 3)
X = sm.add_constant(X)
beta = np.array([[1, 0.5, 0.3, 0]])
Y = X@beta.T + np.random.randn(N, 1)

sm.OLS(Y, X).fit().summary()
result  = sm.OLS(Y, X).fit()
 
# aic bic selection in true model
for kk in range(1, 5):          #kk=1, null model
    result = sm.OLS(Y, X[:, :kk]).fit() 
    print('number of regressors=', kk, 'AIC=', result.aic)

for kk in range(1, 5):
    result = sm.OLS(Y, X[:, :kk]).fit()
    print('number of regressors=', kk, 'BIC=', result.bic)
    
#%% OLS model

np.random.seed(20)
N = 500
k = 5
beta0 = np.array([[0.9,0,0.5,0,0.2]]).T
X = np.random.randn(N,k)
e = np.random.randn(N,1)
y = X@beta0+e

bhat = inv(X.T@X)@X.T@y

#%% forward stepwise model selection
    
SST = y.T@y
remainingIndices = [0, 1, 2, 3, 4]
selectedIndices = []
Xk = X[:, selectedIndices]
AIC = np.zeros((k+1,1))
AIC[0,0] = y.T@y/N      # aic = SST/N + 2*0/N
minAIC = np.inf
minAIC_indices = []

for i in range(k):
    Rsquared = np.zeros([k]) - 999      # 設定一個極小的 R squared
    for m in remainingIndices:
        '''把剩餘的解釋變數計算 R squared'''
        X1 = np.concatenate((Xk, X[:, m:m+1]), axis=1)
        bhats = inv(X1.T@X1)@X1.T@y
        SSR = (y-X1@bhats).T @ (y-X1@bhats)
        Rsquared[m] = (1-SSR/SST).item()
    '''求出使 R squared 最大的 X並加入selectedIndices中'''
    selectedIndices.append(np.argmax(Rsquared))         
    remainingIndices.remove(selectedIndices[-1])        
    Xk = X[:, selectedIndices]
    AIC[i+1, 0] = (y - Xk @ inv(Xk.T@Xk) @ Xk.T@y).T@(y - Xk @ inv(Xk.T@Xk) @ Xk.T@y) / N + 2*(i+1)/N
    print(selectedIndices)      # 得知每次選擇的 X 順序為何
    
    '''select the minimum AIC'''
    if AIC[i+1, 0] < minAIC:
        minAIC = AIC[i+1, 0]
        minAIC_indices = selectedIndices.copy()

#%% 
    
    


    
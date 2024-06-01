# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:27:08 2024

@author: chewei
"""
#%% import
import numpy as np
import statsmodels.api as sm
from numpy.linalg import inv

#%%
np.random.seed(20)

N = 200
X = np.random.randn(N, 3)
X = sm.add_constant(X)
beta = np.array([[1, 0.5, 0.3, 0]])
Y = X@beta.T + np.random.randn(N, 1)

sm.OLS(Y, X).fit().summary()

result  = sm.OLS(Y, X).fit()

for kk in range(1, 5):
    result = sm.OLS(Y, X[:, :kk]).fit()
    print('number of regressors=', kk, 'AIC=', result.aic)
    
for kk in range(1, 5):
    result = sm.OLS(Y, X[:, :kk]).fit()
    print('number of regressors=', kk, 'BIC=', result.bic)

#%%
np.random.seed(20)
N = 500
k = 5
beta0 = np.array([[0.9,0,0.5,0,0.2]]).T
X = np.random.randn(N,k)
e = np.random.randn(N,1)
y = X@beta0+e

bhat = inv(X.T@X)@X.T@y



# model selection
# forward stepwise model selection
SST = y.T@y
list_= [0,1,2,3,4]
idk= []
Xk = X[:,idk]
AIC = np.zeros((k+1,1))
AIC[0,0] = y.T@y/N
for ii in range(k):
    R2 = np.zeros([k])-999
    print(idk)
    for m in list_:
        X1 = np.concatenate((Xk,X[:,m:m+1]),axis=1)
        bhats = inv(X1.T@X1)@X1.T@y
        SSR = (y-X1@bhats).T@(y-X1@bhats)
        R2[m]=(1-SSR/SST).item()
    idk.append(np.argmax(R2)) # index for keeping
    list_.remove(idk[-1]) # index for searching
    Xk = X[:,idk] # selected regressor(s)
    AIC[ii+1,0] = (y-Xk@inv(Xk.T@Xk)@Xk.T@y).T@(y-Xk@inv(Xk.T@Xk)@Xk.T@y)/N+2*(ii+1)/N
    
    

    
    
    
# ridge regression
N_grid = 50
beta_ridge = np.zeros((N_grid,k))
lam = np.linspace(0, 5, num=N_grid)
CV1=np.zeros((N,N_grid))

    
# LOOCV
for ii in np.arange(N):
    idx1=(X[:,0]!=X[ii,0]).flatten()
    idx2=(X[:,0]==X[ii,0]).flatten()
    X1=X[idx1,:]
    y1=y[idx1,:]
    for jj in range(N_grid):
        beta_ridge = inv(lam[jj]*np.eye(k)+X1.T@X1)@X1.T@y1
        CV1[ii,jj]=(y[idx2,:]-X[idx2,:]@beta_ridge)**2
CVm=CV1.mean(axis=0)
lam_o = lam[np.argmin(CVm)]
beta_ridge_cv = inv(lam_o*np.eye(k)+X.T@X)@X.T@y

from sklearn import linear_model
ridge_reg = linear_model.Ridge(alpha=lam_o, fit_intercept=False)
ridge_reg.fit(X, y)
print(ridge_reg.coef_)

lasso_reg = linear_model.Lasso(alpha=0.3, fit_intercept=False)
lasso_reg.fit(X, y)
print(lasso_reg.coef_)

lasso2_reg = linear_model.LassoCV(alphas=np.linspace(0, 5, 100),cv=3, fit_intercept=False).fit(X, y)
print(lasso2_reg.coef_)
 



    


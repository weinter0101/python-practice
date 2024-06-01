# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:34:28 2024

@author: chewei

logistic regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score

#%% 
path = r'C:\Users\chewei\Documents\python-practice\machine learning\data'
dataName = 'creditcard.csv'
dataPath = os.path.join(path, dataName)

data = pd.read_csv(dataPath, header=0)

#%%

data0 = data[data['Class'] == 0]
data1 = data[data['Class'] == 1]

data0Train, data0Test = train_test_split(data0, train_size=300)
data1Train, data1Test = train_test_split(data1, train_size=300)
data0Test = data0Test.sample(192)   # 由於data1的樣本較少，當train_size=300時，test_size僅剩下192

xTrain = pd.concat([data0Train.loc[:, 'V1':'V28'], data1Train.loc[:, 'V1':'V28']], axis=0)
yTrain = pd.concat([data0Train['Class'], data1Train['Class']], axis=0)

xTest = pd.concat([data0Test.loc[:, 'V1':'V28'], data1Test.loc[:, 'V1':'V28']], axis=0)
yTest = pd.concat([data0Test['Class'], data1Test['Class']], axis=0)


#%% Logistic Regression

model_LR = LogisticRegression(fit_intercept=False).fit(xTrain, yTrain)
parameters_LR = model_LR.coef_
yhat = np.exp(xTest@parameters_LR.T) / (1 + np.exp(xTest@parameters_LR.T))

yhat[yhat>=0.5] = 1
yhat[yhat<0.5] = 0

print(confusion_matrix(yTest, yhat))
print(precision_score(yTest, yhat))

#%% Logistic Rrgression：Gradient Descent
bhat0 = np.zeros([28, 1]) + np.random.randn(28, 1)/100
eta = 0.001
criterion = 0.0001
maxIters = 500
count = 0
delta = 999

logL = np.zeros([maxIters+1, 1])
logL[0 ,0] = -999

yTrain = pd.concat([data0Train['Class'], data1Train['Class']], axis=0)
yTrain = yTrain.values.reshape(-1, 1)

while delta > criterion and count < maxIters:
    xBeta = xTrain @ bhat0[:, count:count+1]
    lambda_ = np.exp(xBeta) / (1+np.exp(xBeta))
    gradient = xTrain.T @ (yTrain-lambda_) 
    bhat0 = np.append(bhat0, bhat0[:, count:count+1]+eta*gradient, axis=1)
    logL[count+1, 0] = np.sum(yTrain*np.log(lambda_) + (1-yTrain)*np.log(1-lambda_))
    delta = abs((logL[count+1, 0]-logL[count, 0]) / logL[count, 0])
    count += 1
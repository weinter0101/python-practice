# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:12:46 2024

@author: chewei
"""

#%% import

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf #OLS
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl_lm

#%%
path = r'C:\Users\chewei\Documents\python-practice\machine learning\data'
name = 'Advertising.csv'

advertising = pd.read_csv(os.path.join(path, name))
advertising.info()
advertising.head(3)

#%% OLS

est = smf.ols('sales ~ TV', advertising).fit()
est.summary()

est = smf.ols('sales ~ radio', advertising).fit()
est.summary().tables

#%% OLS
dependentVariable = ['TV', 'radio', 'newspaper']

for variables in dependentVariable:
    est = smf.ols(f"sales ~ {variables}", advertising).fit()
    print(est.summary())
    
est = smf.ols('sales ~ TV + radio + newspaper', advertising).fit()
print(est.summary())
print(advertising.corr())

est = smf.ols('sales ~ TV + radio + TV * radio', advertising).fit()
print(est.summary())

#%% figure

plt.figure(figsize=(18, 6))

for i, variable in enumerate(dependentVariable, 1):
    plt.subplot(1, 3, i) 
    sns.regplot(x=variable, y='sales', data=advertising)
    plt.title(f'Sales to {variable}')
    plt.xlabel(variable)
    plt.ylabel('Sales')

plt.tight_layout()
plt.show()

#%%
#advertising: sklearn
regr = skl_lm.LinearRegression()
featuresCols =['TV','radio','newspaper']
X = advertising[featuresCols]
y = advertising['sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=80)

regr.fit(x_train,y_train)
print([regr.intercept_, regr.coef_])

y_predict = regr.predict(x_test)
print(y_predict)

plt.figure()
plt.plot(range(len(y_predict)),y_predict,'b',label='predict')
plt.plot(range(len(y_predict)),y_test,'r',label='test')
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel("value of sales")
plt.show()

plt.scatter(x_test['TV'], y_test,  color='black')
plt.scatter(x_test['TV'], y_predict,  color='red')
plt.xticks(())
plt.yticks(())
plt.show()

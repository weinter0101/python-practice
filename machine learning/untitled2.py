# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:07:00 2024

@author: chewei
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

%matplotlib inline
plt.style.use('seaborn-white')

# example
data1 =  pd.read_excel(r"C:\Users\chewei\Documents\python-practice\machine learning\data\example2023.xls", sheet_name=0, header=0)
data1.index = pd.to_datetime(data1.iloc[:,0])
data1 = data1.drop('date',axis=1)
idx = np.where(data1.iloc[:,0]<17,1,0)
df_yes = data1[idx == 1] # recession
df_no = data1[idx == 0]


fig = plt.figure(figsize=(8,8),dpi=100)
ax1 = plt.subplot(1,1,1)
ax1.scatter(df_yes.m1b, df_yes.m2, s=40, c='orange', marker='+',linewidths=1)
ax1.scatter(df_no.m1b, df_no.m2, s=40, marker='o', linewidths=1,edgecolors='blue', facecolors='white', alpha=.6)
ax1.set_xlabel('m1b',fontsize=16)
ax1.set_ylabel('m2',fontsize=16)
ax1.legend(['<17', '>17'],fontsize=16)


X = data1[['m1b', 'm2']]
y = idx

mu_yes = np.mean(np.array(df_yes)[:, 1:], axis=0)
mu_no = np.mean(np.array(df_no)[:, 1:], axis=0)
pi_yes = len(df_yes) / (len(df_yes) + len(df_no))
pi_no = len(df_no) / (len(df_yes) + len(df_no))

cov2 = np.aray(df_yes)[:, 1:] - mu_yes
z1 = np.array(df_yes)[:, 1:] - mu_yes
z2 = np.array(df_no)[:, 1:] - mu_no
cov = (z1.T@z1 + z2.T@z2) / (len(df_yes) + len(df_no)-2)

from numpy.linalg import inv
delta_yes = mu_yes.T@inv(cov)@np.array([[3, 10]]).T - mu_yes.T@inv(cov)@mu_yes + np.log(pi_yes)
delta_no = mu_no.T@inv(cov)@np.array([[3, 10]]).T - mu_no.T@inv(cov)@mu_no + np.log(pi_no)




lda = LinearDiscriminantAnalysis()
y_pred = lda.fit(X, y).predict(X)

df_ = pd.DataFrame({'True recession status': y,
                    'Predicted recession status': y_pred})
df_.replace(to_replace={0:'No', 1:'recession'}, inplace=True)

df_.groupby(['Predicted recession status','True recession status']).size().unstack('True recession status')

qda = QuadraticDiscriminantAnalysis()
y_pred2 = qda.fit(X, y).predict(X)
df2_ = pd.DataFrame({'True recession status': y,
                    'Predicted recession status': y_pred2})
df2_.replace(to_replace={0:'No', 1:'recession'}, inplace=True)

df2_.groupby(['Predicted recession status','True recession status']).size().unstack('True recession status')

# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:13:30 2024

@author: chewei
ml 5/8 week7
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from matplotlib. colors import ListedColormap
from matplotlib import colors

#%% read data
path = r"C:\Users\chewei\Documents\python-practice\machine learning\data"
dataName = "example2023.xls"
dataPath = os.path.join(path, dataName)

data1 =  pd.read_excel(dataPath, sheet_name=0, header=0)
data1.index = pd.to_datetime(data1.iloc[:,0])
data1 = data1.drop('date',axis=1)
idx = np.where(data1.iloc[:,0]<17,1,0)

X = data1[['m1b', 'm2']].values
y = idx

#%%

def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(1, 2, fig_index)

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
                                          
    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap,
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')

    return splot


# Colormap
cmap = ListedColormap(('pink','lightblue'))

clf = svm.SVC(kernel='linear',probability=True,C=1000)
clf.fit(X, y)
y_pred_svm1 = clf.predict(X)
plt.figure(figsize=(10, 3), dpi=200)
plot_data(clf, X, y, y_pred_svm1, 1)
plt.title('Support Vector Classifier (linear)',fontsize=20)
plt.show()


clf = svm.SVC(kernel='rbf',probability=True,C=1,gamma=5)
clf.fit(X, y)
y_pred_svm1 = clf.predict(X)
plt.figure(figsize=(10, 3), dpi=200)
plot_data(clf, X, y, y_pred_svm1, 1)
plt.title('Support Vector Classifier (radial basis)'r"$C=1\;,\gamma=5$",fontsize=20)
plt.show()


C=np.linspace(0.01,120,200)
gamma = np.linspace(0.01, 5, 20)
cv=[]
for jj in range(200):
    for rr in range(20):
        clf = svm.SVC(kernel='rbf',probability=True,C=C[jj],gamma=[rr])
        scores = cross_val_score(clf, X, y, cv=2)
        cv.append(np.mean(scores))
plt.plot(C,cv)
plt.ylabel('scores',fontsize=20);
plt.xlabel(r'$C$',fontsize=20);
plt.show()

clf = svm.SVC(kernel='rbf',probability=True,C=C[cv==min(cv)][0],gamma=0.5)
clf.fit(X, y)
y_pred_svm1 = clf.predict(X)
plt.figure(figsize=(10, 3), dpi=200)
plot_data(clf, X, y, y_pred_svm1, 1)
plt.title('Support Vector Classifier (radial basis)'r"$C=best\;,    \gamma=0.5$",fontsize=20)
plt.show()


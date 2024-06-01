# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:40:11 2024

@author: chewei
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split #
from sklearn import tree # tree model
import graphviz #
from sklearn.model_selection import GridSearchCV # 
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

#%%

# example
X =  pd.read_excel(r"C:\Users\chewei\Documents\python-practice\machine learning\data\ndc.xls", sheet_name=0, header=0)

X.index = pd.to_datetime(X.iloc[:,0]) #
X = X.drop('date',axis=1)
y =  pd.read_excel(r"C:\Users\chewei\Documents\python-practice\machine learning\data\ndc_light.xls", sheet_name=0, header=0)

y.index = pd.to_datetime(y.iloc[:,0])
y = y.drop('date',axis=1)

data=pd.concat([X,y],axis=1) 
data_train, data_test = train_test_split(data, train_size=100)

X_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]
X_test = data_test.iloc[:,:-1]
y_test = data_test.iloc[:,-1]

#%%

# first tree model
clf0 = tree.DecisionTreeRegressor(min_samples_split=5) # 當樣本數>=5才繼續往下分支
model0 = clf0.fit(X=X_train, y=y_train)
y0_hat = model0.predict(X_test)
mse0 = np.mean((y0_hat-y_test)**2) 

dot_data = tree.export_graphviz(model0, feature_names=X.columns)  
graph = graphviz.Source(dot_data)  
graph 

#%%

# tree models with different depths
parameters = {'max_depth':range(1,10)} 
clf = GridSearchCV(tree.DecisionTreeRegressor(), parameters, n_jobs = 4, cv = 5, scoring='neg_mean_squared_error') #
GCV_model = clf.fit(X=X_train, y=y_train)
print('The best depth is',GCV_model.best_params_['max_depth']) 
plt.plot(GCV_model.cv_results_['mean_test_score']*(-1))

mse = np.zeros((9,2))
for ii in range(1,10):
    clf0 = tree.DecisionTreeRegressor(max_depth=ii)
    model0 = clf0.fit(X=X_train, y=y_train)
    y0_hat = model0.predict(X_test)
    mse[ii-1,0] = np.mean((y0_hat-y_test)**2) # test sample MSE
    y0_hat = model0.predict(X_train)
    mse[ii-1,1] = np.mean((y0_hat-y_train)**2) # train sample MSE
    
    
fig0 , ax = plt.subplots(1,1,figsize=(7,5),dpi=200)
ax.plot(GCV_model.cv_results_['mean_test_score']*(-1),label='CV') # CV
ax.plot(mse[:,0],label='Testing error') # test MSE
ax.plot(mse[:,1],label='Training error') # train MSE
ax.legend(loc='upper left')
plt.savefig(r"C:\Users\chewei\Desktop\fig1.png", dpi=200, bbox_inches='tight')




tree_model = clf.best_estimator_
dot_data = tree.export_graphviz(tree_model, feature_names=X.columns)  
graph = graphviz.Source(dot_data)  
graph 

#%%



# Bagging
clf_bag = BaggingRegressor(tree.DecisionTreeRegressor(max_depth=3), n_estimators=200, random_state=10)
clf_estimator = clf_bag.fit(X=X_train, y=y_train)
y_bag_hat = clf_estimator.predict(X_test)
mse_bag = np.mean((y_bag_hat-y_test)**2)

# different tree in bagging
dot_data = tree.export_graphviz(clf_estimator.estimators_[3], feature_names=X.columns)  
graph = graphviz.Source(dot_data)   
graph


clf_rf = RandomForestRegressor(max_depth=10,random_state=10, max_features='sqrt')
clf_rf.fit(X=X_train, y=y_train)
clf_rf.predict(X_test)
y_rf_hat = clf_rf.predict(X_test)
mse_rf = np.mean((y_rf_hat-y_test)**2)

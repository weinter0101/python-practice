# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:21:37 2024

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

#%% single tree


reg_tree = tree.DecisionTreeRegressor(max_depth=3)
tree_model = reg_tree.fit(X=X_train, y=y_train)
yhat_tree = tree_model.predict(X_test)
mse_tree = np.mean((y_test-yhat_tree)**2)

print('The mse for tree is', mse_tree)



#%%

reg_bag=BaggingRegressor(tree.DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=10)
bag_model = reg_bag.fit(X=X_train, y=y_train)
yhat_bag = bag_model.predict(X_test)
mse_bag = np.mean((y_test-yhat_bag)**2)

print('The mse for bagging is', mse_bag)


#%%

dot_data = tree.export_graphviz(bag_model.estimators_[0], feature_names = X.columns)
graph = graphviz.Source(dot_data)
graph


#%% bagging importance

features = X.columns
importances = np.zeros((10, len(features)))

for ii in range(10):
    importances[ii, :] = bag_model.estimators_[ii].feature_importances_

bag_importances = np.mean(importances, axis=0)


#%%
reg_rf = RandomForestRegressor(max_depth=3, random_state=10, max_features='sqrt')
rf_model = reg_rf.fit(X=X_train, y=y_train)
yhat_rf = rf_model.predict(X_test)
mse_rf = np.mean((y_test-yhat_rf)**2)
print('The mse for rf is', mse_rf)


rf_importances = rf_model.feature_importances_

indices = np.argsort(rf_importances)
plt.figure()
plt.title("Feature importances (random forest)")

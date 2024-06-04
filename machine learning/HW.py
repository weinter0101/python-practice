# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:40:48 2024

@author: chewei
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline


#%% data
data_train = pd.read_csv(r"C:\Users\chewei\Downloads\loan_train.csv")
data_test = pd.read_csv(r"C:\Users\chewei\Desktop\ya.csv")

#%% 資料前處理

def purpose_transformation(data, method):
    '''將 purpose 的型態轉換'''
    if method == 'label':
        label_encoder = LabelEncoder()
        data['purpose'] = label_encoder.fit_transform(data['purpose'])
    elif method == 'dummy' or method == 'one_hot':
        data = pd.get_dummies(data=data, columns=['purpose'], drop_first=True)
    else:
        raise ValueError("Method must be either 'label' or 'one_hot'")
    return data

def Outlier_IQR(data, column):
    '''透過 IQR 處理極端值'''
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3-Q1
    maxOutlier = Q3 + IQR*1.5
    minOutlier = Q1 - IQR*1.5
    data[column] = data[column].clip(lower=minOutlier, upper=maxOutlier) 
    return data

def Column_ln(data, column):
    ''' column 取 log'''
    data[column] = np.log(data[column])
    return data

def Standard_data(data, method):
    ''' 資料標準化 '''
    stdVars = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'min-max scaling':
        scaler = MinMaxScaler()
    else:
        raise ValueError("standard_data method must be either 'standardization' or 'min-max scaling'")
    data[stdVars] = scaler.fit_transform(data[stdVars])
    return data


purpose_transformation(data_train, 'label')
Outlier_IQR(data_train, 'installment')
Outlier_IQR(data_train, 'days.with.cr.line')
Column_ln(data_train, 'int.rate')
Standard_data(data_train, 'standardization')

purpose_transformation(data_test, 'label')
Outlier_IQR(data_test, 'installment')
Outlier_IQR(data_test, 'days.with.cr.line')
Column_ln(data_test, 'int.rate')
Standard_data(data_test, 'standardization')
#%% data split and feature choose


# =============================================================================
# 正確 slpit data
# def split_data(data, size):
#     '''training data or testing data'''
#     data_paid = data[data['not.fully.paid'] == 0]
#     data_unpaid = data[data['not.fully.paid'] == 1]
#     
#     data_paid_train, data_paid_test = train_test_split(data_paid, train_size=size, random_state=1117)
#     data_unpaid_train, data_unpaid_test = train_test_split(data_unpaid, train_size=size, random_state=1117)
#     
#     data_train = pd.concat([data_paid_train, data_unpaid_train], ignore_index=True)
#     data_test = pd.concat([data_paid_test, data_unpaid_test], ignore_index=True)
#     
#     x_train = data_train.drop('not.fully.paid', axis=1)
#     x_test = data_test.drop('not.fully.paid', axis=1)
#     y_train = data_train['not.fully.paid']
#     y_test = data_test['not.fully.paid']
#     return data_train, data_test, x_train, x_test, y_train, y_test    
# 
# =============================================================================
# 平衡 train data 中的 0 1
data_train0 = data_train[data_train['not.fully.paid'] == 0]
data_train1 = data_train[data_train['not.fully.paid'] == 1]
data_train0 = data_train0.sample(1483, random_state=1020)

data_train = pd.concat([data_train0, data_train1], ignore_index=True)




x_train = data_train.drop('not.fully.paid', axis=1)
x_test = data_test.drop('not.fully.paid', axis=1)
y_train = data_train['not.fully.paid']
y_test = data_test['not.fully.paid']

def forward_stepwise_selection(data):
    '''使用 forward stepwise 進行變數篩選'''
    x = data.drop('not.fully.paid', axis=1)
    x = np.array(x)
    y = data['not.fully.paid'].astype(float)
    y = np.array(y)
    y = y[:, np.newaxis]
    n, k = x.shape
    
    SST = y.T@y
    remainingIndices = list(range(k))
    selectedIndices = []
    xk = x[:, selectedIndices]
    AIC = np.zeros((k+1, 1))
    AIC[0, 0] = SST /n
 
    minAIC = np.inf
    minAIC_indices = []

    for i in range(k):
        Rsquared = np.zeros([k]) - 999
        for m in remainingIndices:
            x1 = np.concatenate((xk, x[:, m:m+1]), axis=1)
            bhats = inv(x1.T@x1)@x1.T@y
            SSR = (y-x1@bhats).T @ (y-x1@bhats)
            Rsquared[m] = (1-SSR/SST).item()
        '''求出使 R squared 最大的 X並加入selectedIndices中'''
        selectedIndices.append(np.argmax(Rsquared))         
        remainingIndices.remove(selectedIndices[-1])        
        xk = x[:, selectedIndices]
        AIC[i+1, 0] = (y - xk @ inv(xk.T@xk) @ xk.T@y).T@(y - xk @ inv(xk.T@xk) @ xk.T@y) / n + 2*(i+1)/n
        
        '''select the minimum AIC'''
        if AIC[i+1, 0] < minAIC:
            minAIC = AIC[i+1, 0]
            minAIC_indices = selectedIndices.copy()
            
        selectedIndices_name = data.columns[selectedIndices].tolist()
        minAIC_indicesName = data.columns[minAIC_indices].tolist()
    return selectedIndices_name, minAIC_indicesName


def lasso_selection(x_train, y_train, alpha):
    '''使用 lasso 進行變數篩選'''
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    coef = lasso.coef_
    selected_features = x_train.columns[coef != 0].tolist()
    return selected_features

def ridge_selection(x_train, y_train, alpha):
    '''使用 ridge 進行變數篩選'''
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train, y_train)
    coef = ridge.coef_
    selected_features = x_train.columns[abs(coef) > 1e-5].tolist()
    return selected_features

forward_columnName, selectedFeatures_forward = forward_stepwise_selection(data_train)
selectedFeatures_lasso = lasso_selection(x_train, y_train, alpha=0.01)
selectedFeatures_ridge = ridge_selection(x_train, y_train, alpha=0.01)

#%% 確定data

def data_features(method):
    if method == 'forward':        
        xTrain = x_train[selectedFeatures_forward]
        xTest = x_test[selectedFeatures_forward]
    elif method == 'lasso':
        xTrain = x_train[selectedFeatures_lasso]
        xTest = x_test[selectedFeatures_lasso]
    elif method == 'ridge':
        xTrain = x_train[selectedFeatures_ridge]
        xTest = x_test[selectedFeatures_ridge]
    return xTrain, xTest

x_train, x_test = data_features(method = 'lasso')

#%% 分類
#%% 模型定義

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,)),
    'Voting Classifier': VotingClassifier(
        estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svm', SVC())],
        voting='hard'
    ),
    'Bagging Classifier': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100
    ),
    'AdaBoost Classifier': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100
    ),
    'XGBoost Classifier': XGBClassifier(
        n_estimators=100
    ),
    'LightGBM': LGBMClassifier(n_estimators=100),
}

param_grid = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'Voting Classifier': {},
    'Bagging Classifier': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0]
    },
    'AdaBoost Classifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'XGBoost Classifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7]
    }
}


random_state = 1117

# =============================================================================
#     'CatBoost': {
#         'iterations': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 1],
#         'depth': [3, 5, 7]
#     }
# =============================================================================


#%%


def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, seed):
    results = []

    for model_name, model in models.items():
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 4)
        precision = round(precision_score(y_test, y_pred) * 100, 4)
        recall = round(recall_score(y_test, y_pred) * 100, 4)
        f1 = round(f1_score(y_test, y_pred) * 100, 4)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        results.append([model_name, accuracy, precision, recall, f1, mean_cv_score])

    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mean CV Score'])
    return df_results

df_model_results = train_and_evaluate_models(x_train, x_test, y_train, y_test, models, random_state)


#%%
# =============================================================================
# def tune_hyperparameters(X_train, X_test, y_train, y_test, models, param_grid, seed):
#     results = []
# 
#     for model_name, model in models.items():
#         if hasattr(model, 'random_state'):
#             model.set_params(random_state=seed)
#         grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy')
#         grid_search.fit(X_train, y_train)
# 
#         best_params = grid_search.best_params_
#         best_score = grid_search.best_score_
# 
#         best_model = model.set_params(**grid_search.best_params_)
#         best_model.fit(X_train, y_train)
#         y_pred = best_model.predict(X_test)
#         test_accuracy = accuracy_score(y_test, y_pred)
# 
#         results.append([model_name, best_params, best_score, test_accuracy])
# 
#     df_results = pd.DataFrame(results, columns=['Model', 'Best Parameters', 'Best Score', 'Test Accuracy'])
#     return df_results
# 
# df_tuning_results = tune_hyperparameters(x_train, x_test, y_train, y_test, models, param_grid, random_state)
#  
# =============================================================================

#%%

def tune_hyperparameters(X_train, X_test, y_train, y_test, models, param_grid, seed):
    results = []

    for model_name, model in models.items():
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        if model_name == 'Voting Classifier':
            best_model = VotingClassifier(estimators=model.estimators, **grid_search.best_params_)
        else:
            best_model = model.__class__(**grid_search.best_params_)
            if hasattr(best_model, 'random_state'): 
                best_model.set_params(random_state=seed)

                
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        results.append([model_name, best_params, best_score, test_accuracy])

    df_results = pd.DataFrame(results, columns=['Model', 'Best Parameters', 'Best Score', 'Test Accuracy'])
    return df_results

df_tuning_results = tune_hyperparameters(x_train, x_test, y_train, y_test, models, param_grid, random_state)

#%%
def evaluate_model_cv(model, X, y, seed, cv=5, stratified=False):
    if stratified:
        cv = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return cv_scores.mean()

cv_results = []
for model_name, model in models.items():
    cv_score = evaluate_model_cv(model, x_train, y_train, random_state)
    stratified_cv_score = evaluate_model_cv(model, x_train, y_train, random_state, stratified=True)
    cv_results.append([model_name, cv_score, stratified_cv_score])

df_cv_results = pd.DataFrame(cv_results, columns=['Model', 'CV Score', 'Stratified CV Score'])

#%%
def hierarchical_clustering_prediction(X_train, X_test, y_train, y_test, n_clusters, random_state=42):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    hc.fit(X_train)
    
    y_train_pred = hc.labels_
    y_test_pred = hc.fit_predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return train_accuracy, test_accuracy

train_acc, test_acc = hierarchical_clustering_prediction(x_train, x_test, y_train, y_test, n_clusters=2, random_state=random_state)
#%%



# 調整參數網格，擴大搜索範圍
param_gridd = {
    'Logistic Regression': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2']
    },
    'Decision Tree': {
        'clf__max_depth': [None, 5, 10, 20],
        'clf__min_samples_split': [2, 5, 10, 20]
    },
    'Random Forest': {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf', 'poly']
    },
    'Gradient Boosting': {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__learning_rate': [0.001, 0.01, 0.1, 1],
        'clf__max_depth': [3, 5, 7, 9]
    },
    'Neural Network': {
        'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'clf__alpha': [0.0001, 0.001, 0.01, 0.1]
    },
    'Voting Classifier': {},
    'Bagging Classifier': {
        'clf__n_estimators': [10, 50, 100, 200],
        'clf__max_samples': [0.5, 0.7, 1.0]
    },
    'AdaBoost Classifier': {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__learning_rate': [0.001, 0.01, 0.1, 1]
    },
    'XGBoost Classifier': {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__learning_rate': [0.001, 0.01, 0.1, 1],
        'clf__max_depth': [3, 5, 7, 9]
    },
    'LightGBM': {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__learning_rate': [0.001, 0.01, 0.1, 1],
        'clf__max_depth': [3, 5, 7, 9]
    },

}


def tune_hyperparameters(X_train, X_test, y_train, y_test, models, param_grid, seed):
    results = []

    for model_name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)  
        ])
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        grid_search = GridSearchCV(pipe, param_grid[model_name], cv=10, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        results.append([model_name, best_params, best_score, test_accuracy])

    df_results = pd.DataFrame(results, columns=['Model', 'Best Parameters', 'Best Score', 'Test Accuracy'])
    return df_results

df_tuninggg_results = tune_hyperparameters(x_train, x_test, y_train, y_test, models, param_gridd, random_state)

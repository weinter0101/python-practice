# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:40:48 2024

@author: chewei
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
import statsmodels.api as sm

from sklearn.linear_model import Lasso, Ridge




#%% data
data = pd.read_csv(r"C:\Users\chewei\Downloads\loan_train.csv")

#%%

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




# =============================================================================
# # 變數 inq.last.6mths \ delinq.2yrs \ pub.rec 是否需要轉換
# # Define the custom binning function
# def custom_binning_delinq(value):
#     if value == 0:
#         return 'bin1'
#     elif value == 1:
#         return 'bin2'
#     else:
#         return 'bin3'

# # Apply the custom binning function to 'delinq.2yrs' feature
# df['delinq_bins'] = df['delinq.2yrs'].apply(custom_binning_delinq)

# # Calculate distribution of new discrete labels with counts and percentages
# delinq_bins_distribution = df['delinq_bins'].value_counts(normalize=True)

# # Define the custom binning function
# def custom_binning_pub_rec(value):
#     if value == 0:
#         return 'bin1'
#     else:
#         return 'bin2'

# # Apply the custom binning function to 'pub.rec' feature
# df['pub_rec_bins'] = df['pub.rec'].apply(custom_binning_pub_re
# =============================================================================


def split_data(data, size):
    '''training data or testing data'''
    data_paid = data[data['not.fully.paid'] == 0]
    data_unpaid = data[data['not.fully.paid'] == 1]
    
    data_paid_train, data_paid_test = train_test_split(data_paid, train_size=size, random_state=1117)
    data_unpaid_train, data_unpaid_test = train_test_split(data_unpaid, train_size=size, random_state=1117)
    
    data_train = pd.concat([data_paid_train, data_unpaid_train], ignore_index=True)
    data_test = pd.concat([data_paid_test, data_unpaid_test], ignore_index=True)
    
    x_train = data_train.drop('not.fully.paid', axis=1)
    x_test = data_test.drop('not.fully.paid', axis=1)
    y_train = data_train['not.fully.paid']
    y_test = data_test['not.fully.paid']
    return data_train, data_test, x_train, x_test, y_train, y_test    



# =============================================================================
# choose feature
# forward stepwise
# lasso
# ridge
# =============================================================================

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
#%%

purpose_transformation(data, 'label')
Outlier_IQR(data, 'installment')
Outlier_IQR(data, 'days.with.cr.line')
Column_ln(data, 'int.rate')
Standard_data(data, 'standardization')
data_train, data_test, x_train, x_test, y_train, y_test = split_data(data, 0.7)
forward_columnName, selectedFeatures_forward = forward_stepwise_selection(data_train)
selectedFeatures_lasso = lasso_selection(x_train, y_train, alpha=0.01)
selectedFeatures_ridge = ridge_selection(x_train, y_train, alpha=0.01)
#%% data_features

def data_features(method):
    if method == 'forward stepwise':        
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

#%%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
    }

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}:")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 4)
        precision = round(precision_score(y_test, y_pred) * 100, 4)
        recall = round(recall_score(y_test, y_pred) * 100, 4)
        f1 = round(f1_score(y_test, y_pred) * 100, 4)
        
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation score:", cv_scores.mean())
        
        print()

# 假設您已經將數據拆分為 X_train, X_test, y_train, y_test
train_and_evaluate_models(x_train, x_test, y_train, y_test)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# 定義參數網格
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
    }
}

# 對每個模型進行網格搜索或隨機搜索
for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}:")
    
    # 使用網格搜索
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    
    # 使用隨機搜索
    # random_search = RandomizedSearchCV(model, param_grid[model_name], n_iter=10, cv=5, scoring='accuracy', random_state=42)
    # random_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    # 使用最佳參數重新訓練模型
    best_model = model.set_params(**grid_search.best_params_)
    best_model.fit(x_train, y_train)
    
    # 在測試集上評估最佳模型
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)
    print()
    
#%%
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# 创建集成模型
voting_classifier = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svm', SVC())],
    voting='hard'
)

bagging_classifier = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

try:
    # 适用于较新版本的scikit-learn
    adaboost_classifier = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    )
except TypeError:
    # 适用于较旧版本的scikit-learn
    adaboost_classifier = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    )

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

xgboost_classifier = XGBClassifier(
    n_estimators=100,
    random_state=42
)

ensemble_models = {
    'Voting Classifier': voting_classifier,
    'Bagging Classifier': bagging_classifier,
    'AdaBoost Classifier': adaboost_classifier,
    'Random Forest Classifier': rf_classifier,
    'XGBoost Classifier': xgboost_classifier
}

# 训练和评估集成模型
for model_name, model in ensemble_models.items():
    print(f"Training and evaluating {model_name}:")
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print()
    
#%%

from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 使用k-fold交叉驗證評估模型性能
def evaluate_model_cv(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")

# 使用分層k-fold交叉驗證評估模型性能
def evaluate_model_stratified_cv(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Stratified cross-validation scores: {cv_scores}")
    print(f"Mean stratified cross-validation score: {cv_scores.mean()}")

# 嘗試不同的分類算法
def try_different_algorithms(X_train, X_test, y_train, y_test):
    algorithms = {
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
    }

    for algo_name, algo in algorithms.items():
        print(f"Training and evaluating {algo_name}:")
        
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy}")
        
        evaluate_model_cv(algo, X_train, y_train)
        evaluate_model_stratified_cv(algo, X_train, y_train)
        
        print()

# 評估原有模型的交叉驗證性能
for model_name, model in models.items():
    print(f"Evaluating {model_name} with cross-validation:")
    evaluate_model_cv(model, x_train, y_train)
    evaluate_model_stratified_cv(model, x_train, y_train)
    print()

# 嘗試不同的分類算法
try_different_algorithms(x_train, x_test, y_train, y_test)

#%%
from sklearn.cluster import AgglomerativeClustering


def hierarchical_clustering_prediction(X_train, X_test, y_train, y_test, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    hc.fit(X_train)
    
    y_train_pred = hc.labels_
    y_test_pred = hc.fit_predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"訓練集準確度: {train_accuracy}")
    print(f"測試集準確度: {test_accuracy}")
    
hierarchical_clustering_prediction(x_train, x_test, y_train, y_test, n_clusters=2)

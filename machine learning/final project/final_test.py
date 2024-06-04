
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:17:18 2024

@author: chewei

final test code

"""

import pandas as pd
import numpy as np


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from itertools import product


data = pd.read_csv(r"C:\Users\chewei\Documents\python-practice\machine learning\final project\loan.csv")
random_state = 1117



def model_parameters(model):
    purpose = None
    log = None
    std = None
    training = None
    select = None
    
    if model == 'Decision Tree':
        purpose = 'dummy'
        log = IQRVars
        std = 'min-max scaling'
        training = 'undersampling'
        select = 'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business'
    elif model == 'Gradient Boosting':
        purpose = 'label'
        log = IQRVars
        std = 'min-max scaling'
        training = 'undersampling'
        select = 'credit.policy', 'purpose', 'int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.util', 'inq.last.6mths', 'pub.rec'
    elif model == 'Bagging Classifier':
        purpose = 'label'
        log = IQRVars
        std = 'standardization'
        training = 'undersampling'
        select = 'purpose', 'inq.last.6mths', 'credit.policy', 'fico', 'int.rate'
    elif model == 'LightGBM_1':
        purpose = 'dummy'
        log = IQRVars
        std = 'min-max scaling'
        training = 'undersampling'
        select = 'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business'
    elif model == 'LightGBM_2':
        purpose = 'dummy'
        log = logVars
        std = 'min-max scaling'
        training = 'undersampling'
        select = 'credit.policy', 'int.rate', 'fico', 'inq.last.6mths' , 'purpose_credit_card', 'purpose_small_business'
    
    return purpose, log, std, training, select

def Purpose_transformation(data, method):
    '''將 purpose 的型態轉換'''
    if method == 'label':
        label_encoder = LabelEncoder()
        data['purpose'] = label_encoder.fit_transform(data['purpose'])
    elif method in ['dummy', 'one_hot']:
        data = pd.get_dummies(data=data, columns=['purpose'], drop_first=True)
        data[dummyVars] = data[dummyVars].astype(int)
        data = data[Vars]
    else:    
        raise ValueError("Method must be either 'label' or 'one_hot'")
    return data

def Outlier_IQR(data, columns):
    '''透過 IQR 處理極端值'''
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        maxOutlier = Q3 + IQR * 1.5
        minOutlier = Q1 - IQR * 1.5
        data[column] = data[column].clip(lower=minOutlier, upper=maxOutlier)
    return data

def Netural_log(data, columns):
    '''取自然對數轉換'''
    for column in columns:
        data[column] = np.log1p(data[column])
    return data

def Standard_data(data, columns, method):
    '''資料標準化'''
    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'min-max scaling':
        scaler = MinMaxScaler()
    else:
        raise ValueError("standard_data method must be either 'standardization' or 'min-max scaling'")
    
    data[columns] = scaler.fit_transform(data[columns])
    return data


def split_data(data, size, seed, method):
    '''split data, training data的方法'''
    data_paid = data[data['not.fully.paid'] == 0]
    data_unpaid = data[data['not.fully.paid'] == 1]
    
    data_paid_train, data_paid_test = train_test_split(data_paid, train_size=size, random_state=seed)
    data_unpaid_train, data_unpaid_test = train_test_split(data_unpaid, train_size=size, random_state=seed)
    
    if method == 'undersampling':
        data_paid = data[data['not.fully.paid'] == 0]
        data_unpaid = data[data['not.fully.paid'] == 1]
        
        data_paid_train, data_paid_test = train_test_split(data_paid, train_size=size, random_state=seed)
        data_unpaid_train, data_unpaid_test = train_test_split(data_unpaid, train_size=size, random_state=seed)
    
        data_paid_train = data_paid_train.sample(1483, random_state=seed)
    
        data_train = pd.concat([data_paid_train, data_unpaid_train], ignore_index=True)
        data_test = pd.concat([data_paid_test, data_unpaid_test], ignore_index=True)
        
        x_train = data_train.drop('not.fully.paid', axis=1)
        x_test = data_test.drop('not.fully.paid', axis=1)
        y_train = data_train['not.fully.paid']
        y_test = data_test['not.fully.paid']
    else:
        x_train = pd.concat([data_paid_train, data_unpaid_train]).drop('not.fully.paid', axis=1)
        y_train = pd.concat([data_paid_train, data_unpaid_train])['not.fully.paid']
        
        if len(y_train.unique()) > 1:
            if method == 'SMOTE':
                sampler = SMOTE(random_state=seed)
            elif method == 'ADASYN':
                sampler = ADASYN(random_state=seed)
            elif method == 'BorderlineSMOTE':
                sampler = BorderlineSMOTE(random_state=seed)
            elif method == 'SVMSMOTE':
                sampler = SVMSMOTE(random_state=seed)
            else:
                raise ValueError(f"Unknown oversampling method: {method}")
            
            x_train, y_train = sampler.fit_resample(x_train, y_train)
        
        data_train = pd.concat([x_train, y_train], axis=1)
        data_test = pd.concat([data_paid_test, data_unpaid_test], ignore_index=True)
        
        x_test = data_test.drop('not.fully.paid', axis=1)
        y_test = data_test['not.fully.paid']
    
    return data_train, data_test, x_train, x_test, y_train, y_test

def data_features(x_train, x_test, selectedFeatures):   
    selectedFeatures = list(selectedFeatures)
    xTrain = x_train[selectedFeatures]
    xTest = x_test[selectedFeatures]
    
    return xTrain, xTest

def model_accuracy_maximum(X_train, X_test, y_train, y_test, models, param_grid, seed):
    results = []
    best_model_overall = None
    best_accuracy_overall = 0
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        if model_name in param_grid:
            param_combinations = [dict(zip(param_grid[model_name].keys(), v)) for v in product(*param_grid[model_name].values())]
            best_model = None
            best_accuracy = 0
            best_params = None
            
            for params in param_combinations:
                model.set_params(**params)
                if hasattr(model, 'random_state'):
                    model.set_params(random_state=seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_model = model
                    best_accuracy = accuracy
                    best_params = params
            
            if best_accuracy > best_accuracy_overall:
                best_model_overall = best_model
                best_accuracy_overall = best_accuracy
        else:
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy_overall:
                best_model_overall = model
                best_accuracy_overall = accuracy
                best_params = {}
                
                
        y_pred = best_model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append([model_name, best_accuracy, precision, recall, f1, best_params])
    
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Best Parameters'])
    
    return df_results

def best_5_model(model_name, X_train, X_test, y_train, y_test, seed):
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=None, min_samples_split=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=7),
        'Bagging Classifier': BaggingClassifier(n_estimators=50, max_samples=0.7),
        'LightGBM_1': LGBMClassifier(n_estimators=200, learning_rate=1, max_depth=7),
        'LightGBM_2': LGBMClassifier(n_estimators=50, learning_rate=1, max_depth=3),
    }
    
    if model_name not in models:
        print(f"無效的模型名稱：{model_name}")
        return None
    
    model = models[model_name]
    
    print(f"訓練 {model_name}...")
    
    if hasattr(model, 'random_state'):
        model.set_params(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })
    
    return results

    
dummyVars = ['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business']
Vars = ['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec'] + dummyVars + ['not.fully.paid']
IQRVars = ['installment', 'days.with.cr.line', 'revol.bal']
logVars = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
stdVars = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']



modelName = 'LightGBM_2' 
purpose, log, std, training, select =  model_parameters(modelName)

data = Purpose_transformation(data, purpose)
data = Outlier_IQR(data, IQRVars)
data = Netural_log(data, log)
data = Standard_data(data, stdVars, std)
data_train, data_test, x_train, x_test, y_train, y_test = split_data(data, 0.7, random_state, training)

x_train, x_test = data_features(x_train, x_test, select)


# =============================================================================
# test = pd.read_csv(r"")
# test = Purpose_transformation(test, purpose)
# test = Outlier_IQR(test, IQRVars)
# test = Netural_log(test, log)
# test = Standard_data(test, stdVars, std)
# 
# x_test = test.drop('not.fully.paid', axis=1)
# y_test = test['not.fully.paid']
# x_train, x_test = data_features(x_train, x_test, select)
# 
# =============================================================================

modelResults = best_5_model(modelName, x_train, x_test, y_train, y_test, random_state)


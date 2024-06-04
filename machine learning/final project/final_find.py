# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:28:16 2024

@author: chewei
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:17:18 2024

@author: chewei

final 

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

def Purpose_transformation(data, method):
    '''將 purpose 的型態轉換'''
    if method == 'label':
        label_encoder = LabelEncoder()
        data['purpose'] = label_encoder.fit_transform(data['purpose'])
    elif method in ['dummy', 'one_hot']:
        data = pd.get_dummies(data=data, columns=['purpose'], drop_first=True)
        data[['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 
              'purpose_small_business']] = data[['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business']].astype(int)
        order = ['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 
                 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business', 'not.fully.paid']
        data = data[order]
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
            
        minAIC_indicesName = data.columns[minAIC_indices].tolist()
    return minAIC_indicesName

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

def SelectKBest_selection(x_train, y_train, k):
    '''使用 kbest 進行變數篩選'''
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_features = selector.fit_transform(data.drop('not.fully.paid', axis=1), data['not.fully.paid'])
    selected_feature_names = data.drop('not.fully.paid', axis=1).columns[selector.get_support()].tolist()
    return selected_feature_names

def data_features(x_train, x_test, method):
    ''' 使用不同 feature selection 的結果更新 data'''
    if method == 'forward':        
        xTrain = x_train[selectedFeatures_forward]
        xTest = x_test[selectedFeatures_forward]
    elif method == 'lasso':
        xTrain = x_train[selectedFeatures_lasso]
        xTest = x_test[selectedFeatures_lasso]
    elif method == 'ridge':
        xTrain = x_train[selectedFeatures_ridge]
        xTest = x_test[selectedFeatures_ridge]
    elif method == 'kbest':
        xTrain = x_train[selectedFeautres_kbest]
        xTest = x_test[selectedFeautres_kbest]
    else:
        raise ValueError("method must be 'forward', 'lasso', 'ridge' or 'kbest'")
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

def run_selected_model(model_name, X_train, X_test, y_train, y_test, seed):
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

    'XGBoost Classifier': XGBClassifier(
        n_estimators=100
    ),
    'LightGBM': LGBMClassifier(n_estimators=100),
}

param_grid = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [3, 5, 7, 9]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'alpha': [0.0001, 0.001, 0.01, 0.1]
    },
    'Voting Classifier': {},
    'Bagging Classifier': {
        'n_estimators': [10, 50, 100, 200],
        'max_samples': [0.5, 0.7, 1.0]
    },
    'XGBoost Classifier': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [3, 5, 7, 9]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [3, 5, 7, 9]
    }
}

IQRVars = ['installment', 'days.with.cr.line', 'revol.bal']
logVars = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
stdVars = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']


#%%

data = Purpose_transformation(data, 'label')
data = Outlier_IQR(data, IQRVars)
data = Netural_log(data, logVars)
data = Standard_data(data, stdVars, 'standardization')
data_train, data_test, x_train, x_test, y_train, y_test = split_data(data, 0.7, random_state, 'undersampling')

selectedFeatures_forward = forward_stepwise_selection(data_train)
selectedFeatures_lasso = lasso_selection(x_train, y_train, alpha=0.01)
selectedFeatures_ridge = ridge_selection(x_train, y_train, alpha=0.01)
selectedFeautres_kbest = SelectKBest_selection(x_train, y_train, k=10)
x_train, x_test = data_features(x_train, x_test, 'forward')


test = pd.read_csv(r"C:\Users\chewei\Desktop\ya.csv")
test = Purpose_transformation(test, 'label')
test = Outlier_IQR(test, IQRVars)
test = Netural_log(test, logVars)
test = Standard_data(test, stdVars, 'standardization')

x_test = test.drop('not.fully.paid', axis=1)
y_test = test['not.fully.paid']
x_train, x_test = data_features(x_train, x_test, 'forward')


df_model_results = model_accuracy_maximum(x_train, x_test, y_train, y_test, models, param_grid, random_state)



# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:28:06 2024

@author: chewei
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline
#%% modelx

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

print(df_model_results)


#%%

def train_and_evaluate_models_with_gridsearch(X_train, X_test, y_train, y_test, models, param_grid, seed):
    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name in param_grid:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}

        y_pred = best_model.predict(X_test)
        
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 4)
        precision = round(precision_score(y_test, y_pred) * 100, 4)
        recall = round(recall_score(y_test, y_pred) * 100, 4)
        f1 = round(f1_score(y_test, y_pred) * 100, 4)
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        results.append([model_name, accuracy, precision, recall, f1, mean_cv_score, best_params])

    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mean CV Score', 'Best Parameters'])
    return df_results

# 使用示例
df_model_results = train_and_evaluate_models_with_gridsearch(x_train, x_test, y_train, y_test, models, param_grid, random_state)
print(df_model_results)

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:00:27 2024

@author: chewei
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import math
import tensorflow as tf
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical
#%% BCT 1m 資料
data_path = r"C:\Users\chewei\Documents\TMBA\ATD\BTCUSDT_1m.csv"
df = pd.read_csv(data_path)

df.head(3)
#%% 前處理
df.index = pd.to_datetime(df['open_time'])
df.sort_index(inplace=True)
df = df[['open', 'high', 'low', 'close', 'quote_volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

rule = '15T'
d1 = df.resample(rule=rule, closed='left', label='left').first()[['open']]
d2 = df.resample(rule=rule, closed='left', label='left').max()[['high']]
d3 = df.resample(rule=rule, closed='left', label='left').min()[['low']]
d4 = df.resample(rule=rule, closed='left', label='left').last()[['close']]
d5 = df.resample(rule=rule, closed='left', label='left').sum()[['volume']]

df = pd.concat([d1, d2, d3, d4, d5], axis=1)
df = df.dropna()
df.head(5)
#%% 固定 random
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seeds(1020) 
#%% triple barrier label
def Triple_Barrier_Label(price, ub, lb, max_period):

    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]
    
    r = np.array(range(max_period))
    
    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period-1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period+1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period+1)
    t = pd.Series([t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT') 
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(1, p.index)
    signal.loc[p > ub] = 2
    signal.loc[p < lb] = 0

    return signal


df['label'] = Triple_Barrier_Label(df['close'], 1.12, 0.95, 48)
#%% 技術指標

data = df.copy(deep=True)

data['MA20'] = data['close'].rolling(window=20, center=False).mean()
data['MA60'] = data['close'].rolling(window=60, center=False).mean()
data['MA120'] = data['close'].rolling(window=120, center=False).mean()

# Bollinger Bands
data['BB_std'] = data['close'].rolling(window=20, center=False).std()
data['upLine'] = data['MA20'] + data['BB_std']*2
data['downLine'] = data['MA20'] - data['BB_std']*2

# EMA
data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()

# MACD
data['MACD'] = data['EMA12'] - data['EMA26']
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_hist'] = data['MACD'] - data['MACD_signal']

# RSI
delta = data['close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss
data['RSI'] = 100 - (100 / (1 + rs))

data['ROC'] = data['close'].pct_change(periods=12) * 100

#%% train and test
data = data.loc['2020-01-01 00:00:00': '2024-06-30 23:59:00']

trainData = data[(data.index >= '2020-01-01 00:00:00') & (data.index <= '2023-06-30 23:59:00')]
testData = data[(data.index >= '2023-07-01 00:00:00') & (data.index <= '2024-06-30 23:59:00')]

features = ['BB_std', 'MACD_hist', 'close', 'ROC', 'volume', 'MACD_signal', 'RSI']

train_time_index = trainData.index
test_time_index = testData.index

x_train = trainData[features]
y_train = trainData['label']    

x_test = testData[features]
y_test = testData['label']

# one-hot
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

#%% lasso 
'''先使用處李過共線性之特徵進行 lasso features selection'''
# =============================================================================
# from sklearn.linear_model import LassoCV
# from sklearn.preprocessing import StandardScaler
# def lasso_feature_selection(X, y, cv=5, max_iter=1000):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     lasso_cv = LassoCV(cv=cv, random_state=42, max_iter=max_iter)
#     lasso_cv.fit(X_scaled, y)
#     feature_importance = np.abs(lasso_cv.coef_)
#     selected_features = X.columns[feature_importance > 0].tolist()
#     return selected_features, feature_importance
# 
# y_single = np.argmax(y_train, axis=1)
# selected_features, feature_importance = lasso_feature_selection(x_train, y_single)
# print("Selected features:", selected_features)
# 
# feature_importance_pairs = sorted(zip(x_train.columns, feature_importance), key=lambda x: x[1], reverse=True)
# sorted_features, sorted_importance = zip(*feature_importance_pairs)
# 
# plt.figure(figsize=(14, 8))
# bar_width = 0.5 
# bars = plt.bar(range(len(sorted_importance)), sorted_importance, width=bar_width)
# plt.title('Feature Importance (LASSO)', fontsize=16)
# plt.xlabel('Features', fontsize=12)
# plt.ylabel('Importance', fontsize=12)
# plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right', fontsize=10)
# plt.ylim(0, max(sorted_importance) * 1.1)
# 
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#               ha='center', va='bottom', fontsize=8, rotation=90)
# 
# plt.tight_layout()
# plt.show()
# 
# x_train_selected = x_train[selected_features]
# x_test_selected = x_test[selected_features]
# =============================================================================
#%% random forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=1020
)

rf_model.fit(x_train, y_train)

all_features = data[features]
all_predictions = rf_model.predict(all_features)
all_predictions = np.argmax(all_predictions, axis=1)

data['predict_label'] = all_predictions

test_mask = (data.index >= '2023-07-01 00:00:00') & (data.index <= '2024-06-30 23:59:00')

y_true = data.loc[test_mask, 'label']
y_pred = data.loc[test_mask, 'predict_label']

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

# confusion matrix
conf_matrix = confusion_matrix(data.loc[test_mask, 'label'], data.loc[test_mask, 'predict_label'])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
label_names = ['-1', '0', '1'] 
plt.xticks(np.arange(3) + 0.5, label_names)
plt.yticks(np.arange(3) + 0.5, label_names)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('confusion matrix')
plt.show()

print(f'Accuracy： {accuracy:.4f}')
print(f'Precision：{precision:.4f}')
print(f'Recall：{recall:.4f}')
print(f'F1 score：{f1:.4f}')

#%% 交易策略
# 初始設定
initial_capital = 10000 
current_capital = initial_capital 
position = 0  
fee_rate = 0.0004
max_investment = 10000 
min_trade_amount = 0.00000001 
buy = []
sell = []
equity = []
trades = []
pending_buy = None 

# 交易策略
for i in range(len(data)):
    current_data = data.iloc[i]
    
    if pending_buy is not None: 
        position = max(pending_buy['shares'], min_trade_amount)
        cost = position * current_data['open']
        fee = cost * fee_rate
        current_capital -= (cost + fee)
        buy.append((current_data.name, current_data['open'], position))
        print(f"Buy executed: {position} shares at {current_data['open']}, time: {current_data.name}")
        pending_buy = None
    
    if i == 0:
        next_data = data.iloc[i+1]
        today_data = data.iloc[i]
        investment_amount = min(initial_capital, max_investment)
        shares_to_buy = max(investment_amount / (today_data['open'] * (1 + fee_rate)), min_trade_amount)
        pending_buy = {'shares': shares_to_buy}
        print(f"Initial Buy planned: {shares_to_buy} shares at next day's open, time: {next_data.name}")
    
    signal = current_data['predict_label']
    
    if signal in [0, 2] and position > 0:
        revenue = position * current_data['close']
        fee = revenue * fee_rate
        profit = revenue - fee - (buy[-1][1] * position) - (buy[-1][1] * position * fee_rate)
        current_capital += revenue - fee
        sell.append((current_data.name, current_data['close'], position))
        trades.append(profit)
        print(f"Sell: {position} shares at {current_data['close']}, time: {current_data.name}")
        print(f"Profit: {profit}")
        position = 0
    
    elif signal == 1 and position == 0 and i < len(data) - 1:
        next_data = data.iloc[i+1]
        investment_amount = min(current_capital, max_investment)
        shares_to_buy = max(investment_amount / (current_data['close'] * (1 + fee_rate)), min_trade_amount)
        pending_buy = {'shares': shares_to_buy}
        print(f"Buy planned: {shares_to_buy} shares at next day's open, time: {next_data.name}")
    
    if position > 0:
        equity.append(current_capital + position * current_data['close'])
    else:
        equity.append(current_capital)

equity_curve = pd.Series(equity, index=data.index)
#%%

train_start = '2020-01-01 00:00:00'
train_end = '2023-06-30 23:59:00'

test_start = '2023-07-01 00:00:00'
test_end = '2024-06-30 23:59:00'

plt.figure(figsize=(24, 8))
train_equity = equity_curve[(equity_curve.index >= train_start) & (equity_curve.index <= train_end)]
plt.plot(train_equity.index, train_equity.values, color='blue', label='train')
test_equity = equity_curve[(equity_curve.index > train_end) & (equity_curve.index <= test_end)]
plt.plot(test_equity.index, test_equity.values, color='red', label='test')

plt.title('equity curve', fontsize=16)
plt.xlabel('date', fontsize=12)
plt.ylabel('equity', fontsize=12)
plt.legend(fontsize=24, loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()  
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=mdates.datestr2num(train_end), color='gray', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

train_start = pd.Timestamp('2020-01-01 00:00:00')
train_end = pd.Timestamp('2023-06-30 23:59:00')
test_start = pd.Timestamp('2023-07-01 00:00:00')
test_end = pd.Timestamp('2024-06-30 23:59:00')
full_start = train_start  # 全部樣本的開始時間
full_end = test_end  # 全部樣本的結束時間

# 函數用於計算指定的指標
def calculate_metrics(equity_curve, trades, sell, start_date, end_date):
    period_equity = equity_curve[(equity_curve.index >= start_date) & (equity_curve.index <= end_date)]
    period_trades = [trade for trade, date in zip(trades, sell) if start_date <= pd.Timestamp(date[0]) <= end_date]
    
    # 基本交易指標
    total_trades = len(period_trades)
    winning_trades = sum(1 for trade in period_trades if trade > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 進階指標
    cumulative_return = (period_equity.iloc[-1] / period_equity.iloc[0]) - 1
    
    years = (end_date - start_date).days / 365.25
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    
    daily_returns = period_equity.pct_change().dropna()
    trading_days = len(daily_returns)
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
    
    cumulative_max = period_equity.cummax()
    drawdown = (period_equity - cumulative_max) / cumulative_max
    mdd = drawdown.min()
    
    risk_free_rate = 0.01 
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    risk_return_ratio = cumulative_return / abs(mdd)
    
    return {
        '累積報酬': cumulative_return,
        '年化報酬': annualized_return,
        '年化波動度': annualized_volatility,
        'MDD': mdd,
        '年化夏普比率': sharpe_ratio,
        '風險報酬比': risk_return_ratio,
        '總交易次數': total_trades,
        '勝率': win_rate,
    }

# 計算各時期的指標
periods = [
    ("訓練期間", train_start, train_end),
    ("測試期間", test_start, test_end),
    ("全部樣本", full_start, full_end)
]

for period_name, start_date, end_date in periods:
    print(f"\n{period_name}:")
    
    metrics = calculate_metrics(equity_curve, trades, sell, start_date, end_date)
    
    print(f"累積報酬: {metrics['累積報酬']:.2%}")
    print(f"年化報酬: {metrics['年化報酬']:.2%}")
    print(f"年化波動度: {metrics['年化波動度']:.2%}")
    print(f"最大回撤 (MDD): {metrics['MDD']:.2%}")
    print(f"年化夏普比率: {metrics['年化夏普比率']:.2f}")
    print(f"風險報酬比: {metrics['風險報酬比']:.2f}")
    print(f"總交易次數: {metrics['總交易次數']}")
    print(f"勝率: {metrics['勝率']:.2%}")
#%%
monthly_returns = equity_curve.resample('M').last().pct_change()

def calculate_monthly_mdd(equity_curve):
    daily_equity = equity_curve.resample('D').last().ffill()
    
    daily_drawdown = 1 - (daily_equity / daily_equity.cummax())
    
    monthly_mdd = daily_drawdown.groupby(pd.Grouper(freq='M')).max() 
    return monthly_mdd
monthly_mdd = calculate_monthly_mdd(equity_curve)


returns_matrix = pd.DataFrame(index=range(2020, 2025), columns=range(1, 13))
for date, value in monthly_returns.items():
    if date.year in returns_matrix.index:
        returns_matrix.loc[date.year, date.month] = value

mdd_matrix = pd.DataFrame(index=range(2020, 2025), columns=range(1, 13))
for date, value in monthly_mdd.items():
    if date.year in mdd_matrix.index:
        mdd_matrix.loc[date.year, date.month] = value

returns_matrix = returns_matrix.apply(pd.to_numeric, errors='coerce')
mdd_matrix = mdd_matrix.apply(pd.to_numeric, errors='coerce')
returns_masked = np.ma.masked_invalid(returns_matrix)
mdd_masked = np.ma.masked_invalid(mdd_matrix)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
year_labels = ['2020', '2021', '2022', '2023', '2024']

plt.figure(figsize=(18, 6))
sns.heatmap(returns_masked, cmap='RdYlGn', center=0, annot=returns_matrix.values, fmt='.2%', 
            cbar_kws={'label': 'Monthly Return'}, mask=returns_masked.mask)
plt.title('Monthly Return (%)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.xticks(np.arange(12) + 0.5, month_labels, rotation=0)
plt.yticks(np.arange(5) + 0.5, year_labels, rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))
sns.heatmap(mdd_masked, cmap='RdYlGn_r', center=None, annot=mdd_matrix.values, fmt='.2%', 
            cbar_kws={'label': 'Monthly MDD (%)'}, mask=mdd_masked.mask)
plt.title('Monthly MDD (%)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.xticks(np.arange(12) + 0.5, month_labels, rotation=0)
plt.yticks(np.arange(5) + 0.5, year_labels, rotation=0)
plt.tight_layout()
plt.show()

#%%

equity_curve = pd.Series(equity, index=data.index)

train_start = '2020-01-01 00:00:00'
train_end = '2023-06-30 23:59:00'
test_start = '2023-07-01 00:00:00'
test_end = '2024-06-30 23:59:00'

bnh_shares = int(initial_capital / (data['close'].iloc[0] * (1 + fee_rate)))
bnh_cost = bnh_shares * data['close'].iloc[0]
bnh_initial_fee = bnh_cost * fee_rate
bnh_remaining_cash = initial_capital - bnh_cost - bnh_initial_fee
bnh_equity = (data['close'] * bnh_shares + bnh_remaining_cash).values
bnh_equity_curve = pd.Series(bnh_equity, index=data.index)

plt.figure(figsize=(24, 8))

train_equity = equity_curve[(equity_curve.index >= train_start) & (equity_curve.index <= train_end)]
plt.plot(train_equity.index, train_equity.values, color='blue', label='Strategy (in-sample)')
test_equity = equity_curve[(equity_curve.index > train_end) & (equity_curve.index <= test_end)]
plt.plot(test_equity.index, test_equity.values, color='red', label='Strategy (out of sample)')
plt.plot(bnh_equity_curve.index, bnh_equity_curve.values, color='lightgray', label='Buy and Hold', linestyle='--')

plt.title('Trading Strategy vs Buy and Hold Equity Curves', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Equity', fontsize=12)
plt.legend(fontsize=24, loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()  
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=mdates.datestr2num(train_end), color='black', linestyle='--', linewidth=2)
plt.tight_layout()
plt.show()


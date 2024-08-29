#!/usr/bin/env python
# coding: utf-8

# In[165]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import warnings


# In[3]:


### 原始data

path = 'C:/NTPU/forecasting/program/metro.taipei.xlsx'
data = pd.read_excel(path)


# In[4]:


### 加上time index data 

data.index = pd.date_range(start = data['date'].iloc[0], end = data['date'].iloc[-1])


# In[173]:


### 日資料data
dataDaily = data.drop(columns = ['date'])


### 月資料data
dataMonth = data.drop(columns = ['date'])
dataMonth = dataMonth.astype(int)

dataMonthly = dataMonth.resample('MS').sum()


### 季資料data
dataQuarter = data.drop(columns = ['date'])
dataQuarter = dataQuarter.astype(int)

dataQuarterly = dataQuarter.resample('QS').sum()


### 年資料data
dataYear = data.drop(columns = ['date'])
dataYear = dataYear.astype(int)

dataYearly = dataYear.resample('YS').sum()


# In[201]:


### 周平均 data


dataDailyinlier = dataDaily.copy()

years = dataDailyinlier.index.year.unique()

for year in years:
    year_data = dataDailyinlier[dataDailyinlier.index.year == year]
    
    largestDailytop3_idx = year_data['rider'].nlargest(3).index
    smallestDailytop3_idx = year_data['rider'].nsmallest(3).index

    dataDailyinlier = dataDailyinlier.drop(largestDailytop3_idx)
    dataDailyinlier = dataDailyinlier.drop(smallestDailytop3_idx)

    
    
dataMonday = dataDailyinlier[dataDailyinlier.index.weekday == 0].mean().round().astype(int)
dataTuesday =  dataDailyinlier[dataDailyinlier.index.weekday == 1].mean().round().astype(int)
dataWednesday = dataDailyinlier[dataDailyinlier.index.weekday == 2].mean().round().astype(int)
dataThursday = dataDailyinlier[dataDailyinlier.index.weekday == 3].mean().round().astype(int)
dataFriday = dataDailyinlier[dataDailyinlier.index.weekday == 4].mean().round().astype(int)
dataSaturday = dataDailyinlier[dataDailyinlier.index.weekday == 5].mean().round().astype(int)
dataSunday = dataDailyinlier[dataDailyinlier.index.weekday == 6].mean().round().astype(int)


weeklyData = pd.DataFrame([dataMonday, dataTuesday, dataWednesday, dataThursday, dataFriday, dataSaturday, dataSunday]
, index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

start_date = '2022-01-03'
    
dates = pd.date_range(start_date, periods=len(weeklyData), freq='D')

weeklyData = pd.Series(weeklyData['rider'].values, index=dates)


# In[202]:


### 月平均 data

dataMonthlyyy = dataDailyinlier.resample('MS').sum()


dataJan = dataMonthlyyy[dataMonthlyyy.index.month == 1].mean().round().astype(int)
dataFeb = dataMonthlyyy[dataMonthlyyy.index.month == 2].mean().round().astype(int)
dataMar = dataMonthlyyy[dataMonthlyyy.index.month == 3].mean().round().astype(int)
dataApr = dataMonthlyyy[dataMonthlyyy.index.month == 4].mean().round().astype(int)
dataMay = dataMonthlyyy[dataMonthlyyy.index.month == 5].mean().round().astype(int)
dataJun = dataMonthlyyy[dataMonthlyyy.index.month == 6].mean().round().astype(int)
dataJul = dataMonthlyyy[dataMonthlyyy.index.month == 7].mean().round().astype(int)
dataAug = dataMonthlyyy[dataMonthlyyy.index.month == 8].mean().round().astype(int)
dataSep = dataMonthlyyy[dataMonthlyyy.index.month == 9].mean().round().astype(int)
dataOct = dataMonthlyyy[dataMonthlyyy.index.month == 10].mean().round().astype(int)
dataNoc = dataMonthlyyy[dataMonthlyyy.index.month == 11].mean().round().astype(int)
dataDec = dataMonthlyyy[dataMonthlyyy.index.month == 12].mean().round().astype(int)

monthlyData = pd.DataFrame([dataJan, dataFeb, dataMar, dataApr, dataMay, dataJun, dataJul, dataAug, dataSep, dataOct, dataNoc, dataDec]
, index= ['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01', '08-01', '09-01', '10-01', '11-01', '12-01'])


# In[142]:


### data 2023

path = 'C:/NTPU/forecasting/program/data2023.xlsx'
data2023 = pd.read_excel(path)


data2023.index = pd.date_range(start = data2023['date'].iloc[0], end = data2023['date'].iloc[-1])
data2023 = data2023.drop(columns = ['Unnamed: 0'])

data2023

### 日資料data2023
dataDaily2023 = data2023.drop(columns = ['date'])


### 月資料data2023
dataMonth2023 = data2023.drop(columns = ['date'])
dataMonth2023 = dataMonth2023.astype(int)

dataMonthly2023 = dataMonth2023.resample('MS').sum()


### 季資料data2023
dataQuarter2023 = data2023.drop(columns = ['date'])
dataQuarter2023 = dataQuarter2023.astype(int)

dataQuarterly2023 = dataQuarter2023.resample('QS').sum()


### 年資料data2023
dataYear2023 = data2023.drop(columns = ['date'])
dataYear2023 = dataYear2023.astype(int)

dataYearly2023 = dataYear2023.resample('YS').sum()


# In[6]:


### hp filter
from statsmodels.tsa.filters.hp_filter import hpfilter as hp


# In[8]:


### hp filter Yearly
cycle, trend = hp(dataYearly, 100)
dataYearly_hp = dataYearly.copy()
dataYearly_hp['cycle'] = cycle
dataYearly_hp['trend'] = trend

dataYearly_hp.plot()
print(dataYearly_hp)


# In[9]:


### hp filter Quarterly
cycle, trend = hp(dataQuarterly, 1600)
dataQuarterly_hp = dataQuarterly.copy()
dataQuarterly_hp['cycle'] = cycle
dataQuarterly_hp['trend'] = trend

dataQuarterly_hp.plot()
print(dataQuarterly_hp)


# In[10]:


### hp filter Monthly
cycle, trend = hp(dataMonthly, 14400)
dataMonthly_hp = dataMonthly.copy()
dataMonthly_hp['cycle'] = cycle
dataMonthly_hp['trend'] = trend

dataMonthly_hp.plot()
print(dataMonthly_hp)


# In[14]:


### STL
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess


# In[150]:


### STL Quarterly
stl = STL(dataQuarterly.iloc[:,0], seasonal=5)
resSTLquarterly = stl.fit()
fig = resSTLquarterly.plot()


# In[151]:


### STL monthly
stl = STL(dataMonthly.iloc[:,0], seasonal=13)
resSTLmonthly = stl.fit()
fig = resSTLmonthly.plot()


# In[49]:


### STL weeklyData
stl = STL(weeklyData, seasonal=53)
res = stl.fit()
fig = res.plot()


# In[51]:


# 進行 STL 分解
stl = STL(weeklyData, seasonal=53)
res = stl.fit()

# 獲取 STL 分解的組件
seasonal = res.seasonal
trend = res.trend
resid = res.resid

# 創建一個新的圖表，自定義 x 軸的標籤為星期名稱
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)



# 趨勢組件
ax[0].plot(trend)
ax[0].set_title('Trend')

# 季節性組件
ax[1].plot(seasonal)
ax[1].set_title('Seasonal')

# 殘差組件
ax[2].plot(resid)
ax[2].set_title('Residual')

# 設定 x 軸刻度標籤為星期名稱
ax[2].set_xticks(weeklyData.index)  # 設定 x 軸的刻度位置
ax[2].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45)  # 設定每個刻度的標籤

plt.tight_layout()
plt.show()


# In[ ]:





# In[92]:


### 尋找最適ARIMA之參數

from pmdarima.model_selection import train_test_split


# In[94]:


### 月資料

train, test = train_test_split(dataMonthly, train_size=0.8)
model_dataMonthly = pm.auto_arima(train)

print(model_dataMonthly.summary())


# In[145]:


train, test = train_test_split(dataQuarterly, train_size=0.8)
model_dataQuarterly = pm.auto_arima(train)

print(model_dataQuarterly.summary())


# In[ ]:





# In[156]:


### 月資料

STLfcast_m = STLForecast(dataMonthly['rider'], ARIMA,  model_kwargs={"order": (4, 1, 1)}, seasonal = 13)
res = STLfcast_m.fit()
STLforecasts_monthly = res.forecast(12)
STLforecasts_monthly = STLforecasts_monthly.astype(int)

STLforecasts_monthly


# In[184]:


rmse_STLforecasts_monthly = np.sqrt(np.mean((STLforecasts_monthly - dataMonthly2023['rider'])**2))

print(f"rmse_STLforecasts_monthly: {rmse_STLforecasts_monthly}")
dataMonthly2023


# In[175]:


### 季資料

STLfcast_q = STLForecast(dataQuarterly['rider'], ARIMA,  model_kwargs={"order": (4, 1, 1)}, seasonal = 13)
res = STLfcast_q.fit()
STLforecasts_quarterly = res.forecast(4)
STLforecasts_quarterly = STLforecasts_quarterly.astype(int)
STLforecasts_quarterly


# In[185]:


rmse_STLforecasts_quarterly = np.sqrt(np.mean((STLforecasts_quarterly - dataQuarterly2023['rider'])**2))

print(f"rmse_STLforecasts_quarterly: {rmse_STLforecasts_quarterly}")


# In[163]:


### ETS

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[166]:


### ETS Model Select 月資料

from itertools import product

y = dataMonthly['rider'].values.flatten() 

error = ['add', 'mul']
trend = [None, 'add', 'mul']
seasonal = [None, 'add', 'mul']
damped_trend = [False, True]

models = list(product(error, trend, seasonal, damped_trend))
successful_models = []
bic_values = []

min_bic = float('inf')
best_params = None

for params in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ETSModel(y, seasonal_periods=12, error=params[0], trend=params[1], seasonal=params[2], damped_trend=params[3])
            fitted_model = model.fit()
            bic_value = fitted_model.bic
            bic_values.append(bic_value)
            successful_models.append(params)
            
            if bic_value < min_bic:
                min_bic = bic_value
                best_params = params
        except ValueError as e:
            print(f"Failed to fit model with params {params}: {e}")

for idx, params in enumerate(successful_models):
    print(f"Model parameters: {params}, BIC: {bic_values[idx]}")

print(f"\nBest Model parameters: {best_params}, Min BIC: {min_bic}")


# In[167]:


### ETS Model Select 季資料

from itertools import product

y = dataQuarterly['rider'].values.flatten() 

error = ['add', 'mul']
trend = [None, 'add', 'mul']
seasonal = [None, 'add', 'mul']
damped_trend = [False, True]

models = list(product(error, trend, seasonal, damped_trend))
successful_models = []
bic_values = []

min_bic = float('inf')
best_params = None

for params in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ETSModel(y, seasonal_periods=12, error=params[0], trend=params[1], seasonal=params[2], damped_trend=params[3])
            fitted_model = model.fit()
            bic_value = fitted_model.bic
            bic_values.append(bic_value)
            successful_models.append(params)
            
            if bic_value < min_bic:
                min_bic = bic_value
                best_params = params
        except ValueError as e:
            print(f"Failed to fit model with params {params}: {e}")

for idx, params in enumerate(successful_models):
    print(f"Model parameters: {params}, BIC: {bic_values[idx]}")

print(f"\nBest Model parameters: {best_params}, Min BIC: {min_bic}")


# In[193]:


ets_model_monthly = ETSModel(dataMonthly['rider'], error = 'add', trend = None, seasonal = 'mul', seasonal_periods=12)
ets_fit_monthly = ets_model_monthly.fit()

forecastETSmonthly = ets_fit_monthly.forecast(steps=12)  
forecastETSmonthly = forecastETSmonthly.astype(int)
print(forecastETSmonthly)


# In[192]:


rmse_ETSforecasts_monthly = np.sqrt(np.mean((forecastETSmonthly - dataMonthly2023['rider'])**2))

print(f"rmse_ETSforecasts_monthly: {rmse_ETSforecasts_monthly}")

print(dataMonthly2023)


# In[199]:


ets_model_quarterly = ETSModel(dataQuarterly['rider'], error = 'add', trend = None, seasonal = 'mul', seasonal_periods=4)
ets_fit_quarterly = ets_model_quarterly.fit()

forecastETSquarterly = ets_fit_quarterly.forecast(steps=4)  
print(forecastETSquarterly)


# In[200]:


rmse_ETSforecasts_quarterly = np.sqrt(np.mean((forecastETSquarterly - dataQuarterly2023['rider'])**2))

print(f"rmse_ETSforecasts_quarterly: {rmse}")


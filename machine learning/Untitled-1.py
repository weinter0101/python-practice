# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:15:55 2024

@author: 張哲瑋
"""
"""
抓取雞蛋批發價及大運輸價(大宗穀物)並存為EXCEL

"""
import pandas as pd
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime
from datetime import timedelta
from selenium.webdriver.support.ui import Select


def scrapeData(cityID, cityName):
    
    driver = webdriver.Chrome()

    webURL = "http://www.foodchina.com.tw/model/marketing/AnaChartNew.aspx?id=52&ChkID=301&Page=0&Type=1&cn=False"
    postURL = "http://www.foodchina.com.tw/model/ajax/getChartData.ashx"
    driver.get(webURL)

    dayselectionID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_RB1"
    yearID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_ddl_Year1"
    monthID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_ddl_Month1"
    dayID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_ddl_Day"

    submitID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_btnSummit"
    startdate = datetime(year=2024, month=2, day=15)
    enddate = datetime(year=2019, month=12, day=20)
    selectID = [yearID, monthID, dayID]

    shippingID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblRows_1"


    dataWholesale = {}
    dataShipping = {}
    driver.find_element(by=By.ID, value=cityID).click()
    driver.find_element(by=By.ID, value=dayselectionID).click()
    driver.find_element(by=By.ID, value=shippingID).click()
    
    
    while startdate > enddate:
        date = [startdate.year, startdate.month, startdate.day]
        for i in range(len(selectID)):
            Select(driver.find_element(by=By.ID, value=selectID[i])).select_by_value(str(date[i]))
        driver.find_element(by= By.ID, value=submitID).click()
        driver.get(postURL)
        
        dictionaryWholesale = json.loads(driver.find_element(by= By.XPATH, value="/html/body/pre").text)['data']['datasets'][0]['data']
        for i in range(len(dictionaryWholesale)-1,-1, -1):
            v = list(dictionaryWholesale[i].values())
            dataWholesale[v[0]] = v[1]
         
        dictionaryShipping = json.loads(driver.find_element(by= By.XPATH, value="/html/body/pre").text)['data']['datasets'][1]['data']
        for i in range(len(dictionaryShipping)-1,-1, -1):
            v = list(dictionaryShipping[i].values())
            dataShipping[v[0]] = v[1]
            
        driver.back()
        driver.back()
        startdate = startdate - timedelta(days=30)
        
    dataWholesale = pd.DataFrame(dataWholesale.items(), columns=[f'{cityName}date', f'{cityName}Wholesale'])
    dataShipping = pd.DataFrame(dataShipping.items(), columns=['shippingdate', f'{cityName}Shipping'])
        
    data = pd.concat([dataWholesale, dataShipping], axis=1)
    data = data.drop(columns=['shippingdate'])
    return data
    

taipeiID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblCols_0"
taichungID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblCols_1"
tainanID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblCols_2"
wholesaleID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblRows_0"
shippingID = "ctl00_ctl00_cpl_MainContent_cpl_BasicMainContent_cblRows_1"

# 抓取資料
taipeiData = scrapeData(taipeiID, 'taipei')
taichungData = scrapeData(taichungID, 'taichung')
tainanData = scrapeData(tainanID, 'tainan')

# 資料合併及刪去多餘的date
mergedData = pd.concat([taipeiData, taichungData, tainanData], axis=1)
mergedData = mergedData.drop(columns=['taichungdate', 'tainandate'])
mergedData = mergedData.rename(columns={'taipeidate': 'date'})

# 設定time index
mergedData['date'] = pd.to_datetime(mergedData['date'], format='%y/%m/%d')
mergedData.set_index('date', inplace=True)

# 把資料轉正且刪除不必要資料
data = mergedData.sort_index(ascending=True)  

startDate = '2020-01-01'
endDate = '2024-01-31'
data = data[startDate:endDate]

data.to_csv('data.csv',index=True)
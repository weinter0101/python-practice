#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
爬蟲777777777777
'''

from datetime import datetime

def date_tf(x): 
    year, month, day = x.split('/')
    y = datetime(int(year) + 1911, int(month), int(day))
    return y

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO


urls = []
date = []
rider = []

urls_input = input("輸入網址 (以空格分隔): ")
urls = urls_input.split()


for url in urls:
        
    response = requests.get(url)     # 使用 requests 獲取網頁內容
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')     # 找到網頁中的表格，您可能需要根據實際的 HTML 結構來調整這個查詢

    table = tables[0]     # 選擇第一個表格或者根據需要選擇其他表格
    table_string_io = StringIO(str(table))     # 將表格轉換為字串，然後使用 StringIO 包裝
    
    df = pd.read_html(table_string_io)[0]
    
    
    mm_data = df.iloc[2:5, 0].apply(date_tf).reset_index(drop=True) 
    mm_pick = mm_data.iloc[0].month        

    yy_data = df.iloc[2:5, 0].apply(date_tf).reset_index(drop=True) 
    yy_pick = yy_data.iloc[0].year
        
    if mm_pick < 8:
        if mm_pick / 2 == 1:
            if yy_pick % 4 == 0:
                end_row = 31
            else:
                end_row = 30
        elif mm_pick % 2 != 0:
                end_row = 33
        else:
                end_row = 32
    else:
        if mm_pick % 2 == 0:
                end_row = 33
        else:
                end_row = 32
                
    date_data = df.iloc[2:end_row, 0].apply(date_tf).reset_index(drop=True)
    rider_data = df.iloc[2:end_row, 2].reset_index(drop=True)
    
    date.append(date_data)
    rider.append(rider_data)
                
                
date = pd.concat(date, axis=0, ignore_index=True)
rider = pd.concat(rider, axis=0, ignore_index=True)

data2023 = pd.concat([date, rider], axis=1, keys=['date', 'rider'])

data2023

data2023.to_excel("data2023.xlsx")


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

def get_history(client, symbol, interval, start, end=None):
    bars = client.get_historical_klines(
        symbol = symbol, 
        interval = interval, 
        start_str = start,
        end_str = end, 
        limit = 1000)
    
    df = pd.DataFrame(bars)
    df['Date'] = pd.to_datetime(df.iloc[:,0], unit='ms')
    df.columns = ["Open Time","Open","High","Low","Close",
                 "Volume", "Close Time", "Quote Asset Volume",
                 "Number of Trades", "Taker Buy Base Asset Volume",
                 "Taker Buy Quote Asset Volume", "Ignore", "Date"
                 ]
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.set_index("Date", inplace = True)
    
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df


# In[4]:


def return_keys(file):
    keys = []
    with open(file, 'r') as f:
        for line in f.readlines():
            keys.append(line.strip())
    return keys


# In[6]:


def live_keys(file):
    keyholder = []
    api_key = return_keys(file)[1].split(":")[1]
    secret_key = return_keys(file)[2].split(":")[1]
    keyholder.append(api_key)
    keyholder.append(secret_key)
    
    return keyholder


# In[8]:


def test_keys(file):
    keyholder = []
    api_key = return_keys(file)[5].split(":")[1]
    secret_key = return_keys(file)[6].split(":")[1]
    keyholder.append(api_key)
    keyholder.append(secret_key)
    
    return keyholder


def future_test_keys(file):
    keyholder = []
    api_key = return_keys(file)[9].split(":")[1]
    secret_key = return_keys(file)[10].split(":")[1]
    keyholder.append(api_key)
    keyholder.append(secret_key)
    
    return keyholder
# In[ ]:





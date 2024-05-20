#!/usr/bin/python3

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


def mean(data):
    total = 0
    for i in data:
        total += i
    return total / len(data)

def variance(data):
    average = mean(data)
    total = 0
    for i in data:
        total += (average - i) ** 2
    return total / len(data)

def sdev(data):
    return variance(data) ** 0.5

def corr(data1, data2):
    total = 0
    mean1 = mean(data1)
    mean2 = mean(data2)
    sdev1 = sdev(data1)
    sdev2 = sdev(data2)
    for (ix, iy) in zip(data1, data2):
        total += ((ix - mean1) * (iy - mean2))
    return total / (len(data1) * sdev1 * sdev2)

def show_price(data):
    sns.set_theme()
    sns.lineplot(data['Adj Close'], dashes=False)
    plt.show()

def show_dist(data):
    sns.set_theme()
    sns.displot(data['Adj Close'].pct_change() * 100, bins=50, kde=True, stat='probability', aspect=2)
    plt.xlabel("Percent Change")
    plt.show()

def show_corr(data1, data2):
    sns.set_theme()
    sns.scatterplot(x=data1, y=data2)
    plt.show()

def show_sdev_dist(data):
    sns.set_theme()
    percent = data.pct_change().dropna()
    average = mean(percent)
    standev = sdev(percent)
    sdevdist = (percent - average) / standev
    sns.displot(sdevdist, bins=50, kde=False, stat='probability', aspect=2)
    plt.show()


interval = '1d'
start_date = '2019-01-01' 
end_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

symbols = ['F', 'GM', 'IPB', 'MS', 'MRK', 'PFE']
symbol_string = ''
for i in symbols:
    symbol_string += i + ' '
symbol_string.strip()

data = yf.download(symbol_string, start=start_date, end=end_date, interval='1d')

if data.isnull().values.any():
    print('WARNING! THIS DATA IS MISSING VALUES! PROCEED WITH CAUTION!')


"""
print(data.info())

print(mean(data['Adj Close']['CRWD']))
print(variance(data['Adj Close']['CRWD']))
print(sdev(data['Adj Close']['CRWD']))

show_price(data)
show_dist(data)
#show_corr(data['Adj Close']['AAPL'], data['Adj Close']['^IXIC'])
print(corr(data['Adj Close']['AAPL'], data['Adj Close']['^IXIC']))
#show_corr(data['Adj Close']['AAPL'].pct_change(), data['Adj Close']['^IXIC'].pct_change())
print(corr(data['Adj Close']['AAPL'].pct_change().dropna(), data['Adj Close']['^IXIC'].pct_change().dropna()))
print(data['Adj Close']['^IXIC'].pct_change())
show_corr(data['Adj Close']['CRWD'], data['Adj Close']['^DJI'])
print(corr(data['Adj Close']['CRWD'], data['Adj Close']['^DJI']))
show_corr(data['Adj Close']['AAPL'], data['Adj Close']['AAPL'] * -1)
print(corr(data['Adj Close']['AAPL'], data['Adj Close']['AAPL'] * -1))

show_corr(data['Adj Close']['NG=F'], data['Adj Close']['^GSPC'])
print(corr(data['Adj Close']['NG=F'].dropna(), data['Adj Close']['^GSPC'].dropna()))
show_corr(data['Adj Close']['NG=F'].pct_change(fill_method=None).dropna(), data['Adj Close']['^GSPC'].pct_change(fill_method=None).dropna())
print(corr(data['Adj Close']['NG=F'].pct_change(fill_method=None).dropna(), data['Adj Close']['^GSPC'].pct_change(fill_method=None).dropna()))
show_corr(data['Adj Close']['F'].pct_change(fill_method=None).dropna(), data['Adj Close']['GM'].pct_change(fill_method=None).dropna())
print(corr(data['Adj Close']['F'].pct_change(fill_method=None).dropna(), data['Adj Close']['GM'].pct_change(fill_method=None).dropna()))
show_corr(data['Adj Close']['IPB'].pct_change(fill_method=None).dropna(), data['Adj Close']['MS'].pct_change(fill_method=None).dropna())
print(corr(data['Adj Close']['IPB'].pct_change(fill_method=None).dropna(), data['Adj Close']['MS'].pct_change(fill_method=None).dropna()))
show_corr(data['Adj Close']['MRK'].pct_change(fill_method=None).dropna(), data['Adj Close']['PFE'].pct_change(fill_method=None).dropna())
print(corr(data['Adj Close']['MRK'].pct_change(fill_method=None).dropna(), data['Adj Close']['PFE'].pct_change(fill_method=None).dropna()))
"""
show_sdev_dist(data['Adj Close']['F'])

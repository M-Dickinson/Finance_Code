#!/usr/bin/python3

import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 1500)


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

def daily_risk(data, value):
    risk = sdev(data[-20:].pct_change(fill_method=None).dropna())
    print('Today\'s Risk : ' + str(risk))
    print('68% Confidence of gain/loss within : ' + str(risk * value))
    print('95% Confidence of gain/loss within : ' + str(risk * value * 2))

def simple_moving_average(data, days):
    return sum(data.iloc[-days:]) / days

def exponential_moving_average(data, days):
    curr = simple_moving_average(data.iloc[:days], days)
    alpha = 2 / (1 + days)
    for i in data.iloc[days:]:
        curr = (alpha * i) + ((1 - alpha) * curr)
    return curr

def weighted_moving_average(data, days):
    total = 0
    for i, j in zip(data.iloc[-days:], range(1, days + 1)):
        total += i * j
    return total * (2 / (days**2 + days))

def show_price(data):
    sns.set_theme()
    sns.lineplot(data, dashes=False)
    plt.show()

def show_dist(data):
    sns.set_theme()
    sns.displot(data['Adj Close'].pct_change(fill_method=None).dropna() * 100, bins=50, kde=True, stat='probability', aspect=2)
    plt.xlabel("Percent Change")
    plt.show()

def show_corr(data1, data2):
    sns.set_theme()
    sns.scatterplot(x=data1, y=data2)
    plt.show()

def show_sdev_dist(data, binwidth):
    sns.set_theme()
    percent = data.pct_change(fill_method=None).dropna()
    average = mean(percent)
    standev = sdev(percent)
    sdev_dist = (percent - average) / standev
    sdev_average = mean(sdev_dist)
    sdev_standev = sdev(sdev_dist)
    curr = sdev_dist.min()
    standev_max = sdev_dist.max()
    step = (standev_max - curr) / 1000
    normal_val = []
    curr_val = []
    for i in range(1000):
        curr_val.append(curr)
        normal_val.append(((1 / (sdev_standev * ((2 * math.pi) ** 0.5))) * (math.exp((((curr - sdev_average) ** 2) * -1) / (2 * (sdev_standev  ** 2)))) * binwidth))
        curr += step
    normal_data = pd.DataFrame({'Sdevs' : curr_val, 'Normal' : normal_val})
    sns.displot(sdev_dist, kde=False, binwidth=binwidth, stat='probability', aspect=2)
    sns.lineplot(normal_data, x='Sdevs', y='Normal')
    plt.show()

def show_volatility(data):
    percent = data.pct_change(fill_method=None).dropna()
    data = []
    index = []
    for i in range(len(percent) - 20):
        data.append(sdev(percent.iloc[i:i+20]) * (252 ** 0.5))
        index.append(percent.index[i+20])
    final_data = pd.DataFrame({'Data' : data}, index=index)
    sns.lineplot(final_data)
    plt.show()

def show_simple_moving_average(data, days):
    average = []
    index = []
    for i in range(days, len(data)+1):
        average.append(simple_moving_average(data.iloc[i-days:i], days))
        index.append(data.index[i-1])
    average_data = pd.DataFrame({'Average' : average}, index=index)
    sns.lineplot(data, dashes=False)
    sns.lineplot(average_data, palette=['orange'])
    plt.show()

def show_exponential_moving_average(data, days):
    average = []
    index = []
    curr = simple_moving_average(data.iloc[:days], days)
    alpha = 2 / (1 + days)
    for i in range(days, len(data)):
        curr = (alpha * data.iloc[i]) + ((1 - alpha) * curr)
        average.append(curr)
        index.append(data.index[i])
    average_data = pd.DataFrame({'Average' : average}, index=index)
    sns.lineplot(data, dashes=False)
    sns.lineplot(average_data, palette=['orange'])
    plt.show()
    print(average_data)

def show_weighted_moving_average(data, days):
    average = []
    index = []
    for i in range(days, len(data)+1):
        average.append(weighted_moving_average(data[i-days:i], days))
        index.append(data.index[i-1])
    average_data = pd.DataFrame({'Average' : average}, index=index)
    sns.lineplot(data, dashes=False)
    sns.lineplot(average_data, palette=['orange'])
    plt.show()

def channels(data, days):
    high = []
    low = []
    index = []
    for i in range(days, len(data)+1):
        high.append(max(data['High'][i-days:i]))
        low.append(min(data['Low'][i-days:i]))
        index.append(data.index[i-1])
    high_data = pd.DataFrame({'High' : high}, index=index)
    low_data = pd.DataFrame({'Low' : low}, index=index)
    sns.lineplot(data['Close'], dashes=False)
    sns.lineplot(high_data, palette=['green'])
    sns.lineplot(low_data, palette=['red'])
    plt.show()

def momentum(data, days):
    momentum = []
    index = []
    for i in range(days, len(data)):
        momentum.append(data['Close'].iloc[i] - data['Close'].iloc[i-days])
        index.append(data.index[i])
    momentum_data = pd.DataFrame({'Momentum' : momentum}, index=index)
    print(momentum_data)
    sns.lineplot(data['Close'], dashes=False)
    sns.lineplot(momentum_data, palette=['orange'])
    plt.show()


interval = '1d'
start_date = '2023-01-01'
end_date = None

symbols = ['CADUSD=X', 'INTC', 'GE', 'NG=F', 'AAPL', '^GSPC', 'CRWD']
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
show_corr(data['Adj Close']['AAPL'], data['Adj Close']['^IXIC'])
print(corr(data['Adj Close']['AAPL'], data['Adj Close']['^IXIC']))
show_corr(data['Adj Close']['AAPL'].pct_change(), data['Adj Close']['^IXIC'].pct_change())
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

stock = 'NG=F'
show_sdev_dist(data['Adj Close'][stock], 1)
show_sdev_dist(data['Adj Close'][stock], 0.75)
show_sdev_dist(data['Adj Close'][stock], 0.5)
show_sdev_dist(data['Adj Close'][stock], 0.25)
show_sdev_dist(data['Adj Close'][stock], 0.1)

show_volatility(data['Adj Close']['AAPL'])
show_volatility(data['Adj Close']['CADUSD=X'])

daily_risk(data['Adj Close']['AAPL'], 20000000)
daily_risk(data['Adj Close']['CADUSD=X'], 20000000)

print(simple_moving_average(data['Close']['AAPL'].dropna(), 20))
print(exponential_moving_average(data['Close']['AAPL'].dropna(), 20))
print(weighted_moving_average(data['Close']['AAPL'].dropna(), 20))
show_simple_moving_average(data['Close']['AAPL'].dropna(), 20)
show_exponential_moving_average(data['Close']['AAPL'].dropna(), 20)
show_weighted_moving_average(data['Close']['AAPL'].dropna(), 20)
"""

newdata = data.swaplevel(axis=1)
#print(newdata['AAPL']['High'].dropna())
#print(newdata.loc[:, ['AAPL', 'CRWD']]['AAPL']['High'].dropna())
#channels(newdata['AAPL'].dropna(), 20)
momentum(newdata['AAPL'].dropna(), 20)

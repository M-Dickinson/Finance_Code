#!/usr/bin/env python3

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2022-12-31'
data = yf.download(symbol, start=start_date, end=end_date)

data['SMA_50'] = data['Close'].rolling(window=50).mean()

def rsi(data, period):
    delta = data.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['RSI'] = rsi(data['Close'], 14)


data['Signal'] = np.where(data['Close'] > data['SMA_50'], 1, 0)

data['Signal2'] = 0
data.loc[data['RSI'] < 30, 'Signal2'] = 1
data.loc[data['RSI'] > 70, 'Signal2'] = -1

data['Daily_Return'] = data['Close'].pct_change()
data['SMA_Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)
data['RSI_Strategy_Return'] = data['Daily_Return'] * data['Signal2'].shift(1)
data['Cumulative_SMA_Return'] = (1 + data['SMA_Strategy_Return']).cumprod()
data['Cumulative_RSI_Return'] = (1 + data['RSI_Strategy_Return']).cumprod()

spy_data = yf.download('SPY', start=start_date, end=end_date)

spy_data['Daily_Return'] = spy_data['Close'].pct_change()
spy_data['Cumulative_Return'] = (1 + spy_data['Daily_Return']).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Cumulative_SMA_Return'], label='SMA Strategy')
plt.plot(data.index, data['Cumulative_RSI_Return'], label='RSI Strategy')
plt.plot(spy_data.index, spy_data['Cumulative_Return'], label='SPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

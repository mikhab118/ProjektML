import ccxt
import pandas as pd
import ta
import matplotlib.pyplot as plt
import numpy as np

# Inicjalizacja obiektu Binance API
exchange = ccxt.binance()

# Parametry
symbol = 'BTC/USDT'
timeframe = '4h'  # lub '1d' dla 1-dniowych danych
since = exchange.parse8601('2023-01-01T00:00:00Z')  # start date
until = exchange.parse8601('2023-01-05T00:00:00Z')  # end date

# Funkcja do pobierania danych w zakresie dat
def fetch_data_in_range(symbol, timeframe, since, until):
    all_data = []
    while since < until:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        last_timestamp = ohlcv[-1][0]
        if last_timestamp >= until:
            ohlcv = [row for row in ohlcv if row[0] < until]
        all_data.extend(ohlcv)
        since = last_timestamp + 1  # update since to fetch next batch
    return all_data

# Pobranie danych historycznych
ohlcv = fetch_data_in_range(symbol, timeframe, since, until)

# Konwersja do DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Obliczanie wskaźników technicznych

# Moving Averages (SMA, EMA)
df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)

# Relative Strength Index (RSI)
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# Moving Average Convergence Divergence (MACD)
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_hist'] = macd.macd_diff()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['bb_upper'] = bollinger.bollinger_hband()
df['bb_middle'] = bollinger.bollinger_mavg()
df['bb_lower'] = bollinger.bollinger_lband()

# Stochastic Oscillator
stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
df['stoch_k'] = stochastic.stoch()
df['stoch_d'] = stochastic.stoch_signal()

# Fibonacci Retracement (zakładamy, że chcesz obliczyć na podstawie ostatnich n świec)
def fibonacci_retracement(high, low):
    diff = high - low
    level1 = high - 0.236 * diff
    level2 = high - 0.382 * diff
    level3 = high - 0.618 * diff
    return level1, level2, level3

high_max = df['high'].max()
low_min = df['low'].min()
df['fib_236'], df['fib_382'], df['fib_618'] = fibonacci_retracement(high_max, low_min)

# Ichimoku Cloud - Chmura Ichimoku
ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
df['ichimoku_a'] = ichimoku.ichimoku_a()
df['ichimoku_b'] = ichimoku.ichimoku_b()
df['ichimoku_base'] = ichimoku.ichimoku_base_line()
df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

# Pivot Points (Classic)
def pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

df['pivot'], df['r1'], df['s1'], df['r2'], df['s2'] = pivot_points(df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1])

# Wyświetlenie pierwszych kilku wierszy
print(df.head())

# Wizualizacja danych
plt.figure(figsize=(18, 22))

# Wykres ceny zamknięcia i SMA, EMA
plt.subplot(5, 2, 1)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.plot(df['timestamp'], df['sma_20'], label='SMA 20', linestyle='--')
plt.plot(df['timestamp'], df['ema_20'], label='EMA 20', linestyle='--')
plt.title(f'{symbol} Close Price with SMA and EMA')
plt.legend()

# Wykres RSI
plt.subplot(5, 2, 2)
plt.plot(df['timestamp'], df['rsi'], label='RSI', color='orange')
plt.axhline(30, linestyle='--', color='red')
plt.axhline(70, linestyle='--', color='red')
plt.title('Relative Strength Index (RSI)')
plt.legend()

# Wykres MACD
plt.subplot(5, 2, 3)
plt.plot(df['timestamp'], df['macd'], label='MACD', color='blue')
plt.plot(df['timestamp'], df['macd_signal'], label='MACD Signal', color='red')
plt.fill_between(df['timestamp'], df['macd_hist'], 0, color='gray', alpha=0.3, label='MACD Histogram')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

# Wykres Bollinger Bands
plt.subplot(5, 2, 4)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.plot(df['timestamp'], df['bb_upper'], label='BB Upper', linestyle='--', color='orange')
plt.plot(df['timestamp'], df['bb_middle'], label='BB Middle', linestyle='--', color='blue')
plt.plot(df['timestamp'], df['bb_lower'], label='BB Lower', linestyle='--', color='orange')
plt.title('Bollinger Bands')
plt.legend()

# Wykres Stochastic Oscillator
plt.subplot(5, 2, 5)
plt.plot(df['timestamp'], df['stoch_k'], label='Stoch %K', color='orange')
plt.plot(df['timestamp'], df['stoch_d'], label='Stoch %D', color='red')
plt.title('Stochastic Oscillator')
plt.legend()

# Wykres Volume
plt.subplot(5, 2, 6)
colors = np.where(df['close'] > df['close'].shift(1), 'green', 'red')  # zielony dla wzrostu, czerwony dla spadku
plt.bar(df['timestamp'], df['volume'], color=colors, label='Volume sell(green - buy)')
plt.title('Volume')
plt.legend()

# Wykres Fibonacci Retracement
plt.subplot(5, 2, 7)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.axhline(df['fib_236'].iloc[-1], linestyle='--', color='orange', label='Fib 23.6%')
plt.axhline(df['fib_382'].iloc[-1], linestyle='--', color='blue', label='Fib 38.2%')
plt.axhline(df['fib_618'].iloc[-1], linestyle='--', color='green', label='Fib 61.8%')
plt.title('Fibonacci Retracement Levels')
plt.legend()

# Wykres Ichimoku Cloud
plt.subplot(5, 2, 8)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.plot(df['timestamp'], df['ichimoku_a'], label='Ichimoku A', linestyle='--', color='green')
plt.plot(df['timestamp'], df['ichimoku_b'], label='Ichimoku B', linestyle='--', color='red')
plt.fill_between(df['timestamp'], df['ichimoku_a'], df['ichimoku_b'], color='gray', alpha=0.3, label='Ichimoku Cloud')
plt.title('Ichimoku Cloud')
plt.legend()

# Wykres Pivot Points
plt.subplot(5, 2, 9)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.axhline(df['pivot'].iloc[-1], linestyle='--', color='orange', label='Pivot')
plt.axhline(df['r1'].iloc[-1], linestyle='--', color='blue', label='R1')
plt.axhline(df['s1'].iloc[-1], linestyle='--', color='green', label='S1')
plt.axhline(df['r2'].iloc[-1], linestyle='--', color='red', label='R2')
plt.axhline(df['s2'].iloc[-1], linestyle='--', color='purple', label='S2')
plt.title('Pivot Points')
plt.legend()

plt.tight_layout()
plt.show()

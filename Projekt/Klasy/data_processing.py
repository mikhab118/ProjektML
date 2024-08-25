import ccxt
import pandas as pd
import ta

def fetch_data_in_range(symbol, timeframe, since, until):
    exchange = ccxt.binance()
    since = exchange.parse8601(since)  # Konwersja daty na milisekundy
    until = exchange.parse8601(until)  # Konwersja daty na milisekundy
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

def calculate_indicators(df, timeframe):
    if timeframe == '1h' or timeframe == '4h':
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.trend.MACD(df['close']).macd(), ta.trend.MACD(df['close']).macd_signal(), ta.trend.MACD(df['close']).macd_diff()

        if timeframe == '4h':
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)

            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()

            stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stochastic.stoch()
            df['stoch_d'] = stochastic.stoch_signal()

            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

            # Oblicz pivot points
            pivot, r1, s1, r2, s2 = pivot_points(df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1])
            df['pivot'] = pivot
            df['r1'] = r1
            df['s1'] = s1
            df['r2'] = r2
            df['s2'] = s2

            # Oblicz Fibonacci Retracement
            high_max = df['high'].max()
            low_min = df['low'].min()
            df['fib_236'], df['fib_382'], df['fib_618'] = fibonacci_retracement(high_max, low_min)

    return df

def fibonacci_retracement(high, low):
    diff = high - low
    level1 = high - 0.236 * diff
    level2 = high - 0.382 * diff
    level3 = high - 0.618 * diff
    return level1, level2, level3

def pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

# data_processing.py

import pandas as pd

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    # Obliczanie krótkiej i długiej średniej wykładniczej
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()

    # Obliczanie MACD i linii sygnałowej
    df['macd'] = short_ema - long_ema
    df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df

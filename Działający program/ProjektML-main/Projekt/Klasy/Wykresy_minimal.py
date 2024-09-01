import ccxt
import pandas as pd
import ta
import matplotlib.pyplot as plt

# Inicjalizacja obiektu Binance API
exchange = ccxt.binance()

# Parametry
symbol = 'BTC/USDT'
timeframe = '1h'  # lub '1d' dla 1-dniowych danych
since = exchange.parse8601('2023-01-01T00:00:00Z')  # start date
until = exchange.parse8601('2023-02-01T00:00:00Z')  # end date

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

# Obliczanie wskaźników technicznych (RSI, MACD)
df['rsi'] = ta.momentum.rsi(df['close'], window=14)
df['macd'], df['macd_signal'], df['macd_hist'] = ta.trend.MACD(df['close']).macd(), ta.trend.MACD(df['close']).macd_signal(), ta.trend.MACD(df['close']).macd_diff()

# Wyświetlenie pierwszych kilku wierszy
print(df.head())

# Wizualizacja danych
plt.figure(figsize=(14, 7))

# Wykres ceny zamknięcia
plt.subplot(3, 1, 1)
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.title(f'{symbol} Close Price')
plt.legend()

# Wykres RSI
plt.subplot(3, 1, 2)
plt.plot(df['timestamp'], df['rsi'], label='RSI', color='orange')
plt.axhline(30, linestyle='--', color='red')
plt.axhline(70, linestyle='--', color='red')
plt.title('RSI')
plt.legend()

# Wykres MACD
plt.subplot(3, 1, 3)
plt.plot(df['timestamp'], df['macd'], label='MACD', color='blue')
plt.plot(df['timestamp'], df['macd_signal'], label='MACD Signal', color='red')
plt.fill_between(df['timestamp'], df['macd_hist'], 0, color='gray', alpha=0.3, label='MACD Histogram')
plt.title('MACD')
plt.legend()

plt.tight_layout()
plt.show()

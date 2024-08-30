import ccxt
import pandas as pd
import ta
from datetime import datetime, timedelta

# Inicjalizacja połączenia z giełdą za pomocą ccxt
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
since = exchange.parse8601('2023-05-14T00:00:00Z')  # Przykładowa data początkowa

# Pobranie danych OHLC
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

# Przetworzenie danych na DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Obliczenie wskaźników technicznych
df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['Stochastic_RSI'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
df['MACD'] = ta.trend.MACD(df['close']).macd()
df['Volume'] = df['volume']
df['SMA'] = ta.trend.SMAIndicator(df['close'], window=14).sma_indicator()
df['EMA'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
df['Ichimoku'] = ta.trend.IchimokuIndicator(df['high'], df['low']).ichimoku_a()

# Wybór tylko 1000h do analizy
df_1000h = df.iloc[-1000:]

# Konwersja ostatniej wartości indeksu na milisekundy
start_time_next_100h = int(df_1000h.index[-1].timestamp() * 1000)

# Pobranie kolejnych 100h po 1000h do analizy cen (close price)
ohlcv_next = exchange.fetch_ohlcv(symbol, timeframe, since=start_time_next_100h, limit=100)
df_next = pd.DataFrame(ohlcv_next, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df_next['timestamp'] = pd.to_datetime(df_next['timestamp'], unit='ms')
df_next.set_index('timestamp', inplace=True)

# Wybór tylko 100h do analizy
df_100h = df_next.iloc[-100:]

# Zapisanie danych z 1000h do pliku CSV
df_1000h.to_csv('btc_usdt_1000h.csv')

# Zapisanie cen zamknięcia z kolejnych 100h do pliku CSV z poprawnym formatowaniem
df_100h.to_csv('btc_usdt_next_100h.csv', columns=['open','close'])

print("Dane zostały zapisane do plików 'btc_usdt_1000h.csv' oraz 'btc_usdt_next_100h.csv'")

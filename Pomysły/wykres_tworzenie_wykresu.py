import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from datetime import datetime
import pandas as pd

# Konfiguracja API Binance
exchange = ccxt.binance()

# Symbol handlowy
symbol = 'BTC/USDT'

# Ustawienie interwału czasowego na 1 godzinę
timeframe = '1h'

def fetch_data():
    # Pobierz dane OHLCV (Open, High, Low, Close, Volume)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

def plot_candlestick(data):
    plt.clf()
    plt.title(f'{symbol} - {timeframe} Candlestick Chart')
    
    width = 0.03  # Szerokość korpusu świecy
    line_width = 1  # Szerokość knotów

    # Rysowanie świec
    for idx, row in data.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Rysowanie knota w kolorze świecy
        plt.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], color=color, lw=line_width)  # Knot
        # Rysowanie korpusu świecy
        plt.bar(row['timestamp'], row['close'] - row['open'], width=width, 
                bottom=min(row['open'], row['close']), color=color)

    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Price (USDT)')
    plt.grid(True)
    plt.tight_layout()

def update(frame):
    data = fetch_data()
    plot_candlestick(data)

# Konfiguracja wykresu
plt.figure(figsize=(10, 5))

# Animacja wykresu, aktualizowana co godzinę
ani = FuncAnimation(plt.gcf(), update, interval=3600000)  # Aktualizacja co godzinę (3600000 ms)

plt.show()

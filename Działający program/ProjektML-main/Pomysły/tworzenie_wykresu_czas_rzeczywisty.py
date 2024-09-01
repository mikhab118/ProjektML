import ccxt
import matplotlib.pyplot as plt
import time

# Wybór giełdy i pary walutowej
exchange = ccxt.binance()  # Możesz wybrać inną giełdę
symbol = 'BTC/USDT'

# Lista do przechowywania danych
timestamps = []
prices = []

# Ustawienie wykresu
plt.ion()  # Włączenie interaktywnego trybu
fig, ax = plt.subplots()

while True:
    # Pobranie danych o aktualnej cenie
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    timestamp = ticker['timestamp']

    # Aktualizacja danych
    timestamps.append(timestamp)
    prices.append(price)

    # Wyczyszczenie wykresu i jego aktualizacja
    ax.clear()
    ax.plot(timestamps, prices)
    ax.set_xlabel('Czas')
    ax.set_ylabel('Cena (USDT)')
    ax.set_title(f'Cena {symbol} na żywo')

    plt.draw()
    plt.pause(1)  # Odświeżanie co 1 sekundę

    # Dodanie opóźnienia, aby nie przeciążyć API
    time.sleep(1)

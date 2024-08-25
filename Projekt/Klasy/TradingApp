import ccxt
import pandas as pd
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from agent import TradingAgent
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_processing import fetch_data_in_range

# Pobranie i przetworzenie danych
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
since = '2023-01-01T00:00:00Z'
until = '2023-02-01T00:00:00Z'

# Pobieranie danych
ohlcv = fetch_data_in_range(symbol, timeframe, since, until)

# Konwersja do DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


class TradingApp:
    def __init__(self, root, data):
        self.root = root
        self.root.title("BTC/USDT Trading Simulator")

        self.balance = 10000  # Początkowy stan konta
        self.data = data  # Wczytane dane
        self.position = None  # Brak pozycji na starcie
        self.agent = TradingAgent(state_size=1, action_size=2)  # 1 cecha (kurs), 2 akcje (long, short)
        self.optimizer = optim.Adam(self.agent.model.parameters())
        self.criterion = nn.MSELoss()

        # Interfejs użytkownika
        self.take_profit_label = tk.Label(root, text="TAKE PROFIT")
        self.take_profit_label.pack()
        self.take_profit_entry = tk.Entry(root)
        self.take_profit_entry.pack()

        self.stop_loss_label = tk.Label(root, text="STOP LOSS")
        self.stop_loss_label.pack()
        self.stop_loss_entry = tk.Entry(root)
        self.stop_loss_entry.pack()

        self.percentage_label = tk.Label(root, text="% stanu konta")
        self.percentage_label.pack()
        self.percentage_entry = tk.Entry(root)
        self.percentage_entry.pack()

        self.long_button = tk.Button(root, text="LONG", bg='green', fg='white', command=self.long_position)
        self.long_button.pack()

        self.short_button = tk.Button(root, text="SHORT", bg='red', fg='white', command=self.short_position)
        self.short_button.pack()

        self.balance_label = tk.Label(root, text=f"STAN KONTA: {self.balance:.2f}")
        self.balance_label.pack()

        # Wykres
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.start_simulation()

    def start_simulation(self):
        self.current_index = 0
        self.update_chart()

    def update_chart(self):
        if self.current_index < len(self.data):
            # Pobieranie bieżącego wiersza danych
            current_data = self.data.iloc[self.current_index]

            # Dodanie punktu do wykresu
            self.ax.plot(self.data['timestamp'][:self.current_index], self.data['close'][:self.current_index],
                         color='blue')

            # Aktualizacja tytułu wykresu
            self.ax.set_title(f"BTC/USDT Trading Simulator")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")

            # Rysowanie wykresu
            self.canvas.draw()

            # Przejście do następnej świeczki
            self.current_index += 1

            # Uruchomienie funkcji update_chart ponownie po 200 ms (1 minuta odpowiada 0,2 sekundy)
            self.root.after(200, self.update_chart)
        else:
            print("Symulacja zakończona.")

    def long_position(self):
        self.place_order("long")

    def short_position(self):
        self.place_order("short")

    def place_order(self, direction):
        if self.position is None:
            entry_price = self.data['close'].iloc[self.current_index]
            self.position = {
                'direction': direction,
                'entry_price': entry_price,
                'take_profit': entry_price * (1 + float(self.take_profit_entry.get()) / 100),
                'stop_loss': entry_price * (1 - float(self.stop_loss_entry.get()) / 100),
                'percentage': float(self.percentage_entry.get())
            }
            print(f"Opened {direction} position at {entry_price}")

    def close_position(self):
        if self.position:
            current_price = self.data['close'].iloc[self.current_index]
            direction = self.position['direction']
            if direction == "long":
                profit_loss = (current_price - self.position['entry_price']) / self.position['entry_price']
            else:  # short
                profit_loss = (self.position['entry_price'] - current_price) / self.position['entry_price']

            self.balance += self.balance * profit_loss * (self.position['percentage'] / 100)
            print(f"Closed position with profit/loss: {profit_loss * 100:.2f}%")
            self.balance_label.config(text=f"STAN KONTA: {self.balance:.2f}")
            self.position = None

    def calculate_reward(self):
        if self.position:
            current_price = self.data['close'].iloc[self.current_index]
            direction = self.position['direction']
            if direction == "long":
                profit_loss = (current_price - self.position['entry_price']) / self.position['entry_price']
            else:  # short
                profit_loss = (self.position['entry_price'] - current_price) / self.position['entry_price']

            if profit_loss > 0:
                return profit_loss * 100  # Nagroda za zysk
            else:
                return profit_loss * 100  # Kara za stratę


root = tk.Tk()

# Przekaż dane do aplikacji tradingowej
app = TradingApp(root, df)
root.mainloop()

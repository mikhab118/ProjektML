import ccxt
import pandas as pd
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from lstm_agent import LSTMTradingAgent  # Import agenta z innego pliku
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_processing import fetch_data_in_range

class TradingApp:
    def __init__(self, root, data):
        self.root = root
        self.root.title("BTC/USDT Trading Simulator")

        self.balance = 10000  # Początkowy stan konta w dolarach
        self.wynik = 10000  # Początkowy wynik
        self.data = data  # Wczytane dane
        self.position = None  # Brak pozycji na starcie
        self.agent = LSTMTradingAgent(input_size=1, hidden_size=50, output_size=2)  # Agent

        # Jeśli istnieje plik modelu, wczytaj go
        model_filepath = 'agent_model.pth'
        if os.path.exists(model_filepath):
            self.agent.load_model(model_filepath)
            print("Model załadowany z pliku:", model_filepath)
        else:
            print("Brak modelu do załadowania, agent zaczyna naukę od zera.")

        self.optimizer = optim.Adam(self.agent.parameters())
        self.criterion = nn.MSELoss()

        # Interfejs użytkownika
        self.take_profit_label = tk.Label(root, text="TAKE PROFIT ($)")
        self.take_profit_label.pack()
        self.take_profit_entry = tk.Entry(root)
        self.take_profit_entry.pack()

        self.stop_loss_label = tk.Label(root, text="STOP LOSS ($)")
        self.stop_loss_label.pack()
        self.stop_loss_entry = tk.Entry(root)
        self.stop_loss_entry.pack()

        self.amount_label = tk.Label(root, text="Kwota inwestycji ($)")
        self.amount_label.pack()
        self.amount_entry = tk.Entry(root)
        self.amount_entry.pack()

        self.long_button = tk.Button(root, text="LONG", bg='green', fg='white', command=self.long_position)
        self.long_button.pack()

        self.short_button = tk.Button(root, text="SHORT", bg='red', fg='white', command=self.short_position)
        self.short_button.pack()

        self.balance_label = tk.Label(root, text=f"STAN KONTA: ${self.balance:.2f}")
        self.balance_label.pack()

        # Wykres
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.start_simulation()

    def start_simulation(self):
        self.current_index = 0
        print("Rozpoczynanie symulacji...")
        self.update_chart()

    def update_chart(self):
        if self.current_index < len(self.data):
            # Pobieranie bieżącego wiersza danych
            current_data = self.data.iloc[self.current_index]
            print(f"Agent analizuje cenę: {current_data['close']}")

            # Dodanie punktu do wykresu
            if self.current_index > 0:
                previous_close = self.data['close'].iloc[self.current_index - 1]
                current_close = self.data['close'].iloc[self.current_index]
                color = 'green' if current_close >= previous_close else 'red'
                self.ax.plot(self.data['timestamp'][self.current_index-1:self.current_index+1],
                             self.data['close'][self.current_index-1:self.current_index+1],
                             color=color)

            # Dodanie siatki do wykresu
            self.ax.grid(True)

            # Aktualizacja tytułu wykresu
            self.ax.set_title(f"BTC/USDT Trading Simulator")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")

            # Rysowanie wykresu
            self.canvas.draw()

            # Agent podejmuje decyzje
            self.agent_act(current_data['close'])

            # Sprawdzenie, czy osiągnięto take profit lub stop loss
            if self.position:
                if self.position['direction'] == "long":
                    if current_data['close'] >= self.position['take_profit'] or current_data['close'] <= self.position['stop_loss']:
                        print("Zamykanie pozycji LONG.")
                        self.close_position(current_data['close'])
                elif self.position['direction'] == "short":
                    if current_data['close'] >= self.position['stop_loss'] or current_data['close'] <= self.position['take_profit']:
                        print("Zamykanie pozycji SHORT.")
                        self.close_position(current_data['close'])

            # Odejmowanie wyniku o 1 co aktualizację wykresu
            self.wynik -= 1
            print(f"Wynik: {self.wynik}")

            # Przejście do następnej świeczki
            self.current_index += 1

            # Uruchomienie funkcji update_chart ponownie po 1000 ms (1 minuta odpowiada 1 sekundzie)
            self.root.after(1000, self.update_chart)
        else:
            model_filepath = 'agent_model.pth'
            self.agent.save_model(model_filepath)
            print("Symulacja zakończona.")

    def agent_act(self, current_price):
        print(f"Agent analizuje cenę: {current_price}")
        action = self.agent.act(current_price)
        if action == 0:
            print("Agent wybrał: LONG")
            self.long_position()
        elif action == 1:
            print("Agent wybrał: SHORT")
            self.short_position()

    def long_position(self):
        self.place_order("long")

    def short_position(self):
        self.place_order("short")

    def place_order(self, direction):
        if self.position is None:
            entry_price = self.data['close'].iloc[self.current_index]
            investment_amount = self.balance * 0.1  # 10% stanu konta
            if investment_amount > self.balance:
                print("Nie masz wystarczającej ilości środków!")
                return
            if direction == "long":
                take_profit = entry_price * 1.03
                stop_loss = entry_price * 0.97
            elif direction == "short":
                take_profit = entry_price * 0.97
                stop_loss = entry_price * 1.03

            self.position = {
                'direction': direction,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'investment_amount': investment_amount
            }
            self.balance -= investment_amount  # Odejmowanie kwoty inwestycji od stanu konta
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")
            print(f"Opened {direction} position at {entry_price}. TP: {take_profit}. SL: {stop_loss}. Investment amount: {investment_amount}")

    def close_position(self, closing_price):
        if self.position:
            direction = self.position['direction']
            initial_balance = self.balance  # Zapamiętanie stanu konta przed zamknięciem pozycji

            if direction == "long":
                profit_loss = (closing_price - self.position['entry_price']) / self.position['entry_price']
            else:  # short
                profit_loss = (self.position['entry_price'] - closing_price) / self.position['entry_price']

            # Dodanie lub odjęcie zysku/straty
            profit_loss_amount = self.position['investment_amount'] * (1 + profit_loss)
            self.balance += profit_loss_amount
            self.wynik += profit_loss_amount - self.position['investment_amount']  # Aktualizacja wyniku
            print(f"Closed position with profit/loss: {profit_loss * 100:.2f}% - ${profit_loss_amount:.2f}")
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")
            print(f"Nowy wynik: {self.wynik}")

            # Przekazanie nagrody lub kary agentowi
            new_state = closing_price
            done = False  # Możesz ustawić True, jeśli to kończy epizod
            self.agent.reward(profit_loss_amount - self.position['investment_amount'], new_state, done)
            self.position = None


if __name__ == "__main__":
    # Pobranie i przetworzenie danych
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    since = '2024-01-01T00:00:00Z'
    until = '2024-01-11T00:00:00Z'

    # Pobieranie danych
    ohlcv = fetch_data_in_range(symbol, timeframe, since, until)

    # Konwersja do DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    root = tk.Tk()
    app = TradingApp(root, df)
    root.mainloop()

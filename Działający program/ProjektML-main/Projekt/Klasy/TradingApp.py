import warnings
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from mpmath import mpf
from torchviz import make_dot
import tkinter as tk
import ccxt
import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torch.optim as optim
import numpy as np
from torch import nn
from lstm_agent import LSTMTradingAgent
from data_processing import fetch_data_in_range
import torch
import datetime
from PIL import Image, ImageTk
import mplfinance as mpf  # upewnij się, że ten import jest na początku pliku
from lstm_agent import detect_head_and_shoulders, detect_double_top_bottom, detect_symmetrical_triangle, detect_flag, detect_wedge

import io

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def log_transaction(transaction_type, balance, profit_loss, entry_date, end_date):
    with open("transaction_log.txt", "a") as f:
        f.write(f"Typ pozycji: {transaction_type}, Balans po zamknieciu pozycji: {balance:.2f}, Profit/strata z pozycji: {profit_loss:.2f}, Data otwarcia pozycji: {entry_date}, Data zamkniecia pozycji: {end_date}, Wykres: {symbol},Przedzial czasowy: {start_date} - {end_date}, Timeframe: {timeframe}\n")



class TradingApp:
    def __init__(self, root, data, start_date, end_date):
        self.root = root
        self.root.title("BTC/USDT Trading Simulator")

        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.data = data
        self.position = None
        self.agent = LSTMTradingAgent(input_size=8, hidden_size=50, output_size=3).to(device)

        self.start_date = start_date
        self.end_date = end_date

        # Inicjalizacja current_index i days_since_last_save
        self.current_index = 0
        self.days_since_last_save = 0

        model_filepath = 'agent_model.pth'
        if os.path.exists(model_filepath):
            self.agent.load_model(model_filepath)
            print("Model załadowany z pliku:", model_filepath)
        else:
            print("Brak modelu do załadowania, agent zaczyna naukę od zera.")

        # Generowanie wizualizacji modelu zaraz po jego załadowaniu
        self.visualize_model()

        self.optimizer = optim.AdamW(self.agent.parameters())
        self.criterion = nn.MSELoss()

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

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)



        self.update_chart()


    def visualize_model(self):
        # Generowanie wizualizacji modelu DuelingDQN
        sample_input = torch.rand((1, 1, 8)).to(device)  # Przykładowe dane wejściowe
        output = self.agent.online_network(sample_input)

        dot = make_dot(output, params=dict(self.agent.online_network.named_parameters()))

        output_filepath = "network_visualization"

        try:
            print(f"Próbuję zapisać wizualizację modelu do pliku network_visualization.png...")
            dot.format = 'png'
            dot.render("network_visualization")
            print(f"Wizualizacja modelu zapisana jako network_visualization.png")
        except Exception as e:
            print(f"Nie udało się zapisać wizualizacji modelu: {e}")

    def fetch_new_data(self, start_date, end_date):
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        timeframe = '1h'

        since = start_date.isoformat()
        until = end_date.isoformat()

        ohlcv = fetch_data_in_range(symbol, timeframe, since, until)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Obliczanie wskaźników technicznych
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9'].astype(float)
            df['MACD_signal'] = macd['MACDs_12_26_9'].astype(float)
            df['MACD_hist'] = macd['MACDh_12_26_9'].astype(float)
        else:
            df['MACD'] = np.nan
            df['MACD_signal'] = np.nan
            df['MACD_hist'] = np.nan

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['Stoch_k'] = stoch['STOCHk_14_3_3'].astype(float)
            df['Stoch_d'] = stoch['STOCHd_14_3_3'].astype(float)
        else:
            df['Stoch_k'] = np.nan
            df['Stoch_d'] = np.nan

        rsi = ta.rsi(df['close'], length=14)
        if rsi is not None:
            df['RSI'] = rsi.astype(float)
        else:
            df['RSI'] = np.nan

        df['Volume_norm'] = df['volume'] / df['volume'].max()

        return df

    def show_training_progress(self):
        self.agent.plot_training_progress()

    def detect_pattern(data_subset):
        """
        Wykrywa formacje świecowe i zwraca identyfikator formacji.
        0 = brak formacji, 1 = formacja głowy z ramionami, 2 = formacja podwójnego szczytu/dna itd.
        """
        # Przykład wykrywania prostej formacji "głowa z ramionami"
        if len(data_subset) >= 5:
            middle = len(data_subset) // 2
            left = data_subset['close'].iloc[:middle]
            right = data_subset['close'].iloc[middle + 1:]
            head = data_subset['close'].iloc[middle]

            if left.max() < head and right.max() < head:
                return 1  # "Głowa z ramionami"
        return 0  # Brak formacji

    def update_chart(self):
        if self.current_index < len(self.data):
            current_data = self.data.iloc[self.current_index]

            # Ustawienie okna na jeden tydzień (168 godzin)
            window_size = 168
            if self.current_index >= window_size:
                moving_average = np.mean(self.data['close'].iloc[self.current_index - window_size:self.current_index])
                data_subset = self.data.iloc[self.current_index - window_size:self.current_index].copy()
                data_subset.index = data_subset['timestamp']
            else:
                moving_average = current_data['close']
                data_subset = self.data.iloc[:self.current_index].copy()
                data_subset.index = data_subset['timestamp']

            volume = current_data['volume']
            rsi = float(current_data['RSI'])
            macd = float(current_data['MACD'])
            macd_signal = float(current_data['MACD_signal'])
            stoch_k = float(current_data['Stoch_k'])
            stoch_d = float(current_data['Stoch_d'])

            # Wykrywanie formacji, jeśli data_subset ma wystarczającą liczbę wierszy
            if len(data_subset) >= 5:  # Dopasuj liczbę do minimalnej liczby wierszy potrzebnej do wykrycia formacji
                data_subset = detect_head_and_shoulders(data_subset)
                data_subset = detect_double_top_bottom(data_subset)
                data_subset = detect_symmetrical_triangle(data_subset)
                data_subset = detect_flag(data_subset)
                data_subset = detect_wedge(data_subset)

                # Dodanie wykrywania formacji z danych świecowych z filtrowaniem NaN
                pattern_info = {
                    'head_and_shoulders': data_subset['head_and_shoulders'].iloc[-1] if pd.notna(
                        data_subset['head_and_shoulders'].iloc[-1]) else 0,
                    'double_top_bottom': data_subset['double_top_bottom'].iloc[-1] if pd.notna(
                        data_subset['double_top_bottom'].iloc[-1]) else 0,
                    'symmetrical_triangle': data_subset['symmetrical_triangle'].iloc[-1] if pd.notna(
                        data_subset['symmetrical_triangle'].iloc[-1]) else 0,
                    'flag': data_subset['flag'].iloc[-1] if pd.notna(data_subset['flag'].iloc[-1]) else 0,
                    'wedge': data_subset['wedge'].iloc[-1] if pd.notna(data_subset['wedge'].iloc[-1]) else 0
                }
            else:
                pattern_info = {
                    'head_and_shoulders': 0,
                    'double_top_bottom': 0,
                    'symmetrical_triangle': 0,
                    'flag': 0,
                    'wedge': 0
                }

            print(f"Detected pattern ID: {pattern_info}")

            # Stan wejściowy dla agenta, rozszerzony o id wykrytej formacji
            state = np.array(
                [current_data['close'], moving_average, volume, rsi, macd, macd_signal, stoch_k, stoch_d])
            state = torch.tensor(state, dtype=torch.float32).to(device)

            print(f"Agent analizuje cenę: {current_data['close']}")

            # Konwersja dat na format wymagany przez matplotlib
            data_subset['timestamp'] = mdates.date2num(data_subset['timestamp'])

            # Wyczyść poprzedni wykres
            self.ax.clear()

            # Rysowanie wykresu świecowego z węższymi świecami
            candle_width = 0.05  # Mniejsza szerokość świecy
            for idx, row in data_subset.iterrows():
                color = 'green' if row['close'] >= row['open'] else 'red'
                # Knot świecy (low do high)
                self.ax.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], color=color)
                # Korpus świecy (open do close)
                self.ax.plot([row['timestamp'] - candle_width / 2, row['timestamp'] + candle_width / 2],
                             [row['open'], row['open']], color=color, linewidth=2)
                self.ax.plot([row['timestamp'] - candle_width / 2, row['timestamp'] + candle_width / 2],
                             [row['close'], row['close']], color=color, linewidth=2)

            # Ustawienia wykresu
            self.ax.xaxis_date()
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            self.ax.xaxis.set_major_locator(ticker.MaxNLocator(7))  # Więcej etykiet na osi czasu
            self.ax.set_title("BTC/USDT Trading Simulator")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.ax.grid(True)

            # Aktualizacja canvas
            self.canvas.draw()

            # Decyzje agenta, z rozszerzonym stanem o wzorzec wykryty z wykresu
            self.agent_act(state, pattern_info)

            # Zamykanie pozycji, jeśli warunki są spełnione
            if self.position:
                if self.position['direction'] == "long" and (
                        current_data['close'] >= self.position['take_profit'] or current_data['close'] <= self.position[
                    'stop_loss']):
                    print("Zamykanie pozycji LONG.")
                    self.close_position(current_data['close'], moving_average, volume, pattern_info)
                elif self.position['direction'] == "short" and (
                        current_data['close'] >= self.position['stop_loss'] or current_data['close'] <= self.position[
                    'take_profit']):
                    print("Zamykanie pozycji SHORT.")
                    self.close_position(current_data['close'], moving_average, volume, pattern_info)

            # Zapis modelu co 30 dni
            self.days_since_last_save += 1
            if self.days_since_last_save >= 30 * 24:
                model_filepath = 'agent_model.pth'
                self.agent.save_model(model_filepath)
                print("Model zapisany po 30 dniach.")
                self.days_since_last_save = 0

            # Przejdź do następnego indeksu
            self.current_index += 1
        else:
            # Koniec danych - reset symulacji
            print("Koniec danych - resetowanie symulacji.")
            self.current_index = 0
            self.balance = self.initial_balance
            self.position = None

            # Czyszczenie osi w przypadku resetowania
            self.ax.clear()
            self.ax.set_title("BTC/USDT Trading Simulator")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.canvas.draw()

            # Restart symulacji
            self.update_chart()

        # Automatyczna aktualizacja co 100 ms
        self.root.after(100, self.update_chart)

    def agent_act(self, state, pattern_info):

        # Inna logika agenta
        action = self.agent.act(state)
        if action == 0:
            print("Agent wybiera: LONG")
            self.long_position()
        elif action == 1:
            print("Agent wybiera: SHORT")
            self.short_position()
        else:
            print("Agent wybiera: NO-OP")

    def long_position(self):
        self.place_order("long")

    def short_position(self):
        self.place_order("short")

    def place_order(self, direction):
        if self.position is None:
            entry_price = self.data['close'].iloc[self.current_index]
            entry_date = self.data['timestamp'].iloc[self.current_index]  # Pobranie daty otwarcia pozycji

            confidence_factor = 0.5
            investment_amount = self.balance * confidence_factor

            if investment_amount > self.balance:
                print("Nie masz wystarczającej ilości środków!")
                return

            confidence = 0.7
            take_profit, stop_loss = self.agent.calculate_dynamic_tp_sl(direction, entry_price, entry_price, 0.7,
                                                                        confidence)


            self.position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_date': entry_date,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'trailing_stop_loss': entry_price * (0.98 if direction == "long" else 1.02),
                'investment_amount': investment_amount
            }
            self.balance -= investment_amount
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")
            print(
                f"Opened {direction} position at {entry_price}. TP: {take_profit}. SL: {stop_loss}. Investment amount: {investment_amount}")

    def close_position(self, closing_price, moving_average, volume, pattern_info):
        if self.position:
            direction = self.position['direction']
            entry_date = self.position['entry_date']  # Data otwarcia pozycji
            end_date = self.data['timestamp'].iloc[self.current_index]  # Pobranie daty otwarcia pozycji

            if direction == "long":
                profit_loss = (closing_price - self.position['entry_price']) / self.position['entry_price']
            else:
                profit_loss = (self.position['entry_price'] - closing_price) / self.position['entry_price']

            if direction == "long" and closing_price < self.position['trailing_stop_loss']:
                self.position['stop_loss'] = closing_price
            elif direction == "short" and closing_price > self.position['trailing_stop_loss']:
                self.position['stop_loss'] = closing_price

            profit_loss_amount = self.position['investment_amount'] * (1 + profit_loss)
            self.balance += profit_loss_amount

            log_transaction(direction, self.balance, profit_loss_amount - self.position['investment_amount'], entry_date, end_date)

            print(
                f"Closed {direction} position with profit/loss: {profit_loss * 100:.2f}% - ${profit_loss_amount - self.position['investment_amount']:.2f}")
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")

            holding_time = 5
            market_volatility = np.std(self.data['close'].iloc[self.current_index - 10:self.current_index])

            new_state = torch.tensor([closing_price, moving_average, volume], dtype=torch.float32).to(device)
            done = False
            self.agent.reward(profit_loss_amount - self.position['investment_amount'], holding_time, market_volatility,
                              new_state, done, pattern_info)

            self.position = None


if __name__ == "__main__":
    start_date = datetime.datetime(2021, 3, 1)
    end_date = datetime.datetime(2023, 4, 1)

    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    ohlcv = fetch_data_in_range(symbol, timeframe, start_date.isoformat(), end_date.isoformat())

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9'].astype(float)
    df['MACD_signal'] = macd['MACDs_12_26_9'].astype(float)
    df['MACD_hist'] = macd['MACDh_12_26_9'].astype(float)

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['Stoch_k'] = stoch['STOCHk_14_3_3'].astype(float)
    df['Stoch_d'] = stoch['STOCHd_14_3_3'].astype(float)

    df['RSI'] = ta.rsi(df['close'], length=14)
    df['Volume_norm'] = df['volume'] / df['volume'].max()

    root = tk.Tk()
    app = TradingApp(root, df, start_date, end_date)
    root.mainloop()

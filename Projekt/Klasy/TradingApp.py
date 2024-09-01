import warnings
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

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TradingApp:
    def __init__(self, root, data):
        self.root = root
        self.root.title("BTC/USDT Trading Simulator")

        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.data = data
        self.position = None
        self.agent = LSTMTradingAgent(input_size=8, hidden_size=50, output_size=3).to(device)

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

        # Dodajemy przycisk do wyświetlania wykresu strat
        self.plot_button = tk.Button(root, text="Pokaż wykres strat", command=self.show_training_progress)
        self.plot_button.pack()

        self.start_simulation()

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

    def start_simulation(self):
        self.current_index = 0
        self.balance = self.initial_balance
        self.ax.clear()
        self.agent.losses = []  # Resetowanie listy strat na początku symulacji
        print("Rozpoczynanie symulacji...")
        self.update_chart()

    def show_training_progress(self):
        self.agent.plot_training_progress()

    def update_chart(self):
        if self.current_index < len(self.data):
            current_data = self.data.iloc[self.current_index]

            window_size = 10
            if self.current_index >= window_size:
                moving_average = np.mean(self.data['close'].iloc[self.current_index - window_size:self.current_index])
            else:
                moving_average = current_data['close']

            volume = current_data['volume']

            rsi = float(current_data['RSI'])
            macd = float(current_data['MACD'])
            macd_signal = float(current_data['MACD_signal'])
            stoch_k = float(current_data['Stoch_k'])
            stoch_d = float(current_data['Stoch_d'])

            state = np.array([current_data['close'], moving_average, volume, rsi, macd, macd_signal, stoch_k, stoch_d])

            state = torch.tensor(state, dtype=torch.float32).to(device)

            print(f"Agent analizuje cenę: {current_data['close']}")

            if self.current_index > 0:
                previous_close = self.data['close'].iloc[self.current_index - 1]
                current_close = self.data['close'].iloc[self.current_index]
                color = 'green' if current_close >= previous_close else 'red'
                self.ax.plot(self.data['timestamp'][self.current_index - 1:self.current_index + 1],
                             self.data['close'][self.current_index - 1:self.current_index + 1],
                             color=color)

            self.ax.grid(True)
            self.ax.set_title(f"BTC/USDT Trading Simulator")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.canvas.draw()

            self.agent_act(state)

            if self.position:
                if self.position['direction'] == "long":
                    if current_data['close'] >= self.position['take_profit'] or current_data['close'] <= self.position['stop_loss']:
                        print("Zamykanie pozycji LONG.")
                        self.close_position(current_data['close'], moving_average, volume)
                elif self.position['direction'] == "short":
                    if current_data['close'] >= self.position['stop_loss'] or current_data['close'] <= self.position['take_profit']:
                        print("Zamykanie pozycji SHORT.")
                        self.close_position(current_data['close'], moving_average, volume)

            self.current_index += 1
            self.root.after(100, self.update_chart)
        else:
            model_filepath = 'agent_model.pth'
            self.agent.save_model(model_filepath)
            with open("final_balance.txt", "a") as file:
                file.write(f"{self.balance}\n")
            print("Symulacja zakończona. Zapisano wynik.")
            self.start_simulation()

    def agent_act(self, state):
        action = self.agent.act(state)
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

            confidence_factor = 0.5
            investment_amount = self.balance * confidence_factor

            if investment_amount > self.balance:
                print("Nie masz wystarczającej ilości środków!")
                return

            confidence = 0.7
            take_profit, stop_loss = self.agent.calculate_dynamic_tp_sl(direction, entry_price, entry_price, 0.7, confidence)

            self.position = {
                'direction': direction,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'trailing_stop_loss': entry_price * (0.98 if direction == "long" else 1.02),
                'investment_amount': investment_amount
            }
            self.balance -= investment_amount
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")
            print(f"Opened {direction} position at {entry_price}. TP: {take_profit}. SL: {stop_loss}. Investment amount: {investment_amount}")

    def close_position(self, closing_price, moving_average, volume):
        if self.position:
            direction = self.position['direction']

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

            print(
                f"Closed {direction} position with profit/loss: {profit_loss * 100:.2f}% - ${profit_loss_amount - self.position['investment_amount']:.2f}")
            self.balance_label.config(text=f"STAN KONTA: ${self.balance:.2f}")

            holding_time = 5
            market_volatility = np.std(self.data['close'].iloc[self.current_index - 10:self.current_index])

            new_state = torch.tensor([closing_price, moving_average, volume], dtype=torch.float32).to(device)
            done = False
            self.agent.reward(profit_loss_amount - self.position['investment_amount'], holding_time, market_volatility,
                              new_state, done)

            self.position = None

if __name__ == "__main__":
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '30m'
    since = '2022-04-01T00:00:00Z'
    until = '2022-06-01T00:00:00Z'

    ohlcv = fetch_data_in_range(symbol, timeframe, since, until)

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
    app = TradingApp(root, df)
    root.mainloop()

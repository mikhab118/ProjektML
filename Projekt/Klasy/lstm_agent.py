import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from matplotlib import pyplot as plt
from torchviz import make_dot
import os

print("Current working directory:", os.getcwd())

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.hidden_size = hidden_size

        # Wspólna warstwa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)

        # Strumień wartości stanu (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Strumień przewagi akcji (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # x powinien mieć wymiary (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(3, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(3, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]

        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class LSTMTradingAgent(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, output_size=3, memory_size=1000, batch_size=256, epsilon_start=1.0,
                 epsilon_min=0.1, epsilon_decay=0.997):
        super(LSTMTradingAgent, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory = []

        self.train_losses = []  # Lista do przechowywania strat podczas treningu

        # Dueling DQN: online network & target network
        self.online_network = DuelingDQN(input_size, hidden_size, output_size)
        self.target_network = DuelingDQN(input_size, hidden_size, output_size)

        self.optimizer = optim.AdamW(self.parameters())
        self.criterion = nn.MSELoss()

        self.target_network.load_state_dict(self.online_network.state_dict())

    def act(self, state):
        actions = [0, 1, 2]  # 0: LONG, 1: SHORT, 2: No-Op

        if random.random() < self.epsilon:
            action = random.choice(actions)
            print("Agent wybiera akcję losowo:", "LONG" if action == 0 else "SHORT" if action == 1 else "No-Op")
        else:
            self.eval()

            # Kopiowanie tensora i przeniesienie go na GPU
            state_tensor = state.clone().detach().unsqueeze(0).unsqueeze(0).to(
                next(self.online_network.parameters()).device)

            # Oblicz wartości Q dla wszystkich akcji
            with torch.no_grad():
                q_values = self.online_network(state_tensor)
                best_q_action = torch.argmax(q_values).item()

            # Symulacja wszystkich akcji
            simulated_rewards = [self.simulate_action(state, action) for action in actions]

            # Połącz decyzje Q-learning z symulacją
            if torch.max(q_values).item() < 0.3 or simulated_rewards[best_q_action] < 0:
                action = np.argmax(simulated_rewards)
            else:
                action = best_q_action

            print("Agent wybiera akcję na podstawie Dueling DQN z Q-values i symulacji:",
                  "LONG" if action == 0 else "SHORT" if action == 1 else "No-Op")

            self.train()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.last_state = state
        self.last_action = action
        return action

    def calculate_dynamic_tp_sl(self, direction, current_price, moving_average, volume, agent_confidence):
        if len(self.replay_memory) < 2:
            print("Za mało danych w replay_memory do obliczenia zmienności.")
            return current_price * 1.03, current_price * 0.97  # Ustawienie domyślnych wartości

        recent_closes = [state[0].cpu().numpy() for state, _, _, _, _ in self.replay_memory[-10:]]
        market_volatility = np.std(recent_closes)

        if np.isnan(market_volatility) or market_volatility == 0:
            print("Nieprawidłowe obliczenie zmienności, ustawienie domyślnych wartości.")
            return current_price * 1.03, current_price * 0.97

        base_tp_sl_distance = market_volatility * agent_confidence

        if direction == "long":
            take_profit = current_price + base_tp_sl_distance
            stop_loss = current_price - base_tp_sl_distance
        elif direction == "short":
            take_profit = current_price - base_tp_sl_distance
            stop_loss = current_price + base_tp_sl_distance

        if current_price > moving_average:
            take_profit *= 1.1
            stop_loss *= 0.9
        elif current_price < moving_average:
            take_profit *= 0.9
            stop_loss *= 1.1

        return take_profit, stop_loss

    def simulate_action(self, state, action):
        # Zmienność rynku i podstawowe wskaźniki techniczne
        price = state[0].item()  # Konwersja tensora do pojedynczej wartości
        moving_average = state[1].item()
        volume = state[2].item()
        rsi = state[3].item()
        macd = state[4].item()

        # Zmienność rynkowa
        if len(self.replay_memory) >= 2:
            market_volatility = np.std(
                [s[0].cpu().numpy() if isinstance(s[0], torch.Tensor) else s[0] for s in self.replay_memory[-10:]])
        else:
            market_volatility = 0  # lub jakaś domyślna wartość

        # Ustal podstawową zmianę ceny na podstawie działania
        if action == 0:  # LONG
            price_change = price * (1 + market_volatility)
        elif action == 1:  # SHORT
            price_change = price * (1 - market_volatility)
        else:  # No-Op
            price_change = price

        # Wykorzystanie wskaźników technicznych do dostosowania prognozowanej zmiany ceny
        if rsi > 70:
            price_change *= 0.98 if action == 0 else 1.02
        elif rsi < 30:
            price_change *= 1.02 if action == 0 else 0.98

        if macd > 0:
            price_change *= 1.01 if action == 0 else 0.99
        elif macd < 0:
            price_change *= 0.99 if action == 0 else 1.01

        if volume > np.mean(
                [s[2].cpu().numpy() if isinstance(s[2], torch.Tensor) else s[2] for s in self.replay_memory[-10:]]):
            price_change *= 1.02 if action == 0 else 0.98

        # Obliczenie zysku/straty na podstawie symulowanej ceny
        if action == 0:  # LONG
            profit_loss = (price_change - price) / price
        elif action == 1:  # SHORT
            profit_loss = (price - price_change) / price
        else:  # No-Op
            profit_loss = 0

        return profit_loss

    def reward(self, profit_loss, holding_time, market_volatility, new_state, done):
        if profit_loss > 0:
            reward_value = profit_loss ** 2
        else:
            reward_value = profit_loss

        if holding_time > 10:
            if profit_loss < 0:
                reward_value -= holding_time * 0.02
            else:
                reward_value -= holding_time * 0.01

        if market_volatility > 0.05:
            if profit_loss > 0:
                reward_value += market_volatility * 0.1
            else:
                reward_value -= market_volatility * 0.05

        if profit_loss > 100:
            reward_value += 50

        if profit_loss < 0:
            reward_value += profit_loss * 1.5

        self.replay_memory.append((self.last_state, self.last_action, reward_value, new_state, done))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)

        self.train_agent()

    def train_agent(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        next_states = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        self.optimizer.zero_grad()
        q_values = self.online_network(states)
        q_target = q_values.clone().detach()

        with torch.no_grad():
            next_q_values_online = self.online_network(next_states)
            best_next_actions = torch.argmax(next_q_values_online, dim=1)
            next_q_values_target = self.target_network(next_states)
            next_q_values_target_selected = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

        for i in range(len(batch)):
            q_target[i, actions[i]] = rewards[i] + (1 - dones[i]) * next_q_values_target_selected[i]

        loss = self.criterion(q_values, q_target)
        loss.backward()
        self.optimizer.step()

        # Zapisanie strat
        self.losses.append(loss.item())

        self.target_network.load_state_dict(self.online_network.state_dict())

    def plot_training_progress(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Over Time")
        plt.show()

    def visualize_model(self, input_size):
        from torchviz import make_dot

        dummy_input = torch.randn(1, 1, input_size).to(next(self.online_network.parameters()).device)
        model_output = self.online_network(dummy_input)
        graph = make_dot(model_output, params=dict(self.online_network.named_parameters()))
        graph.render("network_visualization", format="png")
        print("Wizualizacja modelu zapisana jako 'network_visualization.png'")

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import os

print("Current working directory:", os.getcwd())

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.hidden_size = hidden_size

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.2)

        # Value Stream with dropout
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage Stream with dropout
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
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
    def __init__(self, input_size=8, hidden_size=100, output_size=3, memory_size=5000, batch_size=2048,
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        super(LSTMTradingAgent, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.replay_memory = []
        self.combined_reward = 0
        self.steps = 0

        self.train_losses = []  # Lista do przechowywania strat podczas treningu

        # Dueling DQN: online network & target network
        self.online_network = DuelingDQN(input_size, hidden_size, output_size)
        self.target_network = DuelingDQN(input_size, hidden_size, output_size)

        # Ustawienie learning rate dla optymalizatora
        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.target_network.load_state_dict(self.online_network.state_dict())

        # Inicjalizacja modelu XGBoost
        self.xgb_model = None
        self.train_xgb_model()

    def train_xgb_model(self, retrain_interval=10):
        if len(self.replay_memory) < self.memory_size // 2:
            print(f"Zbyt mało danych w replay_memory ({len(self.replay_memory)}) do trenowania XGBoost.")
            return

        if self.steps % retrain_interval != 0:
            return

        print("Trenowanie modelu XGBoost z GPU...")

        data = np.array(self.replay_memory)
        states = np.array([x[0].cpu().numpy() for x in data])
        rewards = np.array([x[2] for x in data])

        if np.isnan(states).any() or np.isnan(rewards).any():
            print("Warning: Detected NaN in training data, skipping XGBoost training.")
            return

        X_train, X_test, y_train, y_test = train_test_split(states, rewards, test_size=0.2, random_state=42)
        self.xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist')

        self.xgb_model.fit(X_train, y_train)
        predictions = self.xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'XGBoost model retrained. Accuracy: {accuracy * 100:.2f}%')

    def act(self, state):
        actions = [0, 1, 2]  # 0: LONG, 1: SHORT, 2: No-Op
        self.steps += 1

        # Zmniejszanie epsilon, aby szybciej przejść od eksploracji do eksploatacji
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Kopiowanie tensora i przeniesienie go na GPU
        state_tensor = state.clone().detach().unsqueeze(0).unsqueeze(0).to(
            next(self.online_network.parameters()).device)

        # Oblicz wartości Q dla wszystkich akcji
        with torch.no_grad():
            q_values = self.online_network(state_tensor)

            # Wartość Q dla No-Op ustalana na 0
            q_values[0, 2] = 0  # Ustawienie No-Op na wartość 0

            long_q_value = q_values[0, 0].item()  # Q-value dla Long
            short_q_value = q_values[0, 1].item()  # Q-value dla Short

            # Logowanie wartości Q
            print(f"Q-values (Long: {long_q_value}, Short: {short_q_value}, No-Op: 0)")

        # Warunki decyzyjne agenta
        if long_q_value > 0 and short_q_value > 0:
            # Wybierz tę akcję, która ma większą wartość Q
            action = 0 if long_q_value > short_q_value else 1
            print(f"Obie akcje są korzystne, agent wybiera: {'Long' if action == 0 else 'Short'}")
        elif long_q_value > 0:
            action = 0  # Long
            print("Agent wybiera akcję: Long")
        elif short_q_value > 0:
            action = 1  # Short
            print("Agent wybiera akcję: Short")
        else:
            action = 2  # No-Op
            print("Żadna akcja nie jest opłacalna, agent wybiera: No-Op")

        self.last_state = state
        self.last_action = action
        return action

    def reward(self, profit_loss, holding_time, market_volatility, new_state, done):
        reward_value = 0

        # Logowanie nagród i ich modyfikacji
        print(f"Calculating reward for profit/loss: {profit_loss:.4f}, holding_time: {holding_time}, market_volatility: {market_volatility}")

        if profit_loss > 0:
            if profit_loss > 0.01:
                reward_value = profit_loss * 2
            else:
                reward_value = profit_loss
            if holding_time <= 5:
                reward_value += profit_loss * 3
        else:
            reward_value = profit_loss * 2

        if market_volatility > 0.05 and profit_loss < 0:
            reward_value -= market_volatility * 0.5

        if profit_loss == 0 and not done:
            if len(new_state) > 3:
                if new_state[3].item() < 30 or new_state[3].item() > 70:
                    reward_value -= 1.5
            else:
                print("Za mało danych w `new_state` do oceny RSI, pomijam tę część nagrody.")

        if done and profit_loss > 0:
            reward_value += profit_loss * 2
        elif done and profit_loss < 0:
            reward_value -= abs(profit_loss) * 2

        if self.steps % 10 == 0:
            reward_value -= 1

        self.combined_reward += reward_value
        print(f"Final reward for this action: {reward_value}, combined reward: {self.combined_reward}")

        self.replay_memory.append((self.last_state, self.last_action, self.combined_reward, new_state, done))
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

        self.train_losses.append(loss.item())
        print(f"Loss after training: {loss.item()}")

        # Aktualizacja sieci docelowej po każdym trenowaniu
        self.target_network.load_state_dict(self.online_network.state_dict())

    def calculate_expected_profit(self, action, current_price, moving_average, volume, volatility):
        """
        Ocena przewidywanych zysków lub strat dla danej akcji (LONG/SHORT).
        Zwraca wartość przewidywanego zysku/straty dla akcji.
        """
        if action == 0:  # LONG
            expected_price_change = current_price * (1 + volatility)
            expected_profit = (expected_price_change - current_price) / current_price
        elif action == 1:  # SHORT
            expected_price_change = current_price * (1 - volatility)
            expected_profit = (current_price - expected_price_change) / current_price
        else:
            expected_profit = 0  # No-Op doesn't generate profit or loss

        if volume > moving_average:
            expected_profit *= 1.1
        else:
            expected_profit *= 0.9

        return expected_profit

    def calculate_dynamic_tp_sl(self, direction, current_price, moving_average, volume, agent_confidence):
        global take_profit, stop_loss
        if len(self.replay_memory) < 2:
            print("Za mało danych w replay_memory do obliczenia zmienności.")
            return current_price * 1.03, current_price * 0.97

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

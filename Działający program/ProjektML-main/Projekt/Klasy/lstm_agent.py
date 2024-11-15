import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import mplfinance as mpf
import mplfinance as mpf  # upewnij się, że ten import jest na początku pliku

print("Current working directory:", os.getcwd())



# Klasa DuelingDQN
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


# Funkcje wykrywające formacje z dodanym logowaniem
# Dla detekcji formacji Head and Shoulders
# Funkcje wykrywające formacje z dodanym logowaniem
# Dla detekcji formacji Head and Shoulders
def detect_head_and_shoulders(df, window=10):
    # Ustawienie domyślnej wartości 0 dla całej kolumny na początku, aby uniknąć `nan`
    df['head_and_shoulders'] = 0

    # Sprawdzamy wszystkie wiersze, zachowując bezpieczny margines dla `window` po obu stronach
    for i in range(window, len(df) - window):
        # Bezpieczne indeksowanie: upewniamy się, że zakresy są prawidłowe
        if i - window < 0 or i + window >= len(df):
            continue

        # Pobranie lewego ramienia, głowy i prawego ramienia
        left_shoulder = df['close'].iloc[i - window:i].max()
        head = df['close'].iloc[i]
        right_shoulder = df['close'].iloc[i + 1:i + window].max()

        # Logika wykrywania "head and shoulders"
        if left_shoulder < head and right_shoulder < head and abs(left_shoulder - right_shoulder) < 0.02 * head:
            df.at[i, 'head_and_shoulders'] = 1


    return df


def detect_double_top_bottom(df, window=5):
    df['double_top_bottom'] = 0
    for i in range(window, len(df) - window):
        # Zabezpieczenie przed wyjściem poza indeksy
        if i - window < 0 or i + window >= len(df):
            continue

        # Double Top
        if (df['close'].iloc[i] > df['close'].iloc[i - window] and
                df['close'].iloc[i] > df['close'].iloc[i + window] and
                abs(df['close'].iloc[i] - df['close'].iloc[i + window]) < 0.02 * df['close'].iloc[i]):
            df.at[i, 'double_top_bottom'] = 1  # Double Top


        # Double Bottom
        elif (df['close'].iloc[i] < df['close'].iloc[i - window] and
              df['close'].iloc[i] < df['close'].iloc[i + window] and
              abs(df['close'].iloc[i] - df['close'].iloc[i + window]) < 0.02 * df['close'].iloc[i]):
            df.at[i, 'double_top_bottom'] = 2  # Double Bottom


    return df


def detect_symmetrical_triangle(df, window=10):
    df['symmetrical_triangle'] = 0
    for i in range(window, len(df) - window):
        # Zabezpieczenie przed wyjściem poza indeksy
        if i - window < 0 or i + window >= len(df):
            continue

        highs = df['high'].iloc[i - window:i + window]
        lows = df['low'].iloc[i - window:i + window]

        if abs(highs.max() - lows.min()) < 0.02 * df['close'].iloc[i]:  # Szerokość kanału poniżej 2%
            df.at[i, 'symmetrical_triangle'] = 1


    return df


def detect_flag(df, window=5):
    df['flag'] = 0
    for i in range(window, len(df) - window):
        # Zabezpieczenie przed wyjściem poza indeksy
        if i - window < 0 or i + window >= len(df):
            continue

        highs = df['high'].iloc[i - window:i + window]
        lows = df['low'].iloc[i - window:i + window]

        if abs(highs.max() - lows.min()) < 0.02 * df['close'].iloc[i]:  # Kanał poniżej 2%
            df.at[i, 'flag'] = 1


    return df


def detect_wedge(df, window=10):
    df['wedge'] = 0
    for i in range(window, len(df) - window):
        # Zabezpieczenie przed wyjściem poza indeksy
        if i - window < 0 or i + window >= len(df):
            continue

        highs = df['high'].iloc[i - window:i + window]
        lows = df['low'].iloc[i - window:i + window]

        if abs(highs.max() - lows.min()) < 0.02 * df['close'].iloc[i]:  # Kanał poniżej 2%
            df.at[i, 'wedge'] = 1


    return df


# Funkcja do generowania przejrzystych wykresów
def generate_detailed_candlestick_image(data, index, window=30):
    if index < window:
        start = 0
    else:
        start = index - window
    data_subset = data.iloc[start:index].copy()
    data_subset.index = data_subset['timestamp']

    # Dodajemy linie trendu
    ap = [
        mpf.make_addplot(data_subset['close'].rolling(window=5).mean(), color="blue", linestyle="--"),
        mpf.make_addplot(data_subset['close'].rolling(window=10).mean(), color="red", linestyle=":")
    ]
    fig, ax = plt.subplots()
    mpf.plot(data_subset, type='candle', style='charles', addplot=ap, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# CNN do wykrywania formacji
class ChartPatternCNN(nn.Module):
    def __init__(self):
        super(ChartPatternCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 klas: brak, głowa i ramiona, podwójny szczyt/dno, trójkąt, flaga, klin

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Klasa agenta uwzględniająca formacje wykresów
class LSTMTradingAgent(nn.Module):
    def __init__(self, input_size=8, hidden_size=100, output_size=3, memory_size=10000, batch_size=4096,
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

        self.train_losses = []

        self.online_network = DuelingDQN(input_size, hidden_size, output_size)
        self.target_network = DuelingDQN(input_size, hidden_size, output_size)

        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.target_network.load_state_dict(self.online_network.state_dict())

        self.cnn_model = ChartPatternCNN().to(next(self.online_network.parameters()).device)
        self.cnn_optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.0001)
        self.criterion_cnn = nn.CrossEntropyLoss()

    def extract_pattern_features(self, image):
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(next(self.cnn_model.parameters()).device)
        pattern_prediction = self.cnn_model(image_tensor)
        return pattern_prediction.argmax(dim=1).item()

    def agent_act(self, state, index, data):
        candlestick_image = generate_detailed_candlestick_image(data, index)
        pattern_feature = self.extract_pattern_features(candlestick_image)
        combined_state = torch.cat([state, torch.tensor([pattern_feature]).float()])
        action = self.act(combined_state)
        return action

    def act(self, state):
        actions = [0, 1, 2]
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
            q_values[0, 2] = 0

        long_q_value = q_values[0, 0].item()
        short_q_value = q_values[0, 1].item()

        # Logowanie wartości Q

        if long_q_value > 0 and short_q_value > 0:
            action = 0 if long_q_value > short_q_value else 1
        elif long_q_value > 0:
            action = 0
        elif short_q_value > 0:
            action = 1
        else:
            action = 2

        self.last_state = state
        self.last_action = action
        return action

    def reward(self, profit_loss, holding_time, market_volatility, new_state, done, pattern_info=None):
        reward_value = 0
        print(
            f"Calculating reward for profit/loss: {profit_loss:.4f}, holding_time: {holding_time}, market_volatility: {market_volatility}")

        if profit_loss > 0:
            reward_value = profit_loss * 2 if profit_loss > 0.01 else profit_loss
            if holding_time <= 5:
                reward_value += profit_loss * 3
        else:
            reward_value = profit_loss * 2

        if self.last_action == 1 and profit_loss > 0:  # Udane short
            reward_value += profit_loss * 2  # Podwój nagrodę za udane short
        elif self.last_action == 0 and profit_loss < 0:  # Błędne long
            reward_value -= abs(profit_loss) * 2  # Podwój karę za błędne long

            # Kary za błędne decyzje i nagrody za poprawne decyzje w oparciu o formacje
        if pattern_info:
            if pattern_info['head_and_shoulders'] == 1 and self.last_action == 1:  # SHORT na Head & Shoulders
                reward_value += 2  # Nagroda za właściwą reakcję na formację
                print("Reward: Correct SHORT on Head and Shoulders pattern")
            elif pattern_info['double_top_bottom'] == 1 and self.last_action == 1:  # SHORT na Double Top
                reward_value += 2
                print("Reward: Correct SHORT on Double Top pattern")
            elif pattern_info['double_top_bottom'] == 2 and self.last_action == 0:  # LONG na Double Bottom
                reward_value += 2
                print("Reward: Correct LONG on Double Bottom pattern")
            else:
                reward_value -= 1  # Kara za zignorowanie formacji lub złą reakcję
                print("Penalty: Ignored or incorrect reaction to pattern")


        if market_volatility > 0.05 and profit_loss < 0:
            reward_value -= market_volatility * 0.5

        if profit_loss == 0 and not done and len(new_state) > 3:
            if new_state[3].item() < 30 or new_state[3].item() > 70:
                reward_value -= 1.5

        if done:
            reward_value += profit_loss * 2 if profit_loss > 0 else -abs(profit_loss) * 2

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

        self.target_network.load_state_dict(self.online_network.state_dict())

    def calculate_dynamic_tp_sl(self, direction, current_price, moving_average, volume, agent_confidence):
        if len(self.replay_memory) < 2:
            return current_price * 1.03, current_price * 0.97

        recent_closes = [state[0].cpu().numpy() for state, _, _, _, _ in self.replay_memory[-10:]]
        market_volatility = np.std(recent_closes)

        if np.isnan(market_volatility) or market_volatility == 0:
            return current_price * 1.03, current_price * 0.97

        base_tp_sl_distance = market_volatility * agent_confidence

        take_profit = current_price + base_tp_sl_distance if direction == "long" else current_price - base_tp_sl_distance
        stop_loss = current_price - base_tp_sl_distance if direction == "long" else current_price + base_tp_sl_distance

        if current_price > moving_average:
            take_profit *= 1.1
            stop_loss *= 0.9
        elif current_price < moving_average:
            take_profit *= 0.9
            stop_loss *= 1.1

        return take_profit, stop_loss

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        # Załaduj tylko dostępne parametry bez CNN, ignorując brakujące klucze dla nowego CNN
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Sprawdź, czy są jakieś brakujące klucze (powinny dotyczyć CNN)
        if any(missing_keys):
            print("Brakujące klucze w modelu:", missing_keys)
            print("Dodawanie losowych wag do CNN.")

        # Zapisz model z pełną strukturą, aby przyszłe ładowania były już kompletne
        self.save_model(filepath)
        print("Model zapisany z pełną strukturą, w tym CNN.")

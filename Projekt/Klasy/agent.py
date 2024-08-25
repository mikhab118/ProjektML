import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # współczynnik dyskontowania dla przyszłych nagród
        self.epsilon = 1.0  # współczynnik eksploracji
        self.epsilon_min = 0.01  # minimalna wartość epsilon
        self.epsilon_decay = 0.995  # tempo zmniejszania epsilon
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Funkcja straty

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size),
            nn.Softmax(dim=-1)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # eksploracja
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return torch.argmax(act_values).item()  # eksploatacja

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

def prepare_training_data(df):
    # Tworzenie cech na podstawie wskaźników technicznych
    df['price_change'] = df['close'].pct_change()  # Zmiana procentowa ceny zamknięcia
    df['macd_signal_diff'] = df['macd'] - df['macd_signal']  # Różnica między MACD a linią sygnałową
    df['rsi'] = df['rsi'] / 100  # Normalizacja RSI

    # Usuwanie pierwszych wierszy z NaN (które mogły powstać podczas obliczeń wskaźników)
    df.dropna(inplace=True)

    # Wyodrębnienie cech i etykiet
    features = df[['price_change', 'macd', 'macd_signal_diff', 'rsi']].values
    labels = []  # W przyszłości dodamy etykiety (akcje do podjęcia)

    # Można również dodać bardziej zaawansowane cechy
    return features, labels

def train_agent(agent, data, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = data[0]  # Początkowy stan rynku
        for time in range(1, len(data)):
            action = agent.act(state)
            next_state = data[time]
            reward = 0  # Tutaj obliczysz nagrodę na podstawie działania
            done = time == len(data) - 1  # Sprawdzenie, czy to ostatnia iteracja
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

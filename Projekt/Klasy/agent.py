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
        self.gamma = 0.95  # Współczynnik dyskontowania dla przyszłych nagród
        self.epsilon = 1.0  # Współczynnik eksploracji
        self.epsilon_min = 0.01  # Minimalna wartość epsilon
        self.epsilon_decay = 0.995  # Tempo zmniejszania epsilon
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Funkcja straty

    def _build_model(self):
        # Model sieci neuronowej do prognozowania akcji
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Eksploracja
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return torch.argmax(act_values).item()  # Eksploatacja

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

            target_f = self.model(state).detach()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

    def adjust_take_profit_stop_loss(self, state):
        """
        Metoda do ustalania poziomów Take Profit i Stop Loss.
        """
        # Tutaj można dodać logikę do bardziej zaawansowanego ustalania TP/SL, np. na podstawie wskaźników technicznych
        # Na razie po prostu ustalamy je na podstawie odchylenia standardowego
        tp = state[0] * (1 + np.random.rand() * 0.05)  # Przykładowo dodajemy losowy procent do ceny zamknięcia
        sl = state[0] * (1 - np.random.rand() * 0.05)  # Odejmujemy losowy procent
        return tp, sl

    def choose_investment_amount(self, balance):
        """
        Metoda do wyboru kwoty inwestycji.
        """
        # Na przykład, agent może inwestować od 1% do 10% dostępnego balansu
        return balance * (0.01 + np.random.rand() * 0.09)

def prepare_training_data(df):
    # Tworzenie cech na podstawie wskaźników technicznych
    df['price_change'] = df['close'].pct_change()  # Zmiana procentowa ceny zamknięcia
    df['price_momentum'] = df['close'].diff()  # Momentum ceny
    df['volatility'] = df['close'].rolling(window=5).std()  # Zmienność (odchylenie standardowe)
    df['rsi'] = df['rsi'] / 100  # Normalizacja RSI

    # Usuwanie pierwszych wierszy z NaN (które mogły powstać podczas obliczeń wskaźników)
    df.dropna(inplace=True)

    # Wyodrębnienie cech i etykiet
    features = df[['price_change', 'price_momentum', 'volatility', 'rsi']].values
    return features

def calculate_reward(position, closing_price):
    """
    Funkcja do obliczenia nagrody na podstawie aktualnej pozycji i ceny zamknięcia.
    """
    if position['direction'] == "long":
        profit_loss = (closing_price - position['entry_price']) / position['entry_price']
    else:  # short
        profit_loss = (position['entry_price'] - closing_price) / position['entry_price']

    # Nagroda jest proporcjonalna do procentowego zysku lub straty
    return profit_loss * position['investment_amount']

def train_agent(agent, data, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = data[0]  # Początkowy stan rynku
        position = None

        for time in range(1, len(data)):
            action = agent.act(state)
            next_state = data[time]

            if position:
                # Jeśli pozycja jest otwarta, sprawdzamy, czy powinniśmy ją zamknąć
                reward = calculate_reward(position, next_state[0])  # next_state[0] to aktualna cena zamknięcia
                done = next_state[0] >= position['take_profit'] or next_state[0] <= position['stop_loss']
                if done:
                    # Zamknięcie pozycji
                    position = None
            else:
                # Jeśli nie ma otwartej pozycji, otwieramy nową
                reward = 0
                if action == 0:  # Long
                    tp, sl = agent.adjust_take_profit_stop_loss(next_state)
                    investment_amount = agent.choose_investment_amount(agent.balance)
                    position = {
                        'direction': 'long',
                        'entry_price': next_state[0],
                        'take_profit': tp,
                        'stop_loss': sl,
                        'investment_amount': investment_amount
                    }
                    agent.balance -= investment_amount
                elif action == 1:  # Short
                    tp, sl = agent.adjust_take_profit_stop_loss(next_state)
                    investment_amount = agent.choose_investment_amount(agent.balance)
                    position = {
                        'direction': 'short',
                        'entry_price': next_state[0],
                        'take_profit': tp,
                        'stop_loss': sl,
                        'investment_amount': investment_amount
                    }
                    agent.balance -= investment_amount

                done = False

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

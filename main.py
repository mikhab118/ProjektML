import yfinance as yf
import gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Pobieranie danych historycznych dla EUR/USD
data = yf.download('EURUSD=X', start='2023-01-01', end='2024-01-01', interval='1h')

# Wyświetlenie przykładowych danych
print(data.head())


# Tworzenie środowiska
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.current_step:self.current_step + 5].values
        return obs.flatten()

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.data) - 5:
            self.done = True

        # Symulacja zysków/strat na podstawie akcji
        reward = 0
        if action == 0:  # Buy
            reward = self.data['Close'].iloc[self.current_step] - self.data['Close'].iloc[self.current_step - 1]
        elif action == 2:  # Sell
            reward = self.data['Close'].iloc[self.current_step - 1] - self.data['Close'].iloc[self.current_step]

        self.total_reward += reward
        return self._next_observation(), reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Total Reward: {self.total_reward}')


env = TradingEnv(data)


# Tworzenie modelu
model = PPO('MlpPolicy', env, verbose=1)

# Trening modelu
model.learn(total_timesteps=10000)

# Testowanie modelu
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

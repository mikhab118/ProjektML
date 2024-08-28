import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class LSTMTradingAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, memory_size=1000, batch_size=32, epsilon_start=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        super(LSTMTradingAgent, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start  # Startowa wartość epsilon
        self.epsilon_min = epsilon_min  # Minimalna wartość epsilon
        self.epsilon_decay = epsilon_decay  # Współczynnik dekrementacji epsilonu
        self.replay_memory = []  # Lista do przechowywania doświadczeń (stan, akcja, nagroda)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def act(self, state):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor([state]).unsqueeze(0).unsqueeze(0)
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
                print("Eksploracja: agent wybiera losowe działanie:", "LONG" if action == 0 else "SHORT")
            else:
                q_values = self.forward(state)
                action = torch.argmax(q_values, dim=1).item()
                print("Eksploatacja: agent wybiera działanie na podstawie Q-values:",
                      "LONG" if action == 0 else "SHORT")

        # Dekrementacja epsilonu po każdym działaniu
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train()
        self.last_state = state
        self.last_action = action
        return action

    def reward(self, reward_value, new_state, done):
        self.replay_memory.append((self.last_state, self.last_action, reward_value, new_state, done))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)

        self.train_agent()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def train_agent(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        self.optimizer.zero_grad()
        q_values = self(states)
        q_target = q_values.clone().detach()

        next_q_values = self(next_states).max(1)[0]
        for i in range(len(batch)):
            q_target[i, actions[i]] = rewards[i] + (1 - dones[i]) * next_q_values[i]

        loss = self.criterion(q_values, q_target)
        loss.backward()
        self.optimizer.step()

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))

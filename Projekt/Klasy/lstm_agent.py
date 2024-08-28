import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.untried_actions = self.get_possible_actions(state)  # Możliwe akcje na podstawie stanu
        self.visits = 0
        self.value = 0.0

    def get_possible_actions(self, state):
        # Zwraca listę możliwych akcji na podstawie stanu
        return [0, 1]  # Przykładowe akcje: 0 = LONG, 1 = SHORT

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param=1.414):
        choices_weights = [
            (child.value / child.visits) + exploration_param * np.sqrt(np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        action = self.untried_actions.pop(0)
        next_state = self.simulate_action(action)  # Symulacja nowego stanu po wykonaniu akcji
        child_node = Node(next_state, action=action, parent=self)
        self.children.append(child_node)
        return child_node

    def simulate_action(self, action):
        # Tutaj dodaj logikę do symulacji nowego stanu po wykonaniu akcji
        # Dla uproszczenia, użyjemy tego samego stanu, ale w praktyce powinieneś tutaj zaktualizować stan
        return self.state


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

        # Double Q-learning: online network & target network
        self.online_network = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.target_network = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_online = nn.Linear(hidden_size, output_size)
        self.fc_target = nn.Linear(hidden_size, output_size)

        # Zmiana optymalizatora na AdamW
        self.optimizer = optim.AdamW(self.parameters())
        self.criterion = nn.MSELoss()

        # Synchronize the weights of the target network with the online network
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.fc_target.load_state_dict(self.fc_online.state_dict())

        # Inicjalizacja atrybutu last_action
        self.last_action = None

    def act(self, state):
        self.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            action = self.monte_carlo_tree_search(state)

        self.train()
        self.last_state = state
        self.last_action = action  # Ustawienie ostatniej akcji
        return action

    def monte_carlo_tree_search(self, state):
        root = Node(state)
        for _ in range(100):  # Liczba symulacji
            node = self.tree_policy(root)
            reward, _ = self.simulate(node.state)  # Dodano zwracanie nowego stanu
            self.backpropagate(node, reward)

        return self.best_action(root)

    def tree_policy(self, node):
        # Faza selekcji i ekspansji
        while not self.is_terminal(node):
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def is_terminal(self, node):
        # Sprawdza, czy węzeł jest węzłem terminalnym
        return len(node.untried_actions) == 0

    def simulate(self, state):
        """
        Symulacja nowego stanu na podstawie bieżącego stanu oraz wprowadzenie losowej zmienności.
        """
        # Rozpakowanie stanu
        current_price = state[0, 0, 0].item()  # Zakładamy, że cena jest pierwszym elementem w stanie
        moving_average = state[0, 0, 1].item()  # Zakładamy, że średnia krocząca jest drugim elementem
        volume = state[0, 0, 2].item()  # Zakładamy, że wolumen jest trzecim elementem

        # Ustalanie zmienności na podstawie różnicy między ceną a średnią kroczącą
        price_deviation = current_price - moving_average

        # Wprowadzenie losowej zmienności
        volatility_factor = np.random.normal(0, 1)

        # Symulacja nowej ceny: aktualna cena plus zmiana wynikająca z trendu i losowej zmienności
        simulated_price_change = price_deviation * 0.1 + volatility_factor * 0.02
        simulated_price = current_price + simulated_price_change

        # Nagroda zależna od symulowanej zmiany ceny
        if simulated_price > current_price:
            reward = 1  # Zysk dla pozycji long
        else:
            reward = -1  # Strata dla pozycji long

        # Jeśli agent wybrał short, odwracamy logikę nagrody
        if self.last_action == 1:  # 1 = short
            reward = -reward

        # Tworzymy nowy stan na podstawie symulowanych wartości
        new_state = np.array([simulated_price, moving_average, volume])

        return reward, new_state

    def backpropagate(self, node, reward):
        # Propagacja wsteczna wartości nagrody przez drzewo
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self, root):
        if not root.children:
            print("Brak dzieci, sprawdź logikę ekspansji")
            return random.choice([0, 1])  # Domyślna akcja, np. losowy wybór LONG lub SHORT

        best_child = max(root.children, key=lambda child: child.value)
        return best_child.action

    def reward(self, profit_loss, holding_time, market_volatility, new_state, done):
        """
        Dynamiczna funkcja nagrody uwzględniająca zysk/stratę, czas trzymania pozycji oraz zmienność rynku.
        """
        reward_value = profit_loss

        # Kara za długi czas trzymania pozycji
        if holding_time > 10:  # przykład
            reward_value -= holding_time * 0.01

        # Kara za zmienność rynku (większa zmienność = większe ryzyko)
        reward_value -= market_volatility * 0.05

        # Przechowywanie nagrody w pamięci
        self.replay_memory.append((self.last_state, self.last_action, reward_value, new_state, done))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)

        self.train_agent()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.online_network(x, (h0, c0))
        out = self.fc_online(out[:, -1, :])
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

        next_q_values_online = self(next_states).max(1)[0]
        next_q_values_target = self.target_network(next_states)[0].max(1)[0]

        for i in range(len(batch)):
            q_target[i, actions[i]] = rewards[i] + (1 - dones[i]) * next_q_values_target[i]

        loss = self.criterion(q_values, q_target)
        loss.backward()
        self.optimizer.step()

        # Update the target network weights
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.fc_target.load_state_dict(self.fc_online.state_dict())

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))

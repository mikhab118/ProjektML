# lstm_trading_agent.py

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

        # Inicjalizacja atrybutu last_action i trailing stop-loss
        self.last_action = None
        self.trailing_stop_loss = None

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
            print("Agent wybiera akcję losowo:", "LONG" if action == 0 else "SHORT")
        else:
            self.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                action = self.monte_carlo_tree_search(state_tensor)

                print("Agent wybiera akcję na podstawie MCTS z Q-values:", "LONG" if action == 0 else "SHORT")
            self.train()

        # Redukcja epsilon, aby stopniowo zmniejszać częstotliwość losowych wyborów
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.last_state = state
        self.last_action = action  # Ustawienie ostatniej akcji
        return action

    def monte_carlo_tree_search(self, state):
        root = Node(state)
        for _ in range(100):  # Liczba symulacji
            node = self.tree_policy(root)
            reward, new_state = self.simulate(node.state, node.action)  # Symulacja nowego stanu i wartości Q
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

    def simulate(self, state, action):
        """
        Symulacja nowego stanu na podstawie bieżącego stanu oraz użycie Q-values z Double Q-learning.
        Uwzględnienie wolumenu, średnich kroczących i trailing stop-loss.
        """
        current_price = state[0, 0, 0].item()
        moving_average = state[0, 0, 1].item()
        volume = state[0, 0, 2].item()

        # Aktualizacja trailing stop-loss
        if action == 0:  # LONG
            if self.trailing_stop_loss is None:
                self.trailing_stop_loss = current_price * 0.98  # Inicjalny poziom trailing stop-loss
            else:
                self.trailing_stop_loss = max(self.trailing_stop_loss, current_price * 0.98)
        elif action == 1:  # SHORT
            if self.trailing_stop_loss is None:
                self.trailing_stop_loss = current_price * 1.02  # Inicjalny poziom trailing stop-loss
            else:
                self.trailing_stop_loss = min(self.trailing_stop_loss, current_price * 1.02)

        # Ustalanie nagrody na podstawie Q-values i trailing stop-loss
        with torch.no_grad():
            q_values = self.fc_target(self.target_network(state)[0])
            reward = torch.max(q_values).item()  # Użycie najwyższej wartości Q jako proxy dla nagrody

            # Uwzględnienie wolumenu i średnich kroczących
            if current_price > moving_average and volume > np.mean(volume):
                reward *= 1.1  # Preferencja dla silnych sygnałów w kierunku LONG
            elif current_price < moving_average and volume > np.mean(volume):
                reward *= 0.9  # Preferencja dla silnych sygnałów w kierunku SHORT

            # Uwzględnienie trailing stop-loss
            if action == 0 and current_price < self.trailing_stop_loss:
                reward -= 1.0  # Kara za spadek poniżej trailing stop-loss w przypadku LONG
            elif action == 1 and current_price > self.trailing_stop_loss:
                reward -= 1.0  # Kara za wzrost powyżej trailing stop-loss w przypadku SHORT

        return reward, state

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
        Ulepszona dynamiczna funkcja nagrody uwzględniająca zysk/stratę, czas trzymania pozycji, zmienność rynku,
        oraz wprowadzenie nieliniowego skalowania zysków, kar za straty, oraz premiowania wysokiej skuteczności.
        """
        # Nieliniowe skalowanie zysków
        if profit_loss > 0:
            reward_value = profit_loss ** 2  # Nagroda za większe zyski (kwadratowa)
        else:
            reward_value = profit_loss  # Kara za straty pozostaje liniowa

        # Kara za długi czas trzymania pozycji, szczególnie jeśli zysk maleje
        if holding_time > 10:
            if profit_loss < 0:
                reward_value -= holding_time * 0.02  # Większa kara za stratę przy długim czasie trzymania
            else:
                reward_value -= holding_time * 0.01  # Standardowa kara za długi czas trzymania pozycji

        # Dynamiczne karanie lub premiowanie za zmienność rynku
        if market_volatility > 0.05:
            if profit_loss > 0:
                reward_value += market_volatility * 0.1  # Nagroda za zarabianie w warunkach dużej zmienności
            else:
                reward_value -= market_volatility * 0.05  # Kara za stratę w warunkach dużej zmienności

        # Dodatkowa premia za osiągnięcie wysokiego zysku
        if profit_loss > 100:  # Próg zysku
            reward_value += 50  # Dodatkowa premia za wysoką skuteczność

        # Zwiększona kara za straty, aby unikać ryzykownych decyzji
        if profit_loss < 0:
            reward_value += profit_loss * 1.5  # Zwiększenie kary za stratę

        # Przechowywanie nagrody w pamięci replay
        self.replay_memory.append((self.last_state, self.last_action, reward_value, new_state, done))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)

        self.train_agent()

    def forward(self, x):
        # Sprawdzenie, czy x jest trójwymiarowym tensorem, jakiego oczekuje LSTM
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Dodanie dodatkowego wymiaru, jeśli x jest dwuwymiarowy

        # Inicjalizacja stanów ukrytych i komórkowych
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Przekazanie danych przez LSTM
        out, _ = self.online_network(x, (h0, c0))
        out = self.fc_online(out[:, -1, :])  # Pobranie ostatniego wyjścia LSTM
        return out

    def train_agent(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Dodanie kontroli typów i rozmiarów danych
        assert all(isinstance(s, np.ndarray) or isinstance(s, torch.Tensor) for s in
                   states), "Błąd: Niewłaściwy typ danych w states"
        assert len(states) > 0, "Błąd: Pusta lista states"
        assert all(isinstance(ns, np.ndarray) or isinstance(ns, torch.Tensor) for ns in
                   next_states), "Błąd: Niewłaściwy typ danych w next_states"
        assert len(next_states) > 0, "Błąd: Pusta lista next_states"

        # Logowanie danych przed konwersją
        print(f"Konwertowanie states: {states}")
        print(f"Konwertowanie next_states: {next_states}")
        print(f"Konwertowanie actions: {actions}")
        print(f"Konwertowanie rewards: {rewards}")
        print(f"Konwertowanie dones: {dones}")

        # Konwersja numpy arrays do Tensorów
        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        next_states = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        self.optimizer.zero_grad()
        q_values = self(states)
        q_target = q_values.clone().detach()

        # Double Q-Learning
        with torch.no_grad():
            # Wybór akcji z online network
            next_q_values_online = self.fc_online(self.online_network(next_states)[0])
            best_next_actions = torch.argmax(next_q_values_online, dim=1)

            # Obliczenie wartości Q z target network dla wybranej akcji
            next_q_values_target = self.fc_target(self.target_network(next_states)[0])
            next_q_values_target_selected = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

        for i in range(len(batch)):
            q_target[i, actions[i]] = rewards[i] + (1 - dones[i]) * next_q_values_target_selected[i]

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
